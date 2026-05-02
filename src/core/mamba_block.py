# Ported from MambaECAnet — same author, pure-PyTorch selective SSM.
"""
Pure-PyTorch Mamba block — CPU/GPU compatible, no CUDA extensions required.

For production GPU environments, see implementation.md §8 for a drop-in replacement
that uses the official mamba-ssm package with hardware-efficient kernels.

── CONCEPTUAL OVERVIEW ───────────────────────────────────────────────────────
A state space model (SSM) is a sequence model built on this recurrence:

    h_t = A · h_{t-1}  +  B · u_t      ← hidden state update
    y_t = C · h_t      +  D · u_t      ← output

Think of h_t as a "memory vector" that compresses everything seen so far.
A controls how much of the previous memory survives (like a decay).
B controls how much the new input writes into memory.
C controls how memory is read out to produce output.
D is a skip/residual connection that bypasses memory entirely.

The KEY idea in Mamba (vs. older SSMs like S4) is that B, C, and Δ are
computed FROM THE INPUT at each time step.  That is what makes it "selective":
the model learns to decide per-token what to remember and what to ignore,
rather than having fixed weights everywhere.
─────────────────────────────────────────────────────────────────────────────
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    The selective state space model — the mathematical heart of Mamba.

    Dimensions used throughout:
        B   = batch size
        L   = sequence length  (1 for tabular data fed as a single token)
        D   = model dimension  (d_model, or d_inner inside MambaBlock)
        N   = SSM state size   (d_state, default 16)
        R   = dt_rank          (≈ D/16, a small bottleneck dimension)
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # dt_rank is a small bottleneck used to project Δ cheaply.
        # Large Δ = a lot of time passed -> hidden state decays more -> model forgets past
        # small Δ = barely any time passed, state persists, model remembersw the past

        #originally, Δ was fixed. With MAmba, Δ is determined by the input size

        # ceil(D/16) keeps it small while still being expressive.
        dt_rank = dt_rank or math.ceil(d_model / 16)

        # ── Input-dependent projections (the "selective" part) ──────────────
        # A SINGLE linear layer projects each input token x_t into three things:
        #   • a dt_rank-dim vector  → will become Δ_t  (time step / update rate)
        #   • a d_state-dim vector  → will become B_t  (input-to-state write gate)
        #   • a d_state-dim vector  → will become C_t  (state-to-output read gate)
        # Because these depend on x_t, the SSM can selectively attend to tokens.
        self.x_proj  = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False) #2 * d_state -> one copy for B and one for C

        # A second linear expands Δ from the bottleneck (dt_rank) back to full D.
        # This lets every one of the D channels have its own time-step.
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # ── Fixed SSM matrix A ──────────────────────────────────────────────
        # A is the state transition matrix.  Unlike B and C, A is NOT input-
        # dependent — it is a learned but fixed parameter (shared across tokens).
        # We store log(A) so we can enforce A < 0 via A = -exp(A_log), which
        # guarantees the hidden state decays over time (stable recurrence).

        """The recurrence equation:

    `    At every time step the hidden state updates as:
        h_t = A · h_{t-1} + B · u_t
        A is applied repeatedly — once per token, across potentially thousands of tokens. So after T steps, the influence of the very first input on the current state is proportional to A^T (A multiplied by itself T times).

        A must be negative because after many steps:
                A^1 = -0.5
                A^2 =  0.25
                A^3 = -0.125  ...→ 0

        If it was positive: A^1 = 2, A^2 = 4, A^3 = 8  ...→ ∞ ---> old inputs EXPLODE in influecne over time T. This is why stable recurrence requires |A| < 1
        """
        # Shape: (D, N) — each of the D channels has its own N-dim state.
        # Initialized with values 1..N (HiPPO-style: larger indices decay faster).
        A = ( # if d_state = 4: torch.arange(1, 5) → tensor([1., 2., 3., 4.])
            torch.arange(1, d_state + 1, dtype=torch.float32) #1D tensor evenly spaced integers from 1 - d_state size
            .unsqueeze(0)           # (1, N) #inserts new dimension at position 0 [1, 2, 3, 4]        shape: (4,)
                                                                                        #↓  unsqueeze(0)
                                                                                    #[[1, 2, 3, 4]]      shape: (1, 4)
            #HiPPO init: larger indices in state vector should decay faster 
            .expand(d_model, -1)    # (D, N) -> increases each dimension to d_model besides the last one
        )
        self.A_log = nn.Parameter(torch.log(A))   # learned in log space

        # D here is the skip connection weight (not batch size — confusing notation
        # from the original paper; think of it as a residual scalar per channel).
        self.D = nn.Parameter(torch.ones(d_model)) ##nn.Parameter registers this as a learnable weight/enables gradient tracking

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)

# ── Step 1: compute input-dependent parameters ──────────────────────
        # Project x to get the raw Δ, B, C for every token in the sequence.
        x_dbl = self.x_proj(x)       # (B, L, dt_rank + 2*N)

        dt_rank = self.dt_proj.in_features

        # Split the projection into the three parts along the last dimension.
        dt, B_param, C = x_dbl.split([dt_rank, self.d_state, self.d_state], dim=-1) 
        """
        Maps each token from d_model to dt_rank + 2*D_state
        if dt_Rank = 4, and d_state=16, the output has 36 numbers per token/row/record:
                    [  n1, n2, n3, n4,  |  n5...n20  |  n21...n36  ]
                    ←── dt_rank=4 ──→  ← d_state=16 → ← d_state=16 →
                            Δ                 B               C

        """
        # dt:      (B, L, dt_rank) — raw Δ values before expansion
        # B_param: (B, L, N)       — how much each input writes to state
        # C:       (B, L, N)       — how memory is projected to output

        # Expand Δ from bottleneck dim → full D, then pass through softplus.
        # softplus ensures Δ > 0 (negative time-steps make no physical sense).
        """softplus is like ReLU but ensures the output is always positive
                softplus(x) = log(1+exp(x))
                    large positive x = linear
                    large neg x = approaches 0 but never goes negative
                    at 0 : smooth curve, no sharp corner like RELU

            Softplus is used because  Δ(time-step_) must be strictly positive

        """
        dt = F.softplus(self.dt_proj(dt))   # (B, L, D)

# ── Step 2: discretize A ────────────────────────────────────────────
        # The SSM equations above are in *continuous* time.  To run them on
        # discrete tokens we use the Zero-Order Hold (ZOH) discretization:
        #   dA_t = exp(Δ_t · A)
        # A large Δ_t means "long time passed" → state decays more → model
        # forgets the past.  A small Δ_t → model holds on to recent history.

        """ A must be negative because after many steps:
                A^1 = -0.5
                A^2 =  0.25
                A^3 = -0.125  ...→ 0

        If it was positive: A^1 = 2, A^2 = 4, A^3 = 8  ...→ ∞ ---> old inputs EXPLODE in influecne over time T. This is why stable recurrence requires |A| < 1
        """
        A = -torch.exp(self.A_log)   # (D, N)  — enforce A < 0 for stability

        # ── Step 3: run the selective scan ──────────────────────────────────
        y = self._selective_scan(x, dt, A, B_param, C)

        # ── Step 4: skip connection ─────────────────────────────────────────
        # Add the direct pass-through D·x (like a residual) before returning.
        return y + self.D.unsqueeze(0).unsqueeze(0) * x   # (B, L, D)

    """ 
    Walking through selective SSM
        Level 1: What gets written to memory (B gate)
            When the token enters the SSM, B[:, t, :] controls how strongly the input writes into each of the N memory slots. This is input-dependent — computed from the token itself via x_proj. So the network learns: "for this particular pattern of traffic features, write heavily into slot 3 and barely into slot 11.

        Level 2: What gets read from memory (C gate)
            C[:, t, :] controls which memory slots contribute to the output. Also input-dependent. So the network learns: "for this token, slot 3 is relevant to read out, ignore slots 7-10.

        crucially — the raw features are already abstracted away

            By the time data reaches the SSM, it's not "packet size" or "duration" anymore. The embedding layer upstream has already transformed the 41 raw features into D=64 learned dimensions. Each of those D channels is a blend of original features the network found use

            the network might implicitly learn things like: "unusual port patterns are worth remembering across packets, but packet size fluctuations can be forgotten quickly"
        """
    

    def _selective_scan(
        self,
        u: torch.Tensor,    # input:      (B, L, D)
        dt: torch.Tensor,   # time steps: (B, L, D)  — per token, per channel -> D separate values per token (due to dt_proj expanding dt_Rank -> D, giving each of the D channels its own time step)
        A: torch.Tensor,    # decay:      (D, N)     — fixed
        B: torch.Tensor,    # write gate: (B, L, N)  — input-dependent
        C: torch.Tensor,    # read gate:  (B, L, N)  — input-dependent
    ) -> torch.Tensor:
        """
        Sequential selective scan.

        At each time step t the recurrence is:
            dA_t = exp(Δ_t · A)          ← discretized decay, shape (B, D, N)
            dB_t = Δ_t · B_t             ← discretized write gate, shape (B, D, N)
            h_t  = dA_t · h_{t-1}  +  dB_t · u_t    ← state update
            y_t  = sum_N( h_t · C_t )   ← read out, shape (B, D)

        NOTE: This loop is O(L) and straightforward to understand, but slow
        for long sequences.  The official mamba-ssm package uses a parallel
        scan for O(log L) time — same math, much faster on GPU.

        dt holds a float delta value for one specific token in one specific batch in one specific channel
            - dt_proj expands Δ from dt_rank → D, giving every one of the D channels its own time-step. So a single token doesn't have one Δ, it has D of them — one per channel.
            - different channels of the hidden state can decay at different rates for the same token. One channel might decide "remember this" (small Δ), another might decide "forget this" (large Δ), all from the same input token
            - CONTRAST: If Δ were a single float per token, the shape would be (B, L, 1) — every channel forced to use the same time-step. The paper found that per-channel Δ (shape (B, L, D)) works significantly better

        NOTE: channels are features within a single discrete row -> 41 network traffic attributes get transformed into D learned feature dimensions (channels) through the emnbedding layer before reaching SSM
        """
        B_sz, L, D = u.shape
        N = self.d_state

        # Hidden state h starts at zero — no memory at the beginning.
        # (B, D, N) 
        # B=one hidden state per row in batch, 
        # D=one state per channel - each feature dimension tracks its own history
        # N = d_state=16 "memory slots" per channel
        h = torch.zeros(B_sz, D, N, device=u.device, dtype=u.dtype)  

        ys = []
        for t in range(L):
            # ── State update ────────────────────────────────────────────────
            dt_t = dt[:, t, :] #[all B, index t, all D's] --- (B, D)  — grag token t's Δ for every batch and channel

            # Discretized A: how much of the old state survives this step.
            # Broadcast: dt_t (B, D) × A (D, N)  → (B, D, N)
            # a large Δ makes dA small: state decays fast, forget
            # a small Δ keeps dA close to 1: state persists, remember
            """
            unsqueeze(-1): shape trick
                - dt_T.shape = (B,D)
                - dt_t.unsqueeze(-1): (B,D,1)
                - a shape: (D,N) ---> result dA is (B,D,N) because oit broadcats across both D and N
                - without unsqueeze, shapes wouldnt be able to broadcast to other shapes
                - pytorch only broadcast dimensions that are equal or 1
                """
            dA = torch.exp(dt_t.unsqueeze(-1) * A)       # (B, D, N)

            # Discretized B: how strongly this token's input writes to state.
            # B_param[:, t, :] is (B, N); we unsqueeze to (B, 1, N) so it
            # broadcasts across the D channels.
            dB = dt_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (B, D, N)

            # Core SSM recurrence:
            """   
            new_h = decay · old_h  +  write · input
            u[:, t, :] is (B, D); unsqueeze(-1) → (B, D, 1) to broadcast over N.

            dA * h -> decay old memory
            dB * u[t] -> write new input to state (B,D,N)
            """

            h = dA * h + dB * u[:, t, :].unsqueeze(-1)   # (B, D, N)

            # ── Read out ────────────────────────────────────────────────────
            # Project the hidden state back to D-dim using C (the "read gate").
            # C[:, t, :] is (B, N); unsqueeze(1) → (B, 1, N) broadcasts over D.
            # Sum over N collapses the state dimension.

            """
            extracts output from hidden state using C
            C decides which part of hidden state memory to surface as output
            - h shape:                    (B, D, N)   ← current hidden state
            - C[:, t, :].unsqueeze(1):   (B, 1, N)   ← this token's read gate, unsqueeze over D
            - h * C[t]:                  (B, D, N)   ← element-wise: weight each memory slot
            .sum(-1):                  (B, D)      ← collapse N, sum across memory slots

            h[b, d, :] = [m1, m2, m3, ... m16]   ← 16 memory slots for one channel
            C[b, t, :] = [w1, w2, w3, ... w16]   ← 16 read weights
            dot product → one float               ← "what this channel remembers right now"
            """
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        # Stack time steps back into a sequence tensor.
        return torch.stack(ys, dim=1)   # (B, L, D)


class MambaBlock(nn.Module):
    """
    Full Mamba block (Gu & Dao, 2023) as used in MambaECANet.

    ── ARCHITECTURE ──────────────────────────────────────────────────────────
    The block wraps the SSM in a gated architecture (similar in spirit to
    GLU / SwiGLU used in modern transformers):

        Input (B, L, D)
            │
            ▼
        in_proj  ──→  (B, L, 2·d_inner)   ← double width, then split
            │
          split
         ┌──┴──┐
         │     │
         ▼     ▼
      x_branch  z_branch         both (B, L, d_inner)
         │           │
    Conv1D (causal)  SiLU          ← z_branch is a simple gating signal
    SiLU             │
    SelectiveSSM     │
         │           │
         └─── × ─────┘            ← element-wise multiply (the "gate")
               │
           out_proj               ← project back to original D
               │
            Output (B, L, D)

    Why the gating?  The z_branch acts like a learned filter: when z is near
    zero it suppresses the SSM output; when z is near one it passes it through.
    This makes the block more expressive than a bare SSM.

    Both branches (x and z) come from same in_proj output -> x_branch, z_branch = in_proj(input).split([d_inner, d_inner], dim=-1)
            - same input, same linear layer, split in half: 
                    - x branch goes thorugh conv1D, SILU, SSM
                    - z branch goes through SiLU, produces values roughly in (0,1)
                        - ends up being a soft mask -> each of d_inner channels gets a number deciding "how open is this gate"
                        -"Given this input token, is what the SSM found in this feature dimension actually meaningful, or should I discard it?"
                        -It adds a second layer of selectivity on top of the SSM's own Δ-based selectivity — the SSM decides what to remember, and the gate decides what to pass forward

    output = x_branch * z_branch   # element-wise multiply

    What SiLU does to z

            SiLU (also called Swish) is x * sigmoid(x), which produces values that are:

            Near 0 for negative inputs
            Near x for positive inputs
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,   # d_inner = expand × d_model (default 2× width inside block)
    ):
        super().__init__()
        self.d_inner = int(expand * d_model)   # wider internal representation

        # Projects input to TWO parallel streams of width d_inner.

        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise (groups=d_inner) causal convolution.
        # "Depthwise" means each channel is filtered independently — cheap but
        # effective at mixing local temporal context before the SSM.
        # padding=d_conv-1 then truncating to length L gives causal behaviour
        # (no future tokens leak into the past).
        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,      # pad left so output length ≥ input length
            groups=self.d_inner,     # depthwise: one filter per channel
            bias=True,
        )

        self.ssm      = SelectiveSSM(self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)

        # ── Dual projection ─────────────────────────────────────────────────
        xz = self.in_proj(x)                        # (B, L, 2·d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)   # each (B, L, d_inner)

        # ── Branch 1: Conv → SiLU → SSM ─────────────────────────────────────
        # Conv1d expects (B, C, L) but our tensor is (B, L, d_inner) — transpose.
        x_b = x_branch.transpose(1, 2)              # (B, d_inner, L)
        x_b = self.conv1d(x_b)                      # (B, d_inner, L + pad)
        x_b = x_b[..., :x_branch.shape[1]]          # trim padding → (B, d_inner, L)
        x_b = x_b.transpose(1, 2)                   # (B, L, d_inner)
        x_b = F.silu(x_b)                           # smooth non-linearity before SSM
        x_b = self.ssm(x_b)                         # selective state space model

        # ── Branch 2: SiLU gate ──────────────────────────────────────────────
        # z_branch is just activated — it will gate x_branch element-wise.
        z_b = F.silu(z_branch)                      # (B, L, d_inner)

        # ── Gated output ─────────────────────────────────────────────────────
        # The gate z_b modulates how much SSM output passes through.
        # This is the same idea as GLU (Gated Linear Unit) but with SiLU.
        return self.out_proj(x_b * z_b)             # (B, L, D)
