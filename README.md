# Dendroaspis

Code for *"Dendroaspis: Detecting Agent-Orchestrated Attacks on
High-Entropy Workstations via eBPF and Mamba"* (Powers, 2026).

A Mamba state-space sequence detector for kernel-level host intrusion
detection on Linux Tetragon/eBPF telemetry, evaluated against an
ATT&CK-aligned, **agent-driven attack campaign** on a controlled
Proxmox/LXC developer-workstation environment.

> **The dataset is released separately on Hugging Face:**
> https://huggingface.co/datasets/rypow/dendroaspis-tetragon-hids
> (`rypow/dendroaspis-tetragon-hids`, CC BY 4.0)

---

## Pipeline overview

End-to-end data flow, from kernel telemetry to the per-event AUROC numbers
reported in the paper. Each stage's *output* is the next stage's *input*.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: COLLECTION  (offline, in the LXC sandbox — code in configs/)      │
│                                                                             │
│   configs/tetragon/*.yaml  ──►  Tetragon eBPF tracing  ──►  events.json.gz  │
│   (12 kprobes across 6                                       (~140 files,   │
│    monitor policies)                                          row-per-event)│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: PARSE  (src/telemetry/)                                           │
│                                                                             │
│   tetragon_native_parser.py  →  filters (container / sentinel / process)    │
│                              →  extracts process block + kprobe args        │
│   tetragon_native_writer.py  →  writes Parquet shards (raw schema)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: BEHAVIOR ENRICHMENT  (src/processing/)                            │
│                                                                             │
│   v0_2_behavior_builder.py   →  Pass 1: global lineage walk (exec_id graph) │
│                              →  Pass 2: derive 27 columns per row           │
│                                  (action family, path/IP/port categories,   │
│                                   lineage depth + root, log-bucketed deltas)│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: FEATURE ENCODING  (src/features/)                                 │
│                                                                             │
│   v0_2_features.py  →  Tier 1–6 encoders                                    │
│                     →  33 small-int feature columns + 3 identifiers         │
│                     →  Closed vocabularies, embedding-ready dtypes          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: MODEL  (src/core/)                                                │
│                                                                             │
│   v0_2_event_encoder.py  →  per-field embedding tables → d_model=128        │
│   mamba_block.py         →  2-layer selective state-space scan              │
│   v0_2_mamba_scorer.py   →  NLL / MEM / MEM-FA heads                        │
│   v0_2_dataloader.py     →  L=128 sliding windows                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: EVAL  (scripts/eval_v0_2_mamba.py)                                │
│                                                                             │
│   per-event scores  +  labels.csv (ART intervals)                           │
│                     →  per-event AUROC + 95% CI                             │
│                     →  per-window AUROC                                     │
│                     →  per-technique AUROC (22 techniques)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

Stages 1–4 produce the released Hugging Face dataset. Stages 5–6 are what
`scripts/train_v0_2_mamba.py` and `scripts/eval_v0_2_mamba.py` execute on the
downloaded shards.

---

## How to read this codebase

If you want to evaluate the implementation (rather than just run it), read the
files in **pipeline order** — the order data flows through them. Each file is
self-contained enough to read in one sitting:

| Order | File | What to look for |
|---|---|---|
| 1 | [`configs/tetragon/*.yaml`](configs/tetragon/) | Which 12 kernel functions are hooked, and which ATT&CK techniques each one is intended to catch (header comments). |
| 2 | [`src/telemetry/tetragon_native_parser.py`](src/telemetry/tetragon_native_parser.py) | Filter chain + per-kprobe argument extraction. Module docstring describes filter ordering and the procFS-walk corner case. |
| 3 | [`src/telemetry/tetragon_native_writer.py`](src/telemetry/tetragon_native_writer.py) | Parquet schema (Groups A–H). Defines what every downstream stage consumes. |
| 4 | [`src/processing/v0_2_behavior_builder.py`](src/processing/v0_2_behavior_builder.py) | Two-pass lineage walk + 27 derived columns. The categorical vocabularies (`ACTION_FAMILY_VOCAB`, `PATH_CATEGORY_VOCAB`, …) live here. |
| 5 | [`src/features/v0_2_features.py`](src/features/v0_2_features.py) | Tier 1–6 encoders. Single source of truth for every embedding cardinality and bucket boundary. |
| 6 | [`src/core/v0_2_event_encoder.py`](src/core/v0_2_event_encoder.py) | Per-field embedding tables → projected to `d_model=128`. |
| 7 | [`src/core/mamba_block.py`](src/core/mamba_block.py) | The selective state-space scan. |
| 8 | [`src/core/v0_2_mamba_scorer.py`](src/core/v0_2_mamba_scorer.py) | The three head variants (NLL, MEM, MEM-FA). |
| 9 | [`scripts/train_v0_2_mamba.py`](scripts/train_v0_2_mamba.py) / [`scripts/eval_v0_2_mamba.py`](scripts/eval_v0_2_mamba.py) | Training + evaluation entry points. |

---

## Paper section ↔ code map

> Section numbers below mirror the paper draft; if you've revised the paper's
> structure, update this table to match before sharing.

| Paper section | Implementation |
|---|---|
| §3.1 Threat model & telemetry surface | [`configs/tetragon/`](configs/tetragon/) (the 12 kprobes + selectors) |
| §3.2 Tetragon parsing & filtering | [`src/telemetry/tetragon_native_parser.py`](src/telemetry/tetragon_native_parser.py) |
| §3.3 Behavior abstraction (action families, lineage) | [`src/processing/v0_2_behavior_builder.py`](src/processing/v0_2_behavior_builder.py) |
| §3.4 Feature encoding (Tier 1–6) | [`src/features/v0_2_features.py`](src/features/v0_2_features.py) |
| §3.5 Mamba architecture | [`src/core/mamba_block.py`](src/core/mamba_block.py), [`src/core/v0_2_event_encoder.py`](src/core/v0_2_event_encoder.py) |
| §3.6 Training objectives (NLL / MEM / MEM-FA) | [`src/core/v0_2_mamba_scorer.py`](src/core/v0_2_mamba_scorer.py) |
| §4.1 Dataset construction | Released separately — see [HF dataset card](https://huggingface.co/datasets/rypow/dendroaspis-tetragon-hids) |
| §4.2 Baselines (n-gram, IF, XGBoost) | [`src/core/v0_2_baselines/`](src/core/v0_2_baselines/) |
| §4.3 Evaluation protocol | [`scripts/eval_v0_2_mamba.py`](scripts/eval_v0_2_mamba.py) |
| §5 Results (Table 1) | Reproduce via the commands in [Reproducing the paper's results](#reproducing-the-papers-results) above |
| §5.5 Encoder × architecture ablation | `--feature-set {rich,flat}` flag in `scripts/train_v0_2_mamba.py` |
| §6 Discussion (procFS-walk handling) | Module docstring of [`tetragon_native_parser.py`](src/telemetry/tetragon_native_parser.py); see also the procfs-walk note in [`v0_2_behavior_builder.py`](src/processing/v0_2_behavior_builder.py) |

---

## Repository layout

```
src/                        Python package — pipeline + model code
├── telemetry/              Tetragon JSON-Lines parser + raw-parquet writer
├── processing/             Behavior builder (lineage walk + categorical features)
├── features/               Encoder feature constants + integer-code maps
└── core/
    ├── mamba_block.py          Selective state-space scan (PyTorch)
    ├── v0_2_event_encoder.py   Per-field embedding encoder
    ├── v0_2_mamba_scorer.py    NLL + MEM + field-aware MEM heads
    ├── v0_2_dataloader.py
    └── v0_2_baselines/         n-gram, XGBoost, IF (rich + flat)

scripts/                    Train + eval CLIs (Mamba and baselines)
configs/tetragon/           Tetragon tracing policies used during collection
requirements.txt            Python dependencies
LICENSE                     MIT (code only; dataset is CC BY 4.0 on HF)
```

### What each module does (pipeline order)

The tree above is in filesystem order; the list below is in **pipeline order**.
Each entry names what the module *consumes* and *produces*, so the handoffs
between stages are explicit.

1. **`configs/tetragon/`** — six TracingPolicy YAML files declaring the 12
   monitored kprobes. *Produces:* `events.json.gz` files when deployed
   inside the LXC sandbox.

2. **`src/telemetry/`** —
   - `tetragon_native_parser.py`: streams `events.json.gz`, applies three
     filters (target container by PID-namespace inum; sentinel-interval
     dropout; missing-process-binary dropout), then dispatches per-kprobe
     to extract typed args.
   - `tetragon_native_writer.py`: defines the raw Parquet schema and
     buffers parser output into shards.

   *Consumes:* Tetragon JSON-Lines.
   *Produces:* raw Parquet (per-event row, ~70 columns across Groups A–H).

3. **`src/processing/v0_2_behavior_builder.py`** — two-pass enrichment.
   Pass 1 builds a global `exec_id → (parent, basename)` map across the
   whole corpus (necessary because the parser only sees one `events.json`
   file at a time and can't resolve cross-file parents). Pass 2 derives 27
   columns per row: 11 model-feature `f_*` columns and 16 side columns for
   analysis and debugging. Categorical vocabularies and bucket boundaries
   are defined here.

   *Consumes:* raw Parquet from §2.
   *Produces:* enriched Parquet (raw schema + 27 columns).

4. **`src/features/v0_2_features.py`** — applies Tier 1–6 encoders to the
   enriched Parquet. Every output column is a small int (`uint8`/`uint16`)
   in a closed vocabulary with a reserved OOV slot, so embedding tables
   can be sized at design time and never `IndexError` at eval. Single
   source of truth for embedding cardinalities — encoders pull from here,
   the model pulls from here, audit notebooks pull from here.

   *Consumes:* enriched Parquet from §3.
   *Produces:* feature Parquet (3 identifier columns + 33 feature columns).

5. **`src/core/v0_2_event_encoder.py`** — per-field embedding tables for
   each of the 33 feature columns, projected to `d_model=128`. The
   `feature_set={"rich","flat"}` switch drops 7 high-cardinality identity
   hashes for the encoder × architecture ablation in §5.5.

6. **`src/core/mamba_block.py`** — pure-PyTorch selective state-space
   scan (Gu & Dao 2023). 2-layer pre-norm body.

7. **`src/core/v0_2_mamba_scorer.py`** — three heads:
   - **NLL**: autoregressive next-event prediction over the 15-class
     action-family target.
   - **MEM**: vanilla masked event modeling — kept as the run #1 negative
     control.
   - **MEM-FA**: field-aware MEM — 5 categorical + 4 binary heads. The
     headline detector (run #2 candidate).

8. **`src/core/v0_2_dataloader.py`** — `L=128` sliding windows, stride 32;
   chunked all-positions interleaved-mask scoring at eval time.

9. **`src/core/v0_2_baselines/`** — n-gram trigram, Isolation Forest
   (rich + flat), XGBoost. Same feature Parquet input, different scoring
   logic — see §4 of the paper for the XGBoost degenerate-mode detail.

10. **`scripts/train_v0_2_mamba.py` / `scripts/eval_v0_2_mamba.py`** — the
    entry points. Training writes a checkpoint; evaluation joins per-event
    scores against `labels.csv` and emits per-event / per-window /
    per-technique AUROC.

---

## Quick start

### 1. Install dependencies

```bash
git clone https://github.com/ryypow/dendroaspis
cd dendroaspis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended for training the Mamba detector (RTX 5070 Ti
16 GB validated). CPU-only is supported for the classical baselines and
for inference on small windows.

*Optional:* install [Aim](https://aimstack.readthedocs.io)
(`pip install aim`) for live training dashboards. The examples below
pass `--no-aim`, so Aim is **not** required to reproduce results.

### 2. Download the dataset from Hugging Face (required)

The repository ships **code only** — no parquet shards, labels, manifest,
or dataset documentation are bundled. You must download the dataset from
Hugging Face before any of the training or evaluation commands below
will run:

```bash
huggingface-cli login   # if the dataset is gated
huggingface-cli download rypow/dendroaspis-tetragon-hids \
  --repo-type dataset --local-dir data/
```

This pulls the parquet shards, ART intervals, campaign manifest, and the
dataset card / datacard / datasheet. Expected layout under `data/`:

```
data/
├── data/
│   ├── train-{00000..00008}-of-00009.parquet     # 9 shards, ~2M rows each
│   └── test-{00000..00001}-of-00002.parquet      # 2 shards
├── agentic-art-campaign/
│   ├── CAMPAIGN.yaml                             # campaign manifest
│   └── labels.csv                                # 61 ART intervals × 22 techniques
├── README.md                                     # dataset card
├── datacard.md
└── datasheet.md
```

### 3. Train the headline detector

```bash
python scripts/train_v0_2_mamba.py \
  --objective mem-fa --feature-set flat \
  --train-parquet data/data/train-00000-of-00009.parquet \
  --output-dir    artifacts/mamba_mem_fa_flat \
  --no-aim
```

### 4. Evaluate

```bash
python scripts/eval_v0_2_mamba.py \
  --objective mem-fa \
  --checkpoint   artifacts/mamba_mem_fa_flat/best.pt \
  --test-parquet data/data/test-00000-of-00002.parquet \
  --labels-csv   data/agentic-art-campaign/labels.csv \
  --output       artifacts/mamba_mem_fa_flat/evaluation.json \
  --no-aim
```

The eval JSON reports per-event AUROC + 95 % bootstrap CI, per-window
AUROC, and per-technique AUROC across all 22 labeled techniques. Match
against the paper's Table 1.

---

## Anonymization (released corpus)

The dataset published on Hugging Face has been anonymized with a
component-first deep aliasing pass:

- **Basenames** (`proc_binary_basename`, `parent_binary_basename`,
  `root_ancestor_basename`): an allowlist of distro / dev-tool basenames
  passes through verbatim (`bash`, `node`, `python3.12`, `claude`,
  `gh`, `npm`, `code`, `git`, …); custom binaries become
  `custom_tool_NNN`; VSCode commit-hash binaries collapse to
  `vscode_server_bin`.
- **`token`** and **`parent_child_pair`** are regenerated from the
  aliased basenames via the canonical `format_token`.
- **`proc_cwd_normalized`** is path-rewritten (`/home/<user>` →
  `/home/_user_`, `/mnt/projects/<X>` → `/workspace/_project_`).
- **`proc_exec_id`** and **`process_tree_root_exec_id`** are
  pseudonymized via salted HMAC-SHA256 to `exec_<32hex>`. Equality
  joins are preserved (lineage walks still work).
- A defensive global string sweep substitutes operator/host/project
  identifiers across every string column.

Post-write assertions confirm zero PII hits across raw and
base64-decoded forms of all 12 string columns.

---

## Citation

```bibtex
@article{powers2026dendroaspis,
  author  = {Powers, Ryan William},
  title   = {Dendroaspis: Detecting Agent-Orchestrated Attacks on
             High-Entropy Workstations via {eBPF} and {M}amba},
  year    = {2026}
}

@misc{dendroaspis_corpus_v0_2_2026,
  author       = {Powers, Ryan William},
  title        = {Dendroaspis Tetragon HIDS Dataset (Agent-Driven Attack Variant), v0.2.1},
  year         = {2026},
  howpublished = {Hugging Face Datasets},
  note         = {\url{https://huggingface.co/datasets/rypow/dendroaspis-tetragon-hids}}
}
```

---

## License

Code in this repository: **MIT** (see `LICENSE`).
Dataset on Hugging Face: **CC BY 4.0**.

The two licenses are independent; you can use the code without the
dataset and vice versa, subject to each license's terms.

---

## Intended use + dual-use disclaimer

**Defensive research only.** The included parsers, behavior builders,
encoders, scorers, and tracing policies are for studying host-level
anomaly detection on Linux eBPF telemetry. The repository does not
ship attack tooling beyond what is necessary to reproduce the paper's
evaluation; operators who deploy the included Tetragon tracing
policies on their own systems are responsible for their own sandboxing
and consent.

The released attack telemetry (on Hugging Face) is real ATT&CK
technique execution, but generated in an isolated lab LXC container.
The corpus is not an attack template, not a payload library, and not a
substitute for production EDR data.

---

## Contact

Questions, bug reports, reproducibility issues: open a GitHub issue
at https://github.com/ryypow/dendroaspis.
