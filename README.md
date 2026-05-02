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

## Headline results

| Scorer                                       | Per-event AUROC | Per-window AUROC (median) |
|----------------------------------------------|-----------------|---------------------------|
| Mamba MEM-FA — flat encoder (R3 ablation)    | **0.7824**      | **0.8804**                |
| Mamba MEM-FA — rich encoder (run #2)         | 0.7514          | 0.8495                    |
| Mamba NLL — rich encoder (run #2)            | 0.7411          | 0.7755                    |
| Isolation Forest — rich encoder              | 0.6658          | 0.7423                    |
| n-gram trigram                               | 0.5672          | 0.5915                    |
| XGBoost LOOCV (supervised oracle, ceiling reference) | 0.8041  | 0.8421                    |

At each model's own best per-window aggregator, the **unsupervised**
flat-encoder Mamba detector (median = 0.88) outperforms the
leave-one-trial-out supervised XGBoost oracle (mean = 0.84) by **+0.038
AUROC** — the cleanest "Mamba beats classical" claim the paper
supports. See the paper's §5 and §6 for the full results, the
encoder × architecture decomposition, and the labeling-granularity
mechanism story for Unix-Shell techniques.

---

## Why this matters

The dataset and detector together address a question traditional
host-IDS work has not had a corpus for:

> *Can per-event sequence modeling separate attack-window events from
> noisy developer-baseline events when the attacker is itself an
> autonomous LLM agent operating within the same toolchain the
> defender's baseline runs on?*

| Dataset type             | Execution model                       |
|--------------------------|---------------------------------------|
| Traditional HIDS corpora | Human-scripted attacks                |
| This dataset             | **Agent-driven attack orchestration** |

The released corpus captures an end-to-end autonomous attack loop
*(Observation → Reconnaissance → Planning → Tooling → Execution →
Iteration)* across 21 ATT&CK techniques plus a coordinated supply-chain
capstone (`CHAIN.CAP01`, modeled on the 2025 Shai-Hulud npm-worm
incident). The detector is a 2-layer Mamba over per-field embedded
features, trained without attack labels, evaluated against this
campaign.

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

## Reproducing the paper's results

Each scorer is independent; run them in any order. Combined wall time
on a single RTX 5070 Ti: ~6–8 hours.

The commands below are abbreviated. For each row, append the same train
and label paths used in §3/§4 above:

```
--train-parquet data/data/train-00000-of-00009.parquet \
--test-parquet  data/data/test-00000-of-00002.parquet \
--labels-csv    data/agentic-art-campaign/labels.csv \
--output-dir    artifacts/<run_name> \
--no-aim
```

| Paper claim | Command |
|---|---|
| **Mamba NLL run #2** (E1, AUROC 0.741) | `python scripts/train_v0_2_mamba.py --objective nll ...` |
| **Mamba MEM-FA rich** (E2, AUROC 0.751) | `python scripts/train_v0_2_mamba.py --objective mem-fa --feature-set rich ...` |
| **Mamba MEM-FA flat** (R3, AUROC **0.782**) | `python scripts/train_v0_2_mamba.py --objective mem-fa --feature-set flat ...` |
| **n-gram baseline** (AUROC 0.567) | `python scripts/train_v0_2_baseline.py --model ngram ...` |
| **Isolation Forest rich** (AUROC 0.666) | `python scripts/train_v0_2_baseline.py --model isoforest ...` |
| **Isolation Forest flat** (AUROC 0.649) | `python scripts/run_v0_2_if_baseline.py --feature-set flat ...` |
| **XGBoost degenerate** (AUROC 0.500) | `python scripts/train_v0_2_baseline.py --model xgboost ...` |
| **XGBoost LOOCV oracle** (AUROC 0.804) | `python scripts/train_eval_xgboost_loocv.py ...` |

Eval outputs are JSON + per-event Parquet + per-technique CSV; the same
schema across all scorers so you can directly compare to the paper's
tables.

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

### What each module does

- **`src/telemetry/`** — Tetragon JSON-Lines events stream in via
  `tetragon_native_parser.py` (container filter, sentinel-window
  filter, procFS-walk filter), get serialized through
  `tetragon_native_writer.py` to raw Parquet shards.
- **`src/processing/v0_2_behavior_builder.py`** — global lineage walk
  across the entire raw corpus to resolve parent-child chains; derives
  the 14-class action-family abstraction, path/IP/port/object
  categories, lineage features (depth, hashes), and log-bucketed
  timing features. Outputs the model-ready Parquet with 33 categorical
  + 9 scalar columns.
- **`src/features/v0_2_features.py`** — schema constants for every
  embedding cardinality, OOV index, and bucket boundary. Single source
  of truth — encoder pulls from here, parser pulls from here, audit
  notebooks pull from here.
- **`src/core/mamba_block.py`** — pure-PyTorch selective state-space
  scan (Gu & Dao 2023). 2-layer pre-norm body, `d_model=128`.
- **`src/core/v0_2_event_encoder.py`** — per-field embedding tables
  (33 categorical, 9 scalar pass-through) projected to `d_model=128`.
  Supports `feature_set={"rich","flat"}`; flat drops 7 high-cardinality
  identity-hash columns for the encoder × architecture ablation in §5.5.
- **`src/core/v0_2_mamba_scorer.py`** — three head variants:
    - NLL (autoregressive next-event, 15-class action-family target)
    - MEM (vanilla masked event modeling — run #1 negative result, kept as control)
    - MEM-FA (field-aware MEM — 5 categorical + 4 binary heads, the run #2 candidate)
- **`src/core/v0_2_dataloader.py`** — sliding `L=128` windows, stride
  32; chunked all-positions interleaved-mask scoring at eval time.
- **`src/core/v0_2_baselines/`** — n-gram trigram, Isolation Forest
  (rich + flat), XGBoost (with the `XGBoostScorer.fit` degenerate-mode
  branch documented in §4 of the paper).

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
