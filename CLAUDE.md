# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An EEG preprocessing pipeline for OpenNeuro dataset `ds006780` (136 ASD participants). Raw multi-format recordings (`.bdf`, `.vhdr`, `.edf`, `.set`) are converted to clean artifact-free `.fif` files through three sequential Jupyter notebook stages, then segmented into epochs by a fourth stage.

**Hardware context:** BioSemi ActiveTwo, 64 channels (BioSemi64 montage), 512 Hz, 60 Hz power line.

## Pipeline Stages

Run notebooks in order. Each has a **smoke test cell** — always run on a single file first before launching the full batch.

| Notebook | Stage | Input → Output |
|----------|-------|----------------|
| `01_preprocessing_pre_ica_v2.ipynb` | 1 — Filter + bad channels | S3 raw → `desc-preproc_eeg.fif` |
| `02_ica_fit_local_v2.ipynb` | 2 — ICA decomposition | Stage 1 output → `desc-preproc_ica.fif` |
| `03_ica_apply.ipynb` | 3 — ICLabel + artifact removal | Stage 1 + 2 → `desc-clean_eeg.fif` |
| `04_epoching.ipynb` | 4 — Epoching | Stage 3 output → `desc-epo.fif` |
| `00_data_understanding.ipynb` | 0 — Exploratory analysis | S3/GCS metadata |

The ICA decomposition is intentionally split across stages 2 and 3 so the expensive fit runs only once.

## Data Flow Architecture

```
OpenNeuro S3 (raw BIDS)
    → Stage 1: notch → bandpass → resample → bad-channel detect → interpolate
    → ~/asd_eeg_pipeline/derivatives/mne-preproc-pre-ica/
    → Stage 2: average reference → ICA fit (Picard algorithm)
    → ~/asd_eeg_pipeline/derivatives/mne-ica-fit/
    → Stage 3: ICLabel classify → drop artifact ICs → apply ICA
    → ~/asd_eeg_pipeline/derivatives/mne-ica-apply/
    → Stage 4: per-task epoching (fixed-length | event-related from annotations)
    → ~/asd_eeg_pipeline/derivatives/mne-epochs/
                                    + GCS bucket: asd-eeg-dataset
```

**Critical ordering constraint in Stage 3:** Stage 1 raw must be average-referenced *before* applying Stage 2's ICA — the reference must match what Stage 2 was fit on.

**Stage 4 event source:** event-related epochs come **only** from the MNE annotations carried in the Stage 3 `.fif` (originally BIDS `events.tsv`, loaded by `read_raw_bids` in Stage 1 and preserved through filter/resample/ICA). There is no separate BIDS/S3 event path — a recording with no usable annotations fails loudly. Stage 4 does epoching only (no baseline correction, no rejection, no feature extraction).

## Repository Layout

```
asd_eeg/                 # installable package — all reusable code
  infra/                 #   external services & run plumbing
    gcs_io.py            #     GCS I/O (GCSStore)
    wandb_tracking.py    #     W&B experiment tracking
  utils/                 #   generic helpers
    logs.py              #     tolerant readers/repair for the per-stage CSV logs
  metrics/               #   FUTURE: QC / quality metric functions
  plotting/              #   FUTURE: reusable figure builders
notebooks/               # the stage notebooks (00–04) + qc_ica_quality — the canonical copies
pyproject.toml           # declares the package + all runtime deps; enables `pip install -e .`
```

Notebooks import reusable code via the package: `from asd_eeg import gcs_io` /
`from asd_eeg import wandb_tracking as wbt` (both re-exported from `asd_eeg.infra`
by `asd_eeg/__init__.py`). The editable install (below) makes `asd_eeg` importable
from the `notebooks/` working directory.

**Canonical notebooks live in `notebooks/` — there are no notebooks at the repo root.**
If a stage notebook reappears at the root, it is a stale copy from before the package
reorganization (it will use flat imports like `import wandb_tracking`, which no longer
resolve). Delete it rather than editing it.

## Setup

```bash
pip install -e .          # installs all runtime deps (declared in pyproject.toml)
                          # and makes the `asd_eeg` package importable from notebooks/
```

All runtime dependencies (mne, mne-bids, mne-icalabel, python-picard, pyprep, onnxruntime,
boto3, google-cloud-storage, wandb, …) are declared in `pyproject.toml`, so the editable
install pulls them in — there is no separate requirements file to keep in sync.

For GCS access:
```bash
gcloud auth application-default login
```

## Experiment Tracking (Weights & Biases)

All notebooks log to W&B through the shared `asd_eeg/infra/wandb_tracking.py` module,
which is the single source of truth for the schema (so stages can't drift apart). One **run per
stage-batch execution**: run `config` = that stage's hyperparameters, per-file `result`
dicts become a `wandb.Table`, and aggregate stats become summary scalars + histograms.

```bash
wandb login            # once; or set WANDB_MODE='offline' and `wandb sync` later
```

Each notebook has a W&B config cell in its Configuration section:
- `ENABLE_WANDB` — set `False` to make all tracking a no-op (notebooks run unchanged)
- `WANDB_MODE` — `'online'` | `'offline'` | `'disabled'`
- `WANDB_GROUP` — set the **same** id in Stages 1→2→3→4 to group one pipeline configuration

Schema: project `asd-eeg-ds006780`; `job_type` per stage (`stage0-eda`, `stage1-preproc`,
`stage2-ica-fit`, `stage3-ica-apply`, `stage4-epoch`); standard `files/*` + `runtime/*` summary
keys plus a stage-specific `quality/*` namespace. Tracked artifacts: the per-stage CSV logs,
smoke-test / inspect plots, and (Stage 3) the ICLabel component TSVs — never the heavy `.fif` data.

## Key Configuration Parameters

### Stage 1 (`01_preprocessing_pre_ica_v2.ipynb`)
- `TASK_CONFIG` — per-task filter/resample settings: `motor` uses 100 Hz / 500 Hz; all others use 40 Hz / 250 Hz
- `BAD_CHANNEL_METHOD` — `'lof'` (fast) or `'pyprep'` (PREP-style, more thorough)
- `N_JOBS_OUTER` — keep `1` for cloud reads; 4–8 on local SSD
- `UPLOAD_TO_GCS` — set `False` to skip GCS upload
- `OVERWRITE` — set `True` to reprocess already-completed files

### Stage 2 (`02_ica_fit_local_v2.ipynb`)
- `STAGE1_SOURCE` — `'local'` or `'gcs'`
- `ICA_N_COMPONENTS` — `None` = auto-detect rank; do not hardcode unless rank is uniform
- `ICA_RANDOM_STATE` — `42`; keep fixed for reproducibility
- `ICA_DECIM` — `3` (subsamples during fit for speed)
- `N_JOBS_OUTER` — peak RAM is ~3–5× file size per worker; monitor with `htop`

### Stage 3 (`03_ica_apply.ipynb`)
- `INPUT_SOURCE` — `'gcs'` or `'local'`
- `EXCLUDE_LABELS` — ICLabel artifact classes to drop (muscle, eye, heartbeat, line noise, channel noise)
- `LABEL_PROB_THRESHOLD` — `0.80`; minimum confidence to drop a component
- `MAX_BAD_FRACTION_WARN` — warns (does not fail) if >50% of components are dropped

### Stage 4 (`04_epoching.ipynb`)
- `INPUT_SOURCE` — `'gcs'` or `'local'` (reads Stage 3 `desc-clean_eeg.fif`)
- `EPOCH_CONFIG` — per-task segmentation, the analogue of Stage 1's `TASK_CONFIG`. `mode='fixed'`
  (Restingstate: `duration`/`overlap`) or `mode='event'` (FAST/IC/motor: `tmin`/`tmax` + an
  `events` list naming which annotation labels to epoch on). **`tmin`/`tmax` are provisional** —
  validate against task event timing before production.
- `GLOBAL_BASELINE` — `None`; baseline correction is left to downstream analysis
- `GLOBAL_REJECT` — `None`; no epoch rejection (keep all; per-epoch p2p amplitude is logged for QC)
- `N_JOBS_OUTER`, `OVERWRITE`, `UPLOAD_TO_GCS` — as in the other stages

## Logs and Outputs

All outputs under `~/asd_eeg_pipeline/derivatives/logs/`:
- `01_preproc_log.csv`, `01_preproc_errors.txt`
- `02_ica_fit_log.csv`, `02_ica_fit_errors.txt`
- `03_ica_label_apply_log.csv`, `03_ica_label_apply_errors.txt` (older runs: `03_ica_apply_log.csv`)
- `04_epoching_log.csv`, `04_epoching_errors.txt`

Stage 3 also writes per-file `desc-iclabel_components.tsv` alongside the clean `.fif` files.
Stage 4 writes `desc-epo.fif` under `derivatives/mne-epochs/sub-*/eeg/`.

**Reading the logs:** these CSVs accumulate rows across pipeline runs and the writer schema has
changed between eras, so a log can be *ragged* (rows of differing widths) and `pd.read_csv` will
raise `ParserError`. Always read them through `asd_eeg.utils.logs.read_stage_log(path)`, which
reconstructs rows from the known per-stage column orders (stage inferred from the filename) and
returns the canonical `STAGE{N}_LOG_COLUMNS` schema. `logs.repair_log(path)` rewrites a corrupt
log in place (backing up to `*.corrupt.bak`); both are idempotent on already-rectangular files.

## GCS

All GCS reading and writing goes through the shared `gcs_io.GCSStore` (in
`asd_eeg/infra/gcs_io.py`), configured once there — bucket (`asd-eeg-dataset`), dataset id,
retry policy, chunk size, and cache dir. Notebooks do `from asd_eeg import gcs_io` then
`store = gcs_io.GCSStore()` and call its methods:

- `store.upload(path, deriv_root, pipeline)` / `store.upload_tree(deriv_root, pipeline, pattern)` — write derivatives (skips blobs whose size already matches)
- `store.list_files(pipeline, suffix, tasks=...)` — discover a stage's outputs
- `store.staged(blob_name)` — contextmanager that downloads to the cache and deletes after
- `GCSStore.pair_files(primary, secondary, key_fn)` — match raw↔ICA across stages (Stage 3)

The client is created lazily, so importing never fails when `google-cloud-storage` is
missing or the environment is unauthenticated; `gcs_io.HAS_GCS` reports availability.
Auth: `gcloud auth application-default login` (or `GOOGLE_APPLICATION_CREDENTIALS`).
Stage 1 still reads raw input from OpenNeuro **S3** (boto3) — that is a separate source.
