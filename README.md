exit# EEG Preprocessing Pipeline — ds006780

An automated EEG preprocessing pipeline for the OpenNeuro dataset `ds006780` (autism spectrum disorder EEG, BIDS-formatted). The pipeline converts raw multi-format EEG recordings into clean, artifact-free `.fif` files ready for analysis.

## Dataset

- **Source:** [OpenNeuro ds006780](https://openneuro.org/datasets/ds006780) — streamed from S3
- **Population:** Autism spectrum disorder (ASD) participants
- **Format:** BIDS-compliant raw recordings (`.bdf`, `.vhdr`, `.edf`, `.set`)
- **Tasks:** `Restingstate`, `FAST`, `IC`, `motor`
- **Scale:** 136 subjects, ~2,171 files per stage

## Pipeline Overview

The pipeline is split into three stages so the expensive ICA decomposition runs only once. Each stage is an independent Jupyter notebook.

```
Raw EEG (.bdf / .vhdr / .edf / .set)
        │
        ▼
Stage 1 — 01_preprocessing_pre_ica_v2.ipynb
  Notch filter → bandpass filter → resample
  → bad-channel detection → interpolation
        │
        ▼  desc-preproc_eeg.fif
        │
        ▼
Stage 2 — 02_ica_fit_local_v2.ipynb
  Average reference → ICA fit
        │
        ▼  desc-preproc_ica.fif
        │
        ▼
Stage 3 — ica_apply.ipynb
  ICLabel (automated component classification)
  → drop artifact components → apply ICA
        │
        ▼  desc-clean_eeg.fif
```

## Files

| File | Purpose |
|------|---------|
| `01_preprocessing_pre_ica_v2.ipynb` | Stage 1: filtering, resampling, bad-channel detection |
| `02_ica_fit_local_v2.ipynb` | Stage 2: average reference + ICA decomposition |
| `ica_apply.ipynb` | Stage 3: automated ICA labeling (ICLabel) + artifact removal |
| `ds006780_data_understanding.ipynb` | Exploratory analysis and dataset understanding |
| `CLAUDE.md` | Instructions for Claude Code AI assistant |

## Output Structure

All outputs are written under `~/asd_eeg_pipeline/`:

```
~/asd_eeg_pipeline/
└── derivatives/
    ├── mne-preproc-pre-ica/          ← Stage 1
    │   └── sub-XXXXX/eeg/
    │       └── sub-XXXXX_task-TASK_run-NN_desc-preproc_eeg.fif
    ├── mne-ica-fit/                  ← Stage 2
    │   └── sub-XXXXX/eeg/
    │       └── sub-XXXXX_task-TASK_run-NN_desc-preproc_ica.fif
    ├── mne-ica-apply/                ← Stage 3
    │   └── sub-XXXXX/eeg/
    │       └── sub-XXXXX_task-TASK_run-NN_desc-clean_eeg.fif
    └── logs/
        ├── 01_preproc_log.csv
        ├── 02_ica_fit_log.csv
        ├── 02_ica_fit_errors.txt
        └── 03_ica_apply_log.csv
```

## Setup

```bash
pip install mne mne-bids mne-icalabel pandas numpy joblib tqdm \
            python-picard boto3 google-cloud-storage pyprep onnxruntime
```

Optionally authenticate with GCS if using cloud I/O:

```bash
gcloud auth application-default login
```

## Running the Pipeline

Run the notebooks in order. Each has a **smoke test cell** — run it on a single file first before launching the full batch.

```
1. 01_preprocessing_pre_ica_v2.ipynb
2. 02_ica_fit_local_v2.ipynb
3. ica_apply.ipynb
```

To reprocess already-completed files, set `OVERWRITE = True` in the configuration section of the relevant notebook.

## Key Configuration

### Stage 1 (`01_preprocessing_pre_ica_v2.ipynb`)

| Parameter | Description |
|-----------|-------------|
| `TASK_CONFIG` | Per-task filter/resample settings. `motor`: 100 Hz / 500 Hz; others: 40 Hz / 250 Hz |
| `BAD_CHANNEL_METHOD` | `'lof'` (fast) or `'pyprep'` (PREP-style, more thorough) |
| `N_JOBS_OUTER` | Parallel workers. Keep `1` for cloud reads; 4–8 on local SSD |
| `UPLOAD_TO_GCS` | Set `False` to skip GCS upload |

### Stage 2 (`02_ica_fit_local_v2.ipynb`)

| Parameter | Description |
|-----------|-------------|
| `STAGE1_SOURCE` | `'local'` or `'gcs'` |
| `ICA_N_COMPONENTS` | `None` = auto-detect rank. Do not hardcode unless rank is uniform |
| `ICA_RANDOM_STATE` | `42` — keep fixed for reproducibility |
| `ICA_DECIM` | `3` — subsamples during fit for speed |
| `N_JOBS_OUTER` | Peak RAM is ~3–5× file size per worker; monitor with `htop` |

### Stage 3 (`ica_apply.ipynb`)

| Parameter | Description |
|-----------|-------------|
| `INPUT_SOURCE` | `'gcs'` or `'local'` |
| `UPLOAD_TO_GCS` | Upload Stage 3 outputs to GCS |
| `EXCLUDE_LABELS` | ICLabel artifact classes to drop (muscle, eye, heartbeat, line noise, channel noise) |
| `LABEL_PROB_THRESHOLD` | `0.80` — minimum confidence to drop a component |
| `MAX_BAD_FRACTION_WARN` | Warn (don't fail) if >50% of components are dropped |

## GCS

The GCS bucket is `asd-eeg-dataset`. If `google-cloud-storage` is not installed or authentication fails, GCS features are silently skipped. Inputs streamed from GCS are cached in `/tmp` and deleted after use.

## Dependencies

- [MNE-Python](https://mne.tools) — core EEG processing
- [MNE-BIDS](https://mne.tools/mne-bids) — BIDS I/O
- [MNE-ICALabel](https://mne.tools/mne-icalabel) — automated IC classification
- [PyPREP](https://github.com/sappelhoff/pyprep) — PREP-style bad-channel detection
- [python-picard](https://pierreablin.github.io/picard/) — fast ICA solver
- [boto3](https://boto3.amazonaws.com) — OpenNeuro S3 access
- [google-cloud-storage](https://cloud.google.com/python/docs/reference/storage/latest) — GCS I/O
