"""Shared Weights & Biases experiment-tracking helpers for the ds006780 EEG pipeline.

Single source of truth for the W&B schema used by all four notebooks, so the schema
cannot drift between stages. Mirrors the repo's existing optional-dependency pattern
(``HAS_GCS`` / ``HAS_PICARD``): if ``wandb`` is missing or tracking is disabled, every
helper degrades to a no-op and the notebooks run unchanged.

Mapping (this is a batch pipeline, not iterative training):
  * one W&B *run* per stage-batch execution
  * run ``config``           = that stage's hyperparameters
  * per-file ``result`` dicts = rows of a ``wandb.Table``
  * aggregate stats          = run summary scalars + histograms

Usage in a notebook::

    import wandb_tracking as wbt

    run = wbt.init_run(
        stage=1,
        config={...},                 # stage hyperparameters
        group=WANDB_GROUP,            # ties the 3 stages of one config together
        tags=[f'bad_method:{BAD_CHANNEL_METHOD}'],
        enabled=ENABLE_WANDB,
        mode=WANDB_MODE,
    )

    results = []
    for item in inputs:
        results.append(process_one(item))

    wbt.finalize_run(
        run, results,
        namespace='quality',
        log_csv=LOG_FILE,
        plots={'bad_channels': fig},
        quality={'bads_detected_mean': ...},
    )
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─── Schema constants — the contract every notebook follows ──────────────────
WANDB_PROJECT = 'asd-eeg-ds006780'
DATASET_ID = 'ds006780'

# One job_type per pipeline stage. Keyed by stage number for convenience.
JOB_TYPES = {
    0: 'stage0-eda',
    1: 'stage1-preproc',
    2: 'stage2-ica-fit',
    3: 'stage3-ica-apply',
}

# Standard per-file status values produced by every stage's process_one().
STATUS_OK = 'ok'
STATUS_FAILED = 'fail'
STATUS_SKIPPED = 'skipped'


# ─── No-op stub so notebooks call .log()/.finish() unconditionally ───────────
class _DummyDict(dict):
    """dict whose .update() tolerates the extra kwargs wandb accepts.

    Lets notebook code call e.g. ``run.config.update(cfg, allow_val_change=True)``
    unchanged when tracking is disabled.
    """

    def update(self, *args, **kwargs):
        kwargs.pop('allow_val_change', None)
        return super().update(*args, **kwargs)


class NoOpRun:
    """Stand-in returned when tracking is disabled or wandb is unavailable.

    Implements the small slice of the wandb.Run API the notebooks touch, so
    notebook code never needs an ``if run is not None`` guard.
    """

    job_type = None
    group = None

    def __init__(self):
        self.config = _DummyDict()
        self.summary = _DummyDict()
        self.disabled = True

    def log(self, *args, **kwargs):
        return None

    def log_artifact(self, *args, **kwargs):
        return None

    def finish(self, *args, **kwargs):
        return None

    def __bool__(self):
        return False


def _is_active(run) -> bool:
    """True only for a real, enabled wandb run."""
    return HAS_WANDB and run is not None and not isinstance(run, NoOpRun)


# ─── Run lifecycle ───────────────────────────────────────────────────────────
def init_run(
    stage: int,
    config: Mapping[str, Any],
    *,
    group: str | None = None,
    tags: Sequence[str] | None = None,
    name: str | None = None,
    enabled: bool = True,
    mode: str = 'online',
    project: str = WANDB_PROJECT,
):
    """Start a W&B run for one stage-batch, or return a NoOpRun if disabled.

    Parameters
    ----------
    stage    : 0-3, selects the job_type from JOB_TYPES.
    config   : stage hyperparameters; logged as run config.
    group    : pipeline id so the 3 processing stages of one config group together.
               Defaults to DATASET_ID.
    tags     : extra tags; DATASET_ID and the job_type are always added.
    name     : run name; defaults to "<job_type>-<timestamp>".
    enabled  : master switch (e.g. notebook's ENABLE_WANDB).
    mode     : 'online' | 'offline' | 'disabled'.
    """
    if not enabled or not HAS_WANDB:
        return NoOpRun()

    job_type = JOB_TYPES[stage]
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = name or f'{job_type}-{stamp}'
    all_tags = [DATASET_ID, job_type, *(tags or [])]

    return wandb.init(
        project=project,
        job_type=job_type,
        group=group or DATASET_ID,
        name=run_name,
        tags=all_tags,
        config=dict(config),
        mode=mode,
        reinit=True,
    )


# ─── Internal helpers ────────────────────────────────────────────────────────
def _status_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {STATUS_OK: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
    for r in results:
        s = r.get('status')
        if s in counts:
            counts[s] += 1
    return counts


def _table_columns(results: Sequence[Mapping[str, Any]]) -> list[str]:
    """Union of keys across all result dicts, preserving first-seen order."""
    cols: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    return cols


def _numeric_columns(results: Sequence[Mapping[str, Any]], cols: Sequence[str]) -> dict[str, list[float]]:
    """Collect numeric (non-bool) values per column for histogramming."""
    out: dict[str, list[float]] = {}
    for c in cols:
        vals = []
        for r in results:
            v = r.get(c)
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[c] = vals
    return out


def build_table(results: Sequence[Mapping[str, Any]]):
    """Build a wandb.Table from in-memory result dicts.

    Built from memory rather than the CSV log on purpose: stages 1-2 write their
    CSV header from the first row's keys, so a leading 'skipped' row (4 fields)
    followed by 'ok' rows (12+ fields) yields a mixed-width file. In-memory
    results sidestep that entirely.
    """
    if not HAS_WANDB or not results:
        return None
    cols = _table_columns(results)
    table = wandb.Table(columns=cols)
    for r in results:
        table.add_data(*[r.get(c) for c in cols])
    return table


def summarize(results: Sequence[Mapping[str, Any]], **scalars: Any) -> dict[str, Any]:
    """Convenience: namespace arbitrary quality scalars under 'quality/'.

    e.g. summarize(results, namespace='quality', bads_mean=1.2)
    -> {'quality/bads_mean': 1.2}
    """
    namespace = scalars.pop('namespace', 'quality')
    return {f'{namespace}/{k}': v for k, v in scalars.items()}


# ─── Finalize ────────────────────────────────────────────────────────────────
def finalize_run(
    run,
    results: Sequence[Mapping[str, Any]],
    *,
    namespace: str = 'quality',
    wall_seconds: float | None = None,
    log_csv: str | Path | None = None,
    table_artifact_name: str | None = None,
    extra_artifacts: Iterable[str | Path] = (),
    plots: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    histogram_columns: Sequence[str] | None = None,
):
    """Log the standard schema for one stage-batch run, then finish().

    * standard summary scalars: files/total, files/ok, files/failed,
      files/skipped, files/processed_this_run, runtime/wall_seconds
    * per-file wandb.Table (columns = result-dict keys)
    * wandb.Histogram for numeric columns (all, or restricted to
      ``histogram_columns``)
    * ``quality`` dict logged under ``namespace/``
    * ``plots`` (name -> matplotlib Figure) logged as wandb.Image
    * ``log_csv`` + ``extra_artifacts`` uploaded as a W&B Artifact

    Always finishes the run (and is a no-op for NoOpRun).
    """
    if not _is_active(run):
        if run is not None:
            run.finish()
        return

    counts = _status_counts(results)
    n_total = len(results)
    processed = counts[STATUS_OK] + counts[STATUS_FAILED]

    summary: dict[str, Any] = {
        'files/total': n_total,
        'files/ok': counts[STATUS_OK],
        'files/failed': counts[STATUS_FAILED],
        'files/skipped': counts[STATUS_SKIPPED],
        'files/processed_this_run': processed,
    }
    if wall_seconds is not None:
        summary['runtime/wall_seconds'] = round(float(wall_seconds), 1)

    if quality:
        for k, v in quality.items():
            summary[f'{namespace}/{k}'] = v

    # Per-file table.
    table = build_table(results)
    if table is not None:
        run.log({'per_file': table})

    # Numeric distributions.
    cols = _table_columns(results)
    numeric = _numeric_columns(results, cols)
    if histogram_columns is not None:
        numeric = {c: numeric[c] for c in histogram_columns if c in numeric}
    for col, vals in numeric.items():
        try:
            run.log({f'dist/{col}': wandb.Histogram(vals)})
        except Exception:
            pass

    # Plots.
    if plots:
        for key, fig in plots.items():
            if fig is not None:
                run.log({f'plot/{key}': wandb.Image(fig)})

    # Artifacts (CSV logs, TSVs, etc.).
    artifact_paths = [p for p in [log_csv, *extra_artifacts] if p]
    artifact_paths = [Path(p) for p in artifact_paths if Path(p).exists()]
    if artifact_paths:
        art_name = table_artifact_name or f'{run.job_type}-logs'
        artifact = wandb.Artifact(art_name, type='run-logs')
        for p in artifact_paths:
            if p.is_dir():
                artifact.add_dir(str(p))
            else:
                artifact.add_file(str(p))
        run.log_artifact(artifact)

    run.summary.update(summary)
    run.finish()


def log_plot(run, key: str, fig) -> None:
    """Log a single matplotlib figure as a wandb.Image (no-op if inactive)."""
    if _is_active(run) and fig is not None:
        run.log({f'plot/{key}': wandb.Image(fig)})


def log_table(run, key: str, df) -> None:
    """Log a pandas DataFrame as a wandb.Table (no-op if inactive)."""
    if _is_active(run) and df is not None:
        run.log({key: wandb.Table(dataframe=df)})


def log_artifact_files(
    run,
    name: str,
    paths: Iterable[str | Path],
    *,
    artifact_type: str = 'dataset',
) -> None:
    """Bundle files/dirs into a W&B Artifact and log it (no-op if inactive)."""
    if not _is_active(run):
        return
    existing = [Path(p) for p in paths if p and Path(p).exists()]
    if not existing:
        return
    artifact = wandb.Artifact(name, type=artifact_type)
    for p in existing:
        if p.is_dir():
            artifact.add_dir(str(p))
        else:
            artifact.add_file(str(p))
    run.log_artifact(artifact)
