"""Tolerant readers / repair for the pipeline's per-stage CSV logs.

The stage notebooks stream one CSV row per processed recording. When a writer does
not pad every row to a fixed schema (as older Stage‑1 and Stage‑2 ``append_log``
versions did), the header is written once from the first row while later rows carry
more fields — the file becomes ragged and ``pd.read_csv`` raises
``ParserError: Expected N fields, saw M``.

``read_stage_log`` reads such a file regardless, reconstructing each row from the
known column order for that stage/era (keyed by row width), with the per-row
``status`` distinguishing ``ok`` rows from the 6‑field ``skipped`` / ``fail`` rows.
``repair_log`` rewrites a corrupt log in place as a clean fixed-schema CSV (recovering
historical metric rows without re-running the pipeline). Both are idempotent on files
that are already rectangular.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd

# ── Canonical current (fixed) schemas, matching each notebook's append_log ──────────
STAGE1_LOG_COLUMNS = [
    'subject', 'task', 'run', 'status', 'out', 'error', 'sfreq_out', 'n_chans',
    'n_chans_dropped', 'n_chans_retyped', 'n_bads_from_bids', 'n_bads_detected',
    'n_bads_total', 'n_interpolated', 'bads_detected', 'duration_s', 'line_freq',
    'montage_applied', 'bad_method', 'timestamp',
]
STAGE2_LOG_COLUMNS = [
    'subject', 'task', 'run', 'status', 'out', 'gcs_uri', 'n_eeg_channels',
    'rank_reduction', 'expected_components', 'n_components_fit', 'n_iter',
    'fit_seconds', 'method', 'decim', 'converged', 'error', 'timestamp',
]
# Legacy Stage‑3 *apply* log (`03_ica_apply_log.csv`), superseded by the current
# fixed-schema `03_ica_label_apply_log.csv`. The label-count fields are recovered
# best-effort (exact order across that old era is not fully determinable). `out` /
# `error` carry the 6‑field skipped / fail rows.
_STAGE3_OK19_ORDER = [
    'subject', 'task', 'run', 'status', 'cleaned_path', 'labels_path', 'cleaned_gcs',
    'n_components', 'n_excluded', 'excluded_components', 'aux_channel_types',
    'n_brain', 'n_muscle', 'n_eye', 'n_heart', 'n_line', 'n_channel', 'n_other',
    'timestamp',
]
STAGE3_LOG_COLUMNS = _STAGE3_OK19_ORDER[:-1] + ['out', 'error', 'timestamp']
# Stage 4 epoching writes a fixed-schema log from the first row, so it is never
# ragged; this entry is for read_stage_log's fast path + schema consistency.
STAGE4_LOG_COLUMNS = [
    'timestamp', 'subject', 'task', 'run', 'status', 'epoch_mode', 'n_epochs',
    'epoch_duration_s', 'tmin', 'tmax', 'n_event_types', 'event_types', 'sfreq',
    'n_channels', 'mean_p2p_uv', 'max_p2p_uv', 'total_duration_s', 'epochs_path',
    'epochs_gcs', 'warning', 'error',
]

# Per-stage positional order of each historical `ok`-row width seen on disk.
# Verified against the actual logs; widths not listed fall back to end-anchoring.
_STAGE1_OK_ORDERS = {
    20: STAGE1_LOG_COLUMNS,                                   # current schema
    19: [c for c in STAGE1_LOG_COLUMNS if c != 'error'],      # earlier: no `error` col
}
_STAGE2_OK_ORDERS = {
    16: [c for c in STAGE2_LOG_COLUMNS if c != 'error'],      # current schema
    13: ['subject', 'task', 'run', 'status', 'out', 'gcs_uri', 'n_eeg_channels',
         'n_iter', 'fit_seconds', 'method', 'n_components_fit', 'converged',
         'timestamp'],                                        # oldest fit_info order
}
_STAGE3_OK_ORDERS = {
    19: _STAGE3_OK19_ORDER,                                   # legacy apply schema
}
_STAGE4_OK_ORDERS = {
    len(STAGE4_LOG_COLUMNS): STAGE4_LOG_COLUMNS,              # current (only) schema
}

_STAGES = {
    1: (STAGE1_LOG_COLUMNS, _STAGE1_OK_ORDERS),
    2: (STAGE2_LOG_COLUMNS, _STAGE2_OK_ORDERS),
    3: (STAGE3_LOG_COLUMNS, _STAGE3_OK_ORDERS),
    4: (STAGE4_LOG_COLUMNS, _STAGE4_OK_ORDERS),
}


def _infer_stage(path: Path) -> int:
    """Infer pipeline stage from the log filename (``01_*`` → 1, ``02_*`` → 2)."""
    name = path.name
    if name.startswith('01'):
        return 1
    if name.startswith('02'):
        return 2
    if name.startswith('03'):
        return 3
    if name.startswith('04'):
        return 4
    raise ValueError(
        f"Cannot infer stage from {name!r}; pass stage=1, 2, 3 or 4 explicitly."
    )


def _is_clean(path: Path) -> bool:
    """A file is 'clean' if every row has the same field count."""
    with open(path, newline='') as f:
        widths = {len(row) for row in csv.reader(f) if row}
    return len(widths) <= 1


def _map_by_anchors(row: list[str], columns: list[str]) -> dict:
    """Best-effort map for an unknown-width row: anchor identity cols from the left
    and ``timestamp`` (+ a couple of trailing metrics) from the right, leave the
    ambiguous middle as NaN. Used only for rare legacy widths."""
    rec = {}
    head = ['subject', 'task', 'run', 'status', 'out']
    for i, col in enumerate(head):
        if i < len(row):
            rec[col] = row[i]
    # right anchor: timestamp is always last
    if len(row) > len(head):
        rec['timestamp'] = row[-1]
    return rec


def _row_to_record(row: list[str], columns: list[str], ok_orders: dict) -> dict:
    """Map one ragged row to a dict using its status (index 3) and width."""
    status = row[3] if len(row) > 3 else ''
    if status in ('skipped', 'fail') and len(row) == 6:
        # subject, task, run, status, <out|error>, timestamp
        key = 'out' if status == 'skipped' else 'error'
        return dict(zip(['subject', 'task', 'run', 'status', key, 'timestamp'], row))
    order = ok_orders.get(len(row))
    if order is not None:
        return dict(zip(order, row))
    return _map_by_anchors(row, columns)


def read_stage_log(path, stage: int | None = None) -> pd.DataFrame:
    """Read a per-stage CSV log into a DataFrame, tolerating ragged files.

    Clean fixed-schema logs are read with a plain ``pd.read_csv``. A ragged file is
    reconstructed row-by-row from the stage's known column orders and returned on the
    canonical ``STAGE{N}_LOG_COLUMNS`` schema. ``stage`` is inferred from the filename
    when not given.
    """
    path = Path(path).expanduser()
    if _is_clean(path):
        return pd.read_csv(path)

    stage = stage or _infer_stage(path)
    columns, ok_orders = _STAGES[stage]

    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    if rows and rows[0][:4] == ['subject', 'task', 'run', 'status']:
        rows = rows[1:]  # drop the mis-sized header row
    records = [_row_to_record(r, columns, ok_orders) for r in rows]
    return pd.DataFrame(records).reindex(columns=columns)


def repair_log(path, stage: int | None = None, backup: bool = True) -> Path:
    """Rewrite a corrupt per-stage log in place as a clean fixed-schema CSV.

    Backs up the original to ``<name>.corrupt.bak`` (unless ``backup=False``), recovers
    every row via :func:`read_stage_log`, and writes the canonical header. Idempotent:
    a file that is already rectangular is left untouched.
    """
    path = Path(path).expanduser()
    if _is_clean(path):
        return path
    df = read_stage_log(path, stage=stage)
    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + '.corrupt.bak'))
    df.to_csv(path, index=False)
    return path


# Back-compat alias (Stage 2 was the first repaired).
def repair_stage2_log(path, backup: bool = True) -> Path:
    return repair_log(path, stage=2, backup=backup)
