"""Reusable code for the ds006780 ASD EEG preprocessing pipeline.

Notebooks live in ``notebooks/`` and import from this package after an editable
install (``pip install -e .``). The infrastructure modules are re-exported here so
notebook imports stay short and stable even if the internal layout changes:

    from asd_eeg import gcs_io                 # Google Cloud Storage I/O
    from asd_eeg import wandb_tracking as wbt  # Weights & Biases tracking
"""

from asd_eeg.infra import gcs_io, wandb_tracking

__all__ = ["gcs_io", "wandb_tracking"]
