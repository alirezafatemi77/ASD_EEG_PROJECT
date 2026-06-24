"""Shared Google Cloud Storage I/O for the ds006780 EEG pipeline.

One configured place for reading and writing derivatives in GCS, so the bucket,
retry policy, chunk size, cache dir, and the upload/download/discovery helpers are
written once instead of being copy-pasted (and drifting) across the stage notebooks.

Mirrors the repo's optional-dependency pattern (`HAS_GCS` try/except): importing this
module never fails when ``google-cloud-storage`` is missing or the environment is
unauthenticated — the client is created lazily on first real use.

Usage in a notebook::

    import gcs_io
    store   = gcs_io.GCSStore()          # bucket + dataset defaults baked in
    HAS_GCS = gcs_io.HAS_GCS

    # write
    uri = store.upload(out_path, DERIV_ROOT, PIPELINE_NAME)
    store.upload_tree(DERIV_ROOT, PIPELINE_NAME, '*_desc-preproc_ica.fif')

    # read
    inputs = store.list_files(STAGE1_PIPELINE, '_desc-preproc_eeg.fif', tasks=TASKS)
    with store.staged(blob_name) as local_path:
        raw = mne.io.read_raw_fif(local_path)
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google.cloud import storage as gcs_lib
    from google.cloud.storage.retry import DEFAULT_RETRY
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


# ─── BIDS filename helpers (shared by discovery; were duplicated in Stages 2 & 3) ──
def parse_bids_filename(fname: str) -> dict:
    """Pull subject/task/run/desc out of a BIDS filename.

    e.g. 'sub-10003_task-FAST_run-01_desc-preproc_eeg.fif'
        -> {'subject': '10003', 'task': 'FAST', 'run': '01', 'desc': 'preproc'}
    """
    parts = Path(fname).stem.split('_')
    fields = {}
    for p in parts:
        if '-' in p:
            key, val = p.split('-', 1)
            fields[key] = val
    return {
        'subject': fields.get('sub'),
        'task':    fields.get('task'),
        'run':     fields.get('run'),
        'desc':    fields.get('desc'),
    }


def make_key(entities: dict) -> tuple:
    """Stable (subject, task, run) key for matching files across stages."""
    return (entities['subject'], entities['task'], entities.get('run'))


# ─── The store ────────────────────────────────────────────────────────────────
class GCSStore:
    """Configured GCS read/write for one bucket + dataset.

    All knobs (bucket, dataset id, cache dir, chunk/timeout/retry) live here so the
    notebooks don't re-declare them. The underlying client/bucket is created lazily.
    """

    def __init__(self, bucket='asd-eeg-dataset', dataset_id='ds006780',
                 cache_dir='/tmp/gcs_cache', chunk=16 * 1024 * 1024,
                 timeout=600, deadline=600):
        self.bucket_name = bucket
        self.dataset_id = dataset_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk = chunk
        self.timeout = timeout
        self._client = None
        self._bucket = None
        self.retry = (
            DEFAULT_RETRY.with_deadline(deadline).with_delay(
                initial=2, maximum=30, multiplier=2)
            if HAS_GCS else None
        )

    # ── lazy client/bucket ──────────────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
            if not HAS_GCS:
                raise RuntimeError(
                    'google-cloud-storage is not installed. '
                    'Run: pip install google-cloud-storage')
            self._client = gcs_lib.Client()
        return self._client

    @property
    def bucket(self):
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def available(self) -> bool:
        """True if the library is present and the bucket is reachable."""
        if not HAS_GCS:
            return False
        try:
            return self.bucket.exists()
        except Exception:
            return False

    def prefix(self, pipeline: str) -> str:
        """GCS key prefix for a pipeline's derivatives."""
        return f'derivatives/{self.dataset_id}/{pipeline}'

    def blob_name(self, local_path, deriv_root, pipeline: str) -> str:
        """Map a local derivative path to its GCS blob name."""
        rel = Path(local_path).relative_to(deriv_root)
        return f'{self.prefix(pipeline)}/{rel.as_posix()}'

    # ── write ───────────────────────────────────────────────────────────────
    def _upload_one(self, local_path, deriv_root, pipeline):
        """Upload one file; returns (gs_uri, skipped). Skips if size already matches."""
        local_path = Path(local_path)
        blob_name = self.blob_name(local_path, deriv_root, pipeline)
        blob = self.bucket.blob(blob_name, chunk_size=self.chunk)
        uri = f'gs://{self.bucket_name}/{blob_name}'
        if blob.exists(retry=self.retry, timeout=self.timeout):
            blob.reload(retry=self.retry, timeout=self.timeout)
            if blob.size == local_path.stat().st_size:
                return uri, True
        blob.upload_from_filename(str(local_path), timeout=self.timeout, retry=self.retry)
        return uri, False

    def upload(self, local_path, deriv_root, pipeline: str) -> str:
        """Upload a derivative file to GCS (chunked, resumable). Returns the gs:// URI.

        Skips the transfer when a blob of identical size already exists.
        """
        uri, _ = self._upload_one(local_path, deriv_root, pipeline)
        return uri

    def upload_tree(self, deriv_root, pipeline: str, pattern: str,
                    workers: int = 8, keep_local: bool = True,
                    progress: bool = True) -> dict:
        """Upload every file under ``deriv_root`` matching ``pattern`` (rglob).

        Returns {'uploaded': [...], 'skipped': [...], 'failed': [(path, err), ...]}.
        Successful uploads are deleted locally when ``keep_local`` is False.
        Shows a tqdm progress bar when ``progress`` and tqdm is installed.
        """
        deriv_root = Path(deriv_root)
        paths = sorted(deriv_root.rglob(pattern))
        out = {'uploaded': [], 'skipped': [], 'failed': []}

        def _do(p):
            uri, skipped = self._upload_one(p, deriv_root, pipeline)
            return p, uri, skipped

        def _bar(it):
            if progress:
                try:
                    from tqdm.auto import tqdm
                    return tqdm(it, total=len(paths), desc='GCS upload')
                except ImportError:
                    pass
            return it

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do, p): p for p in paths}
            for fut in _bar(as_completed(futures)):
                p = futures[fut]
                try:
                    _, uri, skipped = fut.result()
                    (out['skipped'] if skipped else out['uploaded']).append(uri)
                    if not skipped and not keep_local:
                        p.unlink(missing_ok=True)
                except Exception as exc:
                    out['failed'].append((str(p), f'{type(exc).__name__}: {exc}'))
        return out

    # ── read ────────────────────────────────────────────────────────────────
    def download(self, blob_name: str, dst) -> Path:
        """Download a blob to ``dst`` (skips if the local file already exists)."""
        dst = Path(dst)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob = self.bucket.blob(blob_name, chunk_size=self.chunk)
            blob.download_to_filename(str(dst), timeout=self.timeout, retry=self.retry)
        return dst

    @contextmanager
    def staged(self, blob_name: str, cleanup: bool = True):
        """Download one blob into the local cache, yield its path, delete it after."""
        local_path = self.cache_dir / Path(blob_name).name
        self.download(blob_name, local_path)
        try:
            yield local_path
        finally:
            if cleanup:
                try:
                    local_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def list_files(self, pipeline: str, suffix: str, tasks=None,
                   parse=parse_bids_filename) -> list:
        """List a pipeline's outputs in GCS.

        Returns ``[(blob_name, entities), ...]`` for blobs ending in ``suffix``,
        optionally filtered to ``tasks``.
        """
        out = []
        for blob in self.client.list_blobs(self.bucket_name,
                                            prefix=f'{self.prefix(pipeline)}/'):
            if not blob.name.endswith(suffix):
                continue
            ent = parse(Path(blob.name).name)
            if tasks is not None and ent.get('task') not in tasks:
                continue
            out.append((blob.name, ent))
        return out

    # ── pairing (Stage 3 raw <-> ica) ────────────────────────────────────────
    @staticmethod
    def pair_files(primary, secondary, key_fn=make_key):
        """Match ``primary`` to ``secondary`` by key.

        ``primary``/``secondary`` are ``[(ref, entities), ...]`` lists (as returned by
        ``list_files``). Returns ``(pairs, missing)`` where pairs are
        ``(primary_ref, secondary_ref, entities)`` and missing lists primary refs with
        no secondary match. Reproduces the old Stage-3 ``discover_gcs`` behavior.
        """
        sec_by_key = {key_fn(ent): ref for ref, ent in secondary}
        pairs, missing = [], []
        for ref, ent in sorted(primary, key=lambda x: str(x[0])):
            key = key_fn(ent)
            if key not in sec_by_key:
                missing.append(Path(str(ref)).name)
                continue
            pairs.append((ref, sec_by_key[key], ent))
        return pairs, missing
