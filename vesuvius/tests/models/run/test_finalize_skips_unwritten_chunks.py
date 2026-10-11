"""finalize_outputs must not visit logits chunks that were never written.

After ``vesuvius.predict --bbox`` the merged logits store has the shape of the
whole volume but only the ROI chunks exist on disk. finalize_logits used to
schedule, open and read every chunk of the volume (448k chunks for a level-0
Scroll 1A ROI), all of which came back as fill_value and were dropped as empty.
It now builds the chunk occupancy of the logits store first and only processes
output chunks that overlap a stored chunk. The written output must not change.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run import finalize_outputs as fo

SHAPE = (2, 64, 64, 64)
CHUNKS = (2, 16, 16, 16)
ROI = (slice(None), slice(16, 40), slice(0, 16), slice(32, 48))  # spans 2 x 1 x 1 input chunks


def _sparse_logits(path):
    store = open_zarr(path, mode="w", shape=SHAPE, chunks=CHUNKS, dtype=np.float16,
                      fill_value=0, write_empty_chunks=False)
    rng = np.random.default_rng(0)
    block = rng.normal(size=store[ROI].shape).astype(np.float16)
    block[1] += 2.0  # foreground logit above background so the output is non-trivial
    store[ROI] = block
    return path


def _run(monkeypatch, logits, out, chunk_size, occupancy=True):
    calls = []
    real = fo.process_chunk

    def counting(chunk_info, *a, **k):
        calls.append(chunk_info["indices"])
        return real(chunk_info, *a, **k)

    monkeypatch.setattr(fo, "process_chunk", counting)
    monkeypatch.setattr(fo, "ProcessPoolExecutor", ThreadPoolExecutor)
    if not occupancy:
        monkeypatch.setattr(fo, "build_chunk_occupancy", lambda *a, **k: None, raising=False)
    fo.finalize_logits(str(logits), str(out), mode="binary", chunk_size=chunk_size,
                       num_workers=1, verbose=False)
    return sorted(calls), np.asarray(zarr.open_array(str(out), mode="r")[:])


@pytest.mark.parametrize("chunk_size,expected", [
    (None, [(1, 0, 2), (2, 0, 2)]),          # output grid = input grid
    ((24, 24, 24), [(0, 0, 1), (1, 0, 1)]),  # output grid not aligned to input grid
])
def test_only_chunks_with_stored_logits_are_processed(tmp_path, monkeypatch, chunk_size, expected):
    logits = _sparse_logits(str(tmp_path / "merged.zarr"))
    calls, _ = _run(monkeypatch, logits, tmp_path / "final.zarr", chunk_size)
    assert calls == expected


@pytest.mark.parametrize("chunk_size", [None, (24, 24, 24)])
def test_output_is_unchanged(tmp_path, monkeypatch, chunk_size):
    logits = _sparse_logits(str(tmp_path / "merged.zarr"))
    _, skipped = _run(monkeypatch, logits, tmp_path / "skip.zarr", chunk_size)
    calls_all, full = _run(monkeypatch, logits, tmp_path / "full.zarr", chunk_size, occupancy=False)
    assert len(calls_all) == np.prod([-(-s // c) for s, c in zip(SHAPE[1:], chunk_size or CHUNKS[1:])])
    assert skipped.any()
    np.testing.assert_array_equal(skipped, full)
