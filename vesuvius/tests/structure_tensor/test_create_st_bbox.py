"""compute_st / create_st --bbox: structure tensors for one region of a volume.

predict already restricts its sliding window with --bbox (the shared Inferer
supports it), but the structure-tensor CLIs did not expose it, so a region of a
remote scroll could only be processed by running the whole volume. The eigen
stage also solved every chunk of the full-volume store, written or not.

With --bbox the structure tensor is computed for the ROI only, in the global
frame, and the eigen stage solves only chunks that overlap stored data. Inside
the ROI the result must match a full-volume run.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import zarr

from vesuvius.structure_tensor import create_st as cs

SHAPE = (64, 64, 64)
CHUNK = 16
ROI = (slice(16, 32), slice(32, 48), slice(16, 32))
BBOX = "16:32,32:48,16:32"


@pytest.fixture
def volume(tmp_path):
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    vol = (120 + 60 * np.sin(0.3 * z + 0.1 * y) + rng.normal(0, 5, SHAPE)).clip(0, 255).astype(np.uint8)
    path = str(tmp_path / "vol.zarr")
    zarr.open_array(path, mode="w", shape=SHAPE, chunks=(CHUNK,) * 3, dtype="u1", zarr_format=2)[:] = vol
    return path


def _run(monkeypatch, volume, out, bbox=None):
    solved = []
    real_loader = cs.DataLoader

    def counting_loader(dataset, *a, **k):
        if type(dataset).__name__ == "GroupChunkDataset":
            solved.append(len(dataset))
        return real_loader(dataset, *a, **k)

    monkeypatch.setattr(cs, "DataLoader", counting_loader)
    argv = ["create_st", "--mode", "structure-tensor", "--input_dir", volume, "--output_dir", out,
            "--device", "cpu", "--sigma", "1.0", "--patch_size", f"{CHUNK},{CHUNK},{CHUNK}",
            "--chunk-size", f"{CHUNK},{CHUNK},{CHUNK}"]
    if bbox:
        argv += ["--bbox", bbox]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("VESUVIUS_DISABLE_TORCH_COMPILE", "1")
    cs.main()
    root = zarr.open_group(out + ".zarr" if not out.endswith(".zarr") else out, mode="r")
    return root, solved


def test_bbox_restricts_work_and_matches_full_run(tmp_path, monkeypatch, volume):
    roi_root, roi_solved = _run(monkeypatch, volume, str(tmp_path / "roi"), BBOX)
    full_root, full_solved = _run(monkeypatch, volume, str(tmp_path / "full"))

    n_full = (SHAPE[0] // CHUNK) ** 3
    assert full_solved == [n_full]
    assert roi_solved and roi_solved[0] < n_full  # eigen stage skips chunks with no structure tensor

    st_roi = np.asarray(roi_root["structure_tensor"][(slice(None),) + ROI])
    st_full = np.asarray(full_root["structure_tensor"][(slice(None),) + ROI])
    assert np.abs(st_full).sum() > 0
    np.testing.assert_array_equal(st_roi, st_full)
    for comp in ("z", "y", "x"):  # default OME output of the eigen stage
        n_roi = np.asarray(roi_root[f"normal/{comp}/0"][ROI])
        n_full = np.asarray(full_root[f"normal/{comp}/0"][ROI])
        np.testing.assert_array_equal(n_roi, n_full)
