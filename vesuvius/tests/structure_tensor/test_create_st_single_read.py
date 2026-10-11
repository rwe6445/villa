"""The structure-tensor stage must read each patch once.

_process_batches iterated the inferer's DataLoader, whose dataset reads (and
normalises) every patch from the input volume, but only used each item's
position; it then read the same patch again, with its padding, from the input
array. For a remote volume every patch was downloaded twice. It now walks the
active patch positions directly, so the dataset's __getitem__ is never called,
and the written tensor is unchanged.
"""

from __future__ import annotations

import sys

import numpy as np
import zarr

from vesuvius.data.vc_dataset import VCDataset
from vesuvius.structure_tensor import create_st as cs

SHAPE = (48, 48, 48)
CHUNK = 16


def test_patches_are_not_read_through_the_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    z = np.arange(SHAPE[0])[:, None, None]
    vol = (120 + 60 * np.sin(0.4 * z) + rng.normal(0, 5, SHAPE)).clip(0, 255).astype(np.uint8)
    src = str(tmp_path / "vol.zarr")
    zarr.open_array(src, mode="w", shape=SHAPE, chunks=(CHUNK,) * 3, dtype="u1", zarr_format=2)[:] = vol

    reads = []
    real_getitem = VCDataset.__getitem__

    def counting_getitem(self, idx):
        reads.append(idx)
        return real_getitem(self, idx)

    monkeypatch.setattr(VCDataset, "__getitem__", counting_getitem)
    monkeypatch.setenv("VESUVIUS_DISABLE_TORCH_COMPILE", "1")
    out = str(tmp_path / "st")
    monkeypatch.setattr(sys, "argv", [
        "create_st", "--mode", "structure-tensor", "--structure-tensor-only", "--input_dir", src,
        "--output_dir", out, "--device", "cpu", "--sigma", "1.0",
        "--patch_size", f"{CHUNK},{CHUNK},{CHUNK}"])
    cs.main()

    assert reads == []  # every patch is read once, from the input array, with its padding
    st = np.asarray(zarr.open_group(out + ".zarr", mode="r")["structure_tensor"][:])
    assert st.shape == (6, *SHAPE)
    assert np.isfinite(st).all() and np.abs(st).sum() > 0
    # every patch was written: no all-zero patch-sized block is left behind
    n = SHAPE[0] // CHUNK
    for i in range(n):
        for j in range(n):
            for k in range(n):
                blk = st[:, i * CHUNK:(i + 1) * CHUNK, j * CHUNK:(j + 1) * CHUNK, k * CHUNK:(k + 1) * CHUNK]
                assert np.abs(blk).sum() > 0
