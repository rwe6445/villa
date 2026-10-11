"""``python -m vesuvius.structure_tensor.create_st --mode structure-tensor`` must honour
the eigen/OME options it advertises.

The structure-tensor branch called the eigen step without --keep-eigen, --ome-downsample,
--ome-scale, --confidence-metric or --ome-out, so they were silently replaced by the
defaults while the closing summary still listed eigenvectors and the requested scale.
(The ``vesuvius.compute_st`` runner and ``--mode eigenanalysis`` already passed them.)
"""

from __future__ import annotations

import sys

import numpy as np
import zarr

from vesuvius.structure_tensor import create_st


def test_structure_tensor_mode_passes_eigen_options(tmp_path, monkeypatch):
    src = tmp_path / "in.zarr"
    g = zarr.open_group(str(src), mode="w")
    rng = np.random.default_rng(0)
    g.create_array("0", data=(rng.random((9, 16, 16)) * 255).astype(np.uint8), chunks=(9, 16, 16))
    out = tmp_path / "st"
    monkeypatch.setattr(sys, "argv", [
        "create_st", "--mode", "structure-tensor", "--input_dir", str(src / "0"),
        "--output_dir", str(out), "--device", "cpu", "--keep-eigen",
        "--ome-downsample", "2", "--ome-scale", "1", "--chunk-size", "8,8,8",
        "--patch_size", "9,16,16",
    ])
    monkeypatch.setenv("VESUVIUS_DISABLE_TORCH_COMPILE", "1")
    create_st.main()
    root = zarr.open_group(str(out) + ".zarr", mode="r")
    assert "eigenvectors" in root and "eigenvalues" in root
    assert list(root["normal/z"].keys()) == ["1"]
    assert root["normal/z/1"].shape == (5, 8, 8)


def test_gpu_eigen_oom_falls_back_to_cpu():
    # cuSOLVER's batched eigh asks ~0.5 MB per 3x3 float64 matrix (torch 2.12/CUDA 13),
    # so the 1,048,576-voxel batch requests ~530 GiB and raised OutOfMemoryError on
    # T4/L4/A100 alike; only "cusolver"-message errors used to reach the CPU fallback.
    import torch

    cuda = torch.device("cuda")
    oom = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 530.22 GiB.")
    assert create_st._should_solve_on_cpu(oom, cuda)
    assert create_st._should_solve_on_cpu(RuntimeError("cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR"), cuda)
    assert not create_st._should_solve_on_cpu(RuntimeError("shape mismatch"), cuda)
    assert not create_st._should_solve_on_cpu(oom, torch.device("cpu"))
