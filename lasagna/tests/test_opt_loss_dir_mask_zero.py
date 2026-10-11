from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import opt_loss_dir  # noqa: E402


class NormalMaskZeroTests(unittest.TestCase):
    """fit.py turns this on via opt_cfg.normal_mask_zero; the masking line referenced
    data_nx / data_ny, which no longer exist, so enabling it raised NameError."""

    def tearDown(self):
        opt_loss_dir.set_mask_zero_normals(False)

    def test_zero_data_normals_are_masked_out(self):
        D, H, W = 1, 4, 4
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        xyz = torch.stack([xs, ys, torch.zeros_like(xs)], dim=-1)[None]   # flat sheet
        target = torch.zeros(D, H, W, 3)
        target[..., 0] = 0.6
        target[..., 2] = 0.8
        target[0, 1, 2] = torch.tensor([0.0, 0.0, 1.0])                  # nx == ny == 0
        res = SimpleNamespace(xyz_lr=xyz, gt_normal_lr=target, mask_lr=torch.ones(D, 1, H, W))

        opt_loss_dir.set_mask_zero_normals(False)
        _, mask_off = opt_loss_dir.normal_loss_maps(res=res)
        opt_loss_dir.set_mask_zero_normals(True)
        _, mask_on = opt_loss_dir.normal_loss_maps(res=res)

        self.assertEqual(float(mask_off.sum()), D * H * W)
        self.assertEqual(float(mask_on.sum()), D * H * W - 1)
        self.assertEqual(float(mask_on[0, 0, 1, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
