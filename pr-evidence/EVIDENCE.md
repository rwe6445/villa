# villa fixes — evidence (branch `fix/latent-runtime-errors`, base `d63d3e9`, 2026-10-11)

All runs: CPU-only workspace (2 vCPU, 7 GB RAM), Python 3.14.6, torch 2.14.1+cpu, zarr 3.4.1.
Real data: PHerc. Paris 4 (Scroll 1A) `54keV_7.91um_Scroll1A.zarr`, level `2`, origin (z,y,x) = (1797, 986, 1012),
shape 65×96×96, from `https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/`.
The crop is included as `evidence/scroll1a_L2_crop.zarr`.

| # | Commit | Entry point | Real-data evidence | Regression test fails before / passes after |
|---|---|---|---|---|
| 0 | structure_tensor: solve eigen chunks on CPU when the GPU solve runs out of memory | `vesuvius.compute_st` on any GPU | yes (§0, Modal T4) | yes |
| 1 | structure_tensor: write every downsampled plane | `python -m vesuvius.structure_tensor.compute_vf --downsample 2` | yes (below) | yes |
| 2 | structure_tensor: honour eigen options in create_st | `python -m vesuvius.structure_tensor.create_st --mode structure-tensor` | yes (below) | yes |
| 3 | (dropped — duplicate of open PR #2013) ink-detection entrypoint IndentationError | Docker `ENTRYPOINT ["python", "/app/entrypoint.py"]` | startup only (no model run) | yes |
| 4 | vesuvius.predict malformed `--bbox` | `vesuvius.predict` | yes (GPU, `hf://scrollprize/surface_recto`, §0b) | yes |
| 5 | (issue, not PR) lasagna normal_mask_zero NameError | `fit.py` with `opt_cfg.normal_mask_zero` | no | yes |
| 6 | (issue, not PR) lasagna dataset_vis missing torch import | `lasagna3d/dataset_vis.py` model preview | no | — |

## 0. GPU runs (Modal, default parameters) — added 2026-10-11

Real crop 257×256×256 of the same volume/level/origin; `--chunk-size 128,128,128`, `--patch_size 128,128,128`, compute_vf default `--xi 5`. T4, torch 2.14.1+cu130. Script and full output: `evidence/gpu/pr_a_gpu.py`, `evidence/gpu/pr_a_gpu_result.txt`.

| Step | base (d63d3e9) | fixed |
|---|---|---|
| `vesuvius.compute_st --keep-eigen` | exit 1: `OutOfMemoryError: Tried to allocate 530.22 GiB` in eigenanalysis | exit 0 (86 s) |
| `create_st --mode structure-tensor --keep-eigen --ome-downsample 2 --ome-scale 1` | exit 1 (same OOM); `normal/z/0` 257×256×256, no eigenvectors | exit 0; `normal/z/1` 129×128×128, eigenvectors present |
| `compute_vf --downsample 2` (xi 5) | last plane 0/16384 | 129/129 planes written |

Base compute_vf ran on the zero-filled eigenvector array left by the failed eigen step, so base/fixed planes 0–127 are not comparable on GPU; byte-identity of untouched planes is shown by the CPU run in §1.

### GPU eigen OOM is not T4-specific (commit 891d4d9, patch 0005)
`torch.linalg.eigh` on N symmetric 3×3 float64 matrices, nothing else running (`evidence/gpu/eigh_matrix*.txt`, `eigh_fix.txt`):

| torch | GPU | N=65,536 | N=1,048,576 (villa's batch) |
|---|---|---|---|
| 2.14.1+cu130 | T4 | OOM, asks 33.14 GiB | OOM, asks 530.22 GiB |
| 2.12.1+cu130 (uv.lock) | T4 | OOM, 33.14 GiB | OOM, 530.22 GiB |
| 2.12.1+cu130 | L4 | OOM, 33.14 GiB | OOM, 530.22 GiB |
| 2.12.1+cu130 | A100 40GB | CUSOLVER_STATUS_INTERNAL_ERROR | OOM, 530.22 GiB |

~0.5 MB workspace per matrix; on T4 N ≤ 16,384 succeeds (8.3 GiB). float32 and `preferred_linalg_library("magma")` do not avoid it. CPU solves 1,048,576 in 1.9 s. The fix routes OutOfMemoryError to the existing CPU fallback (which previously only matched "cusolver"); same float64 LAPACK result.

## 0b. vesuvius.predict --bbox on GPU (Modal T4, same 257×256×256 crop, model `hf://scrollprize/surface_recto`, `--disable_tta`)

- base, `--bbox 1,2,3`: exit 1 — `ValueError: --bbox z-range '1' must contain ':'` followed by `NameError: name 'parser' is not defined` (`evidence/gpu/pr_c_bad_bbox.txt`)
- fixed, `--bbox 1,2,3`: exit 2 — `error: --bbox z-range '1' must contain ':' (use ':' alone for the full axis)`
- fixed, `--bbox 64:192,64:192,64:192`: exit 0 in 77 s; 8 patches inferred; `logits_part_0.zarr` (8, 2, 192, 192, 192) float16 (`evidence/gpu/pr_c_valid_bbox.txt`)

## 1. compute_vf --downsample drops the last plane (silent)

Figure: `evidence/pr_a_last_plane.png`. One-command side-by-side reproduction (fresh worktrees of base and patched tree, fresh download of the crop, ~70 s on 2 CPUs): `reproduce_pr_a.sh`.


```
python -m vesuvius.structure_tensor.run_create_st --input_dir scroll1a_L2_crop.zarr/0 --output_dir st_run \
    --keep-eigen --chunk-size 32,32,32 --patch_size 65,96,96
VESUVIUS_DISABLE_TORCH_COMPILE=1 python -m vesuvius.structure_tensor.compute_vf \
    --input-zarr scroll1a_L2_crop.zarr/0 --eigen 0:st_run.zarr --eigen 1:st_run.zarr \
    --output-zarr vf.zarr --chunk-size 32,32,32 --device cpu --ome-only --downsample 2 --xi 2
```
Non-zero voxels per downsampled z-plane of `horizontal/z/0` (shape 33×48×48; 2304 voxels per plane):
- before: planes 0–31 = 2304, **plane 32 = 0** — the run printed "✔ OME-ish uint8 written" (`vf_before.log`, `vf_before_planes.txt`)
- after: all 33 planes = 2304; planes 0–31 are byte-identical to before (`vf_after_planes.txt`)

Cause: target slice `z0//ds : z1//ds` vs samples `block[::ds]`. The last block [64, 65) is clipped to the
volume edge, so the target slice is `[32:32]` (empty). create_st already used the correct on-grid rule
(`_axis_ds`); it is now shared as `vf_format.downsample_axis`.
Note: `--xi 2` and `VESUVIUS_DISABLE_TORCH_COMPILE=1` only to fit the CPU box (`--xi 5` = 31³ kernel exhausts 7 GB on CPU); unrelated to the bug.

## 2. create_st structure-tensor mode ignores --keep-eigen / --ome-downsample / --ome-scale

```
python -m vesuvius.structure_tensor.create_st --mode structure-tensor --input_dir scroll1a_L2_crop.zarr/0 \
    --output_dir st_ds --device cpu --keep-eigen --ome-downsample 2 --ome-scale 1 --chunk-size 32,32,32 --patch_size 65,96,96
```
- before (`st_ds.log`): summary says `scale '1', ds=2` and lists `st_ds.zarr/eigenvectors`; on disk: `normal/z/0` at 65×96×96, no `eigenvectors`/`eigenvalues`
- after (`st_ds_after.log`): `normal/z/1` at 33×48×48, `eigenvectors` (9,65,96,96) and `eigenvalues` present
- `vesuvius.compute_st` (run_create_st) and `--mode eigenanalysis` were already correct; only direct structure-tensor mode was affected.

## 3. ink-detection optimized_inference entrypoint

- before (base): `IndentationError: unexpected indent (entrypoint.py, line 700)` on start — introduced by #1812 (2026-10-08)
- after: starts and reaches config validation (`RuntimeError: Missing required env var: MODEL` without env). A full inference run was not done here.

## Test suites (before → after)

- vesuvius: 990 passed after; remaining 3 collection errors are `ModuleNotFoundError: lasagna` (environment, present before). (`pytest_vesuvius.txt`, `after_vesuvius.txt`)
- lasagna: identical FAILED/ERROR set before and after (43 entries, all pre-existing/env); +1 passing test. (`before_lasagna.txt`, `after_lasagna.txt`)
- New regression tests were run against the original source with `git stash`: each fails there and passes with the fix.

## How these were found

Static analysis (ruff F821/B/PLE rules, `py_compile`) and an LLM pass (Claude haiku over line-numbered source chunks, every quoted line checked against the file) produced candidates. Only candidates reproduced on real scroll data or by a failing test were kept; candidates without callers (e.g. `maybe_swap_triplet_branch_channels(rng=None)`, `utils.utils.generate_positions` on empty ranges, the unpopulated per-volume dilation branch in `train.py`) were not turned into PRs. Cross-family review of the diff: hasa `gpt-oss-120b` raised no objection (weak signal; not counted as verification).
