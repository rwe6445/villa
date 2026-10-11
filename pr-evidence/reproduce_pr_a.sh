#!/usr/bin/env bash
# Reproduce PR-A (structure_tensor fixes) before/after on real scroll data, side by side.
# Usage:  VILLA=/path/to/villa-clone  PATCH=/path/to/patches/PR-A_structure-tensor.patch  [VENV=/path/to/py3.14-venv]  bash reproduce_pr_a.sh
# Needs: Python 3.14 venv with `pip install -e villa/vesuvius` (+ torch CPU, zarr). Runtime ~2 min on 2 CPUs.
set -euo pipefail
: "${VILLA:?set VILLA to a villa clone}"; : "${PATCH:?set PATCH to PR-A_structure-tensor.patch}"
BASE=${BASE:-d63d3e9}
WORK=${WORK:-$PWD/pr_a_repro}
[ -n "${VENV:-}" ] && . "$VENV/bin/activate"
export VESUVIUS_DISABLE_TORCH_COMPILE=1      # CPU-only boxes; unrelated to the bug
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"

git -C "$VILLA" worktree add -f -q "$WORK/base" "$BASE"
git -C "$VILLA" worktree add -f -q "$WORK/fixed" "$BASE"
git -C "$WORK/fixed" apply "$PATCH"

python - <<'EOF'
import zarr, numpy as np
url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/"
a = zarr.open_group(url, mode="r")["2"]
crop = np.asarray(a[1797:1797+65, 986:986+96, 1012:1012+96])
zarr.open_group("crop.zarr", mode="w").create_array("0", data=crop, chunks=crop.shape)
print("real crop", crop.shape, "from", url, "level 2, origin (1797,986,1012)")
EOF

run() {  # $1 = base|fixed
  local tree="$WORK/$1"; export PYTHONPATH="$tree/vesuvius/src"
  python -m vesuvius.structure_tensor.run_create_st --input_dir crop.zarr/0 --output_dir "st_$1" \
      --keep-eigen --chunk-size 32,32,32 --patch_size 65,96,96 > "st_$1.log" 2>&1
  python -m vesuvius.structure_tensor.compute_vf --input-zarr crop.zarr/0 --eigen 0:"st_$1.zarr" --eigen 1:"st_$1.zarr" \
      --output-zarr "vf_$1.zarr" --chunk-size 32,32,32 --device cpu --ome-only --downsample 2 --xi 2 > "vf_$1.log" 2>&1
  python -m vesuvius.structure_tensor.create_st --mode structure-tensor --input_dir crop.zarr/0 --output_dir "cst_$1" \
      --device cpu --keep-eigen --ome-downsample 2 --ome-scale 1 --chunk-size 32,32,32 --patch_size 65,96,96 > "cst_$1.log" 2>&1
  python - "$1" <<'EOF'
import sys, zarr, numpy as np
t = sys.argv[1]
z = np.asarray(zarr.open_group(f"vf_{t}.zarr", mode="r")["horizontal/z/0"])
print(f"[{t}] compute_vf --downsample 2: written voxels in last plane = {int((z[-1]!=0).sum())} / {z[-1].size}")
g = zarr.open_group(f"cst_{t}.zarr", mode="r")
print(f"[{t}] create_st --keep-eigen --ome-downsample 2 --ome-scale 1: normal/z scales =",
      {k: g['normal/z'][k].shape for k in g['normal/z'].keys()}, "| eigenvectors present:", "eigenvectors" in g)
EOF
}
run base
run fixed
git -C "$VILLA" worktree remove --force "$WORK/base"; git -C "$VILLA" worktree remove --force "$WORK/fixed"
