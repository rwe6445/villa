#!/usr/bin/env bash
# Before/after for "zarr_tasks: write the per-level scale OME-NGFF 0.4 requires" on real scroll data.
# Usage:  VILLA=/path/to/villa-clone  [VENV=/path/to/py3.14-venv]  bash reproduce_pr_h.sh
# Needs:  Python 3.14 with `pip install -e villa/vesuvius tifffile opencv-python-headless ome-zarr-models`.
#         Network access to dl.ash2txt.org. Runtime about 1 minute on 2 CPUs.
set -euo pipefail
: "${VILLA:?set VILLA to a villa clone that has upstream main and origin/pr-h-ngff-scale-v2}"
BASE=${BASE:-upstream/main}
FIX=${FIX:-origin/pr-h-ngff-scale-v2}
WORK=${WORK:-$PWD/pr_h_repro}
[ -n "${VENV:-}" ] && . "$VENV/bin/activate"
rm -rf "$WORK" && mkdir -p "$WORK" && cd "$WORK"

git -C "$VILLA" worktree add -f -q "$WORK/base" "$BASE"
git -C "$VILLA" worktree add -f -q "$WORK/fixed" "$FIX"
echo "base  = $(git -C "$WORK/base" log -1 --format='%h %s')"
echo "fixed = $(git -C "$WORK/fixed" log -1 --format='%h %s')"

# Same real crop as the earlier evidence: Scroll 1A, level 2, origin (1797, 986, 1012).
python - <<'EOF'
import zarr, numpy as np
url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/"
src = zarr.open_group(url, mode="r")
crop = np.asarray(src["2"][1797:1797+128, 986:986+128, 1012:1012+128])
zarr.open_group("crop.zarr", mode="w", zarr_format=2).create_array("0", data=crop, chunks=(64, 64, 64))
print("real crop", crop.shape, crop.dtype, "from", url, "level 2, origin (1797,986,1012)")
print("source volume multiscales datasets[0]:", src.attrs["multiscales"][0]["datasets"][0])
EOF

validate() {  # $1 = path of a zarr group
  python - "$1" <<'EOF'
import sys, json, zarr
from ome_zarr_models.v04 import Image
g = zarr.open_group(sys.argv[1], mode="r")
print("  datasets:", json.dumps(g.attrs["multiscales"][0]["datasets"]))
try:
    Image.from_zarr(g)
    print("  OME-NGFF 0.4 validation: PASS")
except Exception as e:
    print("  OME-NGFF 0.4 validation: FAIL ->", type(e).__name__, str(e).splitlines()[0])
EOF
}

for t in base fixed; do
  echo "== [$t] zarr_tasks --task threshold -t 100 --num-levels 3"
  PYTHONPATH="$WORK/$t/vesuvius/src" python -m vesuvius.image_proc.run.zarr_tasks \
      crop.zarr/0 "thr_$t.zarr" --task threshold -t 100 --num-levels 3 -n 2 > "thr_$t.log" 2>&1
  echo "  exit 0"
  validate "thr_$t.zarr"
done

python - <<'EOF'
import zarr, numpy as np
a, b = zarr.open_group("thr_base.zarr", mode="r"), zarr.open_group("thr_fixed.zarr", mode="r")
same = all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ("0", "1", "2"))
print("pixel data identical across all 3 levels:", same)
EOF

git -C "$VILLA" worktree remove --force "$WORK/base"; git -C "$VILLA" worktree remove --force "$WORK/fixed"
