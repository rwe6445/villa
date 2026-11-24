"""Real-scroll-data before/after for the sparse-download resume digest check.

Source data: PHercParis4 20260411134726-2.400um-0.2m-78keV-masked.zarr (public
Vesuvius Challenge open data), scale 2, 16 chunks of 128^3 uint8.

Part 1 (remote, real): copy 16 chunks straight from the public S3 volume, stop
after 8, resume. Reports wall time of the copy and of the resume check.

Part 2 (local mirror of the same real chunks, simulated in-place change): the
mirror is copied, interrupted after 8 chunks, then the 8 already-copied chunk
positions are rewritten with real data from another region of the same volume
(what a re-upload looks like from the reader's side: same path, same schema,
new values). Resume with the code on main vs. this branch.

Usage: PYTHONPATH=<vesuvius/src> python realdata_demo.py <old_module.py> <out_dir>
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from vc_reader.reader import create_group

from vesuvius.ink_detection.preprocessing import download_required_zarr_chunks as new
from vesuvius.ink_detection.volume_io import open_volume, open_volume_root

VOLUME = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
)
SCALE = 2
CHUNK_IDS = tuple((74, y, x) for y in range(30, 34) for x in range(30, 34))
OTHER_Z = 80  # region used as the "new version" values in part 2


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("download_old", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk_slices(chunk_id, chunks=(128, 128, 128)):
    return tuple(slice(i * c, (i + 1) * c) for i, c in zip(chunk_id, chunks))


def interrupted_copy(module, kwargs, stop_after):
    original = module._copy_one_chunk
    calls = 0

    def stop(chunk_id):
        nonlocal calls
        calls += 1
        if calls == stop_after + 1:
            raise KeyboardInterrupt("simulated interruption")
        return original(chunk_id)

    with patch.object(module, "_copy_one_chunk", side_effect=stop):
        try:
            module.copy_chunks_to_output(**kwargs)
        except KeyboardInterrupt:
            pass


def kwargs_for(source, output, **extra):
    return {
        "volume_path": str(source),
        "volume_scale": SCALE,
        "volume_auth_json": None,
        "output_path": Path(output),
        "chunk_ids_by_scale": {SCALE: CHUNK_IDS},
        "worker_count": 1,
        "tqdm_desc": "demo",
        "overwrite": False,
        "recompress": "fast",
        **extra,
    }


def part1_remote(out: Path) -> dict:
    output = out / "remote_copy.zarr"
    kwargs = kwargs_for(VOLUME, output)
    t0 = time.perf_counter()
    interrupted_copy(new, kwargs, stop_after=8)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    stats = new.copy_chunks_to_output(**kwargs)
    t_resume = time.perf_counter() - t0
    # Time the check alone: a second resume has nothing left to copy.
    t0 = time.perf_counter()
    new.copy_chunks_to_output(**kwargs)
    t_check_only = time.perf_counter() - t0
    t0 = time.perf_counter()
    new.copy_chunks_to_output(**{**kwargs, "resume_verify_chunks": 0})
    t_no_check = time.perf_counter() - t0
    digests = new._load_chunk_digests(output)[SCALE]
    return {
        "volume": VOLUME,
        "scale": SCALE,
        "chunks": len(CHUNK_IDS),
        "first_run_seconds_8_chunks": round(t_first, 2),
        "resume_seconds_copy_8_plus_check_8": round(t_resume, 2),
        "rerun_seconds_check_default_8_of_16": round(t_check_only, 2),
        "rerun_seconds_check_disabled": round(t_no_check, 2),
        "resume_stats": {k: {str(s): n for s, n in v.items()} for k, v in stats.items()},
        "digests_recorded": len(digests),
    }


def build_mirror(mirror: Path, old_values: dict) -> None:
    root = open_volume_root(VOLUME)
    new._create_output_arrays(root, mirror, "fast")
    level = new.ZarrWriter.open(f"{mirror}/{SCALE}", auth=None)
    for chunk_id, data in old_values.items():
        level[chunk_slices(chunk_id)] = data


def classify(output_path: Path, version_a: dict, version_b: dict) -> dict:
    level = open_volume(str(output_path), SCALE)
    counts = {"version_A": 0, "version_B": 0, "neither": 0}
    for chunk_id in CHUNK_IDS:
        data = np.asarray(level[chunk_slices(chunk_id)])
        if np.array_equal(data, version_a[chunk_id]):
            counts["version_A"] += 1
        elif np.array_equal(data, version_b[chunk_id]):
            counts["version_B"] += 1
        else:
            counts["neither"] += 1
    return counts


def part2_mirror(old, out: Path) -> dict:
    remote = open_volume(VOLUME, SCALE)
    version_a = {c: np.asarray(remote[chunk_slices(c)]) for c in CHUNK_IDS}
    version_b = {c: np.asarray(remote[chunk_slices((OTHER_Z, c[1], c[2]))]) for c in CHUNK_IDS}
    assert all(not np.array_equal(version_a[c], version_b[c]) for c in CHUNK_IDS)

    results = {}
    for label, module in (("main", old), ("this_branch", new)):
        mirror = out / f"mirror_{label}.zarr"
        output = out / f"mirror_copy_{label}.zarr"
        build_mirror(mirror, version_a)
        kwargs = kwargs_for(mirror, output)
        interrupted_copy(module, kwargs, stop_after=8)
        # Source republished in place: every chunk position now holds version B.
        level = new.ZarrWriter.open(f"{mirror}/{SCALE}", auth=None)
        for chunk_id in CHUNK_IDS:
            level[chunk_slices(chunk_id)] = version_b[chunk_id]
        try:
            module.copy_chunks_to_output(**kwargs)
            outcome = "completed"
            error = None
        except ValueError as exc:
            outcome = "stopped"
            error = str(exc)
        results[label] = {
            "resume_outcome": outcome,
            "error": error,
            "output_chunks_by_version": classify(output, version_a, version_b),
        }
        if module is new and outcome == "stopped":
            module.copy_chunks_to_output(**{**kwargs, "overwrite": True})
            results[label]["after_overwrite_chunks_by_version"] = classify(output, version_a, version_b)
    return results


def main() -> int:
    old = load_old(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    report = {"part1_remote": part1_remote(out), "part2_mirror": part2_mirror(old, out)}
    text = json.dumps(report, indent=2)
    (out / "report.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
