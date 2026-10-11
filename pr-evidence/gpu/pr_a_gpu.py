"""PR-A reproduction on GPU with default parameters (xi=5, chunk 128) on a larger real crop.

Runs the same pipeline against the base tree (d63d3e9) and the fixed tree, inside one Modal T4 container.
Run:  modal run pr_a_gpu.py
"""
import modal

IGNORE = ["**/__pycache__", "**/*.pyc"]
image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install("torch>=2.12,<3", "torchvision")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("fixed/segmentation/models/arch/nnunet", "/pkgs/nnunet", ignore=IGNORE, copy=True)
    .add_local_dir("fixed/segmentation/models/batchgeneratorsv2", "/pkgs/bgv2", ignore=IGNORE, copy=True)
    .run_commands("pip install --no-deps /pkgs/nnunet /pkgs/bgv2")
    .add_local_dir("base/vesuvius/src", "/src/base", ignore=IGNORE)
    .add_local_dir("fixed/vesuvius/src", "/src/fixed", ignore=IGNORE)
)
app = modal.App("villa-pr-a-repro", image=image)

URL = ("https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
       "volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/")
ORIGIN = (1797, 986, 1012)
SIZE = (257, 256, 256)      # odd z: with chunk 128 the last block [256, 257) is edge-clipped


@app.function(gpu="T4", timeout=1800, retries=0)
def run() -> str:
    import os, subprocess, sys, json, time
    import numpy as np, zarr, torch
    out = []
    log = lambda s: (print(s, flush=True), out.append(str(s)))
    log(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    os.chdir("/tmp")
    a = zarr.open_group(URL, mode="r")["2"]
    z, y, x = ORIGIN
    crop = np.asarray(a[z:z + SIZE[0], y:y + SIZE[1], x:x + SIZE[2]])
    zarr.open_group("crop.zarr", mode="w").create_array("0", data=crop, chunks=(128, 128, 128))
    log(f"real crop {crop.shape} from {URL} level 2 origin {ORIGIN}")

    def sh(tree, args, name):
        env = dict(os.environ, PYTHONPATH=f"/src/{tree}")
        t = time.time()
        r = subprocess.run([sys.executable, "-m", *args], env=env, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        log(f"[{tree}] {name}: exit {r.returncode} in {time.time()-t:.0f}s | " + " / ".join(tail)[:300])
        return r.returncode

    for tree in ("base", "fixed"):
        sh(tree, ["vesuvius.structure_tensor.run_create_st", "--input_dir", "crop.zarr/0",
                  "--output_dir", f"st_{tree}", "--keep-eigen", "--chunk-size", "128,128,128",
                  "--patch_size", "128,128,128"], "compute_st runner")
        sh(tree, ["vesuvius.structure_tensor.compute_vf", "--input-zarr", "crop.zarr/0",
                  "--eigen", f"0:st_{tree}.zarr", "--eigen", f"1:st_{tree}.zarr",
                  "--output-zarr", f"vf_{tree}.zarr", "--chunk-size", "128,128,128",
                  "--device", "cuda", "--ome-only", "--downsample", "2"], "compute_vf xi=5 ds=2")
        if not os.path.exists(f"vf_{tree}.zarr"):
            log(f"[{tree}] vf output missing"); continue
        zz = np.asarray(zarr.open_group(f"vf_{tree}.zarr", mode="r")["horizontal/z/0"])
        per = [int((zz[k] != 0).sum()) for k in range(zz.shape[0])]
        log(f"[{tree}] vf shape {zz.shape}; last plane written {per[-1]}/{zz[-1].size}; "
            f"planes fully written {sum(p == zz[0].size for p in per)}/{len(per)}")
        sh(tree, ["vesuvius.structure_tensor.create_st", "--mode", "structure-tensor",
                  "--input_dir", "crop.zarr/0", "--output_dir", f"cst_{tree}", "--device", "cuda",
                  "--keep-eigen", "--ome-downsample", "2", "--ome-scale", "1",
                  "--chunk-size", "128,128,128", "--patch_size", "128,128,128"], "create_st direct")
        if not os.path.exists(f"cst_{tree}.zarr"):
            log(f"[{tree}] create_st output missing"); continue
        g = zarr.open_group(f"cst_{tree}.zarr", mode="r")
        log(f"[{tree}] create_st normal/z scales " +
            json.dumps({k: list(g['normal/z'][k].shape) for k in g['normal/z'].keys()}) +
            f" | eigenvectors present: {'eigenvectors' in g}")
    if os.path.exists("vf_base.zarr") and os.path.exists("vf_fixed.zarr"):
        b = np.asarray(zarr.open_group("vf_base.zarr", mode="r")["horizontal/z/0"])
        f = np.asarray(zarr.open_group("vf_fixed.zarr", mode="r")["horizontal/z/0"])
        log(f"planes 0..{b.shape[0]-2} byte-identical base vs fixed: {np.array_equal(b[:-1], f[:-1])}")
    return "\n".join(out)



@app.function(gpu="T4", timeout=900, retries=0)
def diag() -> str:
    import os, subprocess, sys, numpy as np, zarr
    os.chdir("/tmp")
    a = zarr.open_group(URL, mode="r")["2"]; z, y, x = ORIGIN
    crop = np.asarray(a[z:z+SIZE[0], y:y+SIZE[1], x:x+SIZE[2]])
    zarr.open_group("crop.zarr", mode="w").create_array("0", data=crop, chunks=(128,128,128))
    env = dict(os.environ, PYTHONPATH="/src/fixed")
    r = subprocess.run([sys.executable, "-m", "vesuvius.structure_tensor.create_st", "--mode", "structure-tensor",
        "--input_dir", "crop.zarr/0", "--output_dir", "cst", "--device", "cuda", "--keep-eigen",
        "--chunk-size", "128,128,128", "--patch_size", "128,128,128", "--verbose"], env=env, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    keep = [l for l in lines if ("File \"/src" in l or "Error" in l or "[Eigen]" in l or "---" in l or "Tried" in l or l.strip().startswith(("eig", "block", "_compute", "ST", "J ")))]
    return f"exit {r.returncode}\n" + "\n".join(keep[-60:])


@app.function(gpu="T4", timeout=900, retries=0)
def eigtest() -> str:
    import os, subprocess, sys, time, torch, numpy as np, zarr
    out=[]
    M = torch.randn(1048576, 3, 3, dtype=torch.float64, device="cuda"); M = M + M.transpose(1, 2)
    t=time.time()
    try:
        w, v = torch.linalg.eigh(M); torch.cuda.synchronize()
        out.append(f"eager torch.linalg.eigh float64 batch=1048576 OK {time.time()-t:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB")
    except Exception as e:
        out.append(f"eager eigh FAIL {type(e).__name__}: {str(e)[:160]}")
    os.chdir("/tmp")
    a = zarr.open_group(URL, mode="r")["2"]; z, y, x = ORIGIN
    crop = np.asarray(a[z:z+SIZE[0], y:y+SIZE[1], x:x+SIZE[2]])
    zarr.open_group("crop.zarr", mode="w").create_array("0", data=crop, chunks=(128,128,128))
    for tag, extra in (("compile-off", {"VESUVIUS_DISABLE_TORCH_COMPILE": "1"}),):
        env = dict(os.environ, PYTHONPATH="/src/fixed", **extra)
        t=time.time()
        r = subprocess.run([sys.executable, "-m", "vesuvius.structure_tensor.create_st", "--mode", "structure-tensor",
            "--input_dir", "crop.zarr/0", "--output_dir", f"cst_{tag}", "--device", "cuda", "--keep-eigen",
            "--ome-downsample", "2", "--ome-scale", "1", "--chunk-size", "128,128,128", "--patch_size", "128,128,128"],
            env=env, capture_output=True, text=True)
        tail=" / ".join((r.stdout+r.stderr).strip().splitlines()[-2:])[:200]
        ok = os.path.exists(f"cst_{tag}.zarr")
        info = ""
        if ok:
            g = zarr.open_group(f"cst_{tag}.zarr", mode="r")
            info = f"keys={sorted(g.keys())}"
        out.append(f"create_st {tag}: exit {r.returncode} in {time.time()-t:.0f}s | {tail} | {info}")
    return "\n".join(out)


@app.local_entrypoint()
def eig_main():
    res = eigtest.remote(); open("eig_result.txt","w").write(res); print(res)


@app.function(gpu="T4", timeout=1800, retries=0)
def predict_bbox() -> str:
    import os, subprocess, sys, time, numpy as np, zarr
    out = []
    os.chdir("/tmp")
    a = zarr.open_group(URL, mode="r")["2"]; z, y, x = ORIGIN
    crop = np.asarray(a[z:z+SIZE[0], y:y+SIZE[1], x:x+SIZE[2]])
    zarr.open_group("crop.zarr", mode="w").create_array("0", data=crop, chunks=(128,128,128))
    out.append(f"real crop {crop.shape} level 2 origin {ORIGIN}")
    def run(tree, bbox, tag):
        env = dict(os.environ, PYTHONPATH=f"/src/{tree}")
        t = time.time()
        r = subprocess.run([sys.executable, "-m", "vesuvius.models.run.inference",
            "--model_path", "hf://scrollprize/surface_recto", "--input_dir", "crop.zarr/0",
            "--output_dir", f"pred_{tree}_{tag}", "--device", "cuda", "--bbox", bbox, "--disable_tta"],
            env=env, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).strip().splitlines()
        err = [l for l in lines if "Error" in l or "error:" in l][-2:]
        out.append(f"[{tree}] --bbox {bbox!r}: exit {r.returncode} in {time.time()-t:.0f}s | " + " / ".join(err or lines[-2:])[:260])
        if r.returncode == 0:
            for root, dirs, files in os.walk(f"pred_{tree}_{tag}"):
                if ".zarray" in files or "zarr.json" in files:
                    try:
                        arr = zarr.open_array(root, mode="r")
                        out.append(f"    output array {root} shape {arr.shape} dtype {arr.dtype}")
                    except Exception:
                        pass
    run("fixed", "64:192,64:192,64:192", "ok")
    return "\n".join(out)


@app.local_entrypoint()
def predict_main():
    r = predict_bbox.remote(); open("predict_result.txt","w").write(r); print(r)


@app.local_entrypoint()
def diag_main():
    res = diag.remote(); open("diag_result.txt","w").write(res); print(res)


@app.local_entrypoint()
def main():
    res = run.remote()
    open("pr_a_gpu_result.txt", "w").write(res)
    print("\n=== RESULT ===\n" + res)
