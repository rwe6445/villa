import modal
img = modal.Image.debian_slim(python_version="3.14").pip_install("torch==2.12.1")
app = modal.App("villa-eigh-fix")

@app.function(gpu="T4", image=img, timeout=900)
def probe():
    import torch, time
    r = [f"torch {torch.__version__} {torch.cuda.get_device_name(0)}"]
    def run(n, dtype, lib=None, label=""):
        if lib: torch.backends.cuda.preferred_linalg_library(lib)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        M = torch.randn(n, 3, 3, dtype=dtype, device="cuda"); M = M + M.transpose(1, 2)
        t = time.time()
        try:
            w, v = torch.linalg.eigh(M); torch.cuda.synchronize()
            ref = torch.linalg.eigh(M.double().cpu())[0]
            err = (w.double().cpu() - ref).abs().max().item()
            r.append(f"  {label:<10} {str(dtype)[6:]:<8} batch {n:>8}: OK {time.time()-t:.3f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f}GiB maxerr {err:.1e}")
        except Exception as e:
            m = str(e); i = m.find("Tried to allocate")
            r.append(f"  {label:<10} {str(dtype)[6:]:<8} batch {n:>8}: {type(e).__name__} {m[i:i+30] if i>=0 else m[:50]}")
        finally:
            torch.backends.cuda.preferred_linalg_library("default")
    for n in (1024, 4096, 16384, 32768):
        run(n, torch.float64, label="cusolver")
    for n in (65536, 1048576):
        run(n, torch.float32, label="cusolver")
    for n in (65536, 1048576):
        run(n, torch.float64, lib="magma", label="magma")
    # CPU reference speed for one villa batch
    M = torch.randn(1048576, 3, 3, dtype=torch.float64); M = M + M.transpose(1, 2)
    t = time.time(); torch.linalg.eigh(M); r.append(f"  cpu float64 batch 1048576: {time.time()-t:.2f}s")
    return "\n".join(r)

@app.local_entrypoint()
def main():
    r = probe.remote(); open("eigh_fix.txt", "w").write(r); print(r)
