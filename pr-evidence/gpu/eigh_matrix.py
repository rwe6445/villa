import modal
img214 = modal.Image.debian_slim(python_version="3.14").pip_install("torch==2.14.1")
img212 = modal.Image.debian_slim(python_version="3.14").pip_install("torch==2.12.1")
app = modal.App("villa-eigh-matrix")

def _probe():
    import torch, time
    res = [f"torch {torch.__version__} cuda {torch.version.cuda} {torch.cuda.get_device_name(0)}"]
    for n in (65536, 262144, 524288, 1048576):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        M = torch.randn(n, 3, 3, dtype=torch.float64, device="cuda"); M = M + M.transpose(1, 2)
        t = time.time()
        try:
            torch.linalg.eigh(M); torch.cuda.synchronize()
            res.append(f"  batch {n:>8}: OK {time.time()-t:.2f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        except Exception as e:
            msg = str(e); i = msg.find("Tried to allocate")
            res.append(f"  batch {n:>8}: {type(e).__name__} {msg[i:i+32] if i>=0 else msg[:60]}")
        del M
    return "\n".join(res)

@app.function(gpu="T4", image=img214, timeout=600)
def p214(): return _probe()

@app.function(gpu="T4", image=img212, timeout=600)
def p212(): return _probe()

@app.function(gpu="L4", image=img212, timeout=600)
def p212_l4(): return _probe()

@app.function(gpu="A100", image=img212, timeout=600)
def p212_a100(): return _probe()

@app.local_entrypoint()
def main():
    r = p214.remote() + "\n" + p212.remote()
    open("eigh_matrix.txt", "w").write(r); print(r)

@app.local_entrypoint()
def other_gpus():
    r = p212_l4.remote() + "\n" + p212_a100.remote()
    open("eigh_matrix_gpus.txt", "w").write(r); print(r)
