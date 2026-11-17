"""Confirms PyTorch can run real kernels on this GPU (Blackwell cards need a recent build)."""
import time, torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        a = torch.randn(2048, 2048, device="cuda", dtype=dt)
        t = time.time(); r = (a @ a).float().sum().item(); torch.cuda.synchronize()
        print(f"  matmul {str(dt):15s} ok  ({time.time()-t:.2f}s)")
    print("VRAM total/free GB:", [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()][::-1])
