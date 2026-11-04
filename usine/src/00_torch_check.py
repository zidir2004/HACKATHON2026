"""Vérifie que PyTorch voit le GPU (piste B)."""
import time, torch
print("torch", torch.__version__, "cuda dispo", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30,2), "Go", flush=True)
    a = torch.randn(4096, 4096, device="cuda"); t = time.time()
    for _ in range(50): b = a @ a
    torch.cuda.synchronize(); print("50 matmul 4096 :", round(time.time()-t, 3), "s", flush=True)
