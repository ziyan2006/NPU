"""载入真实发布的预训练权重,给出精确的参数量/体积/位宽构成。"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / "models" / "umxhq_vocals.pth"
sd = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(sd, dict):
    print("type:", type(sd))
elif "state_dict" in sd:
    sd = sd["state_dict"]

total = 0
dtypes = {}
print(f"file: {p}")
print(f"{'tensor':28s} {'shape':22s} {'dtype':12s} {'params':>12s} {'bytes':>12s}")
print("-" * 92)
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    n = v.numel()
    total += n
    dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + n
    if "lstm" in k and "weight_ih" not in k:
        continue
    print(f"{k:28s} {str(tuple(v.shape)):22s} {str(v.dtype):12s} {n:>12,} {n * v.element_size():>12,}")

print("-" * 92)
print(f"TOTAL params: {total:,}  ({total/1e6:.2f} M)")
print("dtype breakdown:", {k: f"{v/1e6:.2f}M" for k, v in dtypes.items()})
for dt, n in dtypes.items():
    sz = n * (4 if "float32" in dt else 2 if "float16" in dt else 1)
    print(f"  若按 {dt} 存储 -> {sz/1024**2:.2f} MB ; 量化到 INT8 -> {n/1024**2:.2f} MB")
