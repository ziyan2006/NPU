"""
统计候选模型的参数量 / 权重体积 / 算力,回答"权重为什么这么大"。
不下载任何预训练权重,直接构建网络结构后统计。
"""
import inspect
import json
from pathlib import Path
import torch
import torch.nn as nn

RESULT = {}


def summarize(name, model, extra=None):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32 = total * 4 / 1024**2
    fp16 = total * 2 / 1024**2
    int8 = total * 1 / 1024**2
    entry = {
        "name": name,
        "params": total,
        "params_M": round(total / 1e6, 3),
        "trainable_M": round(train / 1e6, 3),
        "fp32_MB": round(fp32, 2),
        "fp16_MB": round(fp16, 2),
        "int8_MB": round(int8, 2),
    }
    if extra:
        entry.update(extra)
    RESULT[name] = entry
    print(f"{name:34s} params={total:>12,}  fp32={fp32:7.2f}MB  fp16={fp16:6.2f}MB  int8={int8:6.2f}MB")
    return entry


print("=" * 96)
print("1) Open-Unmix (umxhq) 单 stem 模型 —— 论文里的经典基线")
print("=" * 96)
from openunmix import model as umx_model

sig = inspect.signature(umx_model.OpenUnmix.__init__)
print("OpenUnmix signature:", sig)
print()

# umxhq: STFT n_fft=4096 -> 2049 bins, hidden 512, 3 层 BiLSTM
umx = umx_model.OpenUnmix(
    nb_channels=2,
    nb_bins=4096,
    hidden_size=512,
    nb_layers=3,
    unidirectional=False,
)
summarize("Open-Unmix umxhq (单stem)", umx)

print()
print("   逐模块拆解(看参数堆在哪):")
for mod_name, mod in umx.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n:
        print(f"     {mod_name:14s} {n:>12,}  ({n * 4 / 1024**2:6.2f} MB fp32)")

print()
print("=" * 96)
print("2) Demucs v4 / HTDemucs —— 4 stem SOTA 参考上界")
print("=" * 96)
from demucs import htdemucs

htd = htdemucs.HTDemucs(sources=["drums", "bass", "other", "vocals"])
summarize("HTDemucs (4 stems)", htd)

print()
print("   逐模块拆解:")
for mod_name, mod in htd.named_children():
    n = sum(p.numel() for p in mod.parameters())
    if n:
        print(f"     {mod_name:14s} {n:>12,}  ({n * 4 / 1024**2:7.2f} MB fp32)")

print()
print("=" * 96)
print("3) 目标画像:Zynq 7020 上真正该做的 2-stem 轻量模型")
print("=" * 96)
print("(下面是一个典型轻量频谱域 U-Net 的参数量级,仅作预算参考)")
for name, p in [
    ("轻量 2-stem U-Net (小)",  0.3e6),
    ("轻量 2-stem U-Net (中)",  0.8e6),
    ("轻量 2-stem U-Net (大)",  1.5e6),
]:
    print(f"{name:34s} params={int(p):>12,}  fp32={p * 4 / 1024**2:7.2f}MB  "
          f"fp16={p * 2 / 1024**2:6.2f}MB  int8={p / 1024**2:6.2f}MB  "
          f"BRAM占比(560KB)={p / 1024**2 / 0.547:6.1%}")

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "results" / "model_anatomy.json").open("w", encoding="utf-8") as f:
    json.dump(RESULT, f, ensure_ascii=False, indent=2)
print()
print("saved -> results/model_anatomy.json")
