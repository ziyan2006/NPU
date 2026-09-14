# results/ — 关键实验证据（**精选子集，非全量**）

> 目录名保持 `results/` 是为了让 `scripts/` 里的相对路径约定（`ROOT/results`）开箱即用。
> 原工作目录的 `results/` 有 **24 GB**（含逐曲伪标签缓存 `*_cache/`），此处只保留**能支撑结论的汇总证据**。

## 文件说明

### A/B 对比（低方差三位置协议，逐曲原始数据在 JSON 的 `per_track` 字段）

| 文件 | 对比 |
|---|---|
| `ab_v1_v2.json` | v1 vs v2 |
| `ab_v2_v3.json` | v2 vs v3 |
| `ab_v3_v4.json` | v3 vs v4（**−0.06 dB，打平略负**） |
| `ab_v3_augprobe.json` | v3 vs 数据增强探针（**负结果**） |
| `ab_v4_probeB.json` | v4 vs probeB 能量加权损失（**−2.18 dB，负结果**） |
| `ab_v4_probeC_f0.25 / f0.50 / f0.75.json` | v4 vs probeC 掩码加权损失（**首个正向**） |
| `ab_probeC_k3s_f0.25 / f0.50 / f0.75.json` | probeC vs k3s（K=3）**三位置** |
| `ab_probeC_k3s_pooled.json` | 上述三位置**逐曲合并** → **+0.21 dB（13/19）** |
| `ab_probeC_k3sbest_f0.50 / f0.75.json` | 换检查点复核（`k3s_best`）→ 仍为正 |
| `ab_k3srev_probeCrev.json` | **角色对调对照**（a/b 互换）→ 排除 slot 偏置 |

### 训练运行

| 文件 | 说明 |
|---|---|
| `k3s_report.json` / `k3s.log` | K=3、900 s 等预算对照（**留出集 +0.21 dB，val 更差**） |
| `k3w_report.json` / `k3w.log` | K=3、3 h 长跑（**val 从未改善一步 → 无新权重产出**） |
| `v4_report.json` | v4 训练曲线（88 948 步） |
| `v2_report.json` | ★ **A/B 脚本用它定义留出集**（`--names-from` 默认值）。**删了会改标尺**，务必保留 |
| `loss_attribution_v4_best.json` | 损失归因（86.8% 人声损失落在 `0<mask≤0.5`） |
| `filterbank_audit.json` | 滤波器组审计（49/128 带整列为 0，实际 79 有效带） |

### 量化与后处理

| 文件 | 说明 |
|---|---|
| `quant_v4_best.json` | v4_best 量化核对（deploy12 **29.58 dB**） |
| `quant__probeC_best.json` | probeC_best 量化核对（deploy12 **29.74 dB**） |
| `mask_postproc_lf.json` | 低频保护扫描（`<250 Hz` 最优，`<350 Hz` 人声崩塌） |
| `mask_postproc_probeC.json` / `mask_postproc_k3s.json` | 低频保护在另两个权重上复核 → **与权重无关** |
| `demo_report.json` / `demoK_report.json` | 按键演示渲染记录 |

### 2026-09-14 优化与 NPU 基线

| 文件 | 说明 |
|---|---|
| `ab_k3ship_bott2_true_pooled.json` | 两块瓶颈残差候选在真实未见歌曲三位置汇总 |
| `ab_bott2_temporal_true_pooled.json` / `ab_bott2_comp_true_pooled.json` | 时域块与互补掩码负结果 |
| `ab_bott2_base_mel_full_lf250.json` / `ab_bott2_128_band192_probe.json` | 新滤波器组与 192 频带实验 |
| `filterbank_centers_audit.json` | 滤波器中心与教师投影上限审计 |
| `ab_mir1k_bott2_final.json` / `ab_onair_shipping_mir1k.json` | MIR-1K 真值微调的跨数据集验证 |
| `ab_shipping_mir1k_true_pooled.json` | 新候选相对出货模型的电子音乐三位置汇总 |
| `quant_opt_bott2_mir1k_5m_model.json` | 新候选的 INT8 权重、INT12 激活和累加器验证 |
| `fpga_budget.json` | XC7Z020 早期算力与存储预算 |

目录默认忽略新生成的 JSON/log/PT。新的实验文件只有在进入报告引用的精选证据后，才用
`git add -f` 显式加入，避免把完整运行目录和本机缓存提交到公开仓库。

## 未包含的内容

- `*_cache/`：逐曲伪标签缓存（**353 KB / 音频秒**，24 GB）→ 用 `11_smoke_train.py` 重建（教师筛选 ≈1 h）
- 各轮模型权重的**中间检查点** → 只保留 `models/` 下的 3 个关键权重
- 逐曲输出音频（`vocals.wav` / `accompaniment.wav`）→ 只在 `demo/` 留了出货配置的按键演示
