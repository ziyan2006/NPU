# VEC_ADD 微架构规格

文档版本：`0.1-draft`

状态：P4 可综合原型；真实 residual tile 已位精确验证并通过 100 MHz Vivado OOC

## 1. 数值与数据流

网络 A 的 `VEC_ADD` 固定执行：

```text
residual_q = RNE(src1 * scalar_Q31_multiplier / 2^shift)
residual_q = clamp(residual_q, clamp_min, clamp_max)
dst        = saturate_INT12(src0 + residual_q)
```

`src0` 是 O bank 中卷积 post 后的 8×INT16 行，`src1` 是 A bank 中对应 residual
像素，结果原位写回 O bank。残差只使用一条 scalar quant 参数，硬件将其广播到
8 lane 并复用 `npu_requant_post` 已验证的 Q31/RNE 通路；相加使用 17-bit 中间值，
最后饱和到 `[-2048, 2047]`。

## 2. 地址与控制

模块由真实 `VEC_ADD` command 和当前 tile Operator descriptor 启动。A/W/O bank
分别取自 command immediate。W bank 中参数地址与 DMA/编译器布局一致：

```text
bias_base         = CinBlocks * Kh * Kw * 64
conv_quant_base   = bias_base + 64
vector_quant_base = conv_quant_base + align64(tile_cout * 16)
```

残差地址从已经包含 padding 的 A-bank receptive field 计算：跳过 `pad_top` 行、
`pad_left` 列和 `tile_origin_cout` 个通道，然后按输出 H/W 遍历。启动阶段的行字节
计算复用 radix-4 无 DSP 乘法器。每个 8-lane 向量依次读取 O、A，执行单 lane
requant 后写回 O；完成脉冲只在最后一次 O 写握手后产生。

## 3. 验证与综合

`scripts/_test_npu_vec_add.py` 直接读取真实 tile command、Operator descriptor、
residual quant 参数，覆盖两个 bottleneck block、bank0/bank1、随机读写反压和 96 个
8-lane 结果。Python golden 使用 ISA 的 `requantize` 和 `saturate`，逐 lane 比对。

Vivado 2026.1、参考 `xc7z020clg400-1`、100 MHz OOC：

| 项目 | 结果 |
|---|---:|
| Slice LUT | 2,716 |
| Slice Register | 1,525 |
| RAMB36E1 | 0 |
| DSP48E1 | 4 |
| 100 MHz WNS | `+1.426 ns` |

最终 top 可进一步让 CONV2D 与 VEC_ADD 共享同一 Q31 datapath；当前先保留独立实例，
以简化同步命令的完成与反压证明。

