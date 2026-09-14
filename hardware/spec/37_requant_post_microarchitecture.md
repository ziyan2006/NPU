# Requant 与激活后处理微架构规格

文档版本：`0.1-draft`

状态：P6 可综合原型；已通过位精确 RTL 回归和 100 MHz Vivado 2026.1 OOC 综合

## 1. 数值顺序

`npu_requant_post` 对一组 8-lane INT32 MAC 结果执行固定顺序：

```text
biased  = wrap_INT32(accumulator + bias)
product = biased * signed_Q31_multiplier
scaled  = signed_round_to_nearest_even(product / 2^shift)
clamped = min(max(scaled, clamp_min), clamp_max)
post    = identity / ReLU / LeakyReLU(1/10) / tanh_LUT
output  = signed INT12 in a 16-bit container
```

这与 `scripts/npu_isa.py` 的 `round_shift_rne`、`requantize` 和 `leaky_relu_q`
一致。负数先在绝对值域舍入，再恢复符号；例如 `-0.5 -> 0`、`-1.5 -> -2`。
shift 合法范围是 0 到 63，clamp 必须位于 `[-2048, 2047]` 且下界不大于上界。
尾部无效 lane 不读取其参数，输出为 0。

## 2. 数据通路

为控制 XC7Z020 资源，一套 32x32 signed Q31 通路在 8 lane 间复用。每 lane 依次经过：

1. 乘法；
2. 乘积绝对值；
3. 可变右移并保存 quotient/remainder/halfway；
4. ties-to-even 判定；
5. 符号恢复；
6. clamp；
7. activation 或 tanh BRAM 读取。

内部寄存器切断变量移位、64-bit 舍入加法、符号恢复、clamp 和 activation，确保
100 MHz 时序。一个普通输出向量约 56 cycle；tanh 多一个 BRAM capture cycle/有效
lane。输出使用 ready/valid，反压时数据和 lane mask 保持稳定。

LeakyReLU 的输入已经 clamp 到 INT12，因此精确 `1/10` RNE 只在 12-bit magnitude
上实现。tanh 是可编程的 4096x16 双口 BRAM，索引为 `clamped + 2048`；运行时写口
用于从模型包装载 `tanh_lut_int12.bin`。

## 3. 性能取舍

网络 A 的 248 个 tile 共产生 31,744 个 8-lane 输出向量。按 56 cycle/vector 计，
post 为 1,777,664 cycle；逐层与 MAC 重叠后，计算关键值从 1,986,560 增到约
2,727,936 cycle。加回现有调度的非计算开销后约为 3.23M cycle，即 100 MHz 下
约 32.3 ms，仍低于 46 ms 硬门槛。因此当前保留 1-lane 复用实现，不用 8 倍 post
DSP 换取当前工作负载不需要的吞吐。

## 4. 验证与综合

`scripts/_test_npu_post.py` 使用 `npu_isa.py` 作为独立 golden，固定随机种子生成
100 组 8-lane 向量，覆盖四种 post-op、INT32 bias 回绕、shift 0/63、正负 RNE、
随机 clamp、尾 lane mask、输出反压、非法 shift 和真实 4096 项 tanh LUT。

Vivado 2026.1、临时参考 part `xc7z020clg400-1`、100 MHz 综合后 OOC 结果：

| 项目 | 结果 |
|---|---:|
| Slice LUT | 2,702 |
| Slice Register | 1,639 |
| RAMB36E1 | 2 |
| DSP48E1 | 4 |
| 100 MHz WNS | `+0.197 ns` |

该结果不含 Scratchpad 取 bias/quant 的端口、最终布局布线、PS interconnect 和真实
clock source。下一步要把 W bank 中 `bias_base/conv_quant_base` 接入 CONV2D 数据
链路，并在完整 top 重新检查资源和时序。
