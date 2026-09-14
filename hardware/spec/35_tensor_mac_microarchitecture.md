# 8x8 Tensor MAC 微架构规格

文档版本：`0.1-draft`

状态：P4 可综合数值切片；MAC 算术、流协议、Scratchpad 供数宽度和 CONV2D
loop controller 已验证，bias/requant/post-op 尚未接入

## 1. 模块边界

`npu_tensor_mac_8x8.sv` 每个有效输入 token 接收：

- 8 个 signed 16-bit 激活槽，当前 INT12 激活先符号扩展到该槽；
- 8x8 个 signed INT8 权重；
- 输入和输出各 8-bit lane mask；
- `first/last` 累加边界。

一个 token 为 8 个输出 lane 各执行 8 项点积，即最多 64 次乘法。连续 token
累加到 8 个 INT32 accumulator；`last` token 完成后通过 ready/valid 接口返回
8xINT32。输入和输出 lane 0 均位于总线最低位。

本模块不解析 Command/Operator descriptor，也不生成卷积地址。CONV2D loop
controller 负责按 `H/W/Cin/Kh/Kw/Cout` 顺序产生 token，并保证每组恰有一个
`first` 和一个 `last`。

## 2. 数值语义

每个有效输出 lane 的计算为：

```text
dot[o] = sum(i=0..7, active(i) ? int16(a[i]) * int8(w[o][i]) : 0)
acc[o] = first ? dot[o] : wrap_int32(acc[o] + dot[o])
result[o] = last ? acc[o] : no output
```

- 单乘积为 signed 24 bit；
- 8 项加法树依次使用 25、26、27 bit，点积不会在树内丢位；
- accumulator 使用 32-bit 二补码回绕；当前模型的静态上界仅需 26 bit，正常包
  不会触发回绕，测试仍覆盖回绕以固定硬件语义；
- output mask 为 0 的 lane 返回 0；同一 `first..last` 组内 output mask 必须不变；
- soft reset 丢弃流水中未完成 token，下一组必须重新从 `first=1` 开始。

bias、Q31 requant、RNE、激活和 INT12 饱和位于后续 post pipeline，不在本模块中
隐式执行，避免 MAC 原型改变 ISA 规定的运算顺序。

## 3. 流水和反压

数据依次经过五个寄存阶段：

```text
DSP multiply -> add 2 -> add 4 -> add 8 -> INT32 accumulate/result
```

无反压时每拍可接收一个 token。`result_valid && !result_ready` 时冻结整条流水线，
`input_ready=0`，所有数据和 sideband 保持不变。结果被接收后可在下一拍继续推进；
连续单-token 组允许每拍产生一个结果。

只有 valid 寄存器使用异步硬复位。DSP、加法树和 sideband 数据寄存器不复位，
有效性由独立 valid pipeline 隔离，以避免在 DSP 输入/输出上增加复位 mux。

## 4. Scratchpad 带宽结论

MAC 满速每拍需要 128-bit activation 和 512-bit weight，共 640 bit。Scratchpad
现已采用 64-bit lane striping：A bank 两 lane 并行读 128 bit，W bank 八 lane
并行读 512 bit。O8I8 的一个 8×8 权重 token 恰为连续 64 byte，因此无需八拍预取
或复制 BRAM 数据。

宽口 Scratchpad 在不增加 56 个 RAMB36 容量映射的情况下通过 200 MHz OOC，且回归
证明计算读可连续每拍发射。CONV2D loop controller 已用 4 个真实 tile 验证地址、
尾 lane、INT32 结果和双向反压；下一步是把 Scratchpad 与该控制器接入同一顶层。

## 5. Vivado 2026.1 OOC 证据

临时参考 part：`xc7z020clg400-1`，目标时钟 200 MHz。

| 资源/时序 | 结果 | XC7Z020 占比 |
|---|---:|---:|
| Slice LUT | 1,996 | 3.75% |
| Slice Register | 1,968 | 1.85% |
| DSP48E1 | 64 | 29.09% |
| BRAM36 | 0 | 0% |
| 200 MHz WNS | `+1.701 ns` | 满足 |

乘法显式映射到 64 个 DSP48E1，加法树和 accumulator 显式保留在 LUT/carry
chain，防止综合器额外消耗 16 个 DSP。`synth_design` 为 0 error、0 critical
warning、0 synthesis warning。OOC 顶层没有 input/output delay 和最终 clock source，
结果仍需在 loop controller、Scratchpad 和 SoC 顶层接入后做 post-route 复核。

## 6. 当前验证与下一步

Icarus 回归使用独立 Python golden，覆盖 INT12/INT16 正负值、INT8 权重、1/7/8
尾 lane、单 token/多 token 累加、INT32 回绕和随机结果反压，并检查结果在反压时
保持稳定。

CONV2D 控制器现已执行真实 O8I8 `1x1/1x3/3x3` 和 stride-2 tile，详细证据见
`36_conv2d_controller_microarchitecture.md`。下一步接 bias + per-channel Q31
requant/RNE，并与 Scratchpad 形成首个命令级 bit-exact 计算闭环。
