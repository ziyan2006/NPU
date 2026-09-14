# CONV2D 控制器微架构规格

文档版本：`0.1-draft`

状态：P4 可综合控制/数值垂直切片；真实 tile 已位精确验证，Scratchpad 和 post
pipeline 尚未接入同一顶层

## 1. 模块边界

`npu_conv2d_controller.sv` 接收一个 512-bit Operator descriptor、A/W bank 和完成
event。它生成成对的 128-bit activation 与 512-bit O8I8 weight 读取请求，并直接
驱动 `npu_tensor_mac_8x8`。MAC 的每个结果为 8xINT32，控制器给出 32-byte 递增的
O bank 地址和输出尾 lane mask。

当前支持普通卷积的 `1x1`、`1x3`、`3x3`，stride 1/2、dilation 1、groups 1。
当前网络使用的 padding 已由 DMA 在本地 receptive field 中清零，因此计算循环不做
逐 token 边界判断。depthwise、非 1 dilation 和跨多个 Cout block 的单条命令不在
此控制器版本中。

## 2. 地址和循环

循环顺序固定为：

```text
output_h -> output_w -> cin_block -> kernel_h -> kernel_w
```

令 `CinPad=align8(Cin)`，本地输入尺寸为：

```text
local_h = (tile_h - 1) * stride_h + Kh
local_w = (tile_w - 1) * stride_w + Kw
```

请求地址为：

```text
A = (((oh*stride_h + kh)*local_w + ow*stride_w + kw)*CinPad
     + cin_block*8)*2
W = ((cin_block*Kh + kh)*Kw + kw)*64
```

一个请求 token 对应 8 个 activation lane 和 8x8 个 weight。首尾 token 产生
MAC `first/last`，`Cin` 和 `Cout` 非 8 整数倍时分别生成输入和输出 mask。请求
sideband 进入 8-entry FIFO，因此 Scratchpad 响应和 MAC 结果都可独立反压。

## 3. 启动和容量检查

描述符先同步检查 tile、kernel、stride、dilation、groups 和 channel 边界。合法
descriptor 进入 3 个乘法步骤：row bytes、activation capacity、row advance。
三个步骤复用 `npu_u32_mul_iter`，不占 DSP，每步 17 拍，总启动开销约 51 拍。

启动计算完成前不发 Scratchpad 请求。activation 超过 64 KiB 时返回 capacity
错误；weight 和 accumulator 容量由已限制的 `Cin<=224`、`Kh/Kw<=3`、
`tile_h/tile_w<=16`、`tile_cout<=8` 静态保证。soft reset 会丢弃 setup、请求
metadata、MAC pipeline 和未完成结果，复位后可以立即接收新 descriptor。

## 4. 验证证据

`scripts/_test_npu_conv2d.py` 直接读取当前架构包的 binary descriptor、真实 O8I8
权重、tile schedule 和 program manifest。四类代表 tile 的结果如下：

| layer | kernel/stride | 请求 | 8-lane 结果 |
|---|---|---:|---:|
| out | 1x1 / 1 | 1,024 | 256 |
| bott | 1x3 / 1 | 1,536 | 32 |
| enc0 | 3x3 / 1 | 2,304 | 256 |
| enc1 | 3x3 / 2 | 4,608 | 128 |

共逐项核对 9,472 对 A/W 地址和 672 个 8-lane INT32 结果，覆盖 padding 清零、
输入/输出尾 lane、随机请求反压和随机结果反压。额外 directed test 覆盖 geometry、
mode、capacity 三类错误以及忙态 soft reset 后重新运行。

## 5. Vivado 2026.1 OOC

临时参考 part 为 `xc7z020clg400-1`。综合后资源为 3,610 Slice LUT、2,645 FF、
64 DSP48E1、0 BRAM。控制器初版的组合几何乘法曾使总 DSP 达到 89；改用共享迭代
乘法器后只保留 MAC 的 64 个 DSP。

100 MHz WNS 为 `+2.660 ns`，可作为当前完整集成基线；200 MHz WNS 为
`-2.340 ns`，尚未闭合，不能宣称支持 200 MHz。该结果为 OOC 综合后估算，最终
频率仍需在选定板卡、接入 AXI/PS、布局布线后确认。

## 6. 下一步

将 Scratchpad 的 A/W 计算读口拆成独立并行接口并接入本控制器，再加入 bias、
per-channel Q31 requant、signed RNE、clamp 和激活 pipeline。完成命令处理器、DMA、
compute 和 CSR 顶层后，运行全 1,869 条命令的逐命令位精确仿真。
