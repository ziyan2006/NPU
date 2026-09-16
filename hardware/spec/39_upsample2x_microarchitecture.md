# UPSAMPLE2X 微架构规格

文档版本：`0.1-draft`

状态：P4 可综合 AXI 原型；三条真实命令已位精确验证并通过 100 MHz Vivado OOC

## 1. 执行方式

`UPSAMPLE2X` 对 DDR 中连续 NHWC8、INT12-in-INT16 Tensor 执行两轴最近邻复制：

```text
dst[h, w, c] = src[h >> 1, w >> 1, c]
```

当前网络的源行最大 1 KiB。模块每个源行只发一个 64-bit INCR read burst，读入
`256 × 64-bit` simple-dual-port BRAM；随后按像素边界重复读取行缓冲，生成水平
复制后的目标行，并把同一结果行写两次完成垂直复制。单个目标行最多 2 KiB，正好
是 256 个 AXI beat，不超过 AXI4 burst 上限。

## 2. 接口、检查和完成

模块直接读取 source/destination Tensor descriptor，外部地址为
`activation_base + base_offset`。启动时检查：

- opcode/flags/unused 字段；
- N=1、NHWC8、INT12-in-INT16、stride_c=2；
- 目标 H/W 恰为源的两倍、C 相同；
- 当前 P0 shape 上限、连续行 stride、8-byte 对齐和 allocation 边界。

AXI read/write 各只保留一个 outstanding burst。`RRESP/BRESP`、提前或缺失 RLAST
都会中止命令并产生错误。soft reset 不放弃已经握手的 AXI transaction；外层应先
停止新命令并等待单元 idle。完成只在最后一个 B response 成功后产生。

每次读写 burst 先进入独立 `PLAN_READ` / `PLAN_WRITE` 状态，计算并寄存 AXI 地址、
4 KiB 边界和 burst 长度，下一拍才断言 ARVALID/AWVALID。这样 AXI 地址字段在反压
期间保持稳定，并切断边界/剩余 beat 计算到 arbiter handshake 的长组合路径。

## 3. 验证与综合

`scripts/_test_npu_upsample.py` 使用三条真实命令与 Tensor descriptor，覆盖：

- `128×16×2 → 128×32×4`；
- `96×32×4 → 96×64×8`；
- `64×64×8 → 64×128×16`。

测试对 AR/AW 地址和 burst 属性、每拍 WDATA/WSTRB/WLAST、总字节计数与完整目标
Tensor 逐 beat 比对，并随机施加五通道 AXI 反压。

Vivado 2026.1、参考 `xc7z020clg400-1`、100 MHz OOC：

| 项目 | 结果 |
|---|---:|
| Slice LUT | 1,626 |
| Slice Register | 798 |
| RAMB36E1 | 1 |
| DSP48E1 | 0 |
| 100 MHz WNS | `+0.425 ns` |

该表是单元级基线。上述 burst 规划流水化接入完整 `npu_top` 后，XC7Z020 OOC
post-route 在 100 MHz 下为 WNS `+0.225 ns`、WHS `+0.007 ns`。
