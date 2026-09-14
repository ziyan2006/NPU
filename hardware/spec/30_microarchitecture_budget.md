# NPU v1 微架构与预算草案

文档版本：`0.1-draft`

状态：P2/P3 架构探索输入，不是 RTL 冻结规格

## 1. 顶层结构

```text
                         AXI4-Lite
PS ───────────────────── Control/CSR
                              │
                              ▼
                     ┌─ Command Processor ─┐
DDR ── PS S_AXI_HP ◄─┤ DMA + Desc Cache    │
                     └──────┬───────┬──────┘
                            │       │
                    ┌───────▼─┐   ┌─▼────────┐
                    │Weight SP│   │Activation│
                    │ping/pong│   │bank/tile │
                    └───────┬─┘   └─┬────────┘
                            │       │
                      ┌─────▼───────▼─────┐
                      │ 8×8 Tensor MAC    │
                      │ INT8 × INT8/12/16 │
                      └─────────┬─────────┘
                                ▼
                      INT32 bias/accumulator
                                ▼
                      requant / vector / LUT
                                ▼
                      activation bank / DMA
```

首版采用一个复用的 Tensor MAC 阵列顺序执行各层，不为每层复制硬件。命令处理器和地址发生器使同一阵列服务不同模型。

## 2. 模块职责

| 模块 | 主要职责 | 不承担的职责 |
|---|---|---|
| CSR/doorbell | capability、任务提交、中断、错误、计数器 | 图解析 |
| Command processor | 取指、译码、PC、合法性检查、event/scoreboard | 动态调度/乱序执行 |
| Descriptor cache | 缓存 Tensor/operator/quant 描述符 | 自动 shape 推导 |
| DMA engine | 64-bit AXI burst、2D stride、4 KiB 拆分、错误处理 | cache coherence |
| Activation scratchpad | 多 bank tile、ping-pong、残差/双源读 | 保存完整网络所有 Tensor |
| Weight scratchpad | 重排权重和 bias 的双缓冲 | 保存完整 824 kB 权重 |
| Tensor MAC | 64 lane 卷积/1×1/GEMM 映射 | 浮点、训练 |
| Vector/post unit | add、requant、激活、LUT、upsample 地址复用 | 通用标量程序 |
| Perf/error unit | 周期、stall、流量、高水位、首错锁存 | 软件日志格式化 |

## 3. MAC 阵列

基线为 `Pout=8 × Pin=8`，共 64 lane。一个周期对 8 个输出通道分别累加 8 个输入通道乘积。卷积计算周期的首阶模型为：

```text
C_compute = Hout * Wout * ceil(Cout/8) * ceil(Cin/8) * Kh * Kw
```

depthwise 不应机械套用 8×8 普通卷积数据流，否则利用率很低；P2 需要比较两种方案：

1. 用 64 lane 作为 64 个独立 depthwise MAC；
2. 不增加专用硬件，由编译器接受较低利用率。

只有第二验证网络确实需要 depthwise 后，才决定是否增加 depthwise mode。

每个 lane 支持 signed INT8 权重与符号扩展后的 INT8/INT12/INT16 激活，INT32 累加。实现时要显式约束乘法映射到 DSP48E1，并检查综合是否因可配置位宽复制乘法器。

## 4. 累加与后处理流水线

建议流水级：

```text
SP read → lane multiply → adder tree → INT32 accumulator
        → bias → per-channel multiplier → RNE shift
        → activation/LUT → saturate → SP write
```

需要在 P4 明确：

- DSP、加法树和 accumulator 的寄存级数；
- 同一输出像素跨 `Cin` tile 的 partial sum 保存位置；
- `acc * multiplier` 的完整中间位宽；
- RNE 对正负数和恰好半 LSB 的行为；
- bias、残差加法、激活、clamp 的固定顺序；
- pipeline flush 时命令完成的定义。

当前量化报告显示最坏累加需求为 26 bit，INT32 有余量；这只是当前模型证据。编译器仍须为每个新模型做 accumulator bound 检查。

## 5. 片上存储层次

XC7Z020 提供 140 个 BRAM36。NPU 暂定最多使用 112 个，预留 28 个给系统其它模块。片上存储只保存工作 tile，不尝试容纳完整模型。

| 用途 | BRAM36 提案 | 约合容量 | 说明 |
|---|---:|---:|---|
| Activation banks | 48 | 216 KiB | 多 bank，输入/输出 ping-pong 和双源读 |
| Weight banks | 24 | 108 KiB | 权重双缓冲，bias/scale 可随 tile |
| Accumulator/post | 12 | 54 KiB | partial sum、requant、vector 临时区 |
| DMA/FIFO | 8 | 36 KiB | AXI 读写解耦和 burst 整形 |
| Command/descriptor/debug | 4 | 18 KiB | 取指、描述符和 trace 小缓存 |
| NPU 内部余量 | 16 | 72 KiB | 综合映射、bank 冲突、ECC/调试取舍 |
| **NPU 合计上限** | **112** | **504 KiB** | 不含系统预留 28 块 |

这是分配上限，不是要求全部使用。P2 存储规划器必须用真实 Tensor 生命周期证明 activation bank 容量；若超出，按以下顺序处理：

1. 改 tile 尺寸并把长生命周期 skip Tensor spill 到 DDR；
2. 减小 weight tile；
3. 调整 bank 划分；
4. 最后才评估 12-bit 紧凑打包。

不能未经质量验证把当前模型激活降为 INT8 来节省 BRAM。

### 网络 A 的 tile bank 实例

P3 软件调度器当前采用以下子划分，它仍是逻辑容量提案，尚未换算为最终 BRAM 端口组织：

| 逻辑 bank | 单 bank 容量 | 网络 A 最大占用 | 作用 |
|---|---:|---:|---|
| `A0/A1` | 64 KiB | 57,600 B | 含 receptive-field padding 的输入 tile；按空间 tile ping-pong |
| `W0/W1` | 32 KiB | 16,320 B | 权重、bias、quant；含 64-byte 对齐，按计算 tile ping-pong |
| `P0/P1` | 16 KiB | 8 KiB | INT32 partial sum |
| `O0/O1` | 16 KiB | 4 KiB | requant/post 和异步写回 |

两个 A bank 加两个 O bank 共 160 KiB，低于 activation 类 216 KiB 预算；两个 W bank 共 64 KiB，低于 108 KiB；两个 P bank 共 32 KiB，低于 accumulator/post 类 54 KiB。剩余容量用于边界行、DMA FIFO、端口复制或综合映射损耗，不能在 RTL 前当作可自由分配的净容量。

P4 Scratchpad 原型已按 `A0/A1=2x64 KiB`、`W0/W1=2x32 KiB`、
`O0/O1=2x16 KiB` 实现 6 个 lane-striped bank。DMA 端保持 64 bit，计算端 A/O
为 128 bit、W 为 512 bit。Vivado 2026.1 在参考 `xc7z020clg400-1` 上实际推断为
56 个 RAMB36E1，和容量推导一致；宽口没有复制数据。加上尚未实现的
`P0/P1=2x16 KiB` 理想值 8 个，共 64 个。

### INT12 存储决策

基线提案是用 signed 16-bit 物理槽保存 INT12：地址简单、吞吐规整、BRAM 宽度好映射，代价是相对紧凑 12-bit 多 33% 容量/带宽。是否实现 12-bit packed 格式要比较：

- 当前和第二模型的峰值 tile 容量；
- DDR 读写量及 DMA stall；
- pack/unpack LUT、时序和验证复杂度；
- BRAM 原生宽度下是否真的节省 block 数。

没有综合和周期模型证据前，不冻结 packed12。

## 6. DMA 与数据流

PL 侧 NPU 作为 AXI master 访问 PS 的高性能 DDR 端口。首版接口提案为 64 bit，支持 INCR burst、多个 outstanding read、读写独立通道和 2D stride。

基本 tile 流程：

```text
DMA load input[k+1] / weight[k+1]
          与
compute tile[k]
          与
DMA store output[k-1]
```

DMA 必须检查 4 KiB 边界并拆 burst；片上 FIFO 吸收 AXI back-pressure。命令处理器只在 event 就绪和 bank 无冲突时发射计算。首版采用静态、单任务、受控重叠，不实现复杂乱序调度。

## 7. 当前工作负载性能账本

当前 `bott2_mir1k_v1` 是架构基准，不是硬编码图：

| 项目 | 当前值 |
|---|---:|
| 输入/输出 | `2×128×16` → `4×128×16` |
| 参数 | 824,900 |
| MAC/块 | 123,338,752 |
| 音频时长/块 | 92.88 ms |
| 64-lane 已排程 MAC 周期 | 1,986,560 cycle |
| 200 MHz 纯计算下界 | 9.93 ms |
| 任务周期目标 | ≤4,000,000 cycle |
| 100 MHz 对应目标时间 | ≤40 ms |
| P0 墙钟门槛 | <46 ms |
| 权重每块重读流量 | 约 8.87 MB/s |

`1,986,560 cycle` 只计 MAC 调度，不含命令、DMA stall、bank conflict、requant 和 pipeline 边界。因此规格采用 4,000,000 cycle 的完整任务目标，不能把纯计算下界当成最终性能。

当前逐 tile transaction 模型生成 248 个计算 tile和 1,869 条命令。按已实现的 64-bit AXI、最大 256 beat、4 KiB 拆分、每 burst 16-cycle 保守开销、读写通道可重叠及跨层不重叠的假设，总计 2,485,596 cycle；软件访问区间中 bank conflict 为 0。DMA reference 把 1,078 条命令解析为 45,664 个 burst，并验证 allocation、bank、对齐和边界。该结果关闭了“当前静态分配必然冲突、越界或超 4M cycle”的软件风险，但没有关闭真实 AXI 实效、BRAM 端口数或 post-route Fmax 风险。

周期模型至少拆分为：

```text
C_total = C_command + C_dma_unhidden + C_compute
        + C_post_unhidden + C_bank_stall + C_pipeline
```

每项必须能与硬件性能计数器对应，否则板上变慢时无法定位原因。

## 8. 资源预算

XC7Z020 总量基线为 220 DSP48E1、140 BRAM36、53,200 LUT、106,400 FF。NPU 核首版上限：

| 资源 | NPU 上限 | 基线目标 | 系统余量目的 |
|---|---:|---:|---|
| DSP48E1 | 112 | 64 MAC + 少量 post | FFT、复数乘法和音频链路 |
| BRAM36 | 112 | ≤104，另留 NPU 内部余量 | 系统 FIFO、FFT、调试 |
| LUT | 40,000 | ≤32,000 | AXI、I2S、控制和布线余量 |
| FF | 70,000 | ≤55,000 | 系统控制和时序寄存 |

DSP48E1 资源上限不是阵列规模目标。首版不使用 220 个 DSP 全铺阵列，因为更大的阵列会同时增加 BRAM 端口、路由、权重带宽和尾通道空闲率。

当前 DMA 子系统的 Vivado 2026.1 OOC 综合结果为 7,546 Slice LUT、7,144 FF、
56 RAMB36E1、0 DSP48E1。AGU 使用共享 radix-4 shift/add 乘法器，Descriptor
Cache 的 record size 和 Engine 的 beat size 使用移位，因此控制路径不占用计划留给
Tensor MAC 的 DSP。100 MHz WNS 为 `+1.888 ns`；200 MHz WNS 为 `-3.112 ns`，
故当前只关闭最低验收频率，目标频率仍需在完整约束和布局布线阶段优化。

8x8 Tensor MAC 数值切片单独综合为 1,996 Slice LUT、1,968 FF、64 DSP48E1、
0 BRAM36，200 MHz WNS `+1.701 ns`。乘法固定使用 64 DSP，加法树使用 LUT/carry
chain。DMA 与 MAC 的 OOC 资源简单相加约为 9,542 Slice LUT、9,112 FF、56
BRAM36 和 64 DSP，仍在 NPU 基线预算内；完整集成后的共享逻辑和布线结果才是最终值。
MAC 每拍需要的 128-bit A 和 512-bit W 已由 lane-striped Scratchpad 原生提供；
Scratchpad 单体为 3,147 Slice LUT、596 FF、56 BRAM36，200 MHz WNS `+0.360 ns`。
仍不能依据两个单体结果宣称完整计算路径闭合，必须继续验证 loop controller、
A/W 并行调度和集成后的 place/route。

## 9. 时钟与复位

- 目标 PL 时钟 200 MHz，最低验收点 100 MHz；最终以 post-route timing 为准；
- AXI、NPU core 和音频域若不同时钟，必须用明确的 async FIFO/CDC synchronizer；
- 外部复位异步进入时，内部释放必须同步；
- soft reset 停止新命令、排空已握手 AXI transaction、清内部状态，再置 idle；
- 性能要求是墙钟 deadline 与周期数的组合，不能靠降低时钟掩盖 timing failure。

## 10. P2/P3 必须关闭的架构风险

| 风险 | 所需证据 | 决策结果 |
|---|---|---|
| 16-bit 激活槽导致 BRAM 不足 | 两模型生命周期/tiling 报告 | 保持 16-bit 或增加 packed12 |
| DMA 无法隐藏 | transaction-level 周期模型 | 调 tile、outstanding、burst/FIFO |
| depthwise 利用率过低 | 第二模型逐层周期报告 | 增加 depthwise mode 或移出 P0 |
| 200 MHz 时序困难 | 小型 MAC+SP 原型 post-route | 降目标/加流水，而非猜测 |
| 通用控制器 LUT 过大 | command/DMA/AGU 综合 | 收窄 ISA 或描述符功能 |
| 量化不 bit-exact | Python golden + 随机边界向量 | 修正规格后再写 datapath |
| concat/skip 造成存储峰值 | 编译器生命周期图 | segmented view + DDR spill |

这些风险关闭前，Pout、BRAM bank 数、命令位域和 INT12 存储格式都只能标为提案。

## 11. 器件依据

- AMD 的 Zynq-7000 数据表给出 XC7Z020 的 53,200 LUT、106,400 FF、140 个 36 Kb BRAM 和 220 个 DSP slice；资源预算以该器件表为上限，而不是以早期估算为准：[Zynq-7000 SoC Data Sheet: Overview (DS190)](https://docs.amd.com/api/khub/documents/juMnxca71Tf2gfjmNyjM8A/content)。
- DSP48E1 原生包含 25×18 二补码乘法器和 48-bit accumulator，因此 INT8×INT12/16 可以在其乘法宽度内实现；实际映射仍须由综合报告确认：[7 Series DSP48E1 User Guide (UG479)](https://docs.amd.com/api/khub/documents/gu4oRPFEh_Pm2uaAlfY6Kg/content)。
- Zynq-7000 PS 提供面向 PL master 的 AXI high-performance 接口，HP 数据口支持 32/64-bit 数据路径；本规格首版选择其中一个 64-bit 口，最终带宽必须在目标板上实测：[Zynq-7000 SoC TRM — PL DMA via AXI HP](https://docs.amd.com/r/en-US/ug585-zynq-7000-SoC-TRM/PL-DMA-via-AXI-High-Performance-HP-Interface)、[Datapaths](https://docs.amd.com/r/en-US/ug585-zynq-7000-SoC-TRM/Datapaths)。
