# NPU v1 决策记录与开放问题

文档版本：`0.1-draft`

## 1. 已接受的方向

| ID | 状态 | 决策 | 原因 | 影响 |
|---|---|---|---|---|
| ADR-001 | 已接受 | 从固定人声模型状态机改为轻量可编程 CNN NPU。 | 用户希望后续能更换/放大模型。 | 增加命令、描述符、编译器和验证工作。 |
| ADR-002 | 提案 | v1 的“通用”限定为 batch=1、静态 shape 的小型量化 CNN。 | 在 XC7Z020 上平衡通用性、资源和可验证性。 | Attention、RNN、动态 shape 和浮点移出 v1。 |
| ADR-003 | 已接受 | 图编译和内存规划在部署前完成，PS 提交，PL 不解析框架模型。 | 降低 PL 控制复杂度并保证确定性。 | 必须开发模型 packer/compiler 和模拟器。 |
| ADR-004 | 提案 | STFT/iSTFT、OLA 和音频产品逻辑位于 Tensor NPU 之外。 | 避免 ISA 绑定某种音频前端。 | 完整产品仍需独立音频 IP/软件链。 |
| ADR-005 | 已接受 | 当前人声模型是首要基准，但不是硬编码图。 | 保留已验证质量和实时目标。 | 所有架构提案先用当前模型量化。 |
| ADR-006 | 已接受 | 当前模型激活精度基线为 INT12、累加为 INT32。 | 已有量化实验显示 INT8 激活不足，INT32 覆盖 26-bit 最坏累加。 | 数据通路和存储不能按纯 INT8 设计。 |
| ADR-007 | 已接受 | 完整 Tensor 结果以位精确整数解释器为规范真值。 | 消除 Python/RTL 对舍入和溢出的歧义。 | RTL 前必须先完成软件 golden。 |

## 2. 待批准的架构提案

| ID | 状态 | 提案 | 批准所需证据 |
|---|---|---|---|
| ADR-010 | 提案 | 使用 8×8、64-lane Tensor MAC。 | 两模型周期/带宽模型和 post-route 小原型。 |
| ADR-011 | 提案 | 128-bit 固长粗粒度命令。 | 两模型编译后的字段范围和命令密度。 |
| ADR-012 | 提案 | 激活 `NHWC8`、权重 O8I8 block 布局。 | bank conflict、DMA 连续性和布局转换开销。 |
| ADR-013 | 提案 | INT12 默认存于 signed 16-bit 槽。 | 峰值 BRAM、DDR 流量与 packed12 综合比较。 |
| ADR-014 | 提案 | 一个 64-bit PL M_AXI 连接 PS S_AXI_HP。 | 实际板卡 DDR 配置和并发压力实测。 |
| ADR-015 | 提案 | NPU 核上限 DSP/BRAM 为各 112。 | 完整系统资源分配和 FFT/音频 IP 估算。 |
| ADR-016 | 提案 | concat 使用 segmented Tensor view。 | 当前 U-Net 解码层地址发生器原型和周期结果。 |
| ADR-017 | 提案 | depthwise 是 P0，使用 MAC 阵列的独立 lane 模式。 | 网络 B 确认和实现成本评估。 |
| ADR-018 | 提案 | 任务周期目标 4,000,000 cycle，最低 100 MHz、目标 200 MHz。 | DMA/命令周期模型和 post-route 时序。 |
| ADR-019 | 提案 | tanh 输入采用 max 标定的 signed INT12，step=`0.0046281479`；输出采用 signed INT12、step=`1/2047` 的 4096 项 LUT。 | 2,097,152 个 pre-tanh 观测值无裁剪，tanh 输出量化 SNR 56.31 dB。 |
| ADR-020 | 提案 | `A` 为 2×64 KiB、`W` 为 2×32 KiB、`P` 为 2×16 KiB、`O` 为 2×16 KiB，并用静态 event/WAIT 管理 ping-pong。 | 网络 A 容量和访问区间已通过；仍需 BRAM 端口映射、网络 B 与 RTL assertion。 |

当前网络 A 的软件证据：248 个 tile 展开为 1,869 条命令，64-lane 的 transaction 调度为 2,146,104 cycle，DDR 流量 3,512,000 B；全部 1,078 条 DMA command 已解析为可执行三维地址请求，所有 allocation/bank 容量通过，静态访问区间冲突为 0。O8I8 权重仅增加约 0.23%。这支持 ADR-010/012/016/018/020 继续进入 RTL 原型，但在第二网络、实际 BRAM 端口、AXI 实效和 post-route 结果出现前仍不升级为“已接受”。

P4 已确定一项内部接口约束：DMA engine 接收 `x_bytes × y_count × z_count` 和 DDR/SP 独立 stride 的标准请求；segmented concat 必须按像素通道交织搬运。`DMA_LOAD.imm[10]` 显式区分 vector quant 与 convolution quant，避免从参数数量推断用途。该约束仍随 ISA 处于提案状态，网络 B 通过前不冻结。

## 3. 必须确认的系统问题

| ID | 问题 | 为什么会影响设计 | 关闭时机 |
|---|---|---|---|
| OI-001 | 实际开发板型号、器件封装和 speed grade 是什么？ | 决定 pin、DDR、时钟和实现余量。 | P1 冻结前 |
| OI-002 | DDR 容量、位宽、频率以及 PS 当前负载是多少？ | 决定有效带宽、burst 和 FIFO。 | P1/P2 |
| OI-003 | Vivado/Vitis 具体版本和授权环境是什么？ | 影响 IP、综合结果和可复现性。 | P1 冻结前 |
| OI-004 | 音频输入输出接口、采样率范围和 codec 时钟是什么？ | 决定系统 CDC、FIFO 和产品链路，但不改变 Tensor ISA。 | SoC 规格前 |
| OI-005 | PS 运行 Linux 还是 bare-metal？是否有 CMA/IOMMU？ | 影响驱动、物理内存和 cache maintenance。 | 驱动规格前 |
| OI-006 | 系统可接受功耗和散热上限是多少？ | 可能限制 200 MHz 和 DSP toggle rate。 | P3/P8 |
| OI-007 | 第二个通用性验证模型选什么？ | 决定 depthwise/pool/GEMM 是否真为 P0。 | P1 退出前 |
| OI-008 | 是否要求不中断音频时切换模型？ | 决定双任务上下文和模型加载策略。 | P1 冻结前 |
| OI-009 | 是否需要输入/输出 AXI4-Stream 旁路 DDR？ | 会改变顶层接口、流控和存储规划。 | P3 前 |
| OI-010 | 模型包是否需要硬件 CRC/安全认证？ | 决定包头、DMA 和安全边界。 | P3 前 |

## 4. P2 架构实验队列

| 优先级 | 实验 | 输入 | 输出/判据 |
|---:|---|---|---|
| 1 | 当前图规范化 | manifest + checkpoint | 标准算子图、Tensor 生命周期、P0 算子覆盖 |
| 2 | 整数量化编译 | float scale + calibration | per-channel multiplier/shift，位精确输出 |
| 3 | 布局/tiling 搜索 | 网络 A，NHWC8/O8I8 | 网络 A 已完成；待网络 B 与 BRAM 端口映射 |
| 4 | transaction 周期模型 | tile schedule + AXI 假设 | 网络 A 已完成；待 RTL/板上 AXI 参数校准 |
| 5 | 16-bit 与 packed12 对比 | 相同 tile/网络 | BRAM block、LUT、Fmax、DDR stall |
| 6 | MAC+requant 小原型 | 8×8 datapath | DSP/LUT/FF、post-route Fmax、bit-exact |
| 7 | DMA 小原型 | 实际 Zynq HP 口 | 有效带宽、back-pressure、4 KiB 边界 |
| 8 | 网络 B 编译 | 不同小型 CNN | ISA 缺口和通用性证据 |

## 5. 当前推荐执行顺序

```text
确认板卡/DDR/工具
        ↓
位精确量化规则 + 整数解释器
        ↓
两模型图规范化和 ISA 编译
        ↓
生命周期、tiling、周期/带宽模型
        ↓
冻结数据布局、INT12 存储、ISA 字段
        ↓
MAC/requant 与 DMA 小原型综合
        ↓
冻结 P4 微架构，进入完整 RTL/HLS
```

当前最值得先做的代码不是完整 NPU RTL，而是“整数解释器 + 模型编译/存储规划 + 周期模型”。这三项能用较低成本暴露 ISA、BRAM和性能问题，避免 RTL 写到一半再改外部语义。
