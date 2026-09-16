# XC7Z020 轻量通用 NPU 规格总纲

文档版本：`0.1-draft`

规格状态：网络 A 的 v1 ISA、CSR 和功能微架构已形成实现基线；板级集成约束待具体板卡确认

目标器件：`XC7Z020-1`

首要工作负载：当前实时人声消除网络

## 1. 设计目标

本项目不再把 PL 做成只能运行一个固定人声模型的状态机，而是设计一颗面向小型静态 CNN 的轻量可编程 NPU：模型在部署前由软件编译为命令流、张量描述符和量化参数，NPU 按命令顺序执行。

“通用”在 v1 中的准确含义是：同一套硬件能运行多个满足算子、尺寸和精度边界的 CNN，而不是实现桌面 GPU、完整深度学习框架或任意动态模型。

首版必须先保证当前人声消除网络实时运行，再用第二个不同拓扑的小型 CNN 证明硬件没有写死到当前网络。

## 2. 规格层级与优先级

发生冲突时按以下顺序处理：

1. `10_system_requirements.md` 中状态为“已确认”的 P0 要求；
2. 已批准的决策记录；
3. ISA、描述符和微架构规格；
4. 当前模型的 `manifest.json`；
5. 历史专用加速器文档。

`01_npu_architecture.md` 和 `02_register_map.md` 是专用版基线，用于保留已有性能测算和产品需求，不再冻结通用 NPU 的执行方式或寄存器接口。

状态定义：

- **已确认**：可作为设计与验收依据；修改需要记录原因和影响。
- **提案**：推荐方案，必须经过模型编译器/周期模型或综合证据后才能冻结。
- **待确认**：缺少板卡、系统或实测信息，不能作为硬约束。

## 3. 专业 NPU 开发流程

| 阶段 | 主要输出 | 退出条件 | 当前状态 |
|---|---|---|---|
| P0 概念与用例 | 使用场景、器件、边界、非目标 | 项目目标无歧义 | 已完成 |
| P1 需求基线 | 带编号的功能、性能、接口、资源要求 | 每个 P0 要求有验证方法 | 网络 A 已完成，板级项待确认 |
| P2 工作负载分析 | 算子覆盖、张量尺寸、MAC/带宽/生命周期 | 当前模型和第二模型均可映射 | 部分完成 |
| P3 架构/ISA 探索 | 数据流、存储层次、ISA、周期模型 | 性能和 BRAM 预算闭合 | 网络 A 已完成，网络 B 待验证 |
| P4 微架构规格 | 模块接口、流水线、时序、异常行为 | RTL 接口和逐周期行为可实现 | 网络 A 功能路径已完成 |
| P5 可执行参考 | 图编译器、位精确模拟器、测试向量 | 所有 P0 指令有 golden | 网络 A 完整 task golden 已完成 |
| P6 RTL/HLS 实现 | 可综合模块和软件驱动 | 模块仿真通过 | 功能 RTL 与可移植驱动完成 |
| P7 集成验证 | SoC、DMA、中断、CDC、回归 | 覆盖率和需求回归闭合 | task bit-exact，参考 PS/AXI/IRQ SoC 已综合 |
| P8 实现收敛 | 综合、布局布线、时序、功耗 | 资源和时钟满足规格 | 通用 XC7Z020 完整 SoC 已过 100 MHz；板卡专用重跑待实物信息 |
| P9 上板验收 | 长稳、实时性、音频效果 | 所有 P0 板级测试通过 | 未开始 |

P4/P5/P6 已完成网络 A 的第一条完整垂直路径：真实 task image 经 AXI4-Lite
doorbell 启动，task loader、Command Processor、DMA/Scratchpad、CONV2D/post、
VEC_ADD、UPSAMPLE2X 和共享 AXI 协同执行。1,869 条命令对 32,768-byte 输出与独立
整数参考模型逐字节一致，需 3,254,220 cycle（32.54 ms @100 MHz）。完整 `npu_top`
在临时 `xc7z020clg400-1` 上 OOC 布局布线后使用 25,645 Slice LUT、24,894 FF、
61 BRAM36、72 DSP；100 MHz WNS `+0.225 ns`、TNS `0`、WHS `+0.007 ns`，且
0 条未布通网络、0 条 critical DRC。封装 IP 接入通用 PS7 参考系统后，完整 SoC
post-route 为 25,634 LUT、25,293 FF、61 BRAM36、72 DSP，100 MHz WNS
`+0.005 ns`、WHS `+0.015 ns`，0 unrouted、0 critical DRC。驱动 CSR 契约和交叉编译
也已通过；下一阶段只剩实际板卡 preset、软件平台适配和板级验收。

## 4. 系统边界

```text
模型/量化工具
      │  离线编译
      ▼
命令流 + 张量描述符 + 算子描述符 + 定点量化表 + 权重
      │
      ▼ DDR
┌────────────── Zynq PS ──────────────┐
│ 应用、NPU 驱动、内存分配、提交/中断 │
└──────── GP 控制 ────────┬───────────┘
                           │
┌────────────────────── Zynq PL ──────────────────────┐
│ 命令处理器 ─► DMA/片上存储 ─► Tensor MAC/Vector 单元 │
│                         └────► 量化/激活/性能计数器   │
└────────────────── HP 数据口连接 DDR ────────────────┘
```

v1 的 NPU 输入/输出是量化张量。PCM、STFT/iSTFT、滤波器组、OLA、按键渐变和音频编解码器不进入通用 Tensor ISA；它们由 PS 或独立 PL IP 实现。这样 NPU 不与某一种音频前后端绑定。

## 5. v1 范围

### 必须支持

- batch=1、部署时静态形状的 2D CNN；
- 普通卷积、深度卷积、1×1 卷积映射；
- 残差逐元素加法、定点激活、重定标和饱和；
- 最近邻 2× 上采样；
- 通过张量视图/地址描述实现 concat，避免无意义数据复制；
- DDR 与片上 tile 之间的显式 DMA；
- INT8 权重、至少 INT12 有效激活精度和 INT32 累加；
- 当前 11 层因果 U-Net 的完整执行。

### v1 明确不做

- 训练、反向传播和稀疏训练；
- 动态 shape、动态控制流、抢占和多租户；
- 浮点 Tensor 运算；
- Attention、Softmax、LayerNorm、LSTM/GRU；
- 缓存一致性和虚拟地址；
- 完整 ONNX/PyTorch 算子兼容。

如果后续必须支持 Transformer 或动态模型，应作为 v2 需求重新评估，而不是在 v1 RTL 中预埋大量未经验证的复杂度。

## 6. 当前基线事实与架构假设

### 已有实测/生成事实

- 当前候选模型为 824,900 参数、11 个卷积层；
- 每个 `2×128×16` 输入块为 123,338,752 MAC；
- 16 帧对应 92.88 ms 音频，硬件首阶段门槛为 46 ms；
- `Pout=8, Pin=8` 的 64-lane 调度计算量为 1,986,560 个 MAC 阵列周期；
- 权重 INT8 可接受，当前模型的部署激活不能直接降为 INT8；INT12 是现阶段基线；
- 当前权重 blob 为 824,000 B，必须由 DDR 分块读取。

### 尚需证据的提案

- 64 MAC lane 是 v1 最合适的阵列规模；
- 激活按 16-bit 容器保存 INT8/INT12/INT16，优于 12-bit 紧凑打包；
- 激活采用 `NHWC8`、权重采用 8×8 block 布局；
- 128-bit 固长命令足以覆盖 v1；
- 片上存储不保留完整网络激活，使用 tile + DDR spill 可以满足实时要求。

## 7. 规格文档

| 文档 | 内容 |
|---|---|
| `10_system_requirements.md` | 带编号、优先级、状态和验证方法的需求基线 |
| `11_workload_profile.md` | 当前模型逐层画像、存储压力和第二验证网络候选 |
| `20_programming_model_isa.md` | 执行模型、命令、描述符、算子和数值语义 |
| `21_control_registers.md` | 通用 NPU AXI4-Lite 控制接口草案 |
| `22_operator_instruction_contract.md` | 算子到指令序列、字段约束和 RTL 完成条件 |
| `30_microarchitecture_budget.md` | 模块划分、数据流、片上存储、资源和周期预算 |
| `31_command_processor_microarchitecture.md` | 首个 RTL 控制核心的接口、状态机、event 和异常语义 |
| `32_dma_frontend_microarchitecture.md` | 描述符缓存、section 重定位、三维 DMA AGU 和边界检查 |
| `33_axi_dma_engine_microarchitecture.md` | AXI burst、4 KiB 拆分、scratchpad 接口、错误排空和验证 |
| `34_dma_subsystem_scratchpad_microarchitecture.md` | 描述符 sequencer、DMA 子系统集成、双口 BRAM bank 和计算端口边界 |
| `35_tensor_mac_microarchitecture.md` | 8x8 Tensor MAC 数值语义、流水线、反压、资源和 row-buffer 输入 |
| `36_conv2d_controller_microarchitecture.md` | CONV2D tile 循环、地址、descriptor 检查、位精确验证和综合证据 |
| `37_requant_post_microarchitecture.md` | bias、Q31/RNE、clamp、激活、tanh LUT、资源和周期取舍 |
| `41_npu_top_microarchitecture.md` | 完整 core/top、AXI/CSR、任务生命周期、验证和综合基线 |
| `42_soc_integration_preboard.md` | Vivado IP、Zynq PS/AXI/IRQ 集成、完整 P&R 和上板前清单 |
| `40_verification_plan.md` | 从位精确模型到板级长稳的验证闭环 |
| `50_p2_executable_spec.md` | 当前模型的指令编译、整数语义、存储与周期结果 |
| `90_decision_log.md` | 已决定事项、开放问题和需要补做的实验 |
| `01_npu_architecture.md` | 历史专用人声消除加速器基线 |
| `02_register_map.md` | 历史专用版寄存器草案 |

## 8. 变更控制

冻结 P1 后，任何影响 P0 要求、ISA 二进制兼容、外部接口、数值结果或资源上限的修改，都要在 `90_decision_log.md` 增加决策项，并更新对应需求与验证用例。优化实现可以改变内部流水线，但不能改变已冻结的可观察行为。
