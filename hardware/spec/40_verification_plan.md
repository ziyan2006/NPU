# NPU v1 验证计划

文档版本：`0.1-draft`

状态：验证策略草案

## 1. 验证原则

验证从需求出发，不以“能综合”或“能出声音”代替正确性。每个 P0 需求都要映射到自动化测试、分析报告或板级用例，并保存可复现输入、期望输出、版本和随机种子。

建立三套相互独立但语义一致的参考：

1. 浮点模型：用于判断模型质量；
2. 位精确整数解释器：定义 ISA 数值真值；
3. RTL/HLS DUT：必须逐命令、逐 Tensor 对齐整数解释器。

硬件不是直接对齐 PyTorch 浮点输出，而是先对齐已批准的整数语义；量化误差由算法门槛单独验收。

## 2. 验证层次

| 层次 | 对象 | 主要检查 | 退出条件 |
|---|---|---|---|
| V0 静态规格 | requirement/ISA/descriptor | 完整性、一致性、字段边界 | P0 均可验证，无冲突 |
| V1 软件 golden | packer/compiler/integer simulator | 图映射、布局、舍入、饱和 | 两模型全图可执行 |
| V2 单元仿真 | MAC、requant、AGU、DMA、FIFO、CSR | directed + constrained random | P0 边界与错误注入通过 |
| V3 模块集成 | command + DMA + SP + compute | event、bank、back-pressure | 命令级 bit-exact |
| V4 SoC 仿真 | AXI interconnect、PS driver | 提交、中断、复位、错误恢复 | 完整任务 bit-exact |
| V5 实现检查 | synth/P&R/CDC/timing | 资源、时钟、跨域、警告 | P0 约束闭合 |
| V6 上板 | 当前模型+第二模型 | 正确性、周期、流量、长稳 | 所有 P0 板级测试通过 |
| V7 产品链路 | 音频前后端+NPU | 延迟、连续性、听感/指标 | 产品验收通过 |

## 3. 位精确参考模型

在 RTL 数据通路之前，先实现软件解释器。它至少需要：

- 解析同一二进制 command/descriptor，不建立另一套隐式配置；
- 执行 P0 opcode；
- 精确定义补码扩展、乘法位宽、INT32 wrap/饱和策略；
- 实现 RNE、负数 shift、per-channel requant 和 clamp；
- 实现 NHWC8/O8I8 布局、尾通道、padding、stride 和 segmented view；
- 在每条命令后可导出 Tensor dump 和 hash；
- 支持 trace：PC、opcode、输入/输出描述符、cycle estimate、错误码。

当前 `hardware/generated/bott2_mir1k_v1` 中的 OIHW 权重和 float scale 用于产生编译前参考，不是最终位精确硬件包。必须新增重排权重和整数 requant 参数后，才能形成 RTL golden。

## 4. 单元测试矩阵

### Tensor MAC / CONV2D

- `Cin/Cout` 小于、等于和大于 8，尤其尾通道 1、7、8、9；
- shape 最小值和描述符允许的最大值；
- `1×1`、`1×3`、`3×3`，stride 1/2；
- 上下左右 padding 0 和最大合法值，非对称因果 padding；
- 普通/depthwise 卷积；
- 全 0、全最大正数、全最小负数、交替符号和随机数据；
- accumulator 接近正负边界，bias 后溢出，requant 后饱和；
- 每通道不同 multiplier/shift；
- 多 `Cin` tile 的 partial sum 保持。

### Vector / requant / activation

- 两输入相同和不同 scale 的残差加法；
- 正负恰好位于 0.5 LSB 的 RNE tie；
- shift 为 0、最小和最大合法值；
- ReLU、LeakyReLU 负区间、tanh LUT 端点和插值边界；
- INT8/INT12/INT16 最小最大值和 clamp。

### DMA / scratchpad

- 1 byte 尾部到最大 burst；
- 地址正好位于和跨越 4 KiB 边界；
- 非连续 2D stride、尾块 byte strobe；
- 读写 back-pressure、乱序允许范围、多个 outstanding；
- `SLVERR/DECERR` 注入和 transaction 收尾；
- ping-pong bank 切换、计算/DMA 同 bank 冲突；
- allocation 边界前后 1 byte 的非法访问。

### Command / CSR / reset

- 每个合法 opcode/flag 组合和所有保留编码；
- 描述符索引、shape、dtype、layout 的非法组合；
- WAIT event 已完成、延迟完成、永不完成触发 watchdog；
- 完成/错误 IRQ、清中断、轮询；
- idle、DMA 中、compute 中和 store 中 soft reset；
- 任务 tag、首错 PC/code 锁存和清除；
- ISA/capability 不兼容时拒绝执行。

## 5. 网络级 golden

### 网络 A：当前人声消除模型

固定以下回归输入：

- 随机 Tensor；
- 静音、单频、脉冲和最大幅度边界 Tensor；
- 量化校准集样本；
- 已保存试听歌曲的代表性块。

检查：

- 每条命令后的 Tensor hash；
- 11 层最终 Tensor bit-exact；
- 任务周期 `<4,000,000` 目标和墙钟 `<46 ms` 硬门槛；
- DMA 读写字节数与编译器报告一致；
- 连续块序号、因果边界和首尾块行为。

### 网络 B：通用性证明模型

在 P1 结束前选定，至少与网络 A 有两项差异，例如 depthwise、pool、不同通道尾数、不同 H/W 或不同残差结构。它不要求音频质量，但必须证明：

- 不修改 RTL 即可运行；
- 只替换模型包和输入；
- 每层与整数解释器 bit-exact；
- 性能与存储规划器误差处于批准阈值内。

## 6. 性能验证

硬件计数器和周期模型使用相同分类：

| 计数器 | 用途 |
|---|---|
| `cycles_total` | doorbell 接受到任务完成 |
| `cycles_compute_busy` | MAC/vector/post 实际忙 |
| `cycles_dma_read_wait` | 因 DDR 读数据未到而停顿 |
| `cycles_dma_write_wait` | 因写通道/back-pressure 停顿 |
| `cycles_bank_conflict` | 片上 bank/端口冲突 |
| `bytes_read/bytes_written` | 实际 DDR 流量 |
| `fifo_*_high_water` | FIFO 深度是否合理 |
| `commands_retired` | 命令流完整性与开销 |

板上报告必须同时给出频率、周期、毫秒、输入模型哈希和 bitstream 版本。仅报告“实时”不足以验收。

## 7. 结构与实现验证

- lint：无未驱动、多驱动、隐式锁存和位宽截断未审查告警；
- CDC/RDC：所有跨域和复位释放路径有已批准结构；
- assertions：AXI 协议、FIFO 不溢出/下溢、bank 所有权、event 生命周期、描述符边界；
- 综合：MAC/post 乘法按规划映射 DSP，控制路径不误占 DSP，BRAM 推断与预算一致；
- post-route：目标时钟域 WNS/TNS 满足门槛，无未约束路径；
- 代码覆盖目标：line/branch ≥90%，关键 FSM 状态/转移 100%；合理 exclusion 必须评审；
- 功能覆盖：每条 P0 opcode、dtype、kernel/stride、错误码和同步路径至少命中一次。

覆盖率是发现缺口的手段，不可用高覆盖率替代需求用例。

## 8. 板级验收

| ID | 用例 | 通过标准 | 关联需求 |
|---|---|---|---|
| B-001 | capability/版本读取 | 驱动正确识别，错误版本被拒绝 | IF-004 |
| B-002 | 网络 A 单块 | 输出 bit-exact，`<46 ms` | SYS-004, NUM-007, PERF-001 |
| B-003 | 网络 A 连续运行 | 30 分钟、序号连续、错误计数 0 | PERF-004, REL-004 |
| B-004 | 网络 B | 不换 bitstream，输出 bit-exact | SYS-005 |
| B-005 | AXI 压力 | 注入竞争后仍正确，deadline 行为可观测 | MEM-004, PERF-005 |
| B-006 | 错误注入 | 非法包/DMA 错误报告 code、PC、tag | EXE-005, REL-002 |
| B-007 | soft reset | 不发生越界写，可重新提交 | REL-003 |
| B-008 | 完整音频链路 | 无爆音/掉帧，延迟与质量达产品门槛 | SYS-003, PERF-004 |

## 9. 需求追踪与回归产物

每次可发布构建至少保存：

- requirement → test case → result 的追踪表；
- 模型包、输入向量、期望输出 hash；
- 软件/RTL commit、工具版本、随机种子；
- lint、CDC、综合、利用率、post-route timing 报告；
- 周期计数器、DDR 流量和性能模型对比；
- 板卡、bitstream、驱动和模型版本；
- 失败用例的最小复现。

P0 回归失败、未解释的 bit mismatch、负时序、资源超限或 deadline miss 都是发布阻断项。

## 10. 下一步验证优先级

1. 从现有 manifest 生成真正的定点 Tensor golden；
2. 固定 RNE/饱和/右移语义并建立边界测试；
3. 实现命令和描述符的软件解释器；
4. 实现张量生命周期/tiling/DDR 流量报告；
5. 选择网络 B 并验证 P0 算子覆盖；
6. 再开始 MAC、requant 和 DMA 的 RTL/HLS 单元原型。

当前增量：Command Processor 已接受网络 A 的 1,869 条真实 tile 命令流；Descriptor Cache/Fetch/AGU 已让全部 1,078 条 DMA command 的 RTL 请求与软件 reference 逐 bit 一致。AXI DMA Engine 已覆盖 4 KiB/256-beat 拆分、窄尾部、三维 stride、双向 back-pressure、错误收尾和 busy reset；全网 45,664 个 burst 已静态审计。A/W/O Scratchpad 已覆盖六 bank 路由、byte strobe、同步读、response back-pressure、并行访问和冲突仲裁；集成子系统已完成命令到 AXI read、BRAM 落地、event 以及错误 PC/tag 的端到端测试。Vivado 2026.1 OOC 综合确认 56 RAMB36、0 DSP，100 MHz WNS `+1.888 ns`，200 MHz WNS `-3.112 ns`。尚未覆盖真实 HP interconnect、多个 outstanding、MAC 宽读口、数值计算和 post-route。
