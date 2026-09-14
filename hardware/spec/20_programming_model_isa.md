# NPU v1 编程模型与 Tensor ISA 草案

文档版本：`0.1-draft`

状态：架构提案，二进制编码尚未冻结

## 1. 设计原则

v1 使用“粗粒度 Tensor 指令”，一条卷积命令描述整层或一个 tile，而不是让软件逐条发 MAC。这样既能更换模型，又不会把有限 LUT/BRAM 消耗在复杂的标量处理器上。

命令流由离线编译器生成，PS 只负责装载和提交。硬件不做图优化、不推断 shape、不分配 Tensor 内存，也不解析模型文件。

## 2. 执行对象

一个可执行任务包由以下区段构成：

| 区段 | 作用 | 运行位置 |
|---|---|---|
| Header | magic、版本、区段边界、图哈希、校验 | PS 检查，NPU 可复核 |
| Command stream | 顺序执行的 128-bit 命令 | DDR，命令缓存分块读取 |
| Tensor descriptors | 地址、shape、stride、dtype、layout | DDR/描述符缓存 |
| Operator descriptors | kernel、padding、步幅、group 等 | DDR/描述符缓存 |
| Quant descriptors | multiplier、shift、clamp、LUT 等 | DDR/量化缓存 |
| Constants | 重排后的 INT8 权重、INT32 bias、LUT | DDR/片上 tile |
| Mutable tensors | 输入、中间结果、输出 | DDR/片上 tile |

所有表使用相对任务基址的 offset，禁止模型包携带开发主机的绝对路径。

## 3. 提交和执行语义

1. 驱动验证包头、长度、兼容版本和完整性；
2. 驱动分配物理连续或 DMA 可访问内存，写入任务描述符；
3. 写 doorbell 提交任务；
4. NPU 读取命令并按 PC 顺序发射；
5. DMA 与计算可并行，但对同一片上 bank 的读写依赖必须由 `WAIT` 或硬件 scoreboard 保证；
6. `END` 成功完成任务并回写 tag；任何 fatal error 中止当前任务；
7. v1 一次只执行一个任务，不支持中途抢占。

对同一输入和命令，执行结果必须与位精确参考模拟器一致。性能可以随 tile 方案变化，数值语义不能变化。

## 4. 命令格式提案

所有命令固定 128 bit、16-byte 对齐、小端存储。固定长度便于取指、错误定位和软件生成。字段提案如下：

| bit | 字段 | 含义 |
|---:|---|---|
| `7:0` | `opcode` | 主操作码 |
| `15:8` | `flags` | 异步、饱和、IRQ、扩展语义 |
| `31:16` | `tag` | 指令/调试 tag |
| `47:32` | `dst_td` | 目的 Tensor 描述符索引 |
| `63:48` | `src0_td` | 源 Tensor 0 索引 |
| `79:64` | `src1_td` | 源 Tensor 1 索引或 `0xffff` |
| `95:80` | `op_desc` | 算子/DMA 描述符索引 |
| `111:96` | `quant_desc` | 量化描述符索引或 `0xffff` |
| `127:112` | `imm` | 小立即数、event mask 或保留扩展 |

在 P3 冻结前必须用两个网络的真实命令流验证字段宽度。若 16-bit 描述符索引明显过宽，可回收位用于地址空间或调试，但冻结后不得改变相同主版本的编码。

## 5. 指令集范围

建议按类别分配 opcode，高位表示执行单元，低位表示子操作。下表定义语义范围，不冻结数值 opcode。

| 指令 | v1 等级 | 执行单元 | 语义 |
|---|---|---|---|
| `NOP` | P0 | Command | 无状态变化，可用于对齐 |
| `WAIT` | P0 | Command | 等待 DMA/compute event mask |
| `SIGNAL` | P1 | Command | 软件可见或内部事件置位 |
| `END` | P0 | Command | 成功结束任务，可选触发 IRQ |
| `DMA_LOAD` | P0 | DMA | DDR → 片上 bank/tile |
| `DMA_STORE` | P0 | DMA | 片上 bank/tile → DDR |
| `DMA_COPY2D` | P1 | DMA | 带 source/destination stride 的搬运 |
| `CONV2D` | P0 | MAC | 普通或 depthwise 卷积，含 bias |
| `VEC_ADD` | P0 | Vector | 可独立重标定的逐元素加法 |
| `VEC_SUB` | P1 | Vector | 逐元素减法 |
| `VEC_MUL` | P1 | Vector | 逐元素定点乘法与 requant |
| `VEC_MINMAX` | P1 | Vector | min、max 或 clamp |
| `ACT` | P0 | Post | ReLU、LeakyReLU、tanh-LUT 或 identity |
| `REQUANT` | P0 | Post | INT32/INT16 到目标激活格式 |
| `UPSAMPLE2X` | P0 | Address/Post | 最近邻 2×，单轴或双轴 |
| `POOL2D` | P1 | Post | max/average pool |
| `COPY_LAYOUT` | P1 | DMA/Post | 支持的布局转换/连续化 |

`GEMM` 在 v1 中优先由编译器映射为高宽为 1 的 `CONV2D`，不急于增加重复硬件语义。concat 不设独立计算指令；编译器生成 segmented Tensor view，卷积读地址发生器依次读取各段通道。

## 6. Tensor 描述符

Tensor 描述符建议固定 64 byte 并至少包含：

- `base_offset`：相对任务数据基址的 64-bit byte offset；
- `allocation_bytes`：可访问区长度，用于硬件边界检查；
- `N/H/W/C`：四维逻辑 shape，v1 要求 N=1；
- 四维 byte stride；
- `dtype`：INT8、INT12-in-INT16、INT16、INT32、UINT8；
- `layout`：LINEAR、NHWC8、WEIGHT_O8I8、SEGMENTED；
- `quant_desc`：缺省量化参数索引；
- `flags`：read-only、constant、外部 DDR 或片上 bank；
- segmented view 的段表 offset 和段数。

描述符的每个维度上限由两个真实模型和地址发生器位宽共同确定。在冻结前不能默认“16 bit 一定够”，必须对最大地址、stride 和乘法中间值做溢出分析。

### 主布局提案

激活逻辑形状为 `[N,H,W,C]`，物理通道按 8 补齐：

```text
NHWC8 address = ((((n*H + h)*W + w)*ceil(C/8) + c/8)*8 + c%8)
```

当前音频模型映射为 `H=frequency, W=time`。权重从训练时 `OIHW` 离线重排为：

```text
[O/8][I/8][Kh][Kw][o_lane][i_lane]
```

尾通道补 0。布局服务于 8×8 MAC 阵列，但模型逻辑通道数仍保存真实值，输出尾通道不得写入用户可见 Tensor 范围。

## 7. 卷积描述符和边界

`CONV2D` 描述符至少包含：

- 输入、输出、权重和 bias Tensor 索引；
- `Kh/Kw`：P0 为 1 或 3，并允许 `1×3`；
- `stride_h/stride_w`：1 或 2；
- `dilation_h/dilation_w`：v1 P0 为 1；
- 上、下、左、右显式 padding；
- `groups`：1 或 depthwise (`groups=Cin`)；
- tile 原点和 tile 尺寸；
- padding 填充值，v1 对称量化时为 0；
- post-op 链和量化描述符。

当前因果模型通过显式非对称时间 padding 表示，不为“音频因果卷积”增加专用 opcode。编译器负责保证不会读取未来帧。

当前 P3 二进制原型把 Operator descriptor 固定为 64 B：前 32 B 是 16 个 little-endian `uint16`，依次保存 `src/dst/weight/bias Tensor`、`Kh/Kw`、stride、dilation、四边 padding、groups 和 post-op；后 32 B 是 8 个 little-endian `uint32`，依次保存：

```text
tile_origin_h, tile_origin_w, tile_origin_cout,
tile_h, tile_w, tile_cout,
input_channel_start, input_channel_count
```

网络 A 的最大值全部在范围内，但在网络 B 完成前不缩窄这些字段。

## 8. 量化描述符

硬件不读取 float scale。编译器把实数比例近似为整数乘法和移位：

```text
acc  = sum(input * weight) + bias
wide = acc * multiplier
q    = round_to_nearest_even(wide / 2^shift)
out  = saturate(q, clamp_min, clamp_max)
```

普通卷积允许 `multiplier[Cout]` 和 `shift[Cout]`。逐元素 ADD 的两个输入必须先对齐到公共尺度。所有乘法中间位宽、负数舍入、shift 合法范围和饱和顺序要由位精确模型定义，并生成边界测试向量。

激活可以与卷积融合，但融合前后必须产生与独立命令完全相同的整数结果。否则融合属于不同 opcode/flag，不能由实现静默改变顺序。

## 9. 同步和内存一致性

- 每个异步 DMA 命令分配一个 event bit；完成后置位；
- `WAIT mask` 阻塞到指定事件全部完成；
- 计算命令只有在输入 tile 已就绪、输出 bank 可写时才能发射；
- 对同一外部地址的读写顺序由命令流负责，硬件不实现 cache coherence；
- `END` 隐含等待此前所有写 DMA 完成；
- fatal error 时停止发射新命令，但必须让已经握手的 AXI transaction 合法收尾。

tile 调度原型把命令 `imm[15:0]` 暂定为：`[7:0]` completion event mask，`[8]` activation bank，`[9]` weight bank，`[10]` accumulator bank，`[11]` output bank，`[13:12]` segment index，`[14]` segmented addressing，`[15]` 保留。`WAIT` 只解释 event mask；其它字段由对应执行单元解释。该编码属于 ADR-011/020 的验证载体，尚未冻结。

## 10. 错误与版本兼容

错误至少分为：

- 非法/不支持 opcode；
- ISA 主版本不兼容；
- 描述符索引或字段越界；
- Tensor 地址、长度或对齐错误；
- shape/layout/dtype 组合不支持；
- DMA `SLVERR/DECERR`；
- watchdog 超时；
- 内部 FIFO overflow/underflow 或 scoreboard 冲突。

首个 fatal error 必须锁存 `error_code`、`command_pc`、指令 tag 和任务 tag。ISA 主版本改变二进制语义，次版本只允许增加 capability 可探测的新功能。

## 11. ISA 冻结门槛

P3 前不得宣称 ISA 冻结。至少需要：

1. 当前人声模型成功编译，无隐藏专用操作；
2. 第二个不同拓扑 CNN 成功编译；
3. 位精确解释器运行全部命令并与量化框架对齐；
4. 命令/描述符数量和字段宽度统计完成；
5. 每条 P0 指令的合法/非法边界测试完成；
6. 用周期模型证明 DMA、tile 和命令开销满足实时门槛；
7. 驱动可以从 capability 拒绝不兼容的模型包。
