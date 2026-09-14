# 算子到指令执行契约

文档版本：`0.1-draft`

状态：供编译器、RTL 和驱动并行开发的 P3 接口；网络 B 和 RTL golden 完成前不冻结

## 1. 交付边界

本 NPU 使用粗粒度 Tensor 指令。一条 `CONV2D` 驱动 MAC 阵列完成一个 tile 的全部循环，不为每次乘加生成软件指令。

三类对象分工如下：

| 对象 | 决定什么 | 不决定什么 |
|---|---|---|
| Command | 执行单元、Tensor/描述符索引、同步、bank | kernel/shape 的完整参数 |
| Operator descriptor | kernel、stride、padding、tile 坐标和通道范围 | DDR 任务基址 |
| Tensor/Quant descriptor | 地址、shape、stride、布局、整数 scale/clamp | 指令执行顺序 |

规范源为 `scripts/npu_isa.py`；RTL 使用 `hardware/rtl/include/npu_isa_pkg.sv`，PS/驱动使用 `software/include/npu_isa.h`。两个头文件均由 `scripts/29_generate_npu_isa_headers.py` 生成。

## 2. 当前模型算子覆盖

| 模型算子 | 指令序列 | 备注 |
|---|---|---|
| Conv2D + bias + LeakyReLU | `DMA_LOAD* → WAIT → CONV2D → WAIT → DMA_STORE` | requant 和激活融合在 `CONV2D` |
| Conv2D + bias + tanh | 同上 | 最后一层通过 4096 项 LUT 融合 tanh |
| Residual Conv | Conv 序列，在 store 前增加 `VEC_ADD` | residual 支路使用独立 rescale 参数 |
| Nearest Upsample 2× | `UPSAMPLE2X` | 当前作为同步流式命令 |
| Concat | 无独立指令 | `SEGMENTED` Tensor view，多个 `DMA_LOAD` 写同一 A bank 的不同通道段 |

当前网络 A 的 tile 流只实际发出 `DMA_LOAD`、`WAIT`、`CONV2D`、`VEC_ADD`、`UPSAMPLE2X`、`DMA_STORE` 和 `END`。`NOP`、独立 `ACT/REQUANT` 属于完整 v1 P0 接口但尚未由网络 A 发出；P1 opcode 可以在首版 decoder 中报告 unsupported，不能误执行为 NOP。

## 3. 通用命令规则

- 每条命令 16 B、16-byte 对齐、little-endian；byte 0 为 opcode；
- `0xffff` 表示未使用的描述符索引；
- Command fetch/decoder 必须先检查 opcode、flag 组合和索引范围，再向执行单元发射；
- `CONV2D` command 与 Operator descriptor 中重复的 Tensor 索引必须一致，否则报告 descriptor mismatch；DMA command 只借用 `op_desc` 的 tile 坐标，数据来源由 DMA command 自己的索引决定；
- `ASYNC` 命令在成功进入对应执行队列后允许 command processor 继续取指；完成时置 `imm[7:0]` 指定的 event；
- 发起一个带 completion event 的异步命令时先清该 event，硬件完成后再置位；event mask 为 0 表示不产生完成事件；
- `WAIT` 只读取 `imm[7:0]`，其余 `imm` 位和描述符字段必须为 0/`0xffff`；
- fatal error 后停止发射新命令，已与 AXI 完成握手的 transaction 必须合法收尾。

## 4. 指令契约

### `DMA_LOAD (0x10)`

从 DDR 读取当前 tile 所需数据。`op_desc` 提供 tile 坐标；`imm[8]` 选择 A bank，`imm[9]` 选择 W bank。

支持三种来源：

1. 激活：`src0_td` 指向普通或 segmented source Tensor，`dst_td` 指向逻辑输入 Tensor；
2. 权重/bias：`src0_td=dst_td`，并指向对应 constant Tensor；
3. 量化参数：`src0_td=dst_td=0xffff`，`quant_desc` 指向参数表。

segmented load 时 `imm[14]=1`，`imm[13:12]` 是 segment index；地址发生器从 Segment descriptor 取得源 Tensor 和目的通道偏移。同方向 DMA 按发射顺序完成，因此一组 load 只在最后一条设置 ready event。

W bank 内部采用固定的 64-byte 对齐顺序，命令不再携带一套重复的片上地址：

```text
weight_base      = 0
bias_base        = align64(weight_base + weight_tile_bytes)
conv_quant_base  = align64(bias_base + bias_tile_bytes)
vector_quant_base= align64(conv_quant_base + conv_quant_bytes)  # residual only
```

DDR 地址和长度从描述符计算：weight 按 `tile_origin_cout/8` 选择 O8I8 block，bias 从 `base_offset + tile_origin_cout*4` 开始，卷积 quant 从 `param_offset + tile_origin_cout*16` 开始；residual quant 只有一个 16-byte 参数记录。RTL 必须使用与编译器相同的 64-byte 对齐公式并做 W-bank 容量检查。

在 `DMA_LOAD` 上下文中，`imm[10]` 是 `DMA_VECTOR_QUANT`：0 表示卷积 per-channel quant，1 表示 residual/vector scalar quant。该位在 `CONV2D` 上下文中仍表示 accumulator bank。编译器必须显式设置，RTL 不允许用 `param_count==1` 猜测用途。

### `WAIT (0x01)`

阻塞 command processor，直到 `(event_state & imm[7:0]) == imm[7:0]`。`WAIT` 不清 event；下一次以同一 event 发起异步命令时清除旧值。watchdog 负责检测永不完成的等待。

### `CONV2D (0x20)`

`src0_td/src1_td/dst_td` 分别指向输入、权重和卷积输出；`op_desc` 指向当前 tile，`quant_desc` 指向每输出通道 requant 参数。网络 A 使用 `ASYNC | SATURATE | FUSED_POST_OP`。

数据通路顺序固定为：

```text
INT32 acc = Σ(input × weight) + bias
wide      = acc × Q31_multiplier[channel]
q         = round_to_nearest_even(wide / 2^shift[channel])
post      = identity / ReLU / LeakyReLU(1/10) / tanh-LUT
dst       = saturate_to_INT12(post)
```

`imm[8:11]` 依次选择 A/W/P/O bank，`imm[14]` 选择 segmented input AGU。完成是最后一个有效输出元素写入 O bank且内部 pipeline 不再引用 A/W/P 时；只有此时才能置 compute-done event。

### `VEC_ADD (0x30)`

网络 A 的 residual 语义为：

```text
residual_q = requant(src1, residual_multiplier, residual_shift)
dst        = saturate_INT12(src0 + residual_q)
```

`src0` 是已经完成卷积 post-op、处于输出 scale 的 O-bank tile；`src1` 是 A-bank 中的 residual 输入；`quant_desc` 必须是该层独立的 `residual_rescale` 描述符，不能复用卷积 requant 描述符。命令为同步命令，返回时结果已覆盖写入 O bank。

### `UPSAMPLE2X (0x42)`

对 H/W 两轴做最近邻 2×：`dst[h,w,c] = src[h>>1,w>>1,c]`。当前命令同步完成，返回时输出 Tensor 对后续 segmented load 可见。首版实现可流式访问 DDR，不要求完整 upsample Tensor 驻留片上。

### `DMA_STORE (0x11)`

把 O bank 中由 `op_desc` 指定的输出 tile 写回 `dst_td`。`imm[11]` 选择 O bank，`imm[7:0]` 指定 store-done event。O bank 可包含 `align8(tile_cout)` 个物理 lane，但 DDR 每像素只写 `tile_cout` 个逻辑通道；尾 beat 必须生成正确 byte strobe，不得覆盖 padded lane 或逻辑 Tensor 之外的数据。

### `END (0x03)`

仅当所有已发射写 DMA 完成、执行单元 idle 且没有 fatal error 时成功结束任务。`IRQ` flag 为 1 时置完成中断，并回写 task tag/最终 PC。

## 5. 当前 tile 模板

每层先预取 tile 0，然后在 tile `k` 计算期间预取 `k+1`：

```text
DMA_LOAD input[k]                 # 同一空间 tile 只加载一次
DMA_LOAD weight[k]
DMA_LOAD bias[k]
DMA_LOAD conv_quant[k]
DMA_LOAD residual_quant[k]       # 仅 residual layer
WAIT input_ready | weight_ready | reusable_output_bank
CONV2D k ASYNC
    DMA_LOAD input/weight[k+1]    # 与 compute k 重叠
WAIT compute_done[k]
VEC_ADD k                         # 仅 residual layer
DMA_STORE k ASYNC
```

A bank 按空间 tile ping-pong，同一空间位置的不同输出通道 tile 复用输入；W/P/O bank 按计算 tile ping-pong。层末等待两个 store event 后才进入下一层。

## 6. RTL 首轮实现范围

第一轮可按以下顺序联调，但不能省略非法指令处理：

1. `npu_command_t` 解包、PC、opcode/flag/index 检查；
2. event scoreboard、`WAIT`、`END` 和错误锁存；
3. `DMA_LOAD/DMA_STORE` 请求接口，先用 testbench memory model；
4. `CONV2D` 描述符读取和循环控制，MAC 可先用行为模型；
5. `VEC_ADD` 的独立 residual quant；
6. `UPSAMPLE2X` 和 segmented AGU；
7. 接入真实 MAC/requant pipeline 和 AXI master。

ISA 冻结前仍需网络 B、完整整数解释器、非法编码测试和 RTL bit-exact 测试。实现期间发现字段不足时必须回到本文件、`npu_isa.py` 和决策记录统一修改，不能在 RTL 私自增加隐含语义。
