# P2 可执行规格与编译结果

文档版本：`0.1-draft`

状态：网络 A 已编译；tanh 数值语义已形成，tile 级调度未冻结

## 1. 本阶段产物

`scripts/26_compile_npu_program.py` 将当前专用模型参考包转换为通用 NPU 架构包：

```text
hardware/generated/bott2_mir1k_v1/
    manifest.json
    weights_int8.bin       OIHW
    bias_int32.bin
    scales_f32.bin
             │
             ▼ compile
hardware/generated/bott2_mir1k_v1_program/
    program.json           可读的命令和描述符
    commands.bin           128-bit 指令流
    tensor_desc.bin        64-byte Tensor 描述符
    operator_desc.bin      64-byte 卷积描述符
    quant_desc.bin         32-byte 量化描述符
    quant_params.bin       每通道整数 multiplier/shift
    segments.bin           concat 分段视图
    weights_o8i8.bin       阵列布局权重
    bias_int32.bin
    tanh_lut_int12.bin     4096 项 signed INT12 LUT
    analysis.json          存储、周期和风险报告
```

`scripts/npu_isa.py` 是指令编码和定点原语的单一事实来源。`scripts/_test_npu_isa.py` 验证编码、舍入、scale、权重布局和输出包完整性。

## 2. 暂定二进制结构

| 对象 | 固定大小 | 本模型数量 | 二进制大小 |
|---|---:|---:|---:|
| Command | 16 B | 72 | 1,152 B |
| Tensor descriptor | 64 B | 42 | 2,688 B |
| Operator descriptor | 64 B | 11 | 704 B |
| Quant descriptor | 32 B | 13 | 416 B |
| Segment record | 8 B | 6 | 48 B |

暂定 opcode：

| Opcode | 编码 | Opcode | 编码 |
|---|---:|---|---:|
| `NOP` | `0x00` | `DMA_LOAD` | `0x10` |
| `WAIT` | `0x01` | `DMA_STORE` | `0x11` |
| `SIGNAL` | `0x02` | `DMA_COPY2D` | `0x12` |
| `END` | `0x03` | `CONV2D` | `0x20` |
| `VEC_ADD` | `0x30` | `ACT` | `0x40` |
| `VEC_SUB` | `0x31` | `REQUANT` | `0x41` |
| `VEC_MUL` | `0x32` | `UPSAMPLE2X` | `0x42` |
| `VEC_MINMAX` | `0x33` | `POOL2D` | `0x43` |
|  |  | `COPY_LAYOUT` | `0x44` |

这些数值只用于软件原型和测试。第二网络尚未编译，因此 ADR-011 仍是提案，不能宣称 ISA 已冻结。

## 3. 已固定的整数语义

当前软件 golden 明确定义：

- signed 二补码；
- scale 编码为正的 signed-Q31 `multiplier / 2^shift`；
- 右移使用 round-to-nearest-even；
- 对负数在绝对值域舍入后恢复符号，所以 `-0.5→0`、`-1.5→-2`、`-2.5→-2`；
- requant 在完整乘法中间值上舍入，再 clamp；
- INT12 clamp 为 `[-2048, 2047]`；
- LeakyReLU 暂定精确斜率 `1/10`，同样使用 ties-to-even；
- 每输出通道独立 multiplier/shift；
- residual 支路单独从输入 scale 重标定到输出 scale，再执行 ADD 和饱和。
- tanh 输入是 `out.conv` 的 signed INT12，步长 `0.0046281479`；输出是 signed INT12，步长 `1/2047`。

新生成的整数 scale 最大相对表示误差低于回归门槛 `1e-8`。RTL 必须对同一边界向量 bit-exact。

## 4. 权重布局结果

训练布局 `OIHW` 被重排为：

```text
[O/8][I/8][Kh][Kw][o_lane][i_lane]
```

输入/输出通道不足 8 时补 0。当前权重从 824,000 B 变为 825,856 B，增加约 0.23%。全部 11 层已经通过反向重排逐字节比较，证明转换没有改变有效权重。

## 5. 存储结果

| 项目 | 结果 |
|---|---:|
| 不复用的 mutable DDR arena | 950,272 B |
| Tensor 生命周期峰值 | 524,288 B |
| 暂定 activation scratchpad | 216 KiB |
| O8I8 权重 | 825,856 B |
| bias | 3,600 B |
| 整数量化参数 | 14,432 B |
| tanh LUT | 8,192 B |
| 每任务估算 DDR 流量 | 2,388,992 B |

峰值活跃 Tensor 是 `enc0` skip、`dec1.upsample` 和 `dec1` 输出，共 512 KiB。它不可能完整放进 216 KiB activation scratchpad，因此后续 tile 编译器必须：

1. 让 encoder skip 驻留 DDR；
2. decoder 使用 segmented view 读取 upsample 与 skip；
3. 沿频率轴分 tile，当前建议 `tile_h≤16`；
4. 避免物化完整 concat；
5. 输出通道按 8 计算，partial sum 保存在 accumulator bank。

当前 11 层的建议 tile 均通过 216 KiB activation、108 KiB weight 和 54 KiB accumulator 容量检查。这只是容量证明，bank 端口冲突尚未建模。

## 6. 周期模型结果

| 项目 | 周期 |
|---|---:|
| Tensor MAC | 1,986,560 |
| residual vector | 1,024 |
| upsample | 24,576 |
| DMA 理想总线＋burst 开销 | 335,952 |
| 命令处理 | 288 |
| **全部不重叠估计** | **2,348,400** |
| 预算 | 4,000,000 |

对应时间：

- 200 MHz：约 11.74 ms；
- 100 MHz：约 23.48 ms；
- 当前硬门槛：46 ms。

这个结果说明 64-lane 方案值得继续，并不等于已经通过板级性能验收。DMA 数字假设 64-bit 总线每周期一拍并增加 burst 开销，尚未包含真实 DDR 竞争、HP 口效率和 bank stall。

## 7. tanh 标定结果

早期参考包只有最终输出 scale，无法确定 tanh LUT 输入。现在已经在 MIR-1K 训练缓存上观测最后一层卷积的 pre-tanh 输出：

| 项目 | 结果 |
|---|---:|
| 观测值数量 | 2,097,152 |
| 范围 | `[-5.66438, 9.47382]` |
| 绝对值 P99 / P99.9 / P99.99 | 2.65243 / 3.62528 / 4.77527 |
| max 标定 INT12 step | 0.0046281479 |
| tanh 输出相对浮点 SNR | 56.31 dB |
| tanh 输出最大绝对误差 | 0.0023135 |
| 观测样本裁剪率 | 0% |

选择 max 而不是 P99.9，保持与其它激活相同的 max-calibration 策略并确保校准样本不裁剪。编译器生成 4096 项 LUT，索引覆盖 `[-2048,2047]`；回归验证 LUT 单调、零点正确并在两端饱和。`NUM-TANH-001` 已关闭。

## 8. 尚未完成

- layer command 展开为 tile 级 DMA/compute 命令；
- scratchpad bank 分配和端口冲突模型；
- 第二个不同拓扑网络的编译；
- 单一带 header、section directory 和 CRC 的 `task.bin`；
- 完整命令解释器逐层运行网络 A。

## 9. 复现

```powershell
python scripts/26_compile_npu_program.py
python scripts/_test_npu_isa.py
```

若更换权重或量化校准集，先重新采集 tanh 前范围：

```powershell
python scripts/27_calibrate_tanh.py
python scripts/25_export_npu_package.py
python scripts/26_compile_npu_program.py
```

编译输出完全由参考包决定；`program.json` 不包含本机绝对路径，每个二进制区段均记录字节数和 SHA-256。
