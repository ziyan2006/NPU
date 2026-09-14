# 专用人声消除加速器 AXI4-Lite 寄存器草案 v0.1（历史基线）

> 状态说明：通用 NPU 使用 task/descriptor 接口，新的控制寄存器见 `21_control_registers.md`；本文 offset 不承诺兼容。

所有寄存器为 32 位、小端、4 字节对齐。地址仍是草案，但字段语义从现在起保持兼容。

| 偏移 | 名称 | 访问 | 含义 |
|---:|---|---|---|
| `0x00` | `CONTROL` | R/W | bit0 `start`，bit1 `soft_reset`，bit2 `continuous`，bit3 `irq_enable` |
| `0x04` | `STATUS` | R | bit0 `busy`，bit1 `done`，bit2 `deadline_miss`，bit3 `dma_error`，bit4 `fifo_error` |
| `0x08` | `WEIGHT_BASE_LO` | R/W | 权重 blob 物理地址低32位 |
| `0x0C` | `WEIGHT_BASE_HI` | R/W | 权重 blob 物理地址高32位 |
| `0x10` | `BIAS_BASE_LO` | R/W | INT32 bias blob 地址低32位 |
| `0x14` | `BIAS_BASE_HI` | R/W | INT32 bias blob 地址高32位 |
| `0x18` | `SCALE_BASE_LO` | R/W | scale blob 地址低32位 |
| `0x1C` | `SCALE_BASE_HI` | R/W | scale blob 地址高32位 |
| `0x20` | `INPUT_BASE_LO` | R/W | `2×128×16` INT12 输入块地址低32位 |
| `0x24` | `INPUT_BASE_HI` | R/W | 输入块地址高32位 |
| `0x28` | `OUTPUT_BASE_LO` | R/W | `4×128×16` 输出掩码地址低32位 |
| `0x2C` | `OUTPUT_BASE_HI` | R/W | 输出掩码地址高32位 |
| `0x30` | `BUTTON_GAIN` | R/W | Q1.15，`0` 原混音，`32768` 去人声；硬件内部30ms斜坡 |
| `0x34` | `LF_KILL_BANDS` | R/W | 缺省44；超过范围则置错误 |
| `0x38` | `CHUNK_SEQUENCE` | R/W | 软件提交的块序号，输出完成时原样返回 |
| `0x3C` | `CYCLES_LAST` | R | 上一块 CNN 周期数 |
| `0x40` | `CYCLES_MAX` | R | 复位以来最坏块周期数 |
| `0x44` | `FIFO_HIGH_WATER` | R | 复位以来输入 FIFO 最高占用 |
| `0x48` | `ERROR_COUNT` | R | deadline/DMA/FIFO 错误累计数 |
| `0x4C` | `VERSION` | R | 高16位主版本，低16位 manifest schema |

`start`、`soft_reset` 使用写1触发并自动清零。`STATUS` 的错误位采用写1清除，防止瞬态错误被软件漏读。
