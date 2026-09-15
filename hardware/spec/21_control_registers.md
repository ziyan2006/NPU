# NPU v1 AXI4-Lite 控制接口

文档版本：`1.0-implemented`

状态：P4 RTL 已实现，v1 offset 冻结

所有寄存器为 32 bit、小端、4-byte 对齐。64-bit 值由 LO/HI 两个寄存器组成；软件写入地址时先写 HI/LO，最后写 `DOORBELL`，硬件只在 doorbell 时原子采样提交字段。

## 1. 寄存器表

| Offset | 名称 | 访问 | 含义 |
|---:|---|---|---|
| `0x000` | `IP_ID` | R | 固定 NPU magic/厂商标识 |
| `0x004` | `VERSION` | R | RTL 主/次版本 |
| `0x008` | `ISA_VERSION` | R | 支持的 ISA 主/次版本 |
| `0x00c` | `CAPABILITY0` | R | dtype、算子、DMA capability 位图 |
| `0x010` | `CAPABILITY1` | R | MAC lane、片上 bank 和实现参数 |
| `0x014` | `CONTROL` | R/W | bit0 soft_reset；bit1 irq_global_enable |
| `0x018` | `STATUS` | R | idle、busy、resetting、done、error |
| `0x01c` | `IRQ_STATUS` | R/W1C | done、error、watchdog |
| `0x020` | `IRQ_ENABLE` | R/W | 对应中断使能 |
| `0x024` | `TASK_BASE_LO` | R/W | task header 物理地址低 32 bit |
| `0x028` | `TASK_BASE_HI` | R/W | task header 物理地址高 32 bit |
| `0x02c` | `TASK_BYTES` | R/W | task 可访问区总长度 |
| `0x030` | `TASK_TAG` | R/W | 软件任务 tag |
| `0x034` | `DOORBELL` | W | 写 1 提交；busy 时不提交并锁存 `0xf001` |
| `0x038` | `COMPLETED_TAG` | R | 最近完成或失败的任务 tag |
| `0x03c` | `ERROR_CODE` | R | 首个 fatal error |
| `0x040` | `ERROR_PC` | R | 出错命令 byte offset/PC |
| `0x044` | `ERROR_INST_TAG` | R | 出错命令中的 tag |
| `0x048` | `WATCHDOG_LIMIT` | R/W | 最大任务周期；0 表示仅调试时禁用 |
| `0x04c` | `COMMANDS_RETIRED` | R | 最近任务完成命令数 |
| `0x050/54` | `CYCLES_TOTAL_LO/HI` | R | 最近任务总周期 |
| `0x058/5c` | `CYCLES_COMPUTE_LO/HI` | R | compute busy 周期 |
| `0x060/64` | `CYCLES_RD_WAIT_LO/HI` | R | DDR read stall 周期 |
| `0x068/6c` | `CYCLES_WR_WAIT_LO/HI` | R | DDR write stall 周期 |
| `0x070/74` | `CYCLES_BANK_STALL_LO/HI` | R | scratchpad bank stall 周期 |
| `0x078/7c` | `BYTES_READ_LO/HI` | R | 最近任务 DDR 读字节数 |
| `0x080/84` | `BYTES_WRITTEN_LO/HI` | R | 最近任务 DDR 写字节数 |
| `0x088/8c` | `READ_HIGH_WATER_LO/HI` | R | 最近任务读 burst 最大 beat 数 |
| `0x090` | `ERROR_COUNT` | R | 复位以来失败任务计数 |
| `0x094/98` | `WRITE_HIGH_WATER_LO/HI` | R | 最近任务写 burst 最大 beat 数 |

## 2. 提交约束

- 驱动只允许在 `STATUS.idle=1` 时提交；v1 不提供硬件任务队列；
- `TASK_BASE` 必须 64-byte 对齐，`TASK_BYTES` 必须覆盖 header 声明的全部区段；
- 驱动提交前完成 CPU cache clean，读取输出前完成 invalidate；具体 API 取决于 Linux/bare-metal；
- `DOORBELL` 是提交的唯一生效点，写其它字段不启动任务；
- 成功时更新 `COMPLETED_TAG`；失败上下文使用 `TASK_TAG`、`ERROR_CODE/PC/INST_TAG`；
- `ERROR_CODE/PC/INST_TAG` 在错误时锁存，下一次合法提交或 idle soft reset 清除；
- `IRQ_STATUS[2:0]` 分别为 watchdog/error/done，写 1 清对应 sticky bit；
- 64-bit 只读计数器是直接采样、没有硬件 snapshot。软件使用 `HI-LO-HI` 顺序读取，
  两次 HI 不同时重读，或只在 `STATUS.busy=0` 后读取。

### 首批公共错误码

| Code | 名称 | 来源 |
|---:|---|---|
| `0x0000` | `NONE` | 无错误 |
| `0x0001` | `ILLEGAL_OPCODE` | Command decoder |
| `0x0002` | `ILLEGAL_FLAGS` | Command decoder |
| `0x0003` | `ILLEGAL_FIELDS` | Command/descriptor 前置校验 |
| `0x0004` | `EXECUTION_ERROR` | 执行端未给出更细错误时的兜底 |
| `0x0005` | `WATCHDOG` | 任务超过 `WATCHDOG_LIMIT` |
| `0xf001` | `SUBMISSION_BUSY` | busy 时写 `DOORBELL=1`，任务保持运行 |
| `0xf002` | `RESET_BUSY` | busy 时写 `CONTROL.soft_reset=1`，复位不执行 |

编码由 `scripts/npu_isa.py` 生成到 RTL/C 头文件；后续 DMA、descriptor、bank 等错误只追加新值，不复用已有值。

## 3. soft reset 语义

`CONTROL.soft_reset` 只在 `STATUS.idle=1` 时接受，并产生一个周期的内部 reset pulse，
清除 event、FIFO、任务错误和中断状态。busy 时请求被拒绝并锁存 `0xf002`，因此不会
强行截断已握手 AXI transaction。若 AXI 永久无响应，外部 PS/PL reset 是最终恢复
手段。当前 `STATUS.resetting` 位固定为 0，因为 idle reset 在单周期内完成。

## 4. 与历史寄存器表的关系

`02_register_map.md` 中的 `WEIGHT_BASE`、`INPUT_BASE`、`BUTTON_GAIN` 和 `LF_KILL_BANDS` 属于固定人声消除加速器。通用 NPU 将权重、输入、输出都放入 task/descriptor，音频按键与低频保护移到 NPU 外部，因此旧 offset 不承诺兼容。
