# RTL ISA 交付入口

状态：`P3 draft`，可用于 RTL 原型和联调，不代表 ISA 已冻结。

硬件侧首先引用 `include/npu_isa_pkg.sv`。该 package 由 `scripts/29_generate_npu_isa_headers.py` 从 `scripts/npu_isa.py` 生成，包含：

- opcode、flag、dtype、layout、post-op 和 event 编码；
- 128-bit command packed struct；
- Tensor、Operator、Quant、Segment 描述符 packed struct；
- tile immediate 的 event 和 bank 位定义。

译码器的基本用法：

```systemverilog
import npu_isa_pkg::*;

npu_command_t command;
assign command = npu_command_t'(command_bits);

always_comb begin
  case (npu_opcode_e'(command.opcode))
    NPU_OP_DMA_LOAD:  /* issue read DMA */;
    NPU_OP_CONV2D:    /* issue Tensor MAC */;
    NPU_OP_VEC_ADD:   /* issue residual add */;
    NPU_OP_DMA_STORE: /* issue write DMA */;
    default:          /* reject unsupported opcode */;
  endcase
end
```

命令存储为 little-endian，任务镜像 byte 0 是 `opcode`。Command fetch 把 16 个连续 byte 组装为 `command_bits[127:0]` 时，必须保证 byte 0 对应 bits `[7:0]`；若 AXI 数据拼接方向不同，应在 fetch 边界处理，不能在各执行单元分别交换。

算子到指令序列、字段约束和完成条件见 `../spec/22_operator_instruction_contract.md`。PS/驱动侧使用 `software/include/npu_isa.h`，不能复制一套独立常量。

重新生成并检查：

```powershell
python scripts/29_generate_npu_isa_headers.py
python scripts/29_generate_npu_isa_headers.py --check
python scripts/_test_npu_isa_headers.py
```

修改编码时只编辑 `scripts/npu_isa.py` 和生成器；直接编辑生成的 `.sv`/`.h` 会被 `--check` 回归拒绝。
