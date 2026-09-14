`timescale 1ns/1ps

module tb_npu_isa_pkg;
  import npu_isa_pkg::*;

  logic [127:0] raw_command;
  logic [511:0] raw_operator;
  npu_command_t command;
  npu_operator_desc_t operator_desc;

  assign raw_command = 128'h1234_0005_0004_0003_0002_0001_1234_0d20;
  assign command = npu_command_t'(raw_command);
  assign operator_desc = npu_operator_desc_t'(raw_operator);

  initial begin
    raw_operator = '0;
    raw_operator[15:0] = 16'h0001;
    raw_operator[31:16] = 16'h0002;
    raw_operator[255:240] = NPU_POST_TANH_LUT;
    raw_operator[287:256] = 32'h0000_0010;
    raw_operator[511:480] = 32'h0000_0080;
    if ($bits(npu_command_t) != 128) $fatal(1, "command width");
    if ($bits(npu_tensor_desc_t) != 512) $fatal(1, "tensor descriptor width");
    if ($bits(npu_operator_desc_t) != 512) $fatal(1, "operator descriptor width");
    if ($bits(npu_quant_desc_t) != 256) $fatal(1, "quant descriptor width");
    if ($bits(npu_quant_param_t) != 128) $fatal(1, "quant parameter width");
    if ($bits(npu_segment_desc_t) != 64) $fatal(1, "segment descriptor width");
    #1;
    if (command.opcode != NPU_OP_CONV2D) $fatal(1, "opcode position");
    if (command.flags != 8'h0d) $fatal(1, "flags position");
    if (command.tag != 16'h1234) $fatal(1, "tag position");
    if (command.dst_td != 16'h0001) $fatal(1, "dst position");
    if (command.src0_td != 16'h0002) $fatal(1, "src0 position");
    if (command.src1_td != 16'h0003) $fatal(1, "src1 position");
    if (command.op_desc != 16'h0004) $fatal(1, "operator position");
    if (command.quant_desc != 16'h0005) $fatal(1, "quant position");
    if (command.imm != 16'h1234) $fatal(1, "immediate position");
    if (operator_desc.src_td != 16'h0001) $fatal(1, "operator src position");
    if (operator_desc.dst_td != 16'h0002) $fatal(1, "operator dst position");
    if (operator_desc.post_op_id != NPU_POST_TANH_LUT)
      $fatal(1, "operator post-op position");
    if (operator_desc.tile_origin_h != 32'h0000_0010)
      $fatal(1, "operator tile origin position");
    if (operator_desc.input_channel_count != 32'h0000_0080)
      $fatal(1, "operator input count position");
    $display("npu_isa_pkg RTL layout: PASS");
    $finish;
  end
endmodule
