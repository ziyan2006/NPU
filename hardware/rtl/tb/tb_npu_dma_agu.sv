`timescale 1ns/1ps

module tb_npu_dma_agu;
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  logic command_valid_i = 1'b0;
  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic command_ready_o;
  wire [127:0] command_bits_i;
  wire [511:0] src_tensor_desc_bits_i;
  wire [511:0] dst_tensor_desc_bits_i;
  wire [511:0] operator_desc_bits_i;
  wire [255:0] quant_desc_bits_i;
  wire [63:0] segment_desc_bits_i;
  logic [63:0] activation_base_i = 64'h0000_0000_1000_0000;
  logic [63:0] weight_base_i = 64'h0000_0000_2000_0000;
  logic [63:0] bias_base_i = 64'h0000_0000_3000_0000;
  logic [63:0] quant_param_base_i = 64'h0000_0000_4000_0000;
  logic request_valid_o;
  logic request_ready_i = 1'b0;
  logic [NPU_DMA_REQUEST_BITS-1:0] request_bits_o;
  logic error_valid_o;
  logic [3:0] error_reason_o;

  npu_command_t command;
  npu_tensor_desc_t src_tensor;
  npu_tensor_desc_t dst_tensor;
  npu_operator_desc_t operator_desc;
  npu_quant_desc_t quant_desc;
  npu_segment_desc_t segment_desc;
  npu_dma_request_t request;

  assign command_bits_i = command;
  assign src_tensor_desc_bits_i = src_tensor;
  assign dst_tensor_desc_bits_i = dst_tensor;
  assign operator_desc_bits_i = operator_desc;
  assign quant_desc_bits_i = quant_desc;
  assign segment_desc_bits_i = segment_desc;
  assign request = npu_dma_request_t'(request_bits_o);

  always #5 clk_i = ~clk_i;

  npu_dma_agu dut (.*);

  task automatic run_agu;
    begin
      if (request_valid_o) begin
        request_ready_i = 1'b1;
        @(posedge clk_i);
        #1;
        request_ready_i = 1'b0;
      end else if (error_valid_o) begin
        @(posedge clk_i);
        #1;
      end
      wait (command_ready_o);
      @(negedge clk_i);
      command_valid_i = 1'b1;
      @(posedge clk_i);
      #1;
      command_valid_i = 1'b0;
      wait (request_valid_o || error_valid_o);
      #1;
    end
  endtask

  task automatic clear_inputs;
    begin
      command_valid_i = 1'b0;
      command = '0;
      command.dst_td = NPU_NONE_INDEX;
      command.src0_td = NPU_NONE_INDEX;
      command.src1_td = NPU_NONE_INDEX;
      command.op_desc = NPU_NONE_INDEX;
      command.quant_desc = NPU_NONE_INDEX;
      src_tensor = '0;
      dst_tensor = '0;
      operator_desc = '0;
      quant_desc = '0;
      segment_desc = '0;
      #1;
    end
  endtask

  task automatic common_operator;
    begin
      operator_desc.src_td = 16'd0;
      operator_desc.dst_td = 16'd3;
      operator_desc.weight_td = 16'd1;
      operator_desc.bias_td = 16'd2;
      operator_desc.kh = 16'd3;
      operator_desc.kw = 16'd3;
      operator_desc.stride_h = 16'd1;
      operator_desc.stride_w = 16'd1;
      operator_desc.dilation_h = 16'd1;
      operator_desc.dilation_w = 16'd1;
      operator_desc.pad_top = 16'd1;
      operator_desc.pad_left = 16'd2;
      operator_desc.tile_h = 32'd16;
      operator_desc.tile_w = 32'd16;
      operator_desc.tile_cout = 32'd8;
      operator_desc.input_channel_count = 32'd2;
    end
  endtask

  initial begin
    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;

    // enc0 activation load: clear the padded 18x18x8 local tile, then copy
    // only the two logical source channels from the valid 17x16 region.
    clear_inputs();
    common_operator();
    src_tensor.allocation_bytes = 32'd32768;
    src_tensor.shape_n = 1;
    src_tensor.shape_h = 128;
    src_tensor.shape_w = 16;
    src_tensor.shape_c = 2;
    src_tensor.stride_h = 256;
    src_tensor.stride_w = 16;
    src_tensor.stride_c = 2;
    src_tensor.dtype = NPU_DTYPE_INT12_IN_INT16;
    src_tensor.layout = NPU_LAYOUT_NHWC8;
    dst_tensor = src_tensor;
    command.opcode = NPU_OP_DMA_LOAD;
    command.flags = NPU_FLAG_ASYNC;
    command.tag = 16'h11;
    command.dst_td = 0;
    command.src0_td = 0;
    command.op_desc = 0;
    command.imm = NPU_EVENT_A0_READY;
    run_agu();
    if (!request_valid_o || error_valid_o
        || request.external_address != activation_base_i
        || request.scratchpad != NPU_SPAD_A || request.bank != 0
        || !request.clear_before || request.clear_bytes != 32'd5184
        || request.scratchpad_offset != 32'd320
        || request.x_bytes != 32'd4
        || request.y_count != 16'd16 || request.z_count != 16'd17
        || request.external_y_stride != 32'd16
        || request.external_z_stride != 32'd256
        || request.scratchpad_y_stride != 32'd16
        || request.scratchpad_z_stride != 32'd288)
      $fatal(1, "enc0 activation AGU");

    // A second output-channel block selects the second O8I8 weight block.
    clear_inputs();
    common_operator();
    operator_desc.weight_td = 4;
    operator_desc.input_channel_count = 32;
    operator_desc.tile_origin_cout = 8;
    src_tensor.base_offset = 2304;
    src_tensor.allocation_bytes = 18432;
    src_tensor.dtype = NPU_DTYPE_INT8;
    src_tensor.layout = NPU_LAYOUT_WEIGHT_O8I8;
    command.opcode = NPU_OP_DMA_LOAD;
    command.flags = NPU_FLAG_ASYNC;
    command.dst_td = 4;
    command.src0_td = 4;
    command.op_desc = 1;
    command.imm = 16'h0200;
    run_agu();
    if (!request_valid_o
        || request.memory_space != NPU_DMA_SPACE_WEIGHT
        || request.scratchpad != NPU_SPAD_W || request.bank != 1
        || request.external_address != weight_base_i + 64'd4608
        || request.scratchpad_offset != 0 || request.x_bytes != 2304)
      $fatal(1, "weight AGU");

    // Bias and convolution quant occupy deterministic 64-byte-aligned slots.
    src_tensor.base_offset = 128;
    src_tensor.allocation_bytes = 256;
    src_tensor.dtype = NPU_DTYPE_INT32;
    src_tensor.layout = NPU_LAYOUT_LINEAR;
    operator_desc.bias_td = 5;
    command.dst_td = 5;
    command.src0_td = 5;
    run_agu();
    if (!request_valid_o
        || request.external_address != bias_base_i + 64'd160
        || request.scratchpad_offset != 2304 || request.x_bytes != 32)
      $fatal(1, "bias AGU");

    command.dst_td = NPU_NONE_INDEX;
    command.src0_td = NPU_NONE_INDEX;
    command.quant_desc = 7;
    quant_desc.param_offset = 512;
    quant_desc.param_count = 64;
    run_agu();
    if (!request_valid_o
        || request.external_address != quant_param_base_i + 64'd640
        || request.scratchpad_offset != 2368 || request.x_bytes != 128)
      $fatal(1, "conv quant AGU");

    command.imm = 16'h0600;
    quant_desc.param_offset = 2304;
    quant_desc.param_count = 1;
    run_agu();
    if (!request_valid_o
        || request.external_address != quant_param_base_i + 64'd2304
        || request.scratchpad_offset != 2496 || request.x_bytes != 16)
      $fatal(1, "vector quant AGU");

    // dec3 segment 1 is interleaved into logical channels [128, 224).
    clear_inputs();
    operator_desc.src_td = 25;
    operator_desc.dst_td = 28;
    operator_desc.kh = 3;
    operator_desc.kw = 3;
    operator_desc.stride_h = 1;
    operator_desc.stride_w = 1;
    operator_desc.dilation_h = 1;
    operator_desc.dilation_w = 1;
    operator_desc.pad_top = 1;
    operator_desc.pad_left = 2;
    operator_desc.tile_h = 16;
    operator_desc.tile_w = 4;
    operator_desc.tile_cout = 8;
    operator_desc.input_channel_count = 224;
    src_tensor.base_offset = 229376;
    src_tensor.allocation_bytes = 24576;
    src_tensor.shape_h = 32;
    src_tensor.shape_w = 4;
    src_tensor.shape_c = 96;
    src_tensor.stride_h = 768;
    src_tensor.stride_w = 192;
    src_tensor.stride_c = 2;
    src_tensor.dtype = NPU_DTYPE_INT12_IN_INT16;
    src_tensor.layout = NPU_LAYOUT_NHWC8;
    dst_tensor.shape_h = 32;
    dst_tensor.shape_w = 4;
    dst_tensor.shape_c = 224;
    dst_tensor.segment_count = 2;
    dst_tensor.dtype = NPU_DTYPE_INT12_IN_INT16;
    dst_tensor.layout = NPU_LAYOUT_SEGMENTED;
    segment_desc.tensor_index = 9;
    segment_desc.dst_channel = 128;
    segment_desc.channel_count = 96;
    command.opcode = NPU_OP_DMA_LOAD;
    command.flags = NPU_FLAG_ASYNC;
    command.dst_td = 25;
    command.src0_td = 9;
    command.op_desc = 160;
    command.imm = 16'h5100;
    run_agu();
    if (!request_valid_o || request.clear_before
        || request.external_address != activation_base_i + 64'd229376
        || request.scratchpad_offset != 3840
        || request.x_bytes != 192
        || request.y_count != 4 || request.z_count != 17
        || request.external_y_stride != 192
        || request.external_z_stride != 768
        || request.scratchpad_y_stride != 448
        || request.scratchpad_z_stride != 2688
        || request.clear_bytes != 48384)
      $fatal(1, "segmented activation AGU");

    // Final four-channel output stores only valid lanes with strided pixels.
    clear_inputs();
    operator_desc.dst_td = 41;
    operator_desc.kh = 1;
    operator_desc.kw = 1;
    operator_desc.stride_h = 1;
    operator_desc.stride_w = 1;
    operator_desc.dilation_h = 1;
    operator_desc.dilation_w = 1;
    operator_desc.tile_origin_h = 112;
    operator_desc.tile_h = 16;
    operator_desc.tile_w = 16;
    operator_desc.tile_cout = 4;
    src_tensor.base_offset = 917504;
    src_tensor.allocation_bytes = 32768;
    src_tensor.shape_h = 128;
    src_tensor.shape_w = 16;
    src_tensor.shape_c = 4;
    src_tensor.stride_h = 256;
    src_tensor.stride_w = 16;
    src_tensor.stride_c = 2;
    src_tensor.dtype = NPU_DTYPE_INT12_IN_INT16;
    src_tensor.layout = NPU_LAYOUT_NHWC8;
    command.opcode = NPU_OP_DMA_STORE;
    command.flags = NPU_FLAG_ASYNC;
    command.dst_td = 41;
    command.src0_td = 41;
    command.op_desc = 247;
    command.imm = 16'h0800;
    run_agu();
    if (!request_valid_o || !request.store
        || request.scratchpad != NPU_SPAD_O || request.bank != 1
        || request.external_address
           != activation_base_i + 64'd917504 + 64'd28672
        || request.x_bytes != 8 || request.y_count != 16
        || request.z_count != 16
        || request.external_y_stride != 16
        || request.external_z_stride != 256
        || request.scratchpad_y_stride != 16
        || request.scratchpad_z_stride != 256)
      $fatal(1, "tail-channel store AGU");

    // Corrupt allocation metadata must stop the request before execution.
    src_tensor.allocation_bytes = 16;
    run_agu();
    if (request_valid_o || !error_valid_o
        || error_reason_o != NPU_DMA_AGU_EXTERNAL_BOUNDS)
      $fatal(1, "external bounds rejection");

    $display("npu_dma_agu: PASS");
    $finish;
  end
endmodule
