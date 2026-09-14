`timescale 1ns/1ps

module npu_dma_agu #(
  parameter logic [31:0] ACTIVATION_BANK_BYTES = 32'd65536,
  parameter logic [31:0] WEIGHT_BANK_BYTES     = 32'd32768,
  parameter logic [31:0] OUTPUT_BANK_BYTES     = 32'd16384
) (
  input  logic         command_valid_i,
  input  logic [127:0] command_bits_i,
  input  logic [511:0] src_tensor_desc_bits_i,
  input  logic [511:0] dst_tensor_desc_bits_i,
  input  logic [511:0] operator_desc_bits_i,
  input  logic [255:0] quant_desc_bits_i,
  input  logic [63:0]  segment_desc_bits_i,

  input  logic [63:0]  activation_base_i,
  input  logic [63:0]  weight_base_i,
  input  logic [63:0]  bias_base_i,
  input  logic [63:0]  quant_param_base_i,

  output logic         request_valid_o,
  output logic [npu_dma_pkg::NPU_DMA_REQUEST_BITS-1:0] request_bits_o,
  output logic         error_valid_o,
  output logic [3:0]   error_reason_o
);
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  npu_command_t command;
  npu_tensor_desc_t src_tensor;
  npu_tensor_desc_t dst_tensor;
  npu_operator_desc_t operator_desc;
  npu_quant_desc_t quant_desc;
  npu_segment_desc_t segment_desc;
  npu_dma_request_t request;

  logic is_load;
  logic is_store;
  logic is_quant;
  logic is_weight;
  logic is_bias;
  logic is_activation;
  logic is_segmented;
  logic is_vector_quant;

  logic [31:0] local_h;
  logic [31:0] local_w;
  logic [31:0] local_c_bytes;
  logic [31:0] physical_tile_cout;
  logic [31:0] weight_tile_bytes;
  logic [31:0] bias_tile_bytes;
  logic [31:0] conv_quant_bytes;
  logic [31:0] bias_bank_offset;
  logic [31:0] conv_quant_bank_offset;
  logic [31:0] vector_quant_bank_offset;
  logic [31:0] scratchpad_capacity;

  logic signed [63:0] raw_h0;
  logic signed [63:0] raw_w0;
  logic signed [63:0] raw_h1;
  logic signed [63:0] raw_w1;
  logic [63:0] valid_h0;
  logic [63:0] valid_w0;
  logic [63:0] valid_h1;
  logic [63:0] valid_w1;
  logic [63:0] channel_start;
  logic [63:0] channel_count;
  logic [63:0] destination_channel;
  logic [63:0] external_relative_offset;
  logic [63:0] external_relative_end;
  logic [63:0] scratchpad_end;
  logic [63:0] external_section_base;
  logic [63:0] external_address;
  logic address_overflow;
  logic descriptor_valid;

  assign command = npu_command_t'(command_bits_i);
  assign src_tensor = npu_tensor_desc_t'(src_tensor_desc_bits_i);
  assign dst_tensor = npu_tensor_desc_t'(dst_tensor_desc_bits_i);
  assign operator_desc = npu_operator_desc_t'(operator_desc_bits_i);
  assign quant_desc = npu_quant_desc_t'(quant_desc_bits_i);
  assign segment_desc = npu_segment_desc_t'(segment_desc_bits_i);
  assign request_bits_o = request;

  always @* begin
    request = '0;
    request.tag = command.tag;
    request.event_mask = command.imm[7:0];
    error_reason_o = NPU_DMA_AGU_OK;

    is_load = command.opcode == NPU_OP_DMA_LOAD;
    is_store = command.opcode == NPU_OP_DMA_STORE;
    is_quant = is_load && command.src0_td == NPU_NONE_INDEX
      && command.dst_td == NPU_NONE_INDEX
      && command.quant_desc != NPU_NONE_INDEX;
    is_weight = is_load && !is_quant
      && src_tensor.layout == NPU_LAYOUT_WEIGHT_O8I8;
    is_bias = is_load && !is_quant && !is_weight
      && src_tensor.layout == NPU_LAYOUT_LINEAR
      && src_tensor.dtype == NPU_DTYPE_INT32;
    is_activation = (is_load && !is_quant && !is_weight && !is_bias)
      || is_store;
    is_segmented = is_load && command.imm[NPU_IMM_SEGMENTED_BIT];
    is_vector_quant = is_quant
      && command.imm[NPU_IMM_DMA_VECTOR_QUANT_BIT];

    request.store = is_store;
    request.memory_space = NPU_DMA_SPACE_ACTIVATION;
    request.scratchpad = NPU_SPAD_A;
    request.bank = command.imm[NPU_IMM_ACTIVATION_BANK_BIT];
    request.x_bytes = 32'd0;
    request.y_count = 16'd1;
    request.z_count = 16'd1;

    local_h = 32'd0;
    local_w = 32'd0;
    local_c_bytes = 32'd0;
    physical_tile_cout = npu_align8(operator_desc.tile_cout);
    weight_tile_bytes = 32'd0;
    bias_tile_bytes = operator_desc.tile_cout * 32'd4;
    conv_quant_bytes = operator_desc.tile_cout * NPU_QUANT_PARAM_BYTES;
    bias_bank_offset = 32'd0;
    conv_quant_bank_offset = 32'd0;
    vector_quant_bank_offset = 32'd0;
    scratchpad_capacity = ACTIVATION_BANK_BYTES;

    raw_h0 = 64'sd0;
    raw_w0 = 64'sd0;
    raw_h1 = 64'sd0;
    raw_w1 = 64'sd0;
    valid_h0 = 64'd0;
    valid_w0 = 64'd0;
    valid_h1 = 64'd0;
    valid_w1 = 64'd0;
    channel_start = 64'd0;
    channel_count = 64'd0;
    destination_channel = 64'd0;
    external_relative_offset = 64'd0;
    external_relative_end = 64'd0;
    scratchpad_end = 64'd0;
    external_section_base = activation_base_i;
    external_address = 64'd0;
    address_overflow = 1'b0;
    descriptor_valid = 1'b1;

    if (!is_load && !is_store) begin
      error_reason_o = NPU_DMA_AGU_BAD_COMMAND;
    end else if (command.op_desc == NPU_NONE_INDEX
                 || operator_desc.tile_h == 0
                 || operator_desc.tile_w == 0
                 || operator_desc.tile_cout == 0
                 || operator_desc.kh == 0 || operator_desc.kw == 0
                 || operator_desc.stride_h == 0
                 || operator_desc.stride_w == 0
                 || operator_desc.dilation_h == 0
                 || operator_desc.dilation_w == 0) begin
      error_reason_o = NPU_DMA_AGU_BAD_DESCRIPTOR;
    end else begin
      weight_tile_bytes = (physical_tile_cout
        * npu_align8(operator_desc.input_channel_count)
        * operator_desc.kh * operator_desc.kw);
      bias_bank_offset = npu_align64(weight_tile_bytes);
      conv_quant_bank_offset = npu_align64(
        bias_bank_offset + bias_tile_bytes);
      vector_quant_bank_offset = npu_align64(
        conv_quant_bank_offset + conv_quant_bytes);

      if (is_quant) begin
        request.memory_space = NPU_DMA_SPACE_QUANT;
        request.scratchpad = NPU_SPAD_W;
        request.bank = command.imm[NPU_IMM_WEIGHT_BANK_BIT];
        scratchpad_capacity = WEIGHT_BANK_BYTES;
        external_section_base = quant_param_base_i;
        if (is_vector_quant) begin
          request.scratchpad_offset = vector_quant_bank_offset;
          request.x_bytes = NPU_QUANT_PARAM_BYTES;
          external_relative_offset = quant_desc.param_offset;
          descriptor_valid = quant_desc.param_count >= 1;
        end else begin
          request.scratchpad_offset = conv_quant_bank_offset;
          request.x_bytes = conv_quant_bytes;
          external_relative_offset = quant_desc.param_offset
            + operator_desc.tile_origin_cout * NPU_QUANT_PARAM_BYTES;
          descriptor_valid = operator_desc.tile_origin_cout
            + operator_desc.tile_cout <= quant_desc.param_count;
        end
        external_relative_end = external_relative_offset + request.x_bytes;
      end else if (is_weight) begin
        request.memory_space = NPU_DMA_SPACE_WEIGHT;
        request.scratchpad = NPU_SPAD_W;
        request.bank = command.imm[NPU_IMM_WEIGHT_BANK_BIT];
        scratchpad_capacity = WEIGHT_BANK_BYTES;
        external_section_base = weight_base_i;
        request.scratchpad_offset = 32'd0;
        request.x_bytes = weight_tile_bytes;
        external_relative_offset = src_tensor.base_offset
          + (operator_desc.tile_origin_cout >> 3)
          * npu_align8(operator_desc.input_channel_count)
          * operator_desc.kh * operator_desc.kw * 8;
        external_relative_end = external_relative_offset + request.x_bytes;
        descriptor_valid = command.src0_td == operator_desc.weight_td
          && command.dst_td == operator_desc.weight_td
          && operator_desc.tile_origin_cout[2:0] == 3'b000
          && src_tensor.layout == NPU_LAYOUT_WEIGHT_O8I8;
        if (external_relative_end
            > src_tensor.base_offset + src_tensor.allocation_bytes)
          error_reason_o = NPU_DMA_AGU_EXTERNAL_BOUNDS;
      end else if (is_bias) begin
        request.memory_space = NPU_DMA_SPACE_BIAS;
        request.scratchpad = NPU_SPAD_W;
        request.bank = command.imm[NPU_IMM_WEIGHT_BANK_BIT];
        scratchpad_capacity = WEIGHT_BANK_BYTES;
        external_section_base = bias_base_i;
        request.scratchpad_offset = bias_bank_offset;
        request.x_bytes = bias_tile_bytes;
        external_relative_offset = src_tensor.base_offset
          + operator_desc.tile_origin_cout * 4;
        external_relative_end = external_relative_offset + request.x_bytes;
        descriptor_valid = command.src0_td == operator_desc.bias_td
          && command.dst_td == operator_desc.bias_td;
        if (external_relative_end
            > src_tensor.base_offset + src_tensor.allocation_bytes)
          error_reason_o = NPU_DMA_AGU_EXTERNAL_BOUNDS;
      end else if (is_store) begin
        request.memory_space = NPU_DMA_SPACE_ACTIVATION;
        request.scratchpad = NPU_SPAD_O;
        request.bank = command.imm[NPU_IMM_OUTPUT_BANK_BIT];
        scratchpad_capacity = OUTPUT_BANK_BYTES;
        external_section_base = activation_base_i;
        request.x_bytes = operator_desc.tile_cout * src_tensor.stride_c;
        request.y_count = operator_desc.tile_w[15:0];
        request.z_count = operator_desc.tile_h[15:0];
        request.external_y_stride = src_tensor.stride_w;
        request.external_z_stride = src_tensor.stride_h;
        request.scratchpad_y_stride = physical_tile_cout * src_tensor.stride_c;
        request.scratchpad_z_stride = operator_desc.tile_w
          * physical_tile_cout * src_tensor.stride_c;
        external_relative_offset = src_tensor.base_offset
          + operator_desc.tile_origin_h * src_tensor.stride_h
          + operator_desc.tile_origin_w * src_tensor.stride_w
          + operator_desc.tile_origin_cout * src_tensor.stride_c;
        external_relative_end = external_relative_offset
          + (request.z_count - 1) * request.external_z_stride
          + (request.y_count - 1) * request.external_y_stride
          + request.x_bytes;
        // Residual layers store their post-vector tensor rather than the
        // convolution's intermediate dst_td, so only command self-consistency
        // and geometry are checked here.
        descriptor_valid = command.src0_td == command.dst_td
          && src_tensor.layout == NPU_LAYOUT_NHWC8
          && src_tensor.dtype == NPU_DTYPE_INT12_IN_INT16
          && operator_desc.tile_origin_h + operator_desc.tile_h
             <= src_tensor.shape_h
          && operator_desc.tile_origin_w + operator_desc.tile_w
             <= src_tensor.shape_w
          && operator_desc.tile_origin_cout + operator_desc.tile_cout
             <= src_tensor.shape_c;
        if (external_relative_end
            > src_tensor.base_offset + src_tensor.allocation_bytes)
          error_reason_o = NPU_DMA_AGU_EXTERNAL_BOUNDS;
      end else begin
        // Activation load.  The local bank stores the full receptive field,
        // including explicit zero padding and padded NHWC8 channels.
        local_h = ((operator_desc.tile_h - 1) * operator_desc.stride_h
          + operator_desc.dilation_h * (operator_desc.kh - 1) + 1);
        local_w = ((operator_desc.tile_w - 1) * operator_desc.stride_w
          + operator_desc.dilation_w * (operator_desc.kw - 1) + 1);
        local_c_bytes = npu_align8(operator_desc.input_channel_count)
          * src_tensor.stride_c;
        raw_h0 = $signed({32'd0, operator_desc.tile_origin_h})
          * $signed({48'd0, operator_desc.stride_h})
          - $signed({48'd0, operator_desc.pad_top});
        raw_w0 = $signed({32'd0, operator_desc.tile_origin_w})
          * $signed({48'd0, operator_desc.stride_w})
          - $signed({48'd0, operator_desc.pad_left});
        raw_h1 = raw_h0 + $signed({32'd0, local_h});
        raw_w1 = raw_w0 + $signed({32'd0, local_w});
        valid_h0 = raw_h0 < 0 ? 64'd0 : raw_h0;
        valid_w0 = raw_w0 < 0 ? 64'd0 : raw_w0;
        valid_h1 = raw_h1 > $signed({48'd0, src_tensor.shape_h})
          ? src_tensor.shape_h : raw_h1;
        valid_w1 = raw_w1 > $signed({48'd0, src_tensor.shape_w})
          ? src_tensor.shape_w : raw_w1;

        channel_start = is_segmented ? 64'd0
          : operator_desc.input_channel_start;
        channel_count = is_segmented ? segment_desc.channel_count
          : operator_desc.input_channel_count;
        destination_channel = is_segmented ? segment_desc.dst_channel : 64'd0;
        request.clear_before = !is_segmented
          || command.imm[NPU_IMM_SEGMENT_LSB +: NPU_IMM_SEGMENT_WIDTH] == 0;
        request.clear_bytes = local_h * local_w * local_c_bytes;
        request.x_bytes = channel_count * src_tensor.stride_c;
        request.y_count = valid_w1 - valid_w0;
        request.z_count = valid_h1 - valid_h0;
        request.external_y_stride = src_tensor.stride_w;
        request.external_z_stride = src_tensor.stride_h;
        request.scratchpad_y_stride = local_c_bytes;
        request.scratchpad_z_stride = local_w * local_c_bytes;
        request.scratchpad_offset = ((raw_h0 < 0 ? -raw_h0 : 0) * local_w
          + (raw_w0 < 0 ? -raw_w0 : 0)) * local_c_bytes
          + destination_channel * src_tensor.stride_c;
        external_relative_offset = src_tensor.base_offset
          + valid_h0 * src_tensor.stride_h
          + valid_w0 * src_tensor.stride_w
          + channel_start * src_tensor.stride_c;
        external_relative_end = external_relative_offset
          + (request.z_count - 1) * request.external_z_stride
          + (request.y_count - 1) * request.external_y_stride
          + request.x_bytes;
        descriptor_valid = command.dst_td == operator_desc.src_td
          && src_tensor.layout == NPU_LAYOUT_NHWC8
          && src_tensor.dtype == NPU_DTYPE_INT12_IN_INT16
          && dst_tensor.shape_c == operator_desc.input_channel_count
          && request.y_count != 0 && request.z_count != 0;
        if (is_segmented) begin
          descriptor_valid = descriptor_valid
            && dst_tensor.layout == NPU_LAYOUT_SEGMENTED
            && segment_desc.tensor_index == command.src0_td
            && destination_channel + channel_count <= dst_tensor.shape_c
            && command.imm[NPU_IMM_SEGMENT_LSB +: NPU_IMM_SEGMENT_WIDTH]
               < dst_tensor.segment_count;
        end else begin
          descriptor_valid = descriptor_valid
            && command.src0_td == command.dst_td
            && command.src0_td == operator_desc.src_td
            && dst_tensor.layout == NPU_LAYOUT_NHWC8
            && channel_start + channel_count <= src_tensor.shape_c;
        end
        if (external_relative_end
            > src_tensor.base_offset + src_tensor.allocation_bytes)
          error_reason_o = NPU_DMA_AGU_EXTERNAL_BOUNDS;
      end

      external_address = external_section_base + external_relative_offset;
      address_overflow = external_address < external_section_base;
      request.external_address = external_address;
      scratchpad_end = request.scratchpad_offset
        + (request.z_count - 1) * request.scratchpad_z_stride
        + (request.y_count - 1) * request.scratchpad_y_stride
        + request.x_bytes;

      if (!descriptor_valid && error_reason_o == NPU_DMA_AGU_OK)
        error_reason_o = NPU_DMA_AGU_BAD_DESCRIPTOR;
      else if (address_overflow && error_reason_o == NPU_DMA_AGU_OK)
        error_reason_o = NPU_DMA_AGU_ADDRESS_OVERFLOW;
      else if ((scratchpad_end > scratchpad_capacity
                || request.clear_bytes > scratchpad_capacity)
               && error_reason_o == NPU_DMA_AGU_OK)
        error_reason_o = NPU_DMA_AGU_SPAD_BOUNDS;
    end
  end

  assign request_valid_o = command_valid_i
    && error_reason_o == NPU_DMA_AGU_OK;
  assign error_valid_o = command_valid_i
    && error_reason_o != NPU_DMA_AGU_OK;

endmodule
