`timescale 1ns/1ps

module npu_dma_agu #(
  parameter logic [31:0] ACTIVATION_BANK_BYTES = 32'd65536,
  parameter logic [31:0] WEIGHT_BANK_BYTES     = 32'd32768,
  parameter logic [31:0] OUTPUT_BANK_BYTES     = 32'd16384
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         command_valid_i,
  output logic         command_ready_o,
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
  input  logic         request_ready_i,
  output logic [npu_dma_pkg::NPU_DMA_REQUEST_BITS-1:0] request_bits_o,
  output logic         error_valid_o,
  output logic [3:0]   error_reason_o
);
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  typedef enum logic [2:0] {
    AGU_KIND_ACTIVATION,
    AGU_KIND_STORE,
    AGU_KIND_WEIGHT,
    AGU_KIND_BIAS,
    AGU_KIND_QUANT_CONV,
    AGU_KIND_QUANT_VECTOR
  } agu_kind_e;

  typedef enum logic [5:0] {
    AGU_IDLE,
    AGU_WEIGHT_KERNEL_H,
    AGU_WEIGHT_KERNEL_W,
    AGU_WEIGHT_TILE,
    AGU_WEIGHT_ORIGIN,
    AGU_LAYOUT_BIAS,
    AGU_LAYOUT_BIAS_RETURN,
    AGU_LAYOUT_CONV,
    AGU_LAYOUT_CONV_RETURN,
    AGU_LAYOUT_VECTOR,
    AGU_LAYOUT_VECTOR_RETURN,
    AGU_STORE_X,
    AGU_STORE_SPAD_Y,
    AGU_STORE_SPAD_Z,
    AGU_STORE_EXT_H,
    AGU_STORE_EXT_W,
    AGU_STORE_EXT_C,
    AGU_STORE_END_Z,
    AGU_STORE_END_Y,
    AGU_STORE_SPAD_END_Z,
    AGU_STORE_SPAD_END_Y,
    AGU_ACT_LOCAL_H_STRIDE,
    AGU_ACT_LOCAL_H_DILATION,
    AGU_ACT_LOCAL_W_STRIDE,
    AGU_ACT_LOCAL_W_DILATION,
    AGU_ACT_LOCAL_C,
    AGU_ACT_RAW_H,
    AGU_ACT_RAW_W,
    AGU_ACT_CLAMP,
    AGU_ACT_GEOMETRY,
    AGU_ACT_CLEAR_PLANE,
    AGU_ACT_CLEAR,
    AGU_ACT_X,
    AGU_ACT_PAD_ROW,
    AGU_ACT_PAD_BYTES,
    AGU_ACT_DST_CHANNEL,
    AGU_ACT_SPAD_OFFSET,
    AGU_ACT_EXT_H,
    AGU_ACT_EXT_W,
    AGU_ACT_EXT_C,
    AGU_ACT_END_Z,
    AGU_ACT_END_Y,
    AGU_ACT_SPAD_END_Z,
    AGU_ACT_SPAD_END_Y,
    AGU_ADD_SPAD_OFFSET,
    AGU_ADD_SPAD_X,
    AGU_CHECK_EXTERNAL,
    AGU_FINAL,
    AGU_RETURN
  } agu_state_e;

  agu_state_e state_q;
  agu_kind_e kind_q;
  npu_command_t command_q;
  npu_tensor_desc_t src_tensor_q;
  npu_tensor_desc_t dst_tensor_q;
  npu_operator_desc_t operator_q;
  npu_quant_desc_t quant_q;
  npu_segment_desc_t segment_q;
  npu_dma_request_t request_q;

  npu_command_t command_in;
  npu_tensor_desc_t src_tensor_in;
  npu_tensor_desc_t dst_tensor_in;
  npu_operator_desc_t operator_in;
  npu_quant_desc_t quant_in;
  npu_segment_desc_t segment_in;

  logic [31:0] physical_cout_q;
  logic [31:0] aligned_input_c_q;
  logic [63:0] kernel_bytes_per_cout_q;
  logic [63:0] weight_tile_bytes_q;
  logic [63:0] local_h_q;
  logic [63:0] local_w_q;
  logic [63:0] local_c_bytes_q;
  logic [63:0] local_plane_bytes_q;
  logic signed [63:0] raw_h_q;
  logic signed [63:0] raw_w_q;
  logic [63:0] valid_h0_q;
  logic [63:0] valid_w0_q;
  logic [63:0] valid_h1_q;
  logic [63:0] valid_w1_q;
  logic [63:0] channel_start_q;
  logic [63:0] channel_count_q;
  logic [63:0] destination_channel_q;
  logic [63:0] external_section_base_q;
  logic [63:0] external_relative_offset_q;
  logic [63:0] external_relative_end_q;
  logic [63:0] external_allocation_end_q;
  logic [31:0] scratchpad_end_q;
  logic [31:0] scratchpad_capacity_q;
  logic [31:0] scratchpad_offset_acc_q;
  logic [63:0] temporary_q;
  logic descriptor_valid_q;
  logic [3:0] pending_error_q;

  logic [31:0] mul_a;
  logic [31:0] mul_b;
  logic mul_start;
  logic mul_busy;
  logic mul_done;
  logic [63:0] mul_result;
  logic mul_state;

  logic signed [63:0] raw_h1;
  logic signed [63:0] raw_w1;
  logic [63:0] valid_h0_calc;
  logic [63:0] valid_w0_calc;
  logic [63:0] valid_h1_calc;
  logic [63:0] valid_w1_calc;
  logic [64:0] external_address_sum;
  logic [63:0] bias_tile_bytes;
  logic [63:0] conv_quant_bytes;

  assign command_in = npu_command_t'(command_bits_i);
  assign src_tensor_in = npu_tensor_desc_t'(src_tensor_desc_bits_i);
  assign dst_tensor_in = npu_tensor_desc_t'(dst_tensor_desc_bits_i);
  assign operator_in = npu_operator_desc_t'(operator_desc_bits_i);
  assign quant_in = npu_quant_desc_t'(quant_desc_bits_i);
  assign segment_in = npu_segment_desc_t'(segment_desc_bits_i);

  assign request_bits_o = request_q;
  assign command_ready_o = state_q == AGU_IDLE;
  assign request_valid_o = state_q == AGU_RETURN
    && pending_error_q == NPU_DMA_AGU_OK;
  assign error_valid_o = state_q == AGU_RETURN
    && pending_error_q != NPU_DMA_AGU_OK;
  assign error_reason_o = pending_error_q;

  assign raw_h1 = raw_h_q + $signed(local_h_q);
  assign raw_w1 = raw_w_q + $signed(local_w_q);
  assign valid_h0_calc = raw_h_q < 0 ? 64'd0 : raw_h_q;
  assign valid_w0_calc = raw_w_q < 0 ? 64'd0 : raw_w_q;
  assign valid_h1_calc = raw_h1 > $signed({48'd0, src_tensor_q.shape_h})
    ? src_tensor_q.shape_h : raw_h1;
  assign valid_w1_calc = raw_w1 > $signed({48'd0, src_tensor_q.shape_w})
    ? src_tensor_q.shape_w : raw_w1;
  assign external_address_sum = {1'b0, external_section_base_q}
    + {1'b0, external_relative_offset_q};

  assign bias_tile_bytes = {30'd0, operator_q.tile_cout, 2'b00};
  assign conv_quant_bytes = {28'd0, operator_q.tile_cout, 4'b0000};
  always @* begin
    mul_a = 32'd0;
    mul_b = 32'd0;
    case (state_q)
      AGU_WEIGHT_KERNEL_H: begin mul_a = aligned_input_c_q; mul_b = operator_q.kh; end
      AGU_WEIGHT_KERNEL_W: begin mul_a = temporary_q[31:0]; mul_b = operator_q.kw; end
      AGU_WEIGHT_TILE: begin mul_a = physical_cout_q; mul_b = kernel_bytes_per_cout_q[31:0]; end
      AGU_WEIGHT_ORIGIN: begin mul_a = operator_q.tile_origin_cout; mul_b = kernel_bytes_per_cout_q[31:0]; end
      AGU_STORE_X: begin mul_a = operator_q.tile_cout; mul_b = src_tensor_q.stride_c; end
      AGU_STORE_SPAD_Y: begin mul_a = physical_cout_q; mul_b = src_tensor_q.stride_c; end
      AGU_STORE_SPAD_Z: begin mul_a = operator_q.tile_w; mul_b = request_q.scratchpad_y_stride; end
      AGU_STORE_EXT_H: begin mul_a = operator_q.tile_origin_h; mul_b = src_tensor_q.stride_h; end
      AGU_STORE_EXT_W: begin mul_a = operator_q.tile_origin_w; mul_b = src_tensor_q.stride_w; end
      AGU_STORE_EXT_C: begin mul_a = operator_q.tile_origin_cout; mul_b = src_tensor_q.stride_c; end
      AGU_STORE_END_Z: begin mul_a = request_q.z_count - 1'b1; mul_b = request_q.external_z_stride; end
      AGU_STORE_END_Y: begin mul_a = request_q.y_count - 1'b1; mul_b = request_q.external_y_stride; end
      AGU_STORE_SPAD_END_Z: begin mul_a = request_q.z_count - 1'b1; mul_b = request_q.scratchpad_z_stride; end
      AGU_STORE_SPAD_END_Y: begin mul_a = request_q.y_count - 1'b1; mul_b = request_q.scratchpad_y_stride; end
      AGU_ACT_LOCAL_H_STRIDE: begin mul_a = operator_q.tile_h - 1'b1; mul_b = operator_q.stride_h; end
      AGU_ACT_LOCAL_H_DILATION: begin mul_a = operator_q.kh - 1'b1; mul_b = operator_q.dilation_h; end
      AGU_ACT_LOCAL_W_STRIDE: begin mul_a = operator_q.tile_w - 1'b1; mul_b = operator_q.stride_w; end
      AGU_ACT_LOCAL_W_DILATION: begin mul_a = operator_q.kw - 1'b1; mul_b = operator_q.dilation_w; end
      AGU_ACT_LOCAL_C: begin mul_a = aligned_input_c_q; mul_b = src_tensor_q.stride_c; end
      AGU_ACT_RAW_H: begin mul_a = operator_q.tile_origin_h; mul_b = operator_q.stride_h; end
      AGU_ACT_RAW_W: begin mul_a = operator_q.tile_origin_w; mul_b = operator_q.stride_w; end
      AGU_ACT_CLEAR_PLANE: begin mul_a = local_w_q[31:0]; mul_b = local_c_bytes_q[31:0]; end
      AGU_ACT_CLEAR: begin mul_a = local_h_q[31:0]; mul_b = local_plane_bytes_q[31:0]; end
      AGU_ACT_X: begin mul_a = channel_count_q[31:0]; mul_b = src_tensor_q.stride_c; end
      AGU_ACT_PAD_ROW: begin mul_a = raw_h_q < 0 ? -raw_h_q : 64'd0; mul_b = local_w_q[31:0]; end
      AGU_ACT_PAD_BYTES: begin
        mul_a = temporary_q[31:0] + (raw_w_q < 0 ? -raw_w_q : 64'd0);
        mul_b = local_c_bytes_q[31:0];
      end
      AGU_ACT_DST_CHANNEL: begin mul_a = destination_channel_q[31:0]; mul_b = src_tensor_q.stride_c; end
      AGU_ACT_EXT_H: begin mul_a = valid_h0_q[31:0]; mul_b = src_tensor_q.stride_h; end
      AGU_ACT_EXT_W: begin mul_a = valid_w0_q[31:0]; mul_b = src_tensor_q.stride_w; end
      AGU_ACT_EXT_C: begin mul_a = channel_start_q[31:0]; mul_b = src_tensor_q.stride_c; end
      AGU_ACT_END_Z: begin mul_a = request_q.z_count - 1'b1; mul_b = request_q.external_z_stride; end
      AGU_ACT_END_Y: begin mul_a = request_q.y_count - 1'b1; mul_b = request_q.external_y_stride; end
      AGU_ACT_SPAD_END_Z: begin mul_a = request_q.z_count - 1'b1; mul_b = request_q.scratchpad_z_stride; end
      AGU_ACT_SPAD_END_Y: begin mul_a = request_q.y_count - 1'b1; mul_b = request_q.scratchpad_y_stride; end
      default: begin end
    endcase
  end

  always @* begin
    mul_state = 1'b0;
    case (state_q)
      AGU_WEIGHT_KERNEL_H, AGU_WEIGHT_KERNEL_W, AGU_WEIGHT_TILE,
      AGU_WEIGHT_ORIGIN, AGU_STORE_X, AGU_STORE_SPAD_Y,
      AGU_STORE_SPAD_Z, AGU_STORE_EXT_H, AGU_STORE_EXT_W,
      AGU_STORE_EXT_C, AGU_STORE_END_Z, AGU_STORE_END_Y,
      AGU_STORE_SPAD_END_Z, AGU_STORE_SPAD_END_Y,
      AGU_ACT_LOCAL_H_STRIDE, AGU_ACT_LOCAL_H_DILATION,
      AGU_ACT_LOCAL_W_STRIDE, AGU_ACT_LOCAL_W_DILATION,
      AGU_ACT_LOCAL_C, AGU_ACT_RAW_H, AGU_ACT_RAW_W,
      AGU_ACT_CLEAR_PLANE, AGU_ACT_CLEAR, AGU_ACT_X,
      AGU_ACT_PAD_ROW, AGU_ACT_PAD_BYTES, AGU_ACT_DST_CHANNEL,
      AGU_ACT_EXT_H, AGU_ACT_EXT_W, AGU_ACT_EXT_C,
      AGU_ACT_END_Z, AGU_ACT_END_Y, AGU_ACT_SPAD_END_Z,
      AGU_ACT_SPAD_END_Y: mul_state = 1'b1;
      default: begin end
    endcase
  end

  assign mul_start = mul_state && !mul_busy && !mul_done;

  npu_u32_mul_iter shared_address_multiplier (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_i(mul_start), .operand_a_i(mul_a), .operand_b_i(mul_b),
    .busy_o(mul_busy), .done_o(mul_done), .result_o(mul_result)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= AGU_IDLE;
      kind_q <= AGU_KIND_ACTIVATION;
      command_q <= '0;
      src_tensor_q <= '0;
      dst_tensor_q <= '0;
      operator_q <= '0;
      quant_q <= '0;
      segment_q <= '0;
      request_q <= '0;
      physical_cout_q <= '0;
      aligned_input_c_q <= '0;
      kernel_bytes_per_cout_q <= '0;
      weight_tile_bytes_q <= '0;
      local_h_q <= '0;
      local_w_q <= '0;
      local_c_bytes_q <= '0;
      local_plane_bytes_q <= '0;
      raw_h_q <= '0;
      raw_w_q <= '0;
      valid_h0_q <= '0;
      valid_w0_q <= '0;
      valid_h1_q <= '0;
      valid_w1_q <= '0;
      channel_start_q <= '0;
      channel_count_q <= '0;
      destination_channel_q <= '0;
      external_section_base_q <= '0;
      external_relative_offset_q <= '0;
      external_relative_end_q <= '0;
      external_allocation_end_q <= '0;
      scratchpad_end_q <= '0;
      scratchpad_capacity_q <= '0;
      scratchpad_offset_acc_q <= '0;
      temporary_q <= '0;
      descriptor_valid_q <= 1'b0;
      pending_error_q <= NPU_DMA_AGU_OK;
    end else if (soft_reset_i) begin
      state_q <= AGU_IDLE;
      request_q <= '0;
      pending_error_q <= NPU_DMA_AGU_OK;
    end else begin
      case (state_q)
        AGU_IDLE: begin
          if (command_valid_i) begin
            command_q <= command_in;
            src_tensor_q <= src_tensor_in;
            dst_tensor_q <= dst_tensor_in;
            operator_q <= operator_in;
            quant_q <= quant_in;
            segment_q <= segment_in;
            request_q <= '0;
            request_q.tag <= command_in.tag;
            request_q.event_mask <= command_in.imm[7:0];
            request_q.store <= command_in.opcode == NPU_OP_DMA_STORE;
            request_q.memory_space <= NPU_DMA_SPACE_ACTIVATION;
            request_q.scratchpad <= NPU_SPAD_A;
            request_q.bank <= command_in.imm[NPU_IMM_ACTIVATION_BANK_BIT];
            request_q.y_count <= 16'd1;
            request_q.z_count <= 16'd1;
            physical_cout_q <= (operator_in.tile_cout + 7) & ~32'd7;
            aligned_input_c_q <= (operator_in.input_channel_count + 7) & ~32'd7;
            kernel_bytes_per_cout_q <= '0;
            weight_tile_bytes_q <= '0;
            local_h_q <= '0;
            local_w_q <= '0;
            local_c_bytes_q <= '0;
            local_plane_bytes_q <= '0;
            raw_h_q <= '0;
            raw_w_q <= '0;
            valid_h0_q <= '0;
            valid_w0_q <= '0;
            valid_h1_q <= '0;
            valid_w1_q <= '0;
            external_relative_end_q <= '0;
            external_allocation_end_q <= src_tensor_in.base_offset
              + src_tensor_in.allocation_bytes;
            scratchpad_end_q <= '0;
            scratchpad_offset_acc_q <= '0;
            temporary_q <= '0;
            descriptor_valid_q <= 1'b1;
            pending_error_q <= NPU_DMA_AGU_OK;

            if (command_in.opcode != NPU_OP_DMA_LOAD
                && command_in.opcode != NPU_OP_DMA_STORE) begin
              pending_error_q <= NPU_DMA_AGU_BAD_COMMAND;
              state_q <= AGU_RETURN;
            end else if (command_in.op_desc == NPU_NONE_INDEX
                         || operator_in.tile_h == 0 || operator_in.tile_w == 0
                         || operator_in.tile_cout == 0
                         || operator_in.kh == 0 || operator_in.kw == 0
                         || operator_in.stride_h == 0 || operator_in.stride_w == 0
                         || operator_in.dilation_h == 0 || operator_in.dilation_w == 0) begin
              pending_error_q <= NPU_DMA_AGU_BAD_DESCRIPTOR;
              state_q <= AGU_RETURN;
            end else if (command_in.opcode == NPU_OP_DMA_STORE) begin
              kind_q <= AGU_KIND_STORE;
              request_q.scratchpad <= NPU_SPAD_O;
              request_q.bank <= command_in.imm[NPU_IMM_OUTPUT_BANK_BIT];
              request_q.y_count <= operator_in.tile_w[15:0];
              request_q.z_count <= operator_in.tile_h[15:0];
              request_q.external_y_stride <= src_tensor_in.stride_w;
              request_q.external_z_stride <= src_tensor_in.stride_h;
              external_section_base_q <= activation_base_i;
              external_relative_offset_q <= src_tensor_in.base_offset;
              scratchpad_capacity_q <= OUTPUT_BANK_BYTES;
              descriptor_valid_q <= command_in.src0_td == command_in.dst_td
                && src_tensor_in.layout == NPU_LAYOUT_NHWC8
                && src_tensor_in.dtype == NPU_DTYPE_INT12_IN_INT16
                && operator_in.tile_origin_h + operator_in.tile_h <= src_tensor_in.shape_h
                && operator_in.tile_origin_w + operator_in.tile_w <= src_tensor_in.shape_w
                && operator_in.tile_origin_cout + operator_in.tile_cout <= src_tensor_in.shape_c;
              state_q <= AGU_STORE_X;
            end else if (command_in.src0_td == NPU_NONE_INDEX
                         && command_in.dst_td == NPU_NONE_INDEX
                         && command_in.quant_desc != NPU_NONE_INDEX) begin
              kind_q <= command_in.imm[NPU_IMM_DMA_VECTOR_QUANT_BIT]
                ? AGU_KIND_QUANT_VECTOR : AGU_KIND_QUANT_CONV;
              request_q.memory_space <= NPU_DMA_SPACE_QUANT;
              request_q.scratchpad <= NPU_SPAD_W;
              request_q.bank <= command_in.imm[NPU_IMM_WEIGHT_BANK_BIT];
              external_section_base_q <= quant_param_base_i;
              scratchpad_capacity_q <= WEIGHT_BANK_BYTES;
              state_q <= AGU_WEIGHT_KERNEL_H;
            end else if (src_tensor_in.layout == NPU_LAYOUT_WEIGHT_O8I8) begin
              kind_q <= AGU_KIND_WEIGHT;
              request_q.memory_space <= NPU_DMA_SPACE_WEIGHT;
              request_q.scratchpad <= NPU_SPAD_W;
              request_q.bank <= command_in.imm[NPU_IMM_WEIGHT_BANK_BIT];
              external_section_base_q <= weight_base_i;
              scratchpad_capacity_q <= WEIGHT_BANK_BYTES;
              state_q <= AGU_WEIGHT_KERNEL_H;
            end else if (src_tensor_in.layout == NPU_LAYOUT_LINEAR
                         && src_tensor_in.dtype == NPU_DTYPE_INT32) begin
              kind_q <= AGU_KIND_BIAS;
              request_q.memory_space <= NPU_DMA_SPACE_BIAS;
              request_q.scratchpad <= NPU_SPAD_W;
              request_q.bank <= command_in.imm[NPU_IMM_WEIGHT_BANK_BIT];
              external_section_base_q <= bias_base_i;
              scratchpad_capacity_q <= WEIGHT_BANK_BYTES;
              state_q <= AGU_WEIGHT_KERNEL_H;
            end else begin
              kind_q <= AGU_KIND_ACTIVATION;
              channel_start_q <= command_in.imm[NPU_IMM_SEGMENTED_BIT]
                ? 64'd0 : operator_in.input_channel_start;
              channel_count_q <= command_in.imm[NPU_IMM_SEGMENTED_BIT]
                ? segment_in.channel_count : operator_in.input_channel_count;
              destination_channel_q <= command_in.imm[NPU_IMM_SEGMENTED_BIT]
                ? segment_in.dst_channel : 64'd0;
              request_q.clear_before <= !command_in.imm[NPU_IMM_SEGMENTED_BIT]
                || command_in.imm[NPU_IMM_SEGMENT_LSB +: NPU_IMM_SEGMENT_WIDTH] == 0;
              external_section_base_q <= activation_base_i;
              external_relative_offset_q <= src_tensor_in.base_offset;
              scratchpad_capacity_q <= ACTIVATION_BANK_BYTES;
              state_q <= AGU_ACT_LOCAL_H_STRIDE;
            end
          end
        end

        AGU_WEIGHT_KERNEL_H: if (mul_done) begin
          temporary_q <= mul_result;
          state_q <= AGU_WEIGHT_KERNEL_W;
        end
        AGU_WEIGHT_KERNEL_W: if (mul_done) begin
          kernel_bytes_per_cout_q <= mul_result;
          state_q <= AGU_WEIGHT_TILE;
        end
        AGU_WEIGHT_TILE: if (mul_done) begin
          weight_tile_bytes_q <= mul_result;
          if (kind_q == AGU_KIND_WEIGHT) begin
            request_q.x_bytes <= mul_result[31:0];
            state_q <= AGU_WEIGHT_ORIGIN;
          end else begin
            state_q <= AGU_LAYOUT_BIAS;
          end
        end
        AGU_WEIGHT_ORIGIN: if (mul_done) begin
          external_relative_offset_q <= src_tensor_q.base_offset + mul_result;
          external_relative_end_q <= src_tensor_q.base_offset + mul_result + weight_tile_bytes_q;
          scratchpad_end_q <= weight_tile_bytes_q;
          descriptor_valid_q <= command_q.src0_td == operator_q.weight_td
            && command_q.dst_td == operator_q.weight_td
            && operator_q.tile_origin_cout[2:0] == 3'b000
            && src_tensor_q.layout == NPU_LAYOUT_WEIGHT_O8I8;
          state_q <= AGU_CHECK_EXTERNAL;
        end

        AGU_LAYOUT_BIAS: begin
          temporary_q <= (weight_tile_bytes_q + 63) & ~64'd63;
          state_q <= kind_q == AGU_KIND_BIAS
            ? AGU_LAYOUT_BIAS_RETURN : AGU_LAYOUT_CONV;
        end
        AGU_LAYOUT_BIAS_RETURN: begin
          request_q.scratchpad_offset <= temporary_q[31:0];
          request_q.x_bytes <= bias_tile_bytes[31:0];
          scratchpad_end_q <= temporary_q[31:0] + bias_tile_bytes[31:0];
          external_relative_offset_q <= src_tensor_q.base_offset
            + ({32'd0, operator_q.tile_origin_cout} << 2);
          external_relative_end_q <= src_tensor_q.base_offset
            + ({32'd0, operator_q.tile_origin_cout} << 2) + bias_tile_bytes;
          descriptor_valid_q <= command_q.src0_td == operator_q.bias_td
            && command_q.dst_td == operator_q.bias_td;
          state_q <= AGU_CHECK_EXTERNAL;
        end
        AGU_LAYOUT_CONV: begin
          temporary_q <= (temporary_q + bias_tile_bytes + 63) & ~64'd63;
          state_q <= kind_q == AGU_KIND_QUANT_VECTOR
            ? AGU_LAYOUT_VECTOR : AGU_LAYOUT_CONV_RETURN;
        end
        AGU_LAYOUT_CONV_RETURN: begin
          request_q.scratchpad_offset <= temporary_q[31:0];
          request_q.x_bytes <= conv_quant_bytes[31:0];
          scratchpad_end_q <= temporary_q[31:0] + conv_quant_bytes[31:0];
          external_relative_offset_q <= quant_q.param_offset
            + ({32'd0, operator_q.tile_origin_cout} << 4);
          external_relative_end_q <= quant_q.param_offset
            + ({32'd0, operator_q.tile_origin_cout} << 4) + conv_quant_bytes;
          descriptor_valid_q <= operator_q.tile_origin_cout
            + operator_q.tile_cout <= quant_q.param_count;
          state_q <= AGU_FINAL;
        end
        AGU_LAYOUT_VECTOR: begin
          temporary_q <= (temporary_q + conv_quant_bytes + 63) & ~64'd63;
          state_q <= AGU_LAYOUT_VECTOR_RETURN;
        end
        AGU_LAYOUT_VECTOR_RETURN: begin
          request_q.scratchpad_offset <= temporary_q[31:0];
          request_q.x_bytes <= NPU_QUANT_PARAM_BYTES;
          scratchpad_end_q <= temporary_q[31:0] + NPU_QUANT_PARAM_BYTES;
          external_relative_offset_q <= quant_q.param_offset;
          external_relative_end_q <= quant_q.param_offset
            + NPU_QUANT_PARAM_BYTES;
          descriptor_valid_q <= quant_q.param_count >= 1;
          state_q <= AGU_FINAL;
        end

        AGU_STORE_X: if (mul_done) begin request_q.x_bytes <= mul_result[31:0]; state_q <= AGU_STORE_SPAD_Y; end
        AGU_STORE_SPAD_Y: if (mul_done) begin request_q.scratchpad_y_stride <= mul_result[31:0]; state_q <= AGU_STORE_SPAD_Z; end
        AGU_STORE_SPAD_Z: if (mul_done) begin request_q.scratchpad_z_stride <= mul_result[31:0]; state_q <= AGU_STORE_EXT_H; end
        AGU_STORE_EXT_H: if (mul_done) begin external_relative_offset_q <= external_relative_offset_q + mul_result; state_q <= AGU_STORE_EXT_W; end
        AGU_STORE_EXT_W: if (mul_done) begin external_relative_offset_q <= external_relative_offset_q + mul_result; state_q <= AGU_STORE_EXT_C; end
        AGU_STORE_EXT_C: if (mul_done) begin
          external_relative_offset_q <= external_relative_offset_q + mul_result;
          external_relative_end_q <= external_relative_offset_q + mul_result;
          state_q <= AGU_STORE_END_Z;
        end
        AGU_STORE_END_Z: if (mul_done) begin external_relative_end_q <= external_relative_end_q + mul_result; state_q <= AGU_STORE_END_Y; end
        AGU_STORE_END_Y: if (mul_done) begin
          external_relative_end_q <= external_relative_end_q + mul_result + request_q.x_bytes;
          state_q <= AGU_CHECK_EXTERNAL;
        end
        AGU_STORE_SPAD_END_Z: if (mul_done) begin
          scratchpad_end_q <= mul_result[31:0];
          state_q <= AGU_ADD_SPAD_OFFSET;
        end
        AGU_STORE_SPAD_END_Y: if (mul_done) begin
          scratchpad_end_q <= scratchpad_end_q + mul_result[31:0];
          state_q <= AGU_ADD_SPAD_X;
        end

        AGU_ACT_LOCAL_H_STRIDE: if (mul_done) begin temporary_q <= mul_result; state_q <= AGU_ACT_LOCAL_H_DILATION; end
        AGU_ACT_LOCAL_H_DILATION: if (mul_done) begin local_h_q <= temporary_q + mul_result + 1'b1; state_q <= AGU_ACT_LOCAL_W_STRIDE; end
        AGU_ACT_LOCAL_W_STRIDE: if (mul_done) begin temporary_q <= mul_result; state_q <= AGU_ACT_LOCAL_W_DILATION; end
        AGU_ACT_LOCAL_W_DILATION: if (mul_done) begin local_w_q <= temporary_q + mul_result + 1'b1; state_q <= AGU_ACT_LOCAL_C; end
        AGU_ACT_LOCAL_C: if (mul_done) begin
          local_c_bytes_q <= mul_result;
          request_q.scratchpad_y_stride <= mul_result[31:0];
          state_q <= AGU_ACT_RAW_H;
        end
        AGU_ACT_RAW_H: if (mul_done) begin raw_h_q <= $signed(mul_result) - $signed({48'd0, operator_q.pad_top}); state_q <= AGU_ACT_RAW_W; end
        AGU_ACT_RAW_W: if (mul_done) begin
          raw_w_q <= $signed(mul_result)
            - $signed({48'd0, operator_q.pad_left});
          state_q <= AGU_ACT_CLAMP;
        end
        AGU_ACT_CLAMP: begin
          valid_h0_q <= valid_h0_calc;
          valid_w0_q <= valid_w0_calc;
          valid_h1_q <= valid_h1_calc;
          valid_w1_q <= valid_w1_calc;
          state_q <= AGU_ACT_GEOMETRY;
        end
        AGU_ACT_GEOMETRY: begin
          request_q.y_count <= valid_w1_q - valid_w0_q;
          request_q.z_count <= valid_h1_q - valid_h0_q;
          request_q.external_y_stride <= src_tensor_q.stride_w;
          request_q.external_z_stride <= src_tensor_q.stride_h;
          descriptor_valid_q <= command_q.dst_td == operator_q.src_td
            && src_tensor_q.layout == NPU_LAYOUT_NHWC8
            && src_tensor_q.dtype == NPU_DTYPE_INT12_IN_INT16
            && dst_tensor_q.shape_c == operator_q.input_channel_count
            && valid_w1_q != valid_w0_q && valid_h1_q != valid_h0_q
            && (command_q.imm[NPU_IMM_SEGMENTED_BIT]
                ? (dst_tensor_q.layout == NPU_LAYOUT_SEGMENTED
                   && segment_q.tensor_index == command_q.src0_td
                   && destination_channel_q + channel_count_q <= dst_tensor_q.shape_c
                   && command_q.imm[NPU_IMM_SEGMENT_LSB +: NPU_IMM_SEGMENT_WIDTH] < dst_tensor_q.segment_count)
                : (command_q.src0_td == command_q.dst_td
                   && command_q.src0_td == operator_q.src_td
                   && dst_tensor_q.layout == NPU_LAYOUT_NHWC8
                   && channel_start_q + channel_count_q <= src_tensor_q.shape_c));
          state_q <= AGU_ACT_CLEAR_PLANE;
        end
        AGU_ACT_CLEAR_PLANE: if (mul_done) begin
          local_plane_bytes_q <= mul_result;
          request_q.scratchpad_z_stride <= mul_result[31:0];
          state_q <= AGU_ACT_CLEAR;
        end
        AGU_ACT_CLEAR: if (mul_done) begin request_q.clear_bytes <= mul_result[31:0]; state_q <= AGU_ACT_X; end
        AGU_ACT_X: if (mul_done) begin request_q.x_bytes <= mul_result[31:0]; state_q <= AGU_ACT_PAD_ROW; end
        AGU_ACT_PAD_ROW: if (mul_done) begin temporary_q <= mul_result; state_q <= AGU_ACT_PAD_BYTES; end
        AGU_ACT_PAD_BYTES: if (mul_done) begin
          scratchpad_offset_acc_q <= mul_result[31:0];
          state_q <= AGU_ACT_DST_CHANNEL;
        end
        AGU_ACT_DST_CHANNEL: if (mul_done) begin
          scratchpad_offset_acc_q <= scratchpad_offset_acc_q
            + mul_result[31:0];
          state_q <= AGU_ACT_SPAD_OFFSET;
        end
        AGU_ACT_SPAD_OFFSET: begin
          request_q.scratchpad_offset <= scratchpad_offset_acc_q;
          external_relative_offset_q <= src_tensor_q.base_offset;
          state_q <= AGU_ACT_EXT_H;
        end
        AGU_ACT_EXT_H: if (mul_done) begin external_relative_offset_q <= external_relative_offset_q + mul_result; state_q <= AGU_ACT_EXT_W; end
        AGU_ACT_EXT_W: if (mul_done) begin external_relative_offset_q <= external_relative_offset_q + mul_result; state_q <= AGU_ACT_EXT_C; end
        AGU_ACT_EXT_C: if (mul_done) begin
          external_relative_offset_q <= external_relative_offset_q + mul_result;
          external_relative_end_q <= external_relative_offset_q + mul_result;
          state_q <= AGU_ACT_END_Z;
        end
        AGU_ACT_END_Z: if (mul_done) begin external_relative_end_q <= external_relative_end_q + mul_result; state_q <= AGU_ACT_END_Y; end
        AGU_ACT_END_Y: if (mul_done) begin
          external_relative_end_q <= external_relative_end_q + mul_result + request_q.x_bytes;
          state_q <= AGU_CHECK_EXTERNAL;
        end
        AGU_ACT_SPAD_END_Z: if (mul_done) begin
          scratchpad_end_q <= mul_result[31:0];
          state_q <= AGU_ADD_SPAD_OFFSET;
        end
        AGU_ACT_SPAD_END_Y: if (mul_done) begin
          scratchpad_end_q <= scratchpad_end_q + mul_result[31:0];
          state_q <= AGU_ADD_SPAD_X;
        end

        AGU_ADD_SPAD_X: begin
          scratchpad_end_q <= scratchpad_end_q + request_q.x_bytes;
          state_q <= AGU_FINAL;
        end

        AGU_ADD_SPAD_OFFSET: begin
          scratchpad_end_q <= scratchpad_end_q
            + request_q.scratchpad_offset;
          state_q <= kind_q == AGU_KIND_STORE
            ? AGU_STORE_SPAD_END_Y : AGU_ACT_SPAD_END_Y;
        end

        AGU_CHECK_EXTERNAL: begin
          if (external_relative_end_q > external_allocation_end_q)
            pending_error_q <= NPU_DMA_AGU_EXTERNAL_BOUNDS;
          if (kind_q == AGU_KIND_STORE)
            state_q <= AGU_STORE_SPAD_END_Z;
          else if (kind_q == AGU_KIND_ACTIVATION)
            state_q <= AGU_ACT_SPAD_END_Z;
          else
            state_q <= AGU_FINAL;
        end

        AGU_FINAL: begin
          request_q.external_address <= external_address_sum[63:0];
          if (pending_error_q != NPU_DMA_AGU_OK)
            pending_error_q <= pending_error_q;
          else if (!descriptor_valid_q)
            pending_error_q <= NPU_DMA_AGU_BAD_DESCRIPTOR;
          else if (external_address_sum[64])
            pending_error_q <= NPU_DMA_AGU_ADDRESS_OVERFLOW;
          else if (scratchpad_end_q > scratchpad_capacity_q
                   || request_q.clear_bytes > scratchpad_capacity_q)
            pending_error_q <= NPU_DMA_AGU_SPAD_BOUNDS;
          state_q <= AGU_RETURN;
        end
        AGU_RETURN: if (pending_error_q != NPU_DMA_AGU_OK || request_ready_i)
          state_q <= AGU_IDLE;
        default: begin pending_error_q <= NPU_DMA_AGU_BAD_COMMAND; state_q <= AGU_RETURN; end
      endcase
    end
  end

endmodule
