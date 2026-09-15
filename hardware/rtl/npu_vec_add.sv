`timescale 1ns/1ps

module npu_vec_add (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          soft_reset_i,

  input  logic          start_valid_i,
  output logic          start_ready_o,
  input  logic [127:0]  command_bits_i,
  input  logic [511:0]  operator_desc_bits_i,

  output logic          read_request_valid_o,
  input  logic          read_request_ready_i,
  output logic [1:0]    read_kind_o,
  output logic          read_bank_o,
  output logic [31:0]   read_address_o,
  input  logic          read_response_valid_i,
  output logic          read_response_ready_o,
  input  logic [511:0]  read_data_i,

  output logic          write_valid_o,
  input  logic          write_ready_i,
  output logic [1:0]    write_kind_o,
  output logic          write_bank_o,
  output logic [31:0]   write_address_o,
  output logic [511:0]  write_data_o,
  output logic [63:0]   write_strobe_o,

  output logic          busy_o,
  output logic          done_pulse_o,
  output logic          error_pulse_o,
  output logic [3:0]    error_reason_o
);
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  localparam logic [3:0] VEC_ERROR_NONE = 4'h0;
  localparam logic [3:0] VEC_ERROR_OPCODE = 4'h1;
  localparam logic [3:0] VEC_ERROR_FLAGS = 4'h2;
  localparam logic [3:0] VEC_ERROR_FIELDS = 4'h3;
  localparam logic [3:0] VEC_ERROR_GEOMETRY = 4'h4;
  localparam logic [3:0] VEC_ERROR_MODE = 4'h5;
  localparam logic [3:0] VEC_ERROR_POST = 4'h8;

  typedef enum logic [3:0] {
    VEC_IDLE,
    VEC_SETUP_ROW_BYTES,
    VEC_SETUP_PAD_ROWS,
    VEC_SETUP_PAD_COLUMNS,
    VEC_SETUP_ROW_ADVANCE,
    VEC_PARAM_REQUEST,
    VEC_PARAM_RESPONSE,
    VEC_SRC0_REQUEST,
    VEC_SRC0_RESPONSE,
    VEC_SRC1_REQUEST,
    VEC_SRC1_RESPONSE,
    VEC_POST_SEND,
    VEC_POST_WAIT
  } vec_state_e;

  vec_state_e state_q;
  npu_command_t start_command;
  npu_operator_desc_t start_descriptor;
  npu_operator_desc_t descriptor_q;
  logic activation_bank_q;
  logic weight_bank_q;
  logic output_bank_q;
  logic [7:0] lane_mask_q;
  logic [31:0] activation_pixel_step_q;
  logic [31:0] activation_row_bytes_q;
  logic [31:0] pad_rows_bytes_q;
  logic [31:0] activation_row_advance_q;
  logic [31:0] residual_address_q;
  logic [31:0] output_address_q;
  logic [4:0] output_h_q;
  logic [4:0] output_w_q;
  logic [31:0] vector_quant_address_q;
  logic [6:0] local_w_q;
  logic [127:0] scalar_quant_q;
  logic [127:0] src0_q;
  logic [127:0] src1_q;

  logic [9:0] start_cin_blocks;
  logic [9:0] start_kernel_rows;
  logic [6:0] start_local_w;
  logic [7:0] start_lane_mask;
  logic start_fields_valid;
  logic start_geometry_valid;
  logic start_mode_valid;
  logic start_fire;

  logic [31:0] mul_a;
  logic [31:0] mul_b;
  logic mul_start;
  logic mul_busy;
  logic mul_done;
  logic [63:0] mul_result;

  logic [255:0] post_accumulator;
  logic [1023:0] post_quant_params;
  logic post_input_valid;
  logic post_input_ready;
  logic post_output_valid;
  logic post_output_ready;
  logic [127:0] post_output_data;
  logic [7:0] post_output_mask;
  logic post_error;
  logic [3:0] post_error_reason;
  logic [127:0] add_result;
  logic write_fire;
  integer accumulator_lane;
  integer result_lane;
  integer strobe_lane;

  function automatic logic [15:0] saturating_add_int12(
    input logic [15:0] lhs,
    input logic [15:0] rhs
  );
    logic signed [16:0] sum;
    begin
      sum = $signed(lhs) + $signed(rhs);
      if (sum > 17'sd2047)
        saturating_add_int12 = 16'h07ff;
      else if (sum < -17'sd2048)
        saturating_add_int12 = 16'hf800;
      else
        saturating_add_int12 = sum[15:0];
    end
  endfunction

  assign start_command = npu_command_t'(command_bits_i);
  assign start_descriptor = npu_operator_desc_t'(operator_desc_bits_i);
  assign start_cin_blocks
    = (start_descriptor.input_channel_count + 7) >> 3;
  assign start_local_w = start_descriptor.tile_w - 1 + start_descriptor.kw;
  always_comb begin
    start_kernel_rows = start_cin_blocks;
    if (start_descriptor.kh == 3)
      start_kernel_rows = start_kernel_rows + (start_kernel_rows << 1);
    if (start_descriptor.kw == 3)
      start_kernel_rows = start_kernel_rows + (start_kernel_rows << 1);
    if (start_descriptor.tile_cout[2:0] == 0)
      start_lane_mask = 8'hff;
    else
      start_lane_mask
        = (9'b1 << start_descriptor.tile_cout[2:0]) - 1'b1;
  end

  assign start_fields_valid = start_command.dst_td != NPU_NONE_INDEX
    && start_command.src0_td == start_descriptor.dst_td
    && start_command.src1_td == start_descriptor.src_td
    && start_command.op_desc != NPU_NONE_INDEX
    && start_command.quant_desc != NPU_NONE_INDEX;
  assign start_geometry_valid = start_descriptor.tile_h != 0
    && start_descriptor.tile_h <= 16 && start_descriptor.tile_w != 0
    && start_descriptor.tile_w <= 16 && start_descriptor.tile_cout != 0
    && start_descriptor.tile_cout <= 8
    && start_descriptor.input_channel_count != 0
    && start_descriptor.input_channel_count <= 224
    && start_descriptor.tile_origin_cout + start_descriptor.tile_cout
       <= start_descriptor.input_channel_count;
  assign start_mode_valid = start_descriptor.stride_h == 1
    && start_descriptor.stride_w == 1
    && (start_descriptor.kh == 1 || start_descriptor.kh == 3)
    && (start_descriptor.kw == 1 || start_descriptor.kw == 3)
    && start_descriptor.dilation_h == 1
    && start_descriptor.dilation_w == 1
    && start_descriptor.groups == 1;
  assign start_ready_o = rst_ni && !soft_reset_i && state_q == VEC_IDLE;
  assign start_fire = start_valid_i && start_ready_o;
  assign busy_o = state_q != VEC_IDLE;

  always_comb begin
    mul_a = 32'd0;
    mul_b = 32'd0;
    case (state_q)
      VEC_SETUP_ROW_BYTES: begin
        mul_a = {25'd0, local_w_q};
        mul_b = activation_pixel_step_q;
      end
      VEC_SETUP_PAD_ROWS: begin
        mul_a = {16'd0, descriptor_q.pad_top};
        mul_b = activation_row_bytes_q;
      end
      VEC_SETUP_PAD_COLUMNS: begin
        mul_a = {16'd0, descriptor_q.pad_left};
        mul_b = activation_pixel_step_q;
      end
      VEC_SETUP_ROW_ADVANCE: begin
        mul_a = descriptor_q.tile_w - 1'b1;
        mul_b = activation_pixel_step_q;
      end
      default: begin end
    endcase
  end
  assign mul_start = state_q >= VEC_SETUP_ROW_BYTES
    && state_q <= VEC_SETUP_ROW_ADVANCE && !mul_busy && !mul_done;

  npu_u32_mul_iter setup_multiplier (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_i(mul_start), .operand_a_i(mul_a), .operand_b_i(mul_b),
    .busy_o(mul_busy), .done_o(mul_done), .result_o(mul_result));

  always_comb begin
    read_request_valid_o = 1'b0;
    read_kind_o = NPU_SPAD_A;
    read_bank_o = activation_bank_q;
    read_address_o = residual_address_q;
    case (state_q)
      VEC_PARAM_REQUEST: begin
        read_request_valid_o = 1'b1;
        read_kind_o = NPU_SPAD_W;
        read_bank_o = weight_bank_q;
        read_address_o = vector_quant_address_q;
      end
      VEC_SRC0_REQUEST: begin
        read_request_valid_o = 1'b1;
        read_kind_o = NPU_SPAD_O;
        read_bank_o = output_bank_q;
        read_address_o = output_address_q;
      end
      VEC_SRC1_REQUEST: begin
        read_request_valid_o = 1'b1;
        read_kind_o = NPU_SPAD_A;
        read_bank_o = activation_bank_q;
        read_address_o = residual_address_q;
      end
      default: begin end
    endcase
  end
  assign read_response_ready_o = state_q == VEC_PARAM_RESPONSE
    || state_q == VEC_SRC0_RESPONSE || state_q == VEC_SRC1_RESPONSE;

  always_comb begin
    post_accumulator = '0;
    for (accumulator_lane = 0; accumulator_lane < 8;
         accumulator_lane = accumulator_lane + 1)
      post_accumulator[accumulator_lane*32 +: 32]
        = {{16{src1_q[accumulator_lane*16 + 15]}},
           src1_q[accumulator_lane*16 +: 16]};
  end
  assign post_quant_params = {8{scalar_quant_q}};
  assign post_input_valid = state_q == VEC_POST_SEND;
  assign post_output_ready = state_q == VEC_POST_WAIT && write_ready_i;

  npu_requant_post residual_requant (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .input_valid_i(post_input_valid), .input_ready_o(post_input_ready),
    .accumulator_i(post_accumulator), .bias_i(256'd0),
    .quant_params_i(post_quant_params), .post_op_i(16'h0000),
    .lane_mask_i(lane_mask_q), .output_valid_o(post_output_valid),
    .output_ready_i(post_output_ready), .output_data_o(post_output_data),
    .output_lane_mask_o(post_output_mask), .lut_write_valid_i(1'b0),
    .lut_write_ready_o(), .lut_write_address_i(12'd0),
    .lut_write_data_i(16'd0),
    .error_pulse_o(post_error), .error_reason_o(post_error_reason));

  always_comb begin
    add_result = '0;
    for (result_lane = 0; result_lane < 8; result_lane = result_lane + 1)
      add_result[result_lane*16 +: 16] = saturating_add_int12(
        src0_q[result_lane*16 +: 16],
        post_output_data[result_lane*16 +: 16]);
  end

  assign write_valid_o = state_q == VEC_POST_WAIT && post_output_valid;
  assign write_kind_o = NPU_SPAD_O;
  assign write_bank_o = output_bank_q;
  assign write_address_o = output_address_q;
  assign write_data_o = {384'd0, add_result};
  always_comb begin
    write_strobe_o = '0;
    for (strobe_lane = 0; strobe_lane < 8; strobe_lane = strobe_lane + 1)
      write_strobe_o[strobe_lane*2 +: 2]
        = {2{post_output_mask[strobe_lane]}};
  end
  assign write_fire = write_valid_o && write_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= VEC_IDLE;
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      error_reason_o <= VEC_ERROR_NONE;
    end else if (soft_reset_i) begin
      state_q <= VEC_IDLE;
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      error_reason_o <= VEC_ERROR_NONE;
    end else begin
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;

      if (start_fire) begin
        error_reason_o <= VEC_ERROR_NONE;
        if (start_command.opcode != NPU_OP_VEC_ADD) begin
          error_pulse_o <= 1'b1;
          error_reason_o <= VEC_ERROR_OPCODE;
        end else if (start_command.flags != NPU_FLAG_SATURATE) begin
          error_pulse_o <= 1'b1;
          error_reason_o <= VEC_ERROR_FLAGS;
        end else if (!start_fields_valid) begin
          error_pulse_o <= 1'b1;
          error_reason_o <= VEC_ERROR_FIELDS;
        end else if (!start_geometry_valid) begin
          error_pulse_o <= 1'b1;
          error_reason_o <= VEC_ERROR_GEOMETRY;
        end else if (!start_mode_valid) begin
          error_pulse_o <= 1'b1;
          error_reason_o <= VEC_ERROR_MODE;
        end else begin
          descriptor_q <= start_descriptor;
          activation_bank_q
            <= start_command.imm[NPU_IMM_ACTIVATION_BANK_BIT];
          weight_bank_q <= start_command.imm[NPU_IMM_WEIGHT_BANK_BIT];
          output_bank_q <= start_command.imm[NPU_IMM_OUTPUT_BANK_BIT];
          lane_mask_q <= start_lane_mask;
          local_w_q <= start_local_w;
          activation_pixel_step_q <= {start_cin_blocks, 4'b0};
          vector_quant_address_q <= {start_kernel_rows, 6'b0}
            + (start_descriptor.tile_cout <= 4 ? 32'd128 : 32'd192);
          output_address_q <= '0;
          output_h_q <= '0;
          output_w_q <= '0;
          state_q <= VEC_SETUP_ROW_BYTES;
        end
      end

      if (mul_done) begin
        case (state_q)
          VEC_SETUP_ROW_BYTES: begin
            activation_row_bytes_q <= mul_result[31:0];
            state_q <= VEC_SETUP_PAD_ROWS;
          end
          VEC_SETUP_PAD_ROWS: begin
            pad_rows_bytes_q <= mul_result[31:0];
            state_q <= VEC_SETUP_PAD_COLUMNS;
          end
          VEC_SETUP_PAD_COLUMNS: begin
            residual_address_q <= pad_rows_bytes_q + mul_result[31:0]
              + (descriptor_q.tile_origin_cout << 1);
            state_q <= VEC_SETUP_ROW_ADVANCE;
          end
          VEC_SETUP_ROW_ADVANCE: begin
            activation_row_advance_q
              <= activation_row_bytes_q - mul_result[31:0];
            state_q <= VEC_PARAM_REQUEST;
          end
          default: begin end
        endcase
      end

      case (state_q)
        VEC_PARAM_REQUEST:
          if (read_request_valid_o && read_request_ready_i)
            state_q <= VEC_PARAM_RESPONSE;
        VEC_PARAM_RESPONSE:
          if (read_response_valid_i) begin
            scalar_quant_q <= read_data_i[127:0];
            state_q <= VEC_SRC0_REQUEST;
          end
        VEC_SRC0_REQUEST:
          if (read_request_valid_o && read_request_ready_i)
            state_q <= VEC_SRC0_RESPONSE;
        VEC_SRC0_RESPONSE:
          if (read_response_valid_i) begin
            src0_q <= read_data_i[127:0];
            state_q <= VEC_SRC1_REQUEST;
          end
        VEC_SRC1_REQUEST:
          if (read_request_valid_o && read_request_ready_i)
            state_q <= VEC_SRC1_RESPONSE;
        VEC_SRC1_RESPONSE:
          if (read_response_valid_i) begin
            src1_q <= read_data_i[127:0];
            state_q <= VEC_POST_SEND;
          end
        VEC_POST_SEND:
          if (post_input_ready)
            state_q <= VEC_POST_WAIT;
        VEC_POST_WAIT:
          if (write_fire) begin
            if (output_w_q + 1 == descriptor_q.tile_w) begin
              output_w_q <= '0;
              if (output_h_q + 1 == descriptor_q.tile_h) begin
                state_q <= VEC_IDLE;
                done_pulse_o <= 1'b1;
              end else begin
                output_h_q <= output_h_q + 1'b1;
                residual_address_q
                  <= residual_address_q + activation_row_advance_q;
                output_address_q <= output_address_q + 32'd16;
                state_q <= VEC_SRC0_REQUEST;
              end
            end else begin
              output_w_q <= output_w_q + 1'b1;
              residual_address_q
                <= residual_address_q + activation_pixel_step_q;
              output_address_q <= output_address_q + 32'd16;
              state_q <= VEC_SRC0_REQUEST;
            end
          end
        default: begin end
      endcase

      if (post_error) begin
        state_q <= VEC_IDLE;
        error_pulse_o <= 1'b1;
        error_reason_o <= VEC_ERROR_POST | post_error_reason;
      end
    end
  end

endmodule
