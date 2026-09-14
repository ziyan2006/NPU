`timescale 1ns/1ps

module npu_conv2d_controller #(
  parameter integer METADATA_DEPTH = 8,
  parameter integer METADATA_POINTER_BITS = $clog2(METADATA_DEPTH)
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         start_valid_i,
  output logic         start_ready_o,
  input  logic [511:0] operator_desc_bits_i,
  input  logic         activation_bank_i,
  input  logic         weight_bank_i,
  input  logic [7:0]   completion_event_i,

  output logic         read_request_valid_o,
  input  logic         read_request_ready_i,
  output logic         activation_read_bank_o,
  output logic [31:0]  activation_read_address_o,
  output logic         weight_read_bank_o,
  output logic [31:0]  weight_read_address_o,
  input  logic         read_response_valid_i,
  output logic         read_response_ready_o,
  input  logic [127:0] activation_read_data_i,
  input  logic [511:0] weight_read_data_i,

  output logic         result_valid_o,
  input  logic         result_ready_i,
  output logic [31:0]  result_address_o,
  output logic [7:0]   result_lane_mask_o,
  output logic [255:0] result_data_o,

  output logic         busy_o,
  output logic         done_pulse_o,
  output logic [7:0]   event_set_o,
  output logic         error_pulse_o,
  output logic [3:0]   error_reason_o
);
  import npu_isa_pkg::*;

  localparam logic [3:0] CONV_ERROR_NONE = 4'd0;
  localparam logic [3:0] CONV_ERROR_GEOMETRY = 4'd1;
  localparam logic [3:0] CONV_ERROR_MODE = 4'd2;
  localparam logic [3:0] CONV_ERROR_CAPACITY = 4'd3;

  typedef enum logic [2:0] {
    SETUP_IDLE,
    SETUP_ROW_BYTES,
    SETUP_CAPACITY,
    SETUP_ROW_ADVANCE,
    SETUP_RUN
  } setup_state_e;

  npu_operator_desc_t descriptor;
  assign descriptor = npu_operator_desc_t'(operator_desc_bits_i);

  logic busy_q;
  logic all_requests_issued_q;
  logic activation_bank_q;
  logic weight_bank_q;
  logic [7:0] completion_event_q;
  logic [7:0] output_lane_mask_q;
  logic [7:0] last_input_lane_mask_q;
  logic [4:0] tile_h_q;
  logic [4:0] tile_w_q;
  logic [4:0] cin_blocks_q;
  logic [1:0] kh_q;
  logic [1:0] kw_q;
  logic stride_h_two_q;
  logic [6:0] local_h_q;
  logic [6:0] local_w_q;
  logic [31:0] activation_pixel_step_q;
  logic [31:0] activation_row_bytes_q;
  logic [31:0] activation_kh_advance_q;
  logic [31:0] output_column_step_q;
  logic [31:0] output_row_advance_q;

  logic [4:0] output_h_count_q;
  logic [4:0] output_w_count_q;
  logic [4:0] cin_block_count_q;
  logic [1:0] kh_count_q;
  logic [1:0] kw_count_q;
  logic [31:0] activation_pixel_base_q;
  logic [31:0] activation_address_q;
  logic [31:0] weight_address_q;

  logic [METADATA_DEPTH-1:0] metadata_first_q;
  logic [METADATA_DEPTH-1:0] metadata_last_q;
  logic [METADATA_DEPTH-1:0][7:0] metadata_input_mask_q;
  logic [METADATA_POINTER_BITS-1:0] metadata_write_pointer_q;
  logic [METADATA_POINTER_BITS-1:0] metadata_read_pointer_q;
  logic [METADATA_POINTER_BITS:0] metadata_count_q;
  logic metadata_push;
  logic metadata_pop;
  logic current_first;
  logic current_last;
  logic [7:0] current_input_mask;

  logic mac_input_ready;
  logic mac_result_valid;
  logic [255:0] mac_result_data;
  logic mac_result_fire;
  logic [4:0] result_h_count_q;
  logic [4:0] result_w_count_q;
  logic [31:0] result_address_q;

  setup_state_e setup_state_q;
  logic start_fire;
  logic start_geometry_valid;
  logic start_mode_valid;
  logic [31:0] start_cin_blocks;
  logic [31:0] start_activation_channels_bytes;
  logic [31:0] start_local_h;
  logic [31:0] start_local_w;
  logic [31:0] start_activation_pixel_step;
  logic [31:0] start_output_column_step;
  logic [7:0] start_input_mask;
  logic [7:0] start_output_mask;
  logic mul_start;
  logic mul_busy;
  logic mul_done;
  logic [31:0] mul_a;
  logic [31:0] mul_b;
  logic [63:0] mul_result;

  initial begin
    if (METADATA_DEPTH < 2
        || (METADATA_DEPTH & (METADATA_DEPTH - 1)) != 0)
      $error("METADATA_DEPTH must be a power of two >= 2");
  end

  assign start_cin_blocks = (descriptor.input_channel_count + 7) >> 3;
  assign start_activation_channels_bytes = start_cin_blocks << 4;
  assign start_local_h = (descriptor.stride_h == 2
                           ? ((descriptor.tile_h - 1) << 1)
                           : (descriptor.tile_h - 1)) + descriptor.kh;
  assign start_local_w = (descriptor.stride_w == 2
                           ? ((descriptor.tile_w - 1) << 1)
                           : (descriptor.tile_w - 1)) + descriptor.kw;
  assign start_activation_pixel_step = start_activation_channels_bytes;
  assign start_output_column_step = descriptor.stride_w == 2
    ? start_activation_channels_bytes << 1
    : start_activation_channels_bytes;

  always_comb begin
    if (descriptor.input_channel_count[2:0] == 0)
      start_input_mask = 8'hff;
    else
      start_input_mask
        = (9'b1 << descriptor.input_channel_count[2:0]) - 1'b1;
    if (descriptor.tile_cout[2:0] == 0)
      start_output_mask = 8'hff;
    else
      start_output_mask = (9'b1 << descriptor.tile_cout[2:0]) - 1'b1;
  end

  assign start_geometry_valid = descriptor.tile_h != 0
    && descriptor.tile_h <= 16 && descriptor.tile_w != 0
    && descriptor.tile_w <= 16 && descriptor.tile_cout != 0
    && descriptor.tile_cout <= 8 && descriptor.input_channel_count != 0
    && descriptor.input_channel_count <= 224;
  assign start_mode_valid = (descriptor.kh == 1 || descriptor.kh == 3)
    && (descriptor.kw == 1 || descriptor.kw == 3)
    && (descriptor.stride_h == 1 || descriptor.stride_h == 2)
    && (descriptor.stride_w == 1 || descriptor.stride_w == 2)
    && descriptor.dilation_h == 1 && descriptor.dilation_w == 1
    && descriptor.groups == 1 && descriptor.input_channel_start == 0;
  assign start_ready_o = rst_ni && !soft_reset_i && !busy_q;
  assign start_fire = start_valid_i && start_ready_o;
  assign busy_o = busy_q;

  assign current_first = cin_block_count_q == 0
    && kh_count_q == 0 && kw_count_q == 0;
  assign current_last = cin_block_count_q == cin_blocks_q - 1
    && kh_count_q == kh_q - 1 && kw_count_q == kw_q - 1;
  assign current_input_mask = cin_block_count_q == cin_blocks_q - 1
    ? last_input_lane_mask_q : 8'hff;

  assign read_request_valid_o = setup_state_q == SETUP_RUN
    && !all_requests_issued_q
    && metadata_count_q < METADATA_DEPTH;
  assign activation_read_bank_o = activation_bank_q;
  assign activation_read_address_o = activation_address_q;
  assign weight_read_bank_o = weight_bank_q;
  assign weight_read_address_o = weight_address_q;
  assign metadata_push = read_request_valid_o && read_request_ready_i;

  assign read_response_ready_o = metadata_count_q != 0
    && mac_input_ready;
  assign metadata_pop = read_response_valid_i && read_response_ready_o;

  always_comb begin
    mul_a = 32'd0;
    mul_b = 32'd0;
    case (setup_state_q)
      SETUP_ROW_BYTES: begin
        mul_a = {25'd0, local_w_q};
        mul_b = activation_pixel_step_q;
      end
      SETUP_CAPACITY: begin
        mul_a = {25'd0, local_h_q};
        mul_b = activation_row_bytes_q;
      end
      SETUP_ROW_ADVANCE: begin
        mul_a = {27'd0, tile_w_q} - 1'b1;
        mul_b = output_column_step_q;
      end
      default: begin end
    endcase
  end

  assign mul_start = busy_q && setup_state_q != SETUP_IDLE
    && setup_state_q != SETUP_RUN && !mul_busy && !mul_done;

  npu_u32_mul_iter setup_multiplier (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_i(mul_start), .operand_a_i(mul_a), .operand_b_i(mul_b),
    .busy_o(mul_busy), .done_o(mul_done), .result_o(mul_result)
  );

  npu_tensor_mac_8x8 mac (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .input_valid_i(read_response_valid_i && metadata_count_q != 0),
    .input_ready_o(mac_input_ready),
    .first_i(metadata_first_q[metadata_read_pointer_q]),
    .last_i(metadata_last_q[metadata_read_pointer_q]),
    .input_lane_mask_i(metadata_input_mask_q[metadata_read_pointer_q]),
    .output_lane_mask_i(output_lane_mask_q),
    .activation_i(activation_read_data_i), .weight_i(weight_read_data_i),
    .result_valid_o(mac_result_valid), .result_ready_i(result_ready_i),
    .result_o(mac_result_data));

  assign result_valid_o = mac_result_valid;
  assign result_data_o = mac_result_data;
  assign result_address_o = result_address_q;
  assign result_lane_mask_o = output_lane_mask_q;
  assign mac_result_fire = mac_result_valid && result_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      busy_q <= 1'b0;
      setup_state_q <= SETUP_IDLE;
      all_requests_issued_q <= 1'b0;
      metadata_write_pointer_q <= '0;
      metadata_read_pointer_q <= '0;
      metadata_count_q <= '0;
      result_address_q <= '0;
      result_h_count_q <= '0;
      result_w_count_q <= '0;
      done_pulse_o <= 1'b0;
      event_set_o <= '0;
      error_pulse_o <= 1'b0;
      error_reason_o <= CONV_ERROR_NONE;
    end else begin
      done_pulse_o <= 1'b0;
      event_set_o <= '0;
      error_pulse_o <= 1'b0;

      if (soft_reset_i) begin
        busy_q <= 1'b0;
        setup_state_q <= SETUP_IDLE;
        all_requests_issued_q <= 1'b0;
        metadata_write_pointer_q <= '0;
        metadata_read_pointer_q <= '0;
        metadata_count_q <= '0;
        result_address_q <= '0;
        result_h_count_q <= '0;
        result_w_count_q <= '0;
      end else begin
        case ({metadata_push, metadata_pop})
          2'b10: metadata_count_q <= metadata_count_q + 1'b1;
          2'b01: metadata_count_q <= metadata_count_q - 1'b1;
          default: metadata_count_q <= metadata_count_q;
        endcase
        if (metadata_push) begin
          metadata_first_q[metadata_write_pointer_q] <= current_first;
          metadata_last_q[metadata_write_pointer_q] <= current_last;
          metadata_input_mask_q[metadata_write_pointer_q]
            <= current_input_mask;
          metadata_write_pointer_q <= metadata_write_pointer_q + 1'b1;
        end
        if (metadata_pop)
          metadata_read_pointer_q <= metadata_read_pointer_q + 1'b1;

        if (start_fire) begin
          error_reason_o <= CONV_ERROR_NONE;
          if (!start_geometry_valid) begin
            error_pulse_o <= 1'b1;
            error_reason_o <= CONV_ERROR_GEOMETRY;
          end else if (!start_mode_valid) begin
            error_pulse_o <= 1'b1;
            error_reason_o <= CONV_ERROR_MODE;
          end else begin
            busy_q <= 1'b1;
            setup_state_q <= SETUP_ROW_BYTES;
            all_requests_issued_q <= 1'b0;
            activation_bank_q <= activation_bank_i;
            weight_bank_q <= weight_bank_i;
            completion_event_q <= completion_event_i;
            output_lane_mask_q <= start_output_mask;
            last_input_lane_mask_q <= start_input_mask;
            tile_h_q <= descriptor.tile_h;
            tile_w_q <= descriptor.tile_w;
            cin_blocks_q <= start_cin_blocks;
            kh_q <= descriptor.kh;
            kw_q <= descriptor.kw;
            stride_h_two_q <= descriptor.stride_h == 2;
            local_h_q <= start_local_h;
            local_w_q <= start_local_w;
            activation_pixel_step_q <= start_activation_pixel_step;
            activation_row_bytes_q <= '0;
            activation_kh_advance_q <= '0;
            output_column_step_q <= start_output_column_step;
            output_row_advance_q <= '0;
            output_h_count_q <= '0;
            output_w_count_q <= '0;
            cin_block_count_q <= '0;
            kh_count_q <= '0;
            kw_count_q <= '0;
            activation_pixel_base_q <= '0;
            activation_address_q <= '0;
            weight_address_q <= '0;
            metadata_write_pointer_q <= '0;
            metadata_read_pointer_q <= '0;
            metadata_count_q <= '0;
            result_h_count_q <= '0;
            result_w_count_q <= '0;
            result_address_q <= '0;
          end
        end

        if (mul_done) begin
          case (setup_state_q)
            SETUP_ROW_BYTES: begin
              activation_row_bytes_q <= mul_result[31:0];
              activation_kh_advance_q <= mul_result[31:0]
                - (kw_q == 3 ? activation_pixel_step_q << 1 : 32'd0);
              setup_state_q <= SETUP_CAPACITY;
            end
            SETUP_CAPACITY: begin
              if (mul_result > 64'd65536) begin
                busy_q <= 1'b0;
                setup_state_q <= SETUP_IDLE;
                error_pulse_o <= 1'b1;
                error_reason_o <= CONV_ERROR_CAPACITY;
              end else begin
                setup_state_q <= SETUP_ROW_ADVANCE;
              end
            end
            SETUP_ROW_ADVANCE: begin
              output_row_advance_q
                <= (stride_h_two_q
                    ? activation_row_bytes_q << 1
                    : activation_row_bytes_q) - mul_result[31:0];
              setup_state_q <= SETUP_RUN;
            end
            default: begin end
          endcase
        end

        if (metadata_push) begin
          weight_address_q <= weight_address_q + 64;
          if (kw_count_q != kw_q - 1) begin
            kw_count_q <= kw_count_q + 1'b1;
            activation_address_q
              <= activation_address_q + activation_pixel_step_q;
          end else if (kh_count_q != kh_q - 1) begin
            kw_count_q <= '0;
            kh_count_q <= kh_count_q + 1'b1;
            activation_address_q
              <= activation_address_q + activation_kh_advance_q;
          end else if (cin_block_count_q != cin_blocks_q - 1) begin
            kw_count_q <= '0;
            kh_count_q <= '0;
            cin_block_count_q <= cin_block_count_q + 1'b1;
            activation_address_q <= activation_pixel_base_q
              + (({27'd0, cin_block_count_q} + 1'b1) << 4);
          end else begin
            kw_count_q <= '0;
            kh_count_q <= '0;
            cin_block_count_q <= '0;
            weight_address_q <= '0;
            if (output_h_count_q == tile_h_q - 1
                && output_w_count_q == tile_w_q - 1) begin
              all_requests_issued_q <= 1'b1;
            end else if (output_w_count_q != tile_w_q - 1) begin
              output_w_count_q <= output_w_count_q + 1'b1;
              activation_pixel_base_q
                <= activation_pixel_base_q + output_column_step_q;
              activation_address_q
                <= activation_pixel_base_q + output_column_step_q;
            end else begin
              output_w_count_q <= '0;
              output_h_count_q <= output_h_count_q + 1'b1;
              activation_pixel_base_q
                <= activation_pixel_base_q + output_row_advance_q;
              activation_address_q
                <= activation_pixel_base_q + output_row_advance_q;
            end
          end
        end

        if (mac_result_fire) begin
          if (result_h_count_q == tile_h_q - 1
              && result_w_count_q == tile_w_q - 1) begin
            busy_q <= 1'b0;
            setup_state_q <= SETUP_IDLE;
            done_pulse_o <= 1'b1;
            event_set_o <= completion_event_q;
          end else begin
            result_address_q <= result_address_q + 32;
            if (result_w_count_q != tile_w_q - 1)
              result_w_count_q <= result_w_count_q + 1'b1;
            else begin
              result_w_count_q <= '0;
              result_h_count_q <= result_h_count_q + 1'b1;
            end
          end
        end
      end
    end
  end

endmodule
