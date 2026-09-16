`timescale 1ns/1ps

module npu_conv2d_pipeline (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          soft_reset_i,

  input  logic          start_valid_i,
  output logic          start_ready_o,
  input  logic [511:0]  operator_desc_bits_i,
  input  logic          activation_bank_i,
  input  logic          weight_bank_i,
  input  logic [7:0]    completion_event_i,

  output logic          activation_request_valid_o,
  input  logic          activation_request_ready_i,
  output logic          activation_read_bank_o,
  output logic [31:0]   activation_read_address_o,
  input  logic          activation_response_valid_i,
  output logic          activation_response_ready_o,
  input  logic [127:0]  activation_read_data_i,

  output logic          weight_request_valid_o,
  input  logic          weight_request_ready_i,
  output logic          weight_read_bank_o,
  output logic [31:0]   weight_read_address_o,
  input  logic          weight_response_valid_i,
  output logic          weight_response_ready_o,
  input  logic [511:0]  weight_read_data_i,

  output logic          result_valid_o,
  input  logic          result_ready_i,
  output logic [31:0]   result_address_o,
  output logic [7:0]    result_lane_mask_o,
  output logic [127:0]  result_data_o,

  input  logic          lut_write_valid_i,
  output logic          lut_write_ready_o,
  input  logic [11:0]   lut_write_address_i,
  input  logic [15:0]   lut_write_data_i,

  output logic          busy_o,
  output logic          done_pulse_o,
  output logic [7:0]    event_set_o,
  output logic          error_pulse_o,
  output logic [3:0]    error_reason_o
);
  import npu_isa_pkg::*;

  typedef enum logic [3:0] {
    PARAM_IDLE,
    PARAM_VALIDATE,
    PARAM_BIAS_REQUEST,
    PARAM_BIAS_RESPONSE,
    PARAM_QUANT0_REQUEST,
    PARAM_QUANT0_RESPONSE,
    PARAM_QUANT1_REQUEST,
    PARAM_QUANT1_RESPONSE,
    PARAM_READY
  } param_state_e;

  npu_operator_desc_t descriptor_q;
  param_state_e param_state_q;
  npu_operator_desc_t start_descriptor;
  logic [9:0] start_cin_blocks;
  logic [9:0] start_kernel_rows;
  logic [31:0] bias_address_q;
  logic [31:0] quant0_address_q;
  logic [31:0] quant1_address_q;
  logic weight_bank_q;
  logic [255:0] bias_q;
  logic [1023:0] quant_params_q;

  logic controller_start_ready;
  logic controller_busy;
  logic controller_done;
  logic [7:0] controller_event;
  logic controller_error;
  logic [3:0] controller_error_reason;
  logic controller_soft_reset;
  logic controller_request_valid;
  logic controller_request_ready;
  logic controller_activation_bank;
  logic [31:0] controller_activation_address;
  logic controller_weight_bank;
  logic [31:0] controller_weight_address;
  logic controller_response_valid;
  logic controller_response_ready;
  logic controller_result_valid;
  logic controller_result_ready;
  logic [31:0] controller_result_address;
  logic [7:0] controller_result_mask;
  logic [255:0] controller_result_data;

  logic activation_sent_q;
  logic weight_sent_q;
  logic activation_fire;
  logic weight_fire;
  logic param_request_valid;
  logic param_request_fire;
  logic param_response_state;

  logic post_input_fire;
  logic post_output_valid;
  logic post_output_ready;
  logic [127:0] post_output_data;
  logic [7:0] post_output_mask;
  logic post_error;
  logic [3:0] post_error_reason;
  logic result_buffer_valid_q;
  logic [255:0] result_buffer_data_q;
  logic [7:0] result_buffer_mask_q;
  logic [31:0] result_buffer_address_q;
  logic controller_result_fire;
  logic [31:0] result_address_q;
  logic completion_pending_q;
  logic [7:0] completion_event_q;
  logic post_input_ready;

  assign start_descriptor = npu_operator_desc_t'(operator_desc_bits_i);
  assign start_cin_blocks
    = (start_descriptor.input_channel_count + 7) >> 3;
  always_comb begin
    start_kernel_rows = start_cin_blocks;
    if (start_descriptor.kh == 3)
      start_kernel_rows = start_kernel_rows + (start_kernel_rows << 1);
    if (start_descriptor.kw == 3)
      start_kernel_rows = start_kernel_rows + (start_kernel_rows << 1);
  end

  assign start_ready_o = controller_start_ready
    && param_state_q == PARAM_IDLE && !completion_pending_q;
  assign busy_o = controller_busy || param_state_q != PARAM_IDLE
    || completion_pending_q;
  assign controller_soft_reset = soft_reset_i || post_error;

  assign param_request_valid = param_state_q == PARAM_BIAS_REQUEST
    || param_state_q == PARAM_QUANT0_REQUEST
    || param_state_q == PARAM_QUANT1_REQUEST;
  assign param_request_fire = param_request_valid && weight_request_ready_i;
  assign param_response_state = param_state_q == PARAM_BIAS_RESPONSE
    || param_state_q == PARAM_QUANT0_RESPONSE
    || param_state_q == PARAM_QUANT1_RESPONSE;

  assign activation_request_valid_o = controller_request_valid
    && param_state_q == PARAM_READY && !activation_sent_q;
  assign activation_fire = activation_request_valid_o
    && activation_request_ready_i;
  assign activation_read_bank_o = controller_activation_bank;
  assign activation_read_address_o = controller_activation_address;

  assign weight_request_valid_o = param_request_valid
    || (controller_request_valid && param_state_q == PARAM_READY
        && !weight_sent_q);
  assign weight_fire = controller_request_valid
    && param_state_q == PARAM_READY && !weight_sent_q
    && weight_request_ready_i;
  assign weight_read_bank_o = param_request_valid
    ? weight_bank_q : controller_weight_bank;
  always_comb begin
    case (param_state_q)
      PARAM_BIAS_REQUEST: weight_read_address_o = bias_address_q;
      PARAM_QUANT0_REQUEST: weight_read_address_o = quant0_address_q;
      PARAM_QUANT1_REQUEST: weight_read_address_o = quant1_address_q;
      default: weight_read_address_o = controller_weight_address;
    endcase
  end

  assign controller_request_ready = param_state_q == PARAM_READY
    && (activation_sent_q || activation_fire)
    && (weight_sent_q || weight_fire);
  assign controller_response_valid = activation_response_valid_i
    && weight_response_valid_i;
  assign activation_response_ready_o = param_state_q == PARAM_READY
    && controller_response_ready && weight_response_valid_i;
  assign weight_response_ready_o = param_response_state
    || (param_state_q == PARAM_READY && controller_response_ready
        && activation_response_valid_i);

  // A non-fall-through result buffer cuts the combinational back-pressure
  // path from the slow post unit through MAC/controller into Scratchpad BRAM
  // address selection.  Post latency dominates, so the extra initial cycle
  // has no steady-state throughput cost.
  assign controller_result_ready = !completion_pending_q
    && !result_buffer_valid_q;
  assign controller_result_fire = controller_result_valid
    && controller_result_ready;
  assign post_input_fire = result_buffer_valid_q && post_input_ready;
  assign post_output_ready = result_ready_i;
  assign result_valid_o = post_output_valid;
  assign result_data_o = post_output_data;
  assign result_lane_mask_o = post_output_mask;
  assign result_address_o = result_address_q;

  npu_requant_post post (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .input_valid_i(result_buffer_valid_q),
    .input_ready_o(post_input_ready),
    .accumulator_i(result_buffer_data_q), .bias_i(bias_q),
    .quant_params_i(quant_params_q),
    .post_op_i(descriptor_q.post_op_id),
    .lane_mask_i(result_buffer_mask_q),
    .output_valid_o(post_output_valid),
    .output_ready_i(post_output_ready), .output_data_o(post_output_data),
    .output_lane_mask_o(post_output_mask),
    .lut_write_valid_i(lut_write_valid_i),
    .lut_write_ready_o(lut_write_ready_o),
    .lut_write_address_i(lut_write_address_i),
    .lut_write_data_i(lut_write_data_i),
    .error_pulse_o(post_error), .error_reason_o(post_error_reason));

  npu_conv2d_controller controller (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(controller_soft_reset),
    .start_valid_i(start_valid_i && start_ready_o),
    .start_ready_o(controller_start_ready),
    .operator_desc_bits_i(operator_desc_bits_i),
    .activation_bank_i(activation_bank_i), .weight_bank_i(weight_bank_i),
    .completion_event_i(completion_event_i),
    .read_request_valid_o(controller_request_valid),
    .read_request_ready_i(controller_request_ready),
    .activation_read_bank_o(controller_activation_bank),
    .activation_read_address_o(controller_activation_address),
    .weight_read_bank_o(controller_weight_bank),
    .weight_read_address_o(controller_weight_address),
    .read_response_valid_i(controller_response_valid),
    .read_response_ready_o(controller_response_ready),
    .activation_read_data_i(activation_read_data_i),
    .weight_read_data_i(weight_read_data_i),
    .result_valid_o(controller_result_valid),
    .result_ready_i(controller_result_ready),
    .result_address_o(controller_result_address),
    .result_lane_mask_o(controller_result_mask),
    .result_data_o(controller_result_data),
    .busy_o(controller_busy), .done_pulse_o(controller_done),
    .event_set_o(controller_event), .error_pulse_o(controller_error),
    .error_reason_o(controller_error_reason));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      param_state_q <= PARAM_IDLE;
      activation_sent_q <= 1'b0;
      weight_sent_q <= 1'b0;
      completion_pending_q <= 1'b0;
      result_buffer_valid_q <= 1'b0;
      done_pulse_o <= 1'b0;
      event_set_o <= '0;
      error_pulse_o <= 1'b0;
      error_reason_o <= '0;
    end else if (soft_reset_i) begin
      param_state_q <= PARAM_IDLE;
      activation_sent_q <= 1'b0;
      weight_sent_q <= 1'b0;
      completion_pending_q <= 1'b0;
      result_buffer_valid_q <= 1'b0;
      done_pulse_o <= 1'b0;
      event_set_o <= '0;
      error_pulse_o <= 1'b0;
    end else begin
      done_pulse_o <= 1'b0;
      event_set_o <= '0;
      error_pulse_o <= controller_error || post_error;
      if (controller_error)
        error_reason_o <= controller_error_reason;
      else if (post_error)
        error_reason_o <= 4'h8 | post_error_reason;

      if (start_valid_i && start_ready_o) begin
        descriptor_q <= npu_operator_desc_t'(operator_desc_bits_i);
        bias_address_q <= {start_kernel_rows, 6'b0};
        quant0_address_q <= {start_kernel_rows, 6'b0} + 32'd64;
        quant1_address_q <= {start_kernel_rows, 6'b0} + 32'd128;
        weight_bank_q <= weight_bank_i;
        param_state_q <= PARAM_VALIDATE;
      end
      case (param_state_q)
        PARAM_VALIDATE: begin
          if (controller_error)
            param_state_q <= PARAM_IDLE;
          else if (controller_busy)
            param_state_q <= PARAM_BIAS_REQUEST;
        end
        PARAM_BIAS_REQUEST:
          if (param_request_fire) param_state_q <= PARAM_BIAS_RESPONSE;
        PARAM_BIAS_RESPONSE:
          if (weight_response_valid_i) begin
            bias_q <= weight_read_data_i[255:0];
            param_state_q <= PARAM_QUANT0_REQUEST;
          end
        PARAM_QUANT0_REQUEST:
          if (param_request_fire) param_state_q <= PARAM_QUANT0_RESPONSE;
        PARAM_QUANT0_RESPONSE:
          if (weight_response_valid_i) begin
            quant_params_q[511:0] <= weight_read_data_i;
            if (descriptor_q.tile_cout <= 4)
              param_state_q <= PARAM_READY;
            else
              param_state_q <= PARAM_QUANT1_REQUEST;
          end
        PARAM_QUANT1_REQUEST:
          if (param_request_fire) param_state_q <= PARAM_QUANT1_RESPONSE;
        PARAM_QUANT1_RESPONSE:
          if (weight_response_valid_i) begin
            quant_params_q[1023:512] <= weight_read_data_i;
            param_state_q <= PARAM_READY;
          end
        default: begin end
      endcase

      if (controller_request_valid && controller_request_ready) begin
        activation_sent_q <= 1'b0;
        weight_sent_q <= 1'b0;
      end else begin
        if (activation_fire) activation_sent_q <= 1'b1;
        if (weight_fire) weight_sent_q <= 1'b1;
      end

      if (controller_result_fire) begin
        result_buffer_valid_q <= 1'b1;
        result_buffer_data_q <= controller_result_data;
        result_buffer_mask_q <= controller_result_mask;
        result_buffer_address_q <= controller_result_address;
      end
      if (post_input_fire) begin
        result_buffer_valid_q <= 1'b0;
        result_address_q <= result_buffer_address_q >> 1;
      end
      if (controller_done) begin
        completion_pending_q <= 1'b1;
        completion_event_q <= controller_event;
      end
      if (completion_pending_q && !result_buffer_valid_q
          && post_output_valid && result_ready_i) begin
        completion_pending_q <= 1'b0;
        param_state_q <= PARAM_IDLE;
        done_pulse_o <= 1'b1;
        event_set_o <= completion_event_q;
      end
      if (post_error) begin
        result_buffer_valid_q <= 1'b0;
        completion_pending_q <= 1'b0;
        param_state_q <= PARAM_IDLE;
      end
    end
  end

endmodule
