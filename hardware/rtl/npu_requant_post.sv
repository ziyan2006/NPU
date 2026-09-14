`timescale 1ns/1ps

module npu_requant_post (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          soft_reset_i,

  input  logic          input_valid_i,
  output logic          input_ready_o,
  input  logic [255:0]  accumulator_i,
  input  logic [255:0]  bias_i,
  input  logic [1023:0] quant_params_i,
  input  logic [15:0]   post_op_i,
  input  logic [7:0]    lane_mask_i,

  output logic          output_valid_o,
  input  logic          output_ready_i,
  output logic [127:0]  output_data_o,
  output logic [7:0]    output_lane_mask_o,

  input  logic          lut_write_valid_i,
  output logic          lut_write_ready_o,
  input  logic [11:0]   lut_write_address_i,
  input  logic [15:0]   lut_write_data_i,

  output logic          error_pulse_o,
  output logic [3:0]    error_reason_o
);
  import npu_isa_pkg::*;

  localparam logic [3:0] POST_ERROR_NONE = 4'd0;
  localparam logic [3:0] POST_ERROR_OPERATION = 4'd1;
  localparam logic [3:0] POST_ERROR_SHIFT = 4'd2;
  localparam logic [3:0] POST_ERROR_CLAMP = 4'd3;
  localparam logic [3:0] POST_ERROR_LANE_MASK = 4'd4;

  typedef enum logic [3:0] {
    POST_IDLE,
    POST_MULTIPLY,
    POST_MAGNITUDE,
    POST_SHIFT,
    POST_ROUND_DECIDE,
    POST_ROUND_APPLY,
    POST_CLAMP,
    POST_ACTIVATE,
    POST_TANH_CAPTURE,
    POST_OUTPUT
  } post_state_e;

  post_state_e state_q;
  logic signed [31:0] biased_q [0:7];
  logic signed [31:0] multiplier_q [0:7];
  logic        [7:0]  shift_q [0:7];
  logic signed [31:0] clamp_min_q [0:7];
  logic signed [31:0] clamp_max_q [0:7];
  logic signed [63:0] product_q;
  logic product_negative_q;
  logic [63:0] magnitude_q;
  logic [63:0] quotient_q;
  logic [63:0] remainder_q;
  logic [63:0] halfway_q;
  logic round_up_q;
  logic signed [63:0] rounded_q;
  logic signed [31:0] clamped_q;
  logic [63:0] rounded_bits;
  logic signed [15:0] current_activated;
  logic [15:0] post_op_q;
  logic [7:0] lane_mask_q;
  logic [127:0] output_data_q;
  logic [2:0] lane_q;
  logic signed [15:0] tanh_read_data_q;
  logic [11:0] tanh_read_address;
  logic input_fire;
  logic config_valid;
  logic post_op_valid;
  logic shift_valid;
  logic clamp_valid;
  logic input_clamp_valid;
  integer lane;

  (* ram_style = "block" *) logic signed [15:0] tanh_lut [0:4095];

  function automatic logic signed [31:0] leaky_relu_rne(
    input logic signed [31:0] value
  );
    logic [11:0] magnitude;
    logic [11:0] quotient;
    logic [3:0] remainder;
    begin
      if (value >= 0) begin
        leaky_relu_rne = value;
      end else begin
        magnitude = (~value + 1'b1) & 32'h0000_0fff;
        quotient = magnitude / 10;
        remainder = magnitude % 10;
        if (remainder > 5 || (remainder == 5 && quotient[0]))
          quotient = quotient + 1'b1;
        leaky_relu_rne = -$signed(quotient);
      end
    end
  endfunction

  function automatic logic signed [31:0] clamp_value(
    input logic signed [63:0] value,
    input logic signed [31:0] clamp_min,
    input logic signed [31:0] clamp_max
  );
    begin
      if (value < $signed(clamp_min))
        clamp_value = clamp_min;
      else if (value > $signed(clamp_max))
        clamp_value = clamp_max;
      else
        clamp_value = value[31:0];
    end
  endfunction

  function automatic logic signed [15:0] activate_value(
    input logic signed [31:0] value,
    input logic [15:0] post_op
  );
    begin
      case (post_op)
        NPU_POST_RELU:
          activate_value = value < 0 ? 16'sd0 : value[15:0];
        NPU_POST_LEAKY_RELU_0P1:
          activate_value = leaky_relu_rne(value);
        default:
          activate_value = value[15:0];
      endcase
    end
  endfunction

  assign input_ready_o = rst_ni && !soft_reset_i && state_q == POST_IDLE;
  assign input_fire = input_valid_i && input_ready_o;
  assign output_valid_o = state_q == POST_OUTPUT;
  assign output_data_o = output_data_q;
  assign output_lane_mask_o = lane_mask_q;
  assign lut_write_ready_o = rst_ni && !soft_reset_i
    && state_q != POST_ACTIVATE;

  always_comb begin
    post_op_valid = post_op_i == NPU_POST_NONE
      || post_op_i == NPU_POST_RELU
      || post_op_i == NPU_POST_LEAKY_RELU_0P1
      || post_op_i == NPU_POST_TANH_LUT;
    shift_valid = 1'b1;
    clamp_valid = 1'b1;
    input_clamp_valid = 1'b1;
    for (lane = 0; lane < 8; lane = lane + 1) begin
      if (lane_mask_i[lane]) begin
        if (quant_params_i[lane*128 + 32 +: 8] > 63)
          shift_valid = 1'b0;
        if ($signed(quant_params_i[lane*128 + 64 +: 32])
            > $signed(quant_params_i[lane*128 + 96 +: 32]))
          clamp_valid = 1'b0;
        if ($signed(quant_params_i[lane*128 + 64 +: 32]) < -32'sd2048
            || $signed(quant_params_i[lane*128 + 96 +: 32]) > 32'sd2047)
          input_clamp_valid = 1'b0;
      end
    end
    config_valid = post_op_valid && shift_valid && clamp_valid
      && input_clamp_valid && lane_mask_i != 0;
  end

  assign rounded_bits = (quotient_q ^ {64{product_negative_q}})
    + (product_negative_q ? !round_up_q : round_up_q);
  assign current_activated = activate_value(clamped_q, post_op_q);

  always_comb begin
    if (clamped_q < -32'sd2048)
      tanh_read_address = 12'd0;
    else if (clamped_q > 32'sd2047)
      tanh_read_address = 12'd4095;
    else
      tanh_read_address = clamped_q + 32'sd2048;
  end

  always_ff @(posedge clk_i) begin
    if (lut_write_valid_i && lut_write_ready_o)
      tanh_lut[lut_write_address_i] <= lut_write_data_i;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= POST_IDLE;
      lane_q <= 3'd0;
      error_pulse_o <= 1'b0;
      error_reason_o <= POST_ERROR_NONE;
    end else begin
      error_pulse_o <= 1'b0;
      if (soft_reset_i) begin
        state_q <= POST_IDLE;
        lane_q <= 3'd0;
      end else begin
        case (state_q)
          POST_IDLE: begin
            if (input_fire) begin
              error_reason_o <= POST_ERROR_NONE;
              if (!config_valid) begin
                error_pulse_o <= 1'b1;
                if (!post_op_valid)
                  error_reason_o <= POST_ERROR_OPERATION;
                else if (!shift_valid)
                  error_reason_o <= POST_ERROR_SHIFT;
                else if (!clamp_valid || !input_clamp_valid)
                  error_reason_o <= POST_ERROR_CLAMP;
                else
                  error_reason_o <= POST_ERROR_LANE_MASK;
              end else begin
                lane_q <= 3'd0;
                state_q <= POST_MULTIPLY;
              end
            end
          end

          POST_MULTIPLY: begin
            state_q <= POST_MAGNITUDE;
          end

          POST_MAGNITUDE: begin
            state_q <= POST_SHIFT;
          end

          POST_SHIFT: begin
            state_q <= POST_ROUND_DECIDE;
          end

          POST_ROUND_DECIDE: begin
            state_q <= POST_ROUND_APPLY;
          end

          POST_ROUND_APPLY: begin
            state_q <= POST_CLAMP;
          end

          POST_CLAMP: begin
            state_q <= POST_ACTIVATE;
          end

          POST_ACTIVATE: begin
            if (lane_mask_q[lane_q]
                && post_op_q == NPU_POST_TANH_LUT)
              state_q <= POST_TANH_CAPTURE;
            else if (lane_q == 3'd7)
              state_q <= POST_OUTPUT;
            else begin
              lane_q <= lane_q + 1'b1;
              state_q <= POST_MULTIPLY;
            end
          end

          POST_TANH_CAPTURE: begin
            if (lane_q == 3'd7)
              state_q <= POST_OUTPUT;
            else begin
              lane_q <= lane_q + 1'b1;
              state_q <= POST_MULTIPLY;
            end
          end

          POST_OUTPUT: begin
            if (output_ready_i)
              state_q <= POST_IDLE;
          end

          default: state_q <= POST_IDLE;
        endcase
      end
    end
  end

  // Datapath registers are qualified by control state; valid state is reset
  // separately so wide data flops do not need asynchronous reset muxes.
  always_ff @(posedge clk_i) begin
    if (input_fire && config_valid) begin
      post_op_q <= post_op_i;
      lane_mask_q <= lane_mask_i;
      output_data_q <= '0;
      for (lane = 0; lane < 8; lane = lane + 1) begin
        biased_q[lane]
          <= $signed(accumulator_i[lane*32 +: 32])
           + $signed(bias_i[lane*32 +: 32]);
        multiplier_q[lane] <= quant_params_i[lane*128 +: 32];
        shift_q[lane] <= quant_params_i[lane*128 + 32 +: 8];
        clamp_min_q[lane] <= quant_params_i[lane*128 + 64 +: 32];
        clamp_max_q[lane] <= quant_params_i[lane*128 + 96 +: 32];
      end
    end
    if (state_q == POST_MULTIPLY)
      product_q <= biased_q[lane_q] * multiplier_q[lane_q];
    if (state_q == POST_MAGNITUDE) begin
      product_negative_q <= product_q < 0;
      magnitude_q <= product_q < 0 ? (~product_q + 1'b1) : product_q;
    end
    if (state_q == POST_SHIFT) begin
      quotient_q <= magnitude_q >> shift_q[lane_q];
      if (shift_q[lane_q] == 0) begin
        remainder_q <= 64'd0;
        halfway_q <= 64'd1;
      end else begin
        remainder_q
          <= magnitude_q & ((64'b1 << shift_q[lane_q]) - 1'b1);
        halfway_q <= 64'b1 << (shift_q[lane_q] - 1'b1);
      end
    end
    if (state_q == POST_ROUND_DECIDE)
      round_up_q <= remainder_q > halfway_q
        || (remainder_q == halfway_q && quotient_q[0]);
    if (state_q == POST_ROUND_APPLY)
      rounded_q <= rounded_bits;
    if (state_q == POST_CLAMP)
      clamped_q <= clamp_value(
        rounded_q, clamp_min_q[lane_q], clamp_max_q[lane_q]);
    if (state_q == POST_ACTIVATE) begin
      if (lane_mask_q[lane_q]
          && post_op_q == NPU_POST_TANH_LUT)
        tanh_read_data_q <= tanh_lut[tanh_read_address];
      else if (lane_mask_q[lane_q])
        output_data_q[lane_q*16 +: 16] <= current_activated;
    end
    if (state_q == POST_TANH_CAPTURE)
      output_data_q[lane_q*16 +: 16] <= tanh_read_data_q;
  end

endmodule
