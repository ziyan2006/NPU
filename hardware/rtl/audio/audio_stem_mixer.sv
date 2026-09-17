`timescale 1ns/1ps

module audio_stem_mixer #(
  parameter integer RAMP_SAMPLES = 1323
) (
  input  logic               clk_i,
  input  logic               rst_ni,
  input  logic               sample_tick_i,
  input  logic               press_event_i,
  input  logic signed [15:0] mix_left_i,
  input  logic signed [15:0] mix_right_i,
  input  logic signed [15:0] vocal_left_i,
  input  logic signed [15:0] vocal_right_i,
  output logic               target_stem_o,
  output logic               ramping_o,
  output logic [15:0]        gain_q15_o,
  output logic signed [15:0] out_left_o,
  output logic signed [15:0] out_right_o
);
  localparam integer GAIN_MAX = 32767;
  localparam integer STEP_QUOTIENT = GAIN_MAX / RAMP_SAMPLES;
  localparam integer STEP_REMAINDER = GAIN_MAX % RAMP_SAMPLES;
  localparam integer REMAINDER_WIDTH = $clog2(RAMP_SAMPLES + 1);

  logic target_q;
  logic [15:0] gain_q;
  logic [REMAINDER_WIDTH-1:0] remainder_q;
  logic [REMAINDER_WIDTH:0] remainder_sum;
  logic [15:0] ramp_delta;
  logic target_after_press;

  logic signed [32:0] product_left;
  logic signed [32:0] product_right;
  logic signed [32:0] scaled_left;
  logic signed [32:0] scaled_right;
  logic signed [33:0] mixed_left;
  logic signed [33:0] mixed_right;

  initial begin
    if (RAMP_SAMPLES <= 0 || RAMP_SAMPLES > GAIN_MAX)
      $error("audio_stem_mixer RAMP_SAMPLES is out of range");
  end

  function automatic logic signed [15:0] saturate16(
    input logic signed [33:0] value
  );
    begin
      if (value > 34'sd32767)
        saturate16 = 16'sh7fff;
      else if (value < -34'sd32768)
        saturate16 = 16'sh8000;
      else
        saturate16 = value[15:0];
    end
  endfunction

  always_comb begin
    target_after_press = target_q ^ press_event_i;
    remainder_sum = remainder_q + STEP_REMAINDER;
    ramp_delta = STEP_QUOTIENT;
    if (remainder_sum >= RAMP_SAMPLES) begin
      remainder_sum = remainder_sum - RAMP_SAMPLES;
      ramp_delta = ramp_delta + 1'b1;
    end

    product_left = $signed(vocal_left_i) * $signed({1'b0, gain_q});
    product_right = $signed(vocal_right_i) * $signed({1'b0, gain_q});
    scaled_left = (product_left + 33'sd16384) >>> 15;
    scaled_right = (product_right + 33'sd16384) >>> 15;
    mixed_left = $signed(mix_left_i) - scaled_left;
    mixed_right = $signed(mix_right_i) - scaled_right;
    out_left_o = saturate16(mixed_left);
    out_right_o = saturate16(mixed_right);
  end

  assign target_stem_o = target_q;
  assign gain_q15_o = gain_q;
  assign ramping_o = target_q ? (gain_q != GAIN_MAX) : (gain_q != 0);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      target_q <= 1'b0;
      gain_q <= 16'd0;
      remainder_q <= '0;
    end else begin
      if (press_event_i) begin
        target_q <= !target_q;
        remainder_q <= '0;
      end
      if (sample_tick_i) begin
        if (target_after_press && gain_q != GAIN_MAX) begin
          if (GAIN_MAX - gain_q <= ramp_delta) begin
            gain_q <= GAIN_MAX;
            remainder_q <= '0;
          end else begin
            gain_q <= gain_q + ramp_delta;
            remainder_q <= remainder_sum[REMAINDER_WIDTH-1:0];
          end
        end else if (!target_after_press && gain_q != 0) begin
          if (gain_q <= ramp_delta) begin
            gain_q <= 16'd0;
            remainder_q <= '0;
          end else begin
            gain_q <= gain_q - ramp_delta;
            remainder_q <= remainder_sum[REMAINDER_WIDTH-1:0];
          end
        end
      end
    end
  end
endmodule
