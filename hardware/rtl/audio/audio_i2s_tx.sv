`timescale 1ns/1ps

module audio_i2s_tx (
  input  logic               mclk_i,
  input  logic               rst_ni,
  input  logic               frame_valid_i,
  input  logic signed [15:0] frame_left_i,
  input  logic signed [15:0] frame_right_i,
  output logic               frame_ready_o,
  output logic               sample_tick_o,
  output logic               aud_mclk_o,
  output logic               aud_bclk_o,
  output logic               aud_lrclk_o,
  output logic               aud_dacdat_o
);
  logic bclk_q;
  logic divider_q;
  logic [5:0] bit_index_q;
  logic [23:0] left_sample_q;
  logic [23:0] right_sample_q;
  logic [5:0] next_bit;
  logic [23:0] next_left;
  logic [23:0] next_right;

  assign aud_mclk_o = mclk_i;
  assign aud_bclk_o = bclk_q;

  always_comb begin
    next_bit = bit_index_q + 1'b1;
    next_left = (next_bit == 0 && frame_valid_i)
      ? {frame_left_i, 8'h00} : left_sample_q;
    next_right = (next_bit == 0 && frame_valid_i)
      ? {frame_right_i, 8'h00} : right_sample_q;
  end

  always_ff @(posedge mclk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      bclk_q <= 1'b0;
      divider_q <= 1'b0;
      bit_index_q <= 6'd63;
      left_sample_q <= '0;
      right_sample_q <= '0;
      frame_ready_o <= 1'b0;
      sample_tick_o <= 1'b0;
      aud_lrclk_o <= 1'b1;
      aud_dacdat_o <= 1'b0;
    end else begin
      frame_ready_o <= 1'b0;
      sample_tick_o <= 1'b0;
      divider_q <= !divider_q;
      if (divider_q) begin
        bclk_q <= !bclk_q;
        if (bclk_q) begin
          bit_index_q <= next_bit;
          aud_lrclk_o <= next_bit[5];
          if (next_bit == 0) begin
            left_sample_q <= frame_valid_i ? {frame_left_i, 8'h00} : 24'd0;
            right_sample_q <= frame_valid_i ? {frame_right_i, 8'h00} : 24'd0;
            frame_ready_o <= 1'b1;
            sample_tick_o <= 1'b1;
            aud_dacdat_o <= 1'b0;
          end else if (next_bit <= 24) begin
            aud_dacdat_o <= next_left[24-next_bit];
          end else if (next_bit == 32) begin
            aud_dacdat_o <= 1'b0;
          end else if (next_bit >= 33 && next_bit <= 56) begin
            aud_dacdat_o <= next_right[56-next_bit];
          end else begin
            aud_dacdat_o <= 1'b0;
          end
        end
      end
    end
  end
endmodule
