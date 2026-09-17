`timescale 1ns/1ps

module audio_out_axi #(
  parameter integer AXI_CLOCK_HZ = 100_000_000,
  parameter integer AUDIO_CLOCK_HZ = 11_289_600,
  parameter integer KEY_DEBOUNCE_MS = 20,
  parameter integer CODEC_I2C_HZ = 250_000,
  parameter integer CODEC_FAST_START_CYCLES = AXI_CLOCK_HZ / 20,
  parameter integer CODEC_OUTPUT_SETTLE_CYCLES = AXI_CLOCK_HZ / 100
) (
  input  logic        aclk,
  input  logic        aresetn,
  input  logic [11:0] s_axi_audio_awaddr,
  input  logic        s_axi_audio_awvalid,
  output logic        s_axi_audio_awready,
  input  logic [31:0] s_axi_audio_wdata,
  input  logic [3:0]  s_axi_audio_wstrb,
  input  logic        s_axi_audio_wvalid,
  output logic        s_axi_audio_wready,
  output logic [1:0]  s_axi_audio_bresp,
  output logic        s_axi_audio_bvalid,
  input  logic        s_axi_audio_bready,
  input  logic [11:0] s_axi_audio_araddr,
  input  logic        s_axi_audio_arvalid,
  output logic        s_axi_audio_arready,
  output logic [31:0] s_axi_audio_rdata,
  output logic [1:0]  s_axi_audio_rresp,
  output logic        s_axi_audio_rvalid,
  input  logic        s_axi_audio_rready,

  input  logic        audio_mclk_i,
  input  logic        audio_clock_locked_i,
  input  logic        key_n_i,
  output logic        aud_mclk_o,
  output logic        aud_bclk_o,
  output logic        aud_dac_lrclk_o,
  output logic        aud_dacdat_o,
  inout  wire         aud_scl_io,
  inout  wire         aud_sda_io
);
  logic control_enable;
  logic soft_reset_pulse;
  logic codec_reinit_pulse;
  logic control_tone_enable;
  logic [31:0] tone_control;
  logic frame_wr_valid;
  logic [63:0] frame_wr_data;

  logic fifo_full;
  logic [13:0] fifo_wr_level;
  logic fifo_rd_valid;
  logic [63:0] fifo_rd_data;
  logic fifo_rd_empty;
  logic [13:0] fifo_rd_level;
  logic fifo_rd_ready;
  logic fifo_empty_audio_q;

  logic audio_resetn;
  (* async_reg = "true" *) logic [1:0] audio_reset_sync_q;
  wire audio_async_resetn = aresetn && audio_clock_locked_i;

  logic key_down;
  logic key_press;
  logic stem_target;
  logic stem_ramping;
  logic stem_ramping_audio_q;
  logic [15:0] stem_gain;
  logic signed [15:0] mixed_left;
  logic signed [15:0] mixed_right;

  logic codec_done;
  logic codec_error;
  logic codec_muted;
  logic [7:0] codec_index;
  logic codec_last_ack;
  logic codec_scl_low;
  logic codec_sda_low;
  wire codec_scl_in;
  wire codec_sda_in;

  (* async_reg = "true" *) logic enable_sync1_q;
  (* async_reg = "true" *) logic enable_sync2_q;
  (* async_reg = "true" *) logic tone_enable_sync1_q;
  (* async_reg = "true" *) logic tone_enable_sync2_q;
  (* async_reg = "true" *) logic codec_done_sync1_q;
  (* async_reg = "true" *) logic codec_done_sync2_q;
  (* async_reg = "true" *) logic [31:0] tone_control_sync1_q;
  (* async_reg = "true" *) logic [31:0] tone_control_sync2_q;
  logic frame_ready;
  logic sample_tick;
  logic signed [15:0] tone_left;
  logic signed [15:0] tone_right;
  logic signed [15:0] i2s_left;
  logic signed [15:0] i2s_right;
  logic i2s_valid;

  logic [31:0] played_frames_audio_q;
  logic [31:0] played_gray_audio;
  (* async_reg = "true" *) logic [31:0] played_gray_sync1_q;
  (* async_reg = "true" *) logic [31:0] played_gray_sync2_q;
  logic [31:0] played_frames_axi;
  logic underflow_toggle_audio_q;
  (* async_reg = "true" *) logic underflow_sync1_q;
  (* async_reg = "true" *) logic underflow_sync2_q;
  logic underflow_seen_q;
  logic underflow_pulse_axi;
  (* async_reg = "true" *) logic fifo_empty_sync1_q;
  (* async_reg = "true" *) logic fifo_empty_sync2_q;
  (* async_reg = "true" *) logic key_sync1_q;
  (* async_reg = "true" *) logic key_sync2_q;
  (* async_reg = "true" *) logic target_sync1_q;
  (* async_reg = "true" *) logic target_sync2_q;
  (* async_reg = "true" *) logic ramp_sync1_q;
  (* async_reg = "true" *) logic ramp_sync2_q;
  (* async_reg = "true" *) logic [15:0] gain_sync1_q;
  (* async_reg = "true" *) logic [15:0] gain_sync2_q;

  function automatic logic [31:0] gray_to_binary32(input logic [31:0] gray);
    integer bit_index;
    begin
      gray_to_binary32[31] = gray[31];
      for (bit_index = 30; bit_index >= 0; bit_index = bit_index - 1)
        gray_to_binary32[bit_index] = gray_to_binary32[bit_index+1]
          ^ gray[bit_index];
    end
  endfunction

  IOBUF scl_iobuf (
    .I(1'b0), .T(!codec_scl_low), .O(codec_scl_in), .IO(aud_scl_io)
  );
  IOBUF sda_iobuf (
    .I(1'b0), .T(!codec_sda_low), .O(codec_sda_in), .IO(aud_sda_io)
  );
  assign played_frames_axi = gray_to_binary32(played_gray_sync2_q);
  assign underflow_pulse_axi = underflow_sync2_q ^ underflow_seen_q;

  audio_axi_csr csr (
    .clk_i(aclk), .rst_ni(aresetn),
    .s_axi_awaddr_i(s_axi_audio_awaddr),
    .s_axi_awvalid_i(s_axi_audio_awvalid),
    .s_axi_awready_o(s_axi_audio_awready),
    .s_axi_wdata_i(s_axi_audio_wdata), .s_axi_wstrb_i(s_axi_audio_wstrb),
    .s_axi_wvalid_i(s_axi_audio_wvalid), .s_axi_wready_o(s_axi_audio_wready),
    .s_axi_bresp_o(s_axi_audio_bresp), .s_axi_bvalid_o(s_axi_audio_bvalid),
    .s_axi_bready_i(s_axi_audio_bready), .s_axi_araddr_i(s_axi_audio_araddr),
    .s_axi_arvalid_i(s_axi_audio_arvalid),
    .s_axi_arready_o(s_axi_audio_arready), .s_axi_rdata_o(s_axi_audio_rdata),
    .s_axi_rresp_o(s_axi_audio_rresp), .s_axi_rvalid_o(s_axi_audio_rvalid),
    .s_axi_rready_i(s_axi_audio_rready), .fifo_full_i(fifo_full),
    .fifo_empty_i(fifo_empty_sync2_q), .fifo_level_i(fifo_wr_level),
    .underflow_pulse_i(underflow_pulse_axi), .overflow_pulse_i(1'b0),
    .played_frames_i(played_frames_axi),
    .audio_ready_i(codec_done && audio_clock_locked_i),
    .codec_done_i(codec_done), .codec_error_i(codec_error),
    .codec_muted_i(codec_muted), .codec_index_i(codec_index),
    .codec_last_ack_i(codec_last_ack), .key_pressed_i(key_sync2_q),
    .stem_target_i(target_sync2_q), .stem_ramping_i(ramp_sync2_q),
    .stem_gain_q15_i(gain_sync2_q), .enable_o(control_enable),
    .soft_reset_pulse_o(soft_reset_pulse),
    .codec_reinit_pulse_o(codec_reinit_pulse),
    .tone_enable_o(control_tone_enable), .tone_control_o(tone_control),
    .frame_wr_valid_o(frame_wr_valid), .frame_wr_data_o(frame_wr_data)
  );

  audio_async_fifo fifo (
    .wr_clk_i(aclk), .wr_rst_ni(aresetn && !soft_reset_pulse),
    .wr_valid_i(frame_wr_valid), .wr_data_i(frame_wr_data),
    .wr_full_o(fifo_full), .wr_level_o(fifo_wr_level),
    .rd_clk_i(audio_mclk_i), .rd_rst_ni(audio_resetn),
    .rd_ready_i(fifo_rd_ready), .rd_valid_o(fifo_rd_valid),
    .rd_data_o(fifo_rd_data), .rd_empty_o(fifo_rd_empty),
    .rd_level_o(fifo_rd_level)
  );

  audio_key_debounce #(
    .CLOCK_HZ(AUDIO_CLOCK_HZ), .DEBOUNCE_MS(KEY_DEBOUNCE_MS)
  ) key_debounce (
    .clk_i(audio_mclk_i), .rst_ni(audio_resetn), .key_ni(key_n_i),
    .key_down_o(key_down), .press_event_o(key_press)
  );

  audio_stem_mixer mixer (
    .clk_i(audio_mclk_i), .rst_ni(audio_resetn), .sample_tick_i(sample_tick),
    .press_event_i(key_press),
    .mix_left_i(fifo_rd_data[15:0]), .mix_right_i(fifo_rd_data[31:16]),
    .vocal_left_i(fifo_rd_data[47:32]),
    .vocal_right_i(fifo_rd_data[63:48]), .target_stem_o(stem_target),
    .ramping_o(stem_ramping), .gain_q15_o(stem_gain),
    .out_left_o(mixed_left), .out_right_o(mixed_right)
  );

  wm8960_init #(
    .CLOCK_HZ(AXI_CLOCK_HZ), .I2C_HZ(CODEC_I2C_HZ),
    .FAST_START_CYCLES(CODEC_FAST_START_CYCLES),
    .OUTPUT_SETTLE_CYCLES(CODEC_OUTPUT_SETTLE_CYCLES)
  ) codec (
    .clk_i(aclk), .rst_ni(aresetn), .reinit_i(codec_reinit_pulse),
    .mclk_stable_i(audio_clock_locked_i), .sda_i(codec_sda_in),
    .scl_drive_low_o(codec_scl_low), .sda_drive_low_o(codec_sda_low),
    .codec_done_o(codec_done), .codec_error_o(codec_error),
    .codec_muted_o(codec_muted), .config_index_o(codec_index),
    .last_ack_o(codec_last_ack)
  );

  audio_test_tone tone (
    .clk_i(audio_mclk_i), .rst_ni(audio_resetn), .sample_tick_i(sample_tick),
    .enable_i(tone_enable_sync2_q),
    .phase_step_i({tone_control_sync2_q[15:0], 16'd0}),
    .amplitude_i(tone_control_sync2_q[31:16]),
    .left_o(tone_left), .right_o(tone_right)
  );

  assign i2s_valid = tone_enable_sync2_q
    || (enable_sync2_q && codec_done_sync2_q && fifo_rd_valid);
  assign i2s_left = tone_enable_sync2_q ? tone_left : mixed_left;
  assign i2s_right = tone_enable_sync2_q ? tone_right : mixed_right;
  assign fifo_rd_ready = sample_tick && enable_sync2_q
    && codec_done_sync2_q && !tone_enable_sync2_q && fifo_rd_valid;

  audio_i2s_tx i2s (
    .mclk_i(audio_mclk_i), .rst_ni(audio_resetn), .frame_valid_i(i2s_valid),
    .frame_left_i(i2s_left), .frame_right_i(i2s_right),
    .frame_ready_o(frame_ready), .sample_tick_o(sample_tick),
    .aud_mclk_o(aud_mclk_o), .aud_bclk_o(aud_bclk_o),
    .aud_lrclk_o(aud_dac_lrclk_o), .aud_dacdat_o(aud_dacdat_o)
  );

  always_ff @(posedge audio_mclk_i or negedge audio_async_resetn) begin
    if (!audio_async_resetn) begin
      audio_reset_sync_q <= '0;
    end else begin
      audio_reset_sync_q <= {audio_reset_sync_q[0], 1'b1};
    end
  end
  assign audio_resetn = audio_reset_sync_q[1];

  always_ff @(posedge audio_mclk_i or negedge audio_resetn) begin
    if (!audio_resetn) begin
      enable_sync1_q <= 1'b0;
      enable_sync2_q <= 1'b0;
      tone_enable_sync1_q <= 1'b0;
      tone_enable_sync2_q <= 1'b0;
      codec_done_sync1_q <= 1'b0;
      codec_done_sync2_q <= 1'b0;
      tone_control_sync1_q <= 32'h1000_05ce;
      tone_control_sync2_q <= 32'h1000_05ce;
      played_frames_audio_q <= 32'd0;
      played_gray_audio <= 32'd0;
      underflow_toggle_audio_q <= 1'b0;
      fifo_empty_audio_q <= 1'b1;
      stem_ramping_audio_q <= 1'b0;
    end else begin
      enable_sync1_q <= control_enable;
      enable_sync2_q <= enable_sync1_q;
      tone_enable_sync1_q <= control_tone_enable;
      tone_enable_sync2_q <= tone_enable_sync1_q;
      codec_done_sync1_q <= codec_done;
      codec_done_sync2_q <= codec_done_sync1_q;
      tone_control_sync1_q <= tone_control;
      tone_control_sync2_q <= tone_control_sync1_q;
      fifo_empty_audio_q <= fifo_rd_empty;
      stem_ramping_audio_q <= stem_ramping;
      if (sample_tick && enable_sync2_q && codec_done_sync2_q) begin
        played_frames_audio_q <= played_frames_audio_q + 1'b1;
        played_gray_audio <= ((played_frames_audio_q + 1'b1) >> 1)
          ^ (played_frames_audio_q + 1'b1);
        if (!tone_enable_sync2_q && !fifo_rd_valid)
          underflow_toggle_audio_q <= !underflow_toggle_audio_q;
      end
    end
  end

  always_ff @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      played_gray_sync1_q <= '0;
      played_gray_sync2_q <= '0;
      underflow_sync1_q <= 1'b0;
      underflow_sync2_q <= 1'b0;
      underflow_seen_q <= 1'b0;
      fifo_empty_sync1_q <= 1'b1;
      fifo_empty_sync2_q <= 1'b1;
      key_sync1_q <= 1'b0;
      key_sync2_q <= 1'b0;
      target_sync1_q <= 1'b0;
      target_sync2_q <= 1'b0;
      ramp_sync1_q <= 1'b0;
      ramp_sync2_q <= 1'b0;
      gain_sync1_q <= '0;
      gain_sync2_q <= '0;
    end else begin
      played_gray_sync1_q <= played_gray_audio;
      played_gray_sync2_q <= played_gray_sync1_q;
      underflow_sync1_q <= underflow_toggle_audio_q;
      underflow_sync2_q <= underflow_sync1_q;
      underflow_seen_q <= underflow_sync2_q;
      fifo_empty_sync1_q <= fifo_empty_audio_q;
      fifo_empty_sync2_q <= fifo_empty_sync1_q;
      key_sync1_q <= key_down;
      key_sync2_q <= key_sync1_q;
      target_sync1_q <= stem_target;
      target_sync2_q <= target_sync1_q;
      ramp_sync1_q <= stem_ramping_audio_q;
      ramp_sync2_q <= ramp_sync1_q;
      gain_sync1_q <= stem_gain;
      gain_sync2_q <= gain_sync1_q;
    end
  end
endmodule

`ifndef SYNTHESIS
module IOBUF (
  input  wire I,
  input  wire T,
  output wire O,
  inout  wire IO
);
  assign IO = T ? 1'bz : I;
  assign O = IO;
endmodule
`endif
