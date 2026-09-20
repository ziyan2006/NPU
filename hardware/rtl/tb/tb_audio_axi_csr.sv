`timescale 1ns/1ps

module tb_audio_axi_csr;
  import audio_regs_pkg::*;

  logic clk = 0;
  logic rst_n = 0;
  logic [11:0] awaddr = 0;
  logic awvalid = 0;
  logic awready;
  logic [31:0] wdata = 0;
  logic [3:0] wstrb = 0;
  logic wvalid = 0;
  logic wready;
  logic [1:0] bresp;
  logic bvalid;
  logic bready = 1;
  logic [11:0] araddr = 0;
  logic arvalid = 0;
  logic arready;
  logic [31:0] rdata;
  logic [1:0] rresp;
  logic rvalid;
  logic rready = 0;

  logic fifo_full = 0;
  logic fifo_empty = 1;
  logic [14:0] fifo_level = 0;
  logic underflow_pulse = 0;
  logic overflow_pulse = 0;
  logic [31:0] played_frames = 32'h1234_5678;
  logic audio_ready = 1;
  logic codec_done = 1;
  logic codec_error = 0;
  logic codec_muted = 0;
  logic [7:0] codec_index = 8'h12;
  logic codec_last_ack = 1;
  logic key_pressed = 0;
  logic stem_target = 0;
  logic stem_ramping = 0;
  logic [15:0] stem_gain = 0;

  logic enable;
  logic soft_reset_pulse;
  logic codec_reinit_pulse;
  logic tone_enable;
  logic [31:0] tone_control;
  logic frame_wr_valid;
  logic [63:0] frame_wr_data;
  integer accepted_frames = 0;
  logic [63:0] last_frame = 0;

  always #5 clk = ~clk;

  audio_axi_csr dut (
    .clk_i(clk), .rst_ni(rst_n),
    .s_axi_awaddr_i(awaddr), .s_axi_awvalid_i(awvalid),
    .s_axi_awready_o(awready), .s_axi_wdata_i(wdata),
    .s_axi_wstrb_i(wstrb), .s_axi_wvalid_i(wvalid),
    .s_axi_wready_o(wready), .s_axi_bresp_o(bresp),
    .s_axi_bvalid_o(bvalid), .s_axi_bready_i(bready),
    .s_axi_araddr_i(araddr), .s_axi_arvalid_i(arvalid),
    .s_axi_arready_o(arready), .s_axi_rdata_o(rdata),
    .s_axi_rresp_o(rresp), .s_axi_rvalid_o(rvalid),
    .s_axi_rready_i(rready), .fifo_full_i(fifo_full),
    .fifo_empty_i(fifo_empty), .fifo_level_i(fifo_level),
    .underflow_pulse_i(underflow_pulse),
    .overflow_pulse_i(overflow_pulse), .played_frames_i(played_frames),
    .audio_ready_i(audio_ready), .codec_done_i(codec_done),
    .codec_error_i(codec_error), .codec_muted_i(codec_muted),
    .codec_index_i(codec_index), .codec_last_ack_i(codec_last_ack),
    .key_pressed_i(key_pressed), .stem_target_i(stem_target),
    .stem_ramping_i(stem_ramping), .stem_gain_q15_i(stem_gain),
    .enable_o(enable), .soft_reset_pulse_o(soft_reset_pulse),
    .codec_reinit_pulse_o(codec_reinit_pulse),
    .tone_enable_o(tone_enable), .tone_control_o(tone_control),
    .frame_wr_valid_o(frame_wr_valid), .frame_wr_data_o(frame_wr_data)
  );

  always_ff @(posedge clk) begin
    if (frame_wr_valid) begin
      accepted_frames <= accepted_frames + 1;
      last_frame <= frame_wr_data;
    end
  end

  task automatic axi_write(
    input logic [11:0] address,
    input logic [31:0] data,
    input logic [3:0] strobe,
    input integer order,
    input logic [1:0] expected_response
  );
    begin
      if (order == 0) begin
        @(negedge clk); awaddr = address; awvalid = 1;
        do @(posedge clk); while (!awready);
        @(negedge clk); awvalid = 0;
        repeat (2) @(posedge clk);
        @(negedge clk); wdata = data; wstrb = strobe; wvalid = 1;
        do @(posedge clk); while (!wready);
        @(negedge clk); wvalid = 0;
      end else if (order == 1) begin
        @(negedge clk); wdata = data; wstrb = strobe; wvalid = 1;
        do @(posedge clk); while (!wready);
        @(negedge clk); wvalid = 0;
        repeat (2) @(posedge clk);
        @(negedge clk); awaddr = address; awvalid = 1;
        do @(posedge clk); while (!awready);
        @(negedge clk); awvalid = 0;
      end else begin
        @(negedge clk); awaddr = address; awvalid = 1;
        wdata = data; wstrb = strobe; wvalid = 1;
        do @(posedge clk); while (!(awready && wready));
        @(negedge clk); awvalid = 0; wvalid = 0;
      end
      wait (bvalid);
      if (bresp != expected_response)
        $fatal(1, "write response addr=%h got=%b expected=%b",
               address, bresp, expected_response);
      @(posedge clk);
    end
  endtask

  task automatic axi_read(
    input logic [11:0] address,
    input logic [31:0] expected_data,
    input logic [1:0] expected_response
  );
    begin
      @(negedge clk); araddr = address; arvalid = 1;
      do @(posedge clk); while (!arready);
      @(negedge clk); arvalid = 0;
      repeat (2) @(posedge clk);
      if (!rvalid || rdata != expected_data || rresp != expected_response)
        $fatal(1, "read mismatch addr=%h data=%h/%h resp=%b/%b",
               address, rdata, expected_data, rresp, expected_response);
      @(negedge clk); rready = 1;
      @(posedge clk);
      @(negedge clk); rready = 0;
    end
  endtask

  task automatic pulse_underflow;
    begin
      @(negedge clk); underflow_pulse = 1;
      @(negedge clk); underflow_pulse = 0;
    end
  endtask

  initial begin
    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    axi_read(AUDIO_REG_IP_ID, AUDIO_IP_ID, 2'b00);
    axi_read(AUDIO_REG_VERSION, AUDIO_VERSION, 2'b00);
    axi_read(AUDIO_REG_FIFO_CAPACITY, 32'd16384, 2'b00);
    axi_read(AUDIO_REG_STATUS, 32'h0000_0025, 2'b00);

    axi_write(AUDIO_REG_CONTROL, 32'h0000_000f, 4'h1, 0, 2'b00);
    if (!enable || !tone_enable)
      $fatal(1, "level control bits did not latch");
    repeat (2) @(posedge clk);
    if (soft_reset_pulse || codec_reinit_pulse)
      $fatal(1, "command control bits were not pulses");

    axi_write(AUDIO_REG_MIX_FRAME, 32'h2222_1111, 4'hf, 1, 2'b00);
    if (accepted_frames != 0)
      $fatal(1, "MIX_FRAME committed a partial FIFO entry");
    axi_write(AUDIO_REG_VOCAL_FRAME, 32'h4444_3333, 4'hf, 2, 2'b00);
    repeat (2) @(posedge clk);
    if (accepted_frames != 1 || last_frame != 64'h4444_3333_2222_1111)
      $fatal(1, "atomic frame commit mismatch got=%h", last_frame);

    axi_write(AUDIO_REG_MIX_FRAME, 32'haabb_ccdd, 4'b0101, 0, 2'b00);
    axi_write(AUDIO_REG_VOCAL_FRAME, 32'h5566_7788, 4'b1010, 1, 2'b00);
    repeat (2) @(posedge clk);
    if (last_frame != 64'h5544_7733_22bb_11dd)
      $fatal(1, "WSTRB staging mismatch got=%h", last_frame);

    fifo_full = 1;
    axi_write(AUDIO_REG_VOCAL_FRAME, 32'h9999_8888, 4'hf, 2, 2'b00);
    repeat (2) @(posedge clk);
    if (accepted_frames != 2)
      $fatal(1, "full FIFO accepted a frame");
    axi_read(AUDIO_REG_OVERFLOW_COUNT, 32'd1, 2'b00);
    fifo_full = 0;

    pulse_underflow();
    pulse_underflow();
    axi_read(AUDIO_REG_UNDERFLOW_COUNT, 32'd2, 2'b00);
    axi_write(AUDIO_REG_UNDERFLOW_COUNT, 32'hffff_ffff, 4'hf, 2, 2'b00);
    axi_write(AUDIO_REG_OVERFLOW_COUNT, 32'hffff_ffff, 4'hf, 2, 2'b00);
    axi_read(AUDIO_REG_UNDERFLOW_COUNT, 32'd0, 2'b00);
    axi_read(AUDIO_REG_OVERFLOW_COUNT, 32'd0, 2'b00);

    fifo_level = 14'd123;
    fifo_empty = 0;
    stem_gain = 16'h4567;
    key_pressed = 1;
    stem_ramping = 1;
    stem_target = 1;
    axi_read(AUDIO_REG_FIFO_LEVEL, 32'd123, 2'b00);
    axi_read(AUDIO_REG_PLAYED_FRAMES, 32'h1234_5678, 2'b00);
    axi_read(AUDIO_REG_STEM_STATE, 32'h4567_0007, 2'b00);
    axi_read(AUDIO_REG_CODEC_STATUS, 32'h0000_0112, 2'b00);

    axi_write(AUDIO_REG_TONE_CONTROL, 32'h1234_abcd, 4'hf, 2, 2'b00);
    axi_read(AUDIO_REG_TONE_CONTROL, 32'h1234_abcd, 2'b00);
    axi_write(12'h038, 32'hdead_beef, 4'hf, 2, 2'b10);
    axi_read(12'h038, 32'd0, 2'b10);

    $display("audio_axi_csr: PASS");
    $finish;
  end
endmodule
