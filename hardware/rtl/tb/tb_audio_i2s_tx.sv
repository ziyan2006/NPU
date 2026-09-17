`timescale 1ns/1ps

module tb_audio_i2s_tx;
  logic mclk = 0;
  logic rst_n = 0;
  logic frame_valid = 0;
  logic signed [15:0] frame_left = 16'sh1234;
  logic signed [15:0] frame_right = -16'sh0124;
  logic frame_ready;
  logic sample_tick;
  logic aud_mclk;
  logic aud_bclk;
  logic aud_lrclk;
  logic aud_dacdat;
  logic tone_enable = 0;
  logic [31:0] tone_phase_step = 32'h4000_0000;
  logic [15:0] tone_amplitude = 16'd4096;
  logic signed [15:0] tone_left;
  logic signed [15:0] tone_right;
  logic [63:0] captured_bits;
  logic [63:0] captured_lrclk;
  integer captured_count = 0;
  integer mclk_rises = 0;
  integer bclk_rises = 0;
  integer sample_ticks = 0;
  integer index;
  logic capture_enable = 0;
  logic [23:0] expected_left = 24'h123400;
  logic [23:0] expected_right = 24'hfedc00;

  always #2 mclk = ~mclk;

  audio_i2s_tx tx (
    .mclk_i(mclk), .rst_ni(rst_n), .frame_valid_i(frame_valid),
    .frame_left_i(frame_left), .frame_right_i(frame_right),
    .frame_ready_o(frame_ready), .sample_tick_o(sample_tick),
    .aud_mclk_o(aud_mclk), .aud_bclk_o(aud_bclk),
    .aud_lrclk_o(aud_lrclk), .aud_dacdat_o(aud_dacdat)
  );

  audio_test_tone tone (
    .clk_i(mclk), .rst_ni(rst_n), .sample_tick_i(sample_tick),
    .enable_i(tone_enable), .phase_step_i(tone_phase_step),
    .amplitude_i(tone_amplitude), .left_o(tone_left), .right_o(tone_right)
  );

  always @(posedge mclk)
    if (rst_n)
      mclk_rises = mclk_rises + 1;

  always @(posedge aud_bclk) begin
    if (rst_n) begin
      bclk_rises = bclk_rises + 1;
      if (capture_enable && captured_count < 64) begin
        captured_bits[captured_count] = aud_dacdat;
        captured_lrclk[captured_count] = aud_lrclk;
        captured_count = captured_count + 1;
      end
    end
  end

  always @(posedge mclk)
    if (sample_tick)
      sample_ticks = sample_ticks + 1;

  initial begin
    repeat (5) @(posedge mclk);
    rst_n = 1;
    frame_valid = 1;
    wait (sample_tick);
    capture_enable = 1;
    @(negedge mclk); frame_valid = 0;
    wait (captured_count == 64);
    #1;

    if (aud_mclk !== mclk)
      $fatal(1, "MCLK output is not forwarded");
    if (bclk_rises < 64 || mclk_rises < bclk_rises * 4 - 3)
      $fatal(1, "MCLK:BCLK ratio mismatch mclk=%0d bclk=%0d",
             mclk_rises, bclk_rises);
    for (index = 0; index < 32; index = index + 1)
      if (captured_lrclk[index] !== 1'b0)
        $fatal(1, "LRCLK left slot mismatch bit=%0d", index);
    for (index = 32; index < 64; index = index + 1)
      if (captured_lrclk[index] !== 1'b1)
        $fatal(1, "LRCLK right slot mismatch bit=%0d", index);

    if (captured_bits[0] !== 1'b0 || captured_bits[32] !== 1'b0)
      $fatal(1, "I2S one-bit delay is missing");
    for (index = 0; index < 24; index = index + 1) begin
      if (captured_bits[index+1] !== expected_left[23-index])
        $fatal(1, "left sample bit mismatch index=%0d", index);
      if (captured_bits[index+33] !== expected_right[23-index])
        $fatal(1, "right sample bit mismatch index=%0d", index);
    end
    for (index = 25; index < 32; index = index + 1)
      if (captured_bits[index] !== 1'b0 || captured_bits[index+32] !== 1'b0)
        $fatal(1, "slot padding is not zero index=%0d", index);

    // No second frame is queued: clocks continue and one zero frame follows.
    captured_count = 0;
    captured_bits = '0;
    captured_lrclk = '0;
    wait (captured_count == 64);
    #1;
    if (captured_bits !== 64'd0 || sample_ticks < 2)
      $fatal(1, "underflow frame was not zero");

    // Quarter-cycle phase steps visit 0, +A, 0, -A and repeat.
    tone_enable = 1;
    force tx.sample_tick_o = 0;
    repeat (2) @(posedge mclk);
    if (tone_left != 0 || tone_right != 0)
      $fatal(1, "tone did not start at zero");
    for (index = 0; index < 4; index = index + 1) begin
      force tx.sample_tick_o = 1;
      @(posedge mclk);
      force tx.sample_tick_o = 0;
      @(posedge mclk);
      if (tone_left != tone_right)
        $fatal(1, "tone channels differ");
    end
    if (tone_left != 0)
      $fatal(1, "tone phase did not repeat after four ticks got=%0d", tone_left);
    release tx.sample_tick_o;

    $display("audio_i2s_tx: PASS");
    $finish;
  end
endmodule
