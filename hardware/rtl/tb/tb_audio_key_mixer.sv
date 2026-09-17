`timescale 1ns/1ps

module tb_audio_key_mixer;
  logic clk = 0;
  logic rst_n = 0;
  logic key_n = 1;
  logic key_down;
  logic press_event;
  logic sample_tick = 0;
  logic signed [15:0] mix_left = 0;
  logic signed [15:0] mix_right = 0;
  logic signed [15:0] vocal_left = 0;
  logic signed [15:0] vocal_right = 0;
  logic target;
  logic ramping;
  logic [15:0] gain;
  logic signed [15:0] out_left;
  logic signed [15:0] out_right;
  integer press_count = 0;
  integer tick;
  logic [15:0] reversal_gain;

  always #5 clk = ~clk;

  audio_key_debounce #(
    .CLOCK_HZ(1000),
    .DEBOUNCE_MS(20)
  ) debounce (
    .clk_i(clk), .rst_ni(rst_n), .key_ni(key_n),
    .key_down_o(key_down), .press_event_o(press_event)
  );

  audio_stem_mixer #(
    .RAMP_SAMPLES(1323)
  ) mixer (
    .clk_i(clk), .rst_ni(rst_n), .sample_tick_i(sample_tick),
    .press_event_i(press_event), .mix_left_i(mix_left),
    .mix_right_i(mix_right), .vocal_left_i(vocal_left),
    .vocal_right_i(vocal_right), .target_stem_o(target),
    .ramping_o(ramping), .gain_q15_o(gain),
    .out_left_o(out_left), .out_right_o(out_right)
  );

  always_ff @(posedge clk)
    if (press_event)
      press_count <= press_count + 1;

  task automatic wait_cycles(input integer count);
    repeat (count) @(posedge clk);
  endtask

  task automatic sample_once;
    begin
      @(negedge clk); sample_tick = 1;
      @(posedge clk);
      @(negedge clk); sample_tick = 0;
    end
  endtask

  task automatic clean_press;
    begin
      @(negedge clk); key_n = 0;
      wait_cycles(23);
      @(negedge clk); key_n = 1;
      wait_cycles(23);
    end
  endtask

  initial begin
    wait_cycles(5);
    rst_n = 1;
    wait_cycles(4);

    // Chatter for less than 20 ms must not count as a press.
    repeat (5) begin
      @(negedge clk); key_n = 0; wait_cycles(3);
      @(negedge clk); key_n = 1; wait_cycles(2);
    end
    if (press_count != 0 || key_down)
      $fatal(1, "bounce was accepted as a key press");

    // A stable low level emits one event; holding it cannot auto-repeat.
    @(negedge clk); key_n = 0;
    wait_cycles(24);
    if (press_count != 1 || !key_down || !target)
      $fatal(1, "debounced press missing count=%0d", press_count);
    wait_cycles(40);
    if (press_count != 1)
      $fatal(1, "held key emitted repeated presses");
    @(negedge clk); key_n = 1;
    wait_cycles(24);
    if (key_down)
      $fatal(1, "debounced release missing");

    // Exactly 1323 sample ticks reach Q1.15 full scale without overshoot.
    for (tick = 1; tick <= 1323; tick = tick + 1) begin
      sample_once();
      if (tick < 1323 && gain >= 16'd32767)
        $fatal(1, "ramp reached target early at tick %0d", tick);
    end
    if (gain != 16'd32767 || ramping)
      $fatal(1, "up-ramp endpoint mismatch gain=%0d ramp=%b", gain, ramping);

    mix_left = 16'sh7fff;
    vocal_left = -16'sh8000;
    mix_right = -16'sh8000;
    vocal_right = 16'sh7fff;
    #1;
    if (out_left != 16'sh7fff || out_right != -16'sh8000)
      $fatal(1, "saturation mismatch left=%0d right=%0d", out_left, out_right);

    // Toggle off, descend for 400 samples, then reverse from the current gain.
    clean_press();
    if (target)
      $fatal(1, "second press did not select bypass");
    for (tick = 0; tick < 400; tick = tick + 1)
      sample_once();
    reversal_gain = gain;
    clean_press();
    if (!target || gain != reversal_gain)
      $fatal(1, "reversal changed gain discontinuously");
    sample_once();
    if (gain <= reversal_gain)
      $fatal(1, "reversed ramp did not move upward");

    // Reset/default bypass is bit-exact mixture.
    rst_n = 0;
    wait_cycles(3);
    rst_n = 1;
    wait_cycles(3);
    mix_left = -16'sd12345;
    mix_right = 16'sd23456;
    vocal_left = 16'sd30000;
    vocal_right = -16'sd30000;
    #1;
    if (target || gain != 0 || out_left != mix_left || out_right != mix_right)
      $fatal(1, "reset bypass mismatch");

    $display("audio_key_mixer: PASS");
    $finish;
  end
endmodule
