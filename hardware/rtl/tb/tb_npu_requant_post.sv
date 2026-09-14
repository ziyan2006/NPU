`timescale 1ns/1ps

module tb_npu_requant_post;
  localparam integer MAX_CASES = 100;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic input_valid_i = 1'b0;
  logic input_ready_o;
  logic [255:0] accumulator_i = '0;
  logic [255:0] bias_i = '0;
  logic [1023:0] quant_params_i = '0;
  logic [15:0] post_op_i = '0;
  logic [7:0] lane_mask_i = '0;
  logic output_valid_o;
  logic output_ready_i = 1'b0;
  logic [127:0] output_data_o;
  logic [7:0] output_lane_mask_o;
  logic lut_write_valid_i = 1'b0;
  logic lut_write_ready_o;
  logic [11:0] lut_write_address_i = '0;
  logic [15:0] lut_write_data_i = '0;
  logic error_pulse_o;
  logic [3:0] error_reason_o;

  logic [255:0] accumulator_mem [0:MAX_CASES-1];
  logic [255:0] bias_mem [0:MAX_CASES-1];
  logic [1023:0] quant_mem [0:MAX_CASES-1];
  logic [23:0] meta_mem [0:MAX_CASES-1];
  logic [127:0] expected_mem [0:MAX_CASES-1];
  logic [15:0] lut_mem [0:4095];
  logic [127:0] held_output;
  integer case_count;
  integer case_index;
  integer lut_index;
  integer hold_cycle;
  integer timeout;

  always #5 clk_i = ~clk_i;

  npu_requant_post dut (.*);

  initial begin
    if (!$value$plusargs("CASE_COUNT=%d", case_count))
      $fatal(1, "CASE_COUNT plusarg is required");
    if (case_count <= 0 || case_count > MAX_CASES)
      $fatal(1, "invalid CASE_COUNT=%0d", case_count);
    $readmemh("post_accumulator.hex", accumulator_mem);
    $readmemh("post_bias.hex", bias_mem);
    $readmemh("post_quant.hex", quant_mem);
    $readmemh("post_meta.hex", meta_mem);
    $readmemh("post_expected.hex", expected_mem);
    $readmemh("post_lut.hex", lut_mem);
    for (lut_index = 0; lut_index < 4096; lut_index = lut_index + 1)
      dut.tanh_lut[lut_index] = lut_mem[lut_index];

    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;

    // Exercise the run-time LUT programming port, then restore the entry.
    @(negedge clk_i);
    lut_write_address_i = 12'd2048;
    lut_write_data_i = 16'h1234;
    lut_write_valid_i = 1'b1;
    @(posedge clk_i);
    if (!lut_write_ready_o)
      $fatal(1, "LUT write stalled while post unit was idle");
    @(negedge clk_i);
    lut_write_data_i = lut_mem[2048];
    @(posedge clk_i);
    @(negedge clk_i);
    lut_write_valid_i = 1'b0;
    for (case_index = 0; case_index < case_count;
         case_index = case_index + 1) begin
      accumulator_i = accumulator_mem[case_index];
      bias_i = bias_mem[case_index];
      quant_params_i = quant_mem[case_index];
      post_op_i = meta_mem[case_index][23:8];
      lane_mask_i = meta_mem[case_index][7:0];
      input_valid_i = 1'b1;
      do @(posedge clk_i); while (!input_ready_o);
      @(negedge clk_i);
      input_valid_i = 1'b0;

      timeout = 0;
      while (!output_valid_o && timeout < 100) begin
        @(negedge clk_i);
        timeout = timeout + 1;
      end
      if (timeout == 100)
        $fatal(1, "case %0d output timeout", case_index);
      if (output_data_o !== expected_mem[case_index])
        $fatal(1, "case %0d data mismatch got=%032x expected=%032x",
               case_index, output_data_o, expected_mem[case_index]);
      if (output_lane_mask_o !== meta_mem[case_index][7:0])
        $fatal(1, "case %0d lane mask mismatch", case_index);

      held_output = output_data_o;
      for (hold_cycle = 0; hold_cycle < 3; hold_cycle = hold_cycle + 1) begin
        @(negedge clk_i);
        if (!output_valid_o || output_data_o !== held_output)
          $fatal(1, "case %0d changed under output back-pressure",
                 case_index);
      end
      output_ready_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      output_ready_i = 1'b0;
    end

    // Reject an out-of-contract shift without producing an output token.
    accumulator_i = '0;
    bias_i = '0;
    quant_params_i = quant_mem[0];
    quant_params_i[32 +: 8] = 8'd64;
    post_op_i = 16'd0;
    lane_mask_i = 8'h01;
    input_valid_i = 1'b1;
    do @(posedge clk_i); while (!input_ready_o);
    @(negedge clk_i);
    input_valid_i = 1'b0;
    if (!error_pulse_o || error_reason_o != 4'd2 || output_valid_o)
      $fatal(1, "illegal shift was not rejected");

    // Invalid padding-lane parameters are ignored when the lane is masked.
    quant_params_i = quant_mem[0];
    quant_params_i[7*128 + 32 +: 8] = 8'd255;
    lane_mask_i = 8'h7f;
    input_valid_i = 1'b1;
    do @(posedge clk_i); while (!input_ready_o);
    @(negedge clk_i);
    input_valid_i = 1'b0;
    timeout = 0;
    while (!output_valid_o && timeout < 100) begin
      @(negedge clk_i);
      timeout = timeout + 1;
    end
    if (timeout == 100 || error_pulse_o)
      $fatal(1, "masked padding lane parameter was incorrectly rejected");
    output_ready_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    output_ready_i = 1'b0;

    $display("npu_requant_post: %0d bit-exact vectors PASS", case_count);
    $finish;
  end
endmodule
