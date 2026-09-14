`timescale 1ns/1ps

module tb_npu_conv2d_controller_errors;
  import npu_isa_pkg::*;

  localparam logic [3:0] CONV_ERROR_GEOMETRY = 4'd1;
  localparam logic [3:0] CONV_ERROR_MODE = 4'd2;
  localparam logic [3:0] CONV_ERROR_CAPACITY = 4'd3;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_valid_i = 1'b0;
  logic start_ready_o;
  logic [511:0] operator_desc_bits_i;
  logic activation_bank_i = 1'b1;
  logic weight_bank_i = 1'b1;
  logic [7:0] completion_event_i = NPU_EVENT_C1_DONE;
  logic read_request_valid_o;
  logic read_request_ready_i = 1'b0;
  logic activation_read_bank_o;
  logic [31:0] activation_read_address_o;
  logic weight_read_bank_o;
  logic [31:0] weight_read_address_o;
  logic read_response_valid_i = 1'b0;
  logic read_response_ready_o;
  logic [127:0] activation_read_data_i = '0;
  logic [511:0] weight_read_data_i = '0;
  logic result_valid_o;
  logic result_ready_i = 1'b1;
  logic [31:0] result_address_o;
  logic [7:0] result_lane_mask_o;
  logic [255:0] result_data_o;
  logic busy_o;
  logic done_pulse_o;
  logic [7:0] event_set_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;
  npu_operator_desc_t descriptor;

  assign operator_desc_bits_i = descriptor;

  npu_conv2d_controller dut (.*);
  always #2.5 clk_i = ~clk_i;

  task automatic set_valid_descriptor;
    begin
      descriptor = '0;
      descriptor.kh = 1;
      descriptor.kw = 1;
      descriptor.stride_h = 1;
      descriptor.stride_w = 1;
      descriptor.dilation_h = 1;
      descriptor.dilation_w = 1;
      descriptor.groups = 1;
      descriptor.tile_h = 1;
      descriptor.tile_w = 1;
      descriptor.tile_cout = 1;
      descriptor.input_channel_count = 1;
    end
  endtask

  task automatic start_and_expect_error(input logic [3:0] reason);
    integer wait_cycles;
    begin
      @(negedge clk_i);
      start_valid_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      start_valid_i = 1'b0;
      wait_cycles = 0;
      while (!error_pulse_o && wait_cycles < 100) begin
        if (read_request_valid_o)
          $fatal(1, "invalid descriptor issued a read request");
        @(negedge clk_i);
        wait_cycles = wait_cycles + 1;
      end
      if (!error_pulse_o || error_reason_o != reason)
        $fatal(1, "expected error reason=%0d, got pulse=%0b reason=%0d",
               reason, error_pulse_o, error_reason_o);
      if (busy_o || read_request_valid_o)
        $fatal(1, "invalid descriptor started execution");
      @(posedge clk_i);
      @(negedge clk_i);
      if (error_pulse_o)
        $fatal(1, "error pulse lasted more than one cycle");
    end
  endtask

  initial begin
    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;

    set_valid_descriptor();
    descriptor.tile_h = 0;
    start_and_expect_error(CONV_ERROR_GEOMETRY);

    set_valid_descriptor();
    descriptor.kh = 2;
    start_and_expect_error(CONV_ERROR_MODE);

    set_valid_descriptor();
    descriptor.kh = 3;
    descriptor.kw = 3;
    descriptor.stride_h = 2;
    descriptor.stride_w = 2;
    descriptor.tile_h = 16;
    descriptor.tile_w = 16;
    descriptor.tile_cout = 8;
    descriptor.input_channel_count = 224;
    start_and_expect_error(CONV_ERROR_CAPACITY);

    set_valid_descriptor();
    @(negedge clk_i);
    start_valid_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    start_valid_i = 1'b0;
    if (!busy_o)
      $fatal(1, "valid descriptor did not start");
    soft_reset_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    soft_reset_i = 1'b0;
    #1;
    if (busy_o || read_request_valid_o || result_valid_o)
      $fatal(1, "soft reset did not flush controller");
    if (!start_ready_o)
      $fatal(1, "controller not ready after soft reset");

    // Prove that a flushed controller can execute a new one-token job.
    @(negedge clk_i);
    start_valid_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    start_valid_i = 1'b0;
    read_request_ready_i = 1'b1;
    wait (read_request_valid_o);
    @(posedge clk_i);
    @(negedge clk_i);
    read_request_ready_i = 1'b0;
    if (activation_read_address_o != 0 || weight_read_address_o != 0
        || !activation_read_bank_o || !weight_read_bank_o)
      $fatal(1, "recovery request metadata mismatch");
    read_response_valid_i = 1'b1;
    wait (read_response_ready_o);
    @(posedge clk_i);
    @(negedge clk_i);
    read_response_valid_i = 1'b0;
    wait (result_valid_o);
    if (result_address_o != 0 || result_lane_mask_o != 8'h01
        || result_data_o != 0)
      $fatal(1, "recovery result mismatch");
    @(posedge clk_i);
    @(negedge clk_i);
    if (!done_pulse_o || event_set_o != completion_event_i || busy_o)
      $fatal(1, "recovery completion mismatch");

    $display("npu_conv2d_controller errors/reset: PASS");
    $finish;
  end
endmodule
