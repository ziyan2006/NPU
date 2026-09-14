`timescale 1ns/1ps

module tb_npu_conv2d_controller;
  localparam integer MAX_ACTIVATION_ROWS = 4096;
  localparam integer MAX_WEIGHT_ROWS = 512;
  localparam integer MAX_REQUESTS = 32768;
  localparam integer MAX_RESULTS = 256;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_valid_i = 1'b0;
  logic start_ready_o;
  logic [511:0] operator_desc_bits_i;
  logic activation_bank_i = 1'b0;
  logic weight_bank_i = 1'b0;
  logic [7:0] completion_event_i = 8'h20;
  logic read_request_valid_o;
  logic read_request_ready_i;
  logic activation_read_bank_o;
  logic [31:0] activation_read_address_o;
  logic weight_read_bank_o;
  logic [31:0] weight_read_address_o;
  logic read_response_valid_i;
  logic read_response_ready_o;
  logic [127:0] activation_read_data_i;
  logic [511:0] weight_read_data_i;
  logic result_valid_o;
  logic result_ready_i;
  logic [31:0] result_address_o;
  logic [7:0] result_lane_mask_o;
  logic [255:0] result_data_o;
  logic busy_o;
  logic done_pulse_o;
  logic [7:0] event_set_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;

  logic [511:0] descriptor_memory [0:0];
  logic [127:0] activation_memory [0:MAX_ACTIVATION_ROWS-1];
  logic [511:0] weight_memory [0:MAX_WEIGHT_ROWS-1];
  logic [63:0] request_memory [0:MAX_REQUESTS-1];
  logic [255:0] result_memory [0:MAX_RESULTS-1];
  logic response_valid_q = 1'b0;
  logic [127:0] response_activation_q;
  logic [511:0] response_weight_q;
  logic [31:0] lfsr_q = 32'h6d5a_2c13;
  integer request_count = 0;
  integer result_count = 0;
  integer request_index = 0;
  integer result_index = 0;
  integer cycles = 0;
  reg [1023:0] descriptor_hex;
  reg [1023:0] activation_hex;
  reg [1023:0] weight_hex;
  reg [1023:0] request_hex;
  reg [1023:0] result_hex;

  npu_conv2d_controller dut (.*);
  always #2.5 clk_i = ~clk_i;

  assign read_request_ready_i = (!response_valid_q
                                 || read_response_ready_o)
                                && (lfsr_q[0] || lfsr_q[4]);
  assign read_response_valid_i = response_valid_q;
  assign activation_read_data_i = response_activation_q;
  assign weight_read_data_i = response_weight_q;
  assign result_ready_i = lfsr_q[1] || lfsr_q[7];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      response_valid_q <= 1'b0;
      request_index <= 0;
      result_index <= 0;
      cycles <= 0;
      lfsr_q <= 32'h6d5a_2c13;
    end else begin
      cycles <= cycles + 1;
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};

      if (response_valid_q && read_response_ready_o)
        response_valid_q <= 1'b0;
      if (read_request_valid_o && read_request_ready_i) begin
        if (request_index >= request_count)
          $fatal(1, "unexpected read request %0d", request_index);
        if ({weight_read_address_o, activation_read_address_o}
            !== request_memory[request_index])
          $fatal(1, "request %0d got A=%0d W=%0d expected=%016x",
                 request_index, activation_read_address_o,
                 weight_read_address_o, request_memory[request_index]);
        if (activation_read_address_o[3:0] != 0
            || weight_read_address_o[5:0] != 0)
          $fatal(1, "unaligned paired read request");
        response_activation_q
          <= activation_memory[activation_read_address_o >> 4];
        response_weight_q <= weight_memory[weight_read_address_o >> 6];
        response_valid_q <= 1'b1;
        request_index <= request_index + 1;
      end

      if (result_valid_o && result_ready_i) begin
        if (result_index >= result_count)
          $fatal(1, "unexpected result %0d", result_index);
        if (result_address_o !== result_index * 32)
          $fatal(1, "result address %0d got=%0d", result_index,
                 result_address_o);
        if (result_data_o !== result_memory[result_index])
          $fatal(1, "result %0d got=%064x expected=%064x",
                 result_index, result_data_o, result_memory[result_index]);
        result_index <= result_index + 1;
      end

      if (error_pulse_o)
        $fatal(1, "controller error reason=%0d", error_reason_o);
      if (done_pulse_o) begin
        if (event_set_o !== completion_event_i)
          $fatal(1, "completion event mismatch");
        if (request_index != request_count || result_index != result_count)
          $fatal(1, "early done requests=%0d/%0d results=%0d/%0d",
                 request_index, request_count, result_index, result_count);
        $display("npu_conv2d_controller: PASS (%0d requests, %0d results)",
                 request_count, result_count);
        $finish;
      end
      if (cycles > 200000)
        $fatal(1, "timeout requests=%0d/%0d results=%0d/%0d",
               request_index, request_count, result_index, result_count);
    end
  end

  initial begin
    if (!$value$plusargs("DESCRIPTOR_HEX=%s", descriptor_hex)
        || !$value$plusargs("ACTIVATION_HEX=%s", activation_hex)
        || !$value$plusargs("WEIGHT_HEX=%s", weight_hex)
        || !$value$plusargs("REQUEST_HEX=%s", request_hex)
        || !$value$plusargs("RESULT_HEX=%s", result_hex)
        || !$value$plusargs("REQUEST_COUNT=%d", request_count)
        || !$value$plusargs("RESULT_COUNT=%d", result_count))
      $fatal(1, "missing vector plusargs");
    if (request_count <= 0 || request_count > MAX_REQUESTS
        || result_count <= 0 || result_count > MAX_RESULTS)
      $fatal(1, "invalid vector counts");
    $readmemh(descriptor_hex, descriptor_memory);
    $readmemh(activation_hex, activation_memory);
    $readmemh(weight_hex, weight_memory);
    $readmemh(request_hex, request_memory);
    $readmemh(result_hex, result_memory);
    operator_desc_bits_i = descriptor_memory[0];
    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;
    @(negedge clk_i);
    start_valid_i = 1'b1;
    do @(posedge clk_i); while (!start_ready_o);
    @(negedge clk_i);
    start_valid_i = 1'b0;
  end
endmodule
