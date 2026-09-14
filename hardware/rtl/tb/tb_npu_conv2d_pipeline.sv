`timescale 1ns/1ps

module tb_npu_conv2d_pipeline;
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
  logic activation_request_valid_o;
  logic activation_request_ready_i;
  logic activation_read_bank_o;
  logic [31:0] activation_read_address_o;
  logic activation_response_valid_i;
  logic activation_response_ready_o;
  logic [127:0] activation_read_data_i;
  logic weight_request_valid_o;
  logic weight_request_ready_i;
  logic weight_read_bank_o;
  logic [31:0] weight_read_address_o;
  logic weight_response_valid_i;
  logic weight_response_ready_o;
  logic [511:0] weight_read_data_i;
  logic result_valid_o;
  logic result_ready_i;
  logic [31:0] result_address_o;
  logic [7:0] result_lane_mask_o;
  logic [127:0] result_data_o;
  logic lut_write_valid_i = 1'b0;
  logic lut_write_ready_o;
  logic [11:0] lut_write_address_i = '0;
  logic [15:0] lut_write_data_i = '0;
  logic busy_o;
  logic done_pulse_o;
  logic [7:0] event_set_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;

  logic [511:0] descriptor_memory [0:0];
  logic [127:0] activation_memory [0:MAX_ACTIVATION_ROWS-1];
  logic [511:0] weight_memory [0:MAX_WEIGHT_ROWS-1];
  logic [63:0] request_memory [0:MAX_REQUESTS-1];
  logic [127:0] result_memory [0:MAX_RESULTS-1];
  logic [15:0] lut_memory [0:4095];
  logic activation_response_valid_q = 1'b0;
  logic weight_response_valid_q = 1'b0;
  logic [127:0] activation_response_data_q;
  logic [511:0] weight_response_data_q;
  logic [31:0] lfsr_q = 32'h95e7_0210;
  integer request_count = 0;
  integer result_count = 0;
  integer activation_request_index = 0;
  integer weight_request_index = 0;
  integer parameter_request_index = 0;
  integer parameter_request_count = 0;
  integer bias_address = 0;
  integer result_index = 0;
  integer cycles = 0;
  integer lut_index;
  reg [1023:0] descriptor_hex;
  reg [1023:0] activation_hex;
  reg [1023:0] weight_hex;
  reg [1023:0] request_hex;
  reg [1023:0] result_hex;
  reg [1023:0] lut_hex;

  npu_conv2d_pipeline dut (.*);
  always #2.5 clk_i = ~clk_i;

  assign activation_request_ready_i
    = (!activation_response_valid_q || activation_response_ready_o)
      && (lfsr_q[0] || lfsr_q[5]);
  assign weight_request_ready_i
    = (!weight_response_valid_q || weight_response_ready_o)
      && (lfsr_q[1] || lfsr_q[6]);
  assign activation_response_valid_i = activation_response_valid_q;
  assign activation_read_data_i = activation_response_data_q;
  assign weight_response_valid_i = weight_response_valid_q;
  assign weight_read_data_i = weight_response_data_q;
  assign result_ready_i = lfsr_q[2] || lfsr_q[9];

  function automatic integer expected_parameter_address(input integer index);
    if (index == 0)
      expected_parameter_address = bias_address;
    else
      expected_parameter_address = bias_address + index * 64;
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      activation_response_valid_q <= 1'b0;
      weight_response_valid_q <= 1'b0;
      activation_request_index <= 0;
      weight_request_index <= 0;
      parameter_request_index <= 0;
      result_index <= 0;
      cycles <= 0;
      lfsr_q <= 32'h95e7_0210;
    end else begin
      cycles <= cycles + 1;
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};

      if (activation_response_valid_q && activation_response_ready_o)
        activation_response_valid_q <= 1'b0;
      if (activation_request_valid_o && activation_request_ready_i) begin
        if (activation_request_index >= request_count)
          $fatal(1, "unexpected activation request");
        if (activation_read_address_o
            !== request_memory[activation_request_index][31:0])
          $fatal(1, "activation request %0d got=%0d expected=%0d",
                 activation_request_index, activation_read_address_o,
                 request_memory[activation_request_index][31:0]);
        activation_response_data_q
          <= activation_memory[activation_read_address_o >> 4];
        activation_response_valid_q <= 1'b1;
        activation_request_index <= activation_request_index + 1;
      end

      if (weight_response_valid_q && weight_response_ready_o)
        weight_response_valid_q <= 1'b0;
      if (weight_request_valid_o && weight_request_ready_i) begin
        if (parameter_request_index < parameter_request_count) begin
          if (weight_read_address_o
              !== expected_parameter_address(parameter_request_index))
            $fatal(1, "parameter request %0d got=%0d expected=%0d",
                   parameter_request_index, weight_read_address_o,
                   expected_parameter_address(parameter_request_index));
          parameter_request_index <= parameter_request_index + 1;
        end else begin
          if (weight_request_index >= request_count)
            $fatal(1, "unexpected weight request");
          if (weight_read_address_o
              !== request_memory[weight_request_index][63:32])
            $fatal(1, "weight request %0d got=%0d expected=%0d",
                   weight_request_index, weight_read_address_o,
                   request_memory[weight_request_index][63:32]);
          weight_request_index <= weight_request_index + 1;
        end
        weight_response_data_q <= weight_memory[weight_read_address_o >> 6];
        weight_response_valid_q <= 1'b1;
      end

      if (result_valid_o && result_ready_i) begin
        if (result_index >= result_count)
          $fatal(1, "unexpected result");
        if (result_address_o !== result_index * 16)
          $fatal(1, "result address %0d got=%0d", result_index,
                 result_address_o);
        if (result_data_o !== result_memory[result_index])
          $fatal(1, "result %0d got=%032x expected=%032x",
                 result_index, result_data_o, result_memory[result_index]);
        result_index <= result_index + 1;
      end

      if (error_pulse_o)
        $fatal(1, "pipeline error reason=%0d", error_reason_o);
      if (done_pulse_o) begin
        if (event_set_o !== completion_event_i || busy_o)
          $fatal(1, "completion state mismatch");
        if (activation_request_index != request_count
            || weight_request_index != request_count
            || parameter_request_index != parameter_request_count
            || result_index != result_count)
          $fatal(1, "early completion");
        $display("npu_conv2d_pipeline: PASS (%0d requests, %0d results)",
                 request_count, result_count);
        $finish;
      end
      if (cycles > 400000)
        $fatal(1, "pipeline timeout");
    end
  end

  initial begin
    if (!$value$plusargs("DESCRIPTOR_HEX=%s", descriptor_hex)
        || !$value$plusargs("ACTIVATION_HEX=%s", activation_hex)
        || !$value$plusargs("WEIGHT_HEX=%s", weight_hex)
        || !$value$plusargs("REQUEST_HEX=%s", request_hex)
        || !$value$plusargs("RESULT_HEX=%s", result_hex)
        || !$value$plusargs("LUT_HEX=%s", lut_hex)
        || !$value$plusargs("BIAS_ADDRESS=%d", bias_address)
        || !$value$plusargs("PARAM_REQUEST_COUNT=%d", parameter_request_count)
        || !$value$plusargs("REQUEST_COUNT=%d", request_count)
        || !$value$plusargs("RESULT_COUNT=%d", result_count))
      $fatal(1, "missing vector plusargs");
    $readmemh(descriptor_hex, descriptor_memory);
    $readmemh(activation_hex, activation_memory);
    $readmemh(weight_hex, weight_memory);
    $readmemh(request_hex, request_memory);
    $readmemh(result_hex, result_memory);
    $readmemh(lut_hex, lut_memory);
    for (lut_index = 0; lut_index < 4096; lut_index = lut_index + 1)
      dut.post.tanh_lut[lut_index] = lut_memory[lut_index];
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
