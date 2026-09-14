`timescale 1ns/1ps

module tb_npu_tensor_mac_8x8;
  localparam integer MAX_TOKENS = 1024;
  localparam integer MAX_RESULTS = 256;

  logic clk;
  logic rst_n;
  logic soft_reset;
  logic input_valid;
  logic input_ready;
  logic first_token;
  logic last_token;
  logic [7:0] input_lane_mask;
  logic [7:0] output_lane_mask;
  logic [127:0] activation;
  logic [511:0] weight;
  logic result_valid;
  logic result_ready;
  logic [255:0] result_data;

  logic [659:0] token_memory [0:MAX_TOKENS-1];
  logic [255:0] expected_memory [0:MAX_RESULTS-1];
  logic [31:0] lfsr_q;
  logic held_result_q;
  logic [255:0] held_data_q;
  integer token_count;
  integer result_count;
  integer token_index;
  integer result_index;
  integer cycles;
  reg [1023:0] token_hex;
  reg [1023:0] expected_hex;

  npu_tensor_mac_8x8 dut (
    .clk_i(clk), .rst_ni(rst_n), .soft_reset_i(soft_reset),
    .input_valid_i(input_valid), .input_ready_o(input_ready),
    .first_i(first_token), .last_i(last_token),
    .input_lane_mask_i(input_lane_mask),
    .output_lane_mask_i(output_lane_mask),
    .activation_i(activation), .weight_i(weight),
    .result_valid_o(result_valid), .result_ready_i(result_ready),
    .result_o(result_data)
  );

  always #2.5 clk = ~clk;

  always @* begin
    input_valid = token_index < token_count;
    activation = '0;
    weight = '0;
    input_lane_mask = '0;
    output_lane_mask = '0;
    first_token = 1'b0;
    last_token = 1'b0;
    if (token_index < token_count) begin
      activation = token_memory[token_index][127:0];
      weight = token_memory[token_index][639:128];
      input_lane_mask = token_memory[token_index][647:640];
      output_lane_mask = token_memory[token_index][655:648];
      first_token = token_memory[token_index][656];
      last_token = token_memory[token_index][657];
    end
    result_ready = lfsr_q[0] || lfsr_q[5];
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      token_index <= 0;
      result_index <= 0;
      cycles <= 0;
      lfsr_q <= 32'h1ace_b00c;
      held_result_q <= 1'b0;
      held_data_q <= '0;
    end else begin
      cycles <= cycles + 1;
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};

      if (held_result_q) begin
        if (!result_valid || result_data !== held_data_q) begin
          $display("FAIL: result changed while back-pressured");
          $fatal(1);
        end
        if (result_ready)
          held_result_q <= 1'b0;
      end else if (result_valid && !result_ready) begin
        held_result_q <= 1'b1;
        held_data_q <= result_data;
      end

      if (input_valid && input_ready)
        token_index <= token_index + 1;

      if (result_valid && result_ready) begin
        if (result_index >= result_count) begin
          $display("FAIL: unexpected result %0d", result_index);
          $fatal(1);
        end
        if (result_data !== expected_memory[result_index]) begin
          $display("FAIL: result %0d got=%064x expected=%064x",
                   result_index, result_data, expected_memory[result_index]);
          $fatal(1);
        end
        result_index <= result_index + 1;
      end

      if (token_index == token_count && result_index == result_count
          && !result_valid && !held_result_q) begin
        $display("npu_tensor_mac_8x8: PASS (%0d tokens, %0d results)",
                 token_count, result_count);
        $finish;
      end
      if (cycles > 20000) begin
        $display("FAIL: timeout tokens=%0d/%0d results=%0d/%0d",
                 token_index, token_count, result_index, result_count);
        $fatal(1);
      end
    end
  end

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    soft_reset = 1'b0;
    token_count = 0;
    result_count = 0;
    if (!$value$plusargs("TOKEN_HEX=%s", token_hex)
        || !$value$plusargs("EXPECTED_HEX=%s", expected_hex)
        || !$value$plusargs("TOKEN_COUNT=%d", token_count)
        || !$value$plusargs("RESULT_COUNT=%d", result_count)) begin
      $display("FAIL: missing vector plusargs");
      $fatal(1);
    end
    if (token_count <= 0 || token_count > MAX_TOKENS
        || result_count <= 0 || result_count > MAX_RESULTS) begin
      $display("FAIL: invalid vector counts");
      $fatal(1);
    end
    $readmemh(token_hex, token_memory);
    $readmemh(expected_hex, expected_memory);
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
  end

endmodule
