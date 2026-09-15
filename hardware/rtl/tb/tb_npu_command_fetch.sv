`timescale 1ns/1ps

module tb_npu_command_fetch;
  localparam integer MAX_COMMANDS = 2048;
  localparam logic [63:0] COMMAND_BASE = 64'h0000_0000_2000_0100;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_valid_i = 1'b0;
  logic start_ready_o;
  logic memory_request_valid_o;
  logic memory_request_ready_i;
  logic [63:0] memory_request_address_o;
  logic [6:0] memory_request_bytes_o;
  logic memory_response_valid_i = 1'b0;
  logic memory_response_ready_o;
  logic [511:0] memory_response_data_i;
  logic memory_response_error_i = 1'b0;
  logic command_valid_o;
  logic command_ready_i;
  logic [127:0] command_bits_o;
  logic [31:0] command_pc_o;
  logic busy_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;
  logic [31:0] error_pc_o;
  logic [15:0] error_tag_o;

  logic [127:0] command_memory [0:MAX_COMMANDS-1];
  logic [31:0] lfsr_q = 32'h62a1_55d9;
  logic request_pending_q = 1'b0;
  integer pending_index_q = 0;
  integer response_delay_q = 0;
  integer accepted_q = 0;
  integer cycles_q = 0;
  integer command_count;
  integer expected_error;
  integer expected_reason;
  string command_hex;

  always #5 clk_i = ~clk_i;
  assign memory_request_ready_i = !request_pending_q
    && !memory_response_valid_i && lfsr_q[0];
  assign memory_response_data_i = {384'd0, command_memory[pending_index_q]};
  assign command_ready_i = lfsr_q[2] || lfsr_q[3];

  npu_command_fetch dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_valid_i(start_valid_i), .start_ready_o(start_ready_o),
    .command_base_i(COMMAND_BASE), .command_bytes_i(command_count * 16),
    .memory_request_valid_o(memory_request_valid_o),
    .memory_request_ready_i(memory_request_ready_i),
    .memory_request_address_o(memory_request_address_o),
    .memory_request_bytes_o(memory_request_bytes_o),
    .memory_response_valid_i(memory_response_valid_i),
    .memory_response_ready_o(memory_response_ready_o),
    .memory_response_data_i(memory_response_data_i),
    .memory_response_error_i(memory_response_error_i),
    .command_valid_o(command_valid_o), .command_ready_i(command_ready_i),
    .command_bits_o(command_bits_o), .command_pc_o(command_pc_o),
    .busy_o(busy_o), .error_pulse_o(error_pulse_o),
    .error_reason_o(error_reason_o), .error_pc_o(error_pc_o),
    .error_tag_o(error_tag_o));

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      lfsr_q <= 32'h62a1_55d9;
      request_pending_q <= 1'b0;
      response_delay_q <= 0;
      memory_response_valid_i <= 1'b0;
      accepted_q <= 0;
      cycles_q <= 0;
    end else begin
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      cycles_q <= cycles_q + 1;
      if (cycles_q > 100000)
        $fatal(1, "command fetch timeout");

      if (memory_request_valid_o && memory_request_ready_i) begin
        if (memory_request_bytes_o != 16
            || memory_request_address_o != COMMAND_BASE + accepted_q * 16)
          $fatal(1, "memory request mismatch at %0d", accepted_q);
        request_pending_q <= 1'b1;
        pending_index_q <= accepted_q;
        response_delay_q <= lfsr_q[6:4];
      end
      if (request_pending_q && !memory_response_valid_i) begin
        if (response_delay_q == 0) begin
          memory_response_valid_i <= 1'b1;
          request_pending_q <= 1'b0;
        end else begin
          response_delay_q <= response_delay_q - 1;
        end
      end
      if (memory_response_valid_i && memory_response_ready_o)
        memory_response_valid_i <= 1'b0;

      if (command_valid_o && command_ready_i) begin
        if (command_pc_o != accepted_q * 16
            || command_bits_o != command_memory[accepted_q])
          $fatal(1, "command output mismatch at %0d", accepted_q);
        accepted_q <= accepted_q + 1;
      end
    end
  end

  initial begin
    if (!$value$plusargs("COMMANDS=%s", command_hex)
        || !$value$plusargs("COMMAND_COUNT=%d", command_count)
        || !$value$plusargs("EXPECT_ERROR=%d", expected_error)
        || !$value$plusargs("ERROR_REASON=%d", expected_reason))
      $fatal(1, "missing plusargs");
    $readmemh(command_hex, command_memory);
    repeat (5) @(posedge clk_i);
    rst_ni <= 1'b1;
    @(posedge clk_i);
    start_valid_i <= 1'b1;
    do @(posedge clk_i); while (!start_ready_o);
    start_valid_i <= 1'b0;

    if (expected_error) begin
      wait(error_pulse_o);
      if (error_reason_o != expected_reason)
        $fatal(1, "expected error %0d, got %0d", expected_reason, error_reason_o);
    end else begin
      wait(accepted_q == command_count);
      @(posedge clk_i);
      if (error_pulse_o || busy_o)
        $fatal(1, "valid stream did not terminate cleanly");
    end
    $display("command fetch: PASS commands=%0d error=%0d cycles=%0d",
             accepted_q, expected_error, cycles_q);
    $finish;
  end
endmodule
