`timescale 1ns/1ps

module tb_npu_command_stream;
  import npu_isa_pkg::*;

  localparam int MAX_COMMANDS = 4096;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_i = 1'b0;
  logic command_valid_i;
  logic command_ready_o;
  logic [127:0] command_bits_i;
  logic dma_command_valid_o;
  logic dma_command_ready_i = 1'b1;
  logic [127:0] dma_command_bits_o;
  logic compute_command_valid_o;
  logic compute_command_ready_i = 1'b1;
  logic [127:0] compute_command_bits_o;
  logic vector_command_valid_o;
  logic vector_command_ready_i = 1'b1;
  logic [127:0] vector_command_bits_o;
  logic [7:0] event_set_i;
  logic vector_sync_done_i = 1'b0;
  logic units_idle_i = 1'b1;
  logic [31:0] watchdog_limit_i = 32'd20000;
  logic execution_error_i = 1'b0;
  logic [15:0] execution_error_code_i = NPU_ERR_NONE;
  logic [31:0] execution_error_pc_i = '0;
  logic [15:0] execution_error_tag_i = '0;
  logic busy_o;
  logic done_pulse_o;
  logic irq_pulse_o;
  logic error_o;
  logic [15:0] error_code_o;
  logic [31:0] error_pc_o;
  logic [15:0] error_inst_tag_o;
  logic [31:0] command_pc_o;
  logic [31:0] commands_retired_o;
  logic [63:0] cycles_total_o;
  logic [7:0] event_state_o;

  logic [127:0] command_memory [0:MAX_COMMANDS-1];
  logic [7:0] dma_event_q0, dma_event_q1;
  logic [7:0] compute_event_q0, compute_event_q1;
  integer command_count;
  integer command_index = 0;
  integer cycle_count = 0;
  string command_hex;

  always #5 clk_i = ~clk_i;

  assign command_valid_i = busy_o && (command_index < command_count);
  assign command_bits_i = command_memory[command_index];
  assign event_set_i = dma_event_q1 | compute_event_q1;

  npu_command_processor dut (.*);

  always_ff @(posedge clk_i) begin
    dma_event_q0 <= (dma_command_valid_o && dma_command_ready_i)
      ? dma_command_bits_o[119:112] : 8'h00;
    dma_event_q1 <= dma_event_q0;
    compute_event_q0 <= (compute_command_valid_o && compute_command_ready_i)
      ? compute_command_bits_o[119:112] : 8'h00;
    compute_event_q1 <= compute_event_q0;
    vector_sync_done_i <= vector_command_valid_o && vector_command_ready_i;

    if (command_valid_i && command_ready_o)
      command_index <= command_index + 1;
    cycle_count <= cycle_count + 1;

    if (error_o)
      $fatal(1, "stream rejected at pc=%0d code=%0h tag=%0h",
             error_pc_o, error_code_o, error_inst_tag_o);
    if (cycle_count > 20000)
      $fatal(1, "stream timeout at command %0d pc=%0d events=%0h",
             command_index, command_pc_o, event_state_o);
  end

  initial begin
    dma_event_q0 = 0;
    dma_event_q1 = 0;
    compute_event_q0 = 0;
    compute_event_q1 = 0;
    if (!$value$plusargs("COMMAND_HEX=%s", command_hex))
      $fatal(1, "missing +COMMAND_HEX");
    if (!$value$plusargs("COMMAND_COUNT=%d", command_count))
      $fatal(1, "missing +COMMAND_COUNT");
    if (command_count <= 0 || command_count > MAX_COMMANDS)
      $fatal(1, "invalid command count %0d", command_count);
    $readmemh(command_hex, command_memory, 0, command_count - 1);

    repeat (3) @(posedge clk_i);
    rst_ni = 1'b1;
    @(negedge clk_i);
    start_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    start_i = 1'b0;

    @(posedge done_pulse_o);
    #1;
    if (!irq_pulse_o) $fatal(1, "stream END did not request IRQ");
    if (command_index != command_count)
      $fatal(1, "consumed %0d of %0d commands", command_index, command_count);
    if (commands_retired_o != command_count)
      $fatal(1, "retired %0d of %0d commands",
             commands_retired_o, command_count);
    if (command_pc_o != command_count * NPU_COMMAND_BYTES)
      $fatal(1, "final pc %0d", command_pc_o);
    $display("tile command stream RTL decode: PASS (%0d commands, %0d cycles)",
             command_count, cycle_count);
    $finish;
  end
endmodule
