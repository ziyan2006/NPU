`timescale 1ns/1ps

module tb_npu_command_processor;
  import npu_isa_pkg::*;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_i = 1'b0;
  logic command_valid_i = 1'b0;
  logic command_ready_o;
  logic [127:0] command_bits_i = '0;
  logic dma_command_valid_o;
  logic dma_command_ready_i = 1'b1;
  logic [127:0] dma_command_bits_o;
  logic compute_command_valid_o;
  logic compute_command_ready_i = 1'b1;
  logic [127:0] compute_command_bits_o;
  logic vector_command_valid_o;
  logic vector_command_ready_i = 1'b1;
  logic [127:0] vector_command_bits_o;
  logic [7:0] event_set_i = 8'h00;
  logic vector_sync_done_i = 1'b0;
  logic units_idle_i = 1'b1;
  logic [31:0] watchdog_limit_i = 32'd100;
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

  always #5 clk_i = ~clk_i;

  npu_command_processor dut (.*);

  function automatic logic [127:0] make_command(
    input logic [7:0] opcode,
    input logic [7:0] flags,
    input logic [15:0] tag,
    input logic [15:0] dst_td,
    input logic [15:0] src0_td,
    input logic [15:0] src1_td,
    input logic [15:0] op_desc,
    input logic [15:0] quant_desc,
    input logic [15:0] imm
  );
    make_command = {imm, quant_desc, op_desc, src1_td, src0_td,
                    dst_td, tag, flags, opcode};
  endfunction

  task automatic send_command(input logic [127:0] bits);
    begin
      @(negedge clk_i);
      command_bits_i = bits;
      command_valid_i = 1'b1;
      do @(posedge clk_i); while (!command_ready_o);
      @(negedge clk_i);
      command_valid_i = 1'b0;
      command_bits_i = '0;
      // CP_DISPATCH deliberately separates capture/decode from unit issue.
      @(posedge clk_i);
      @(negedge clk_i);
    end
  endtask

  task automatic start_task;
    begin
      @(negedge clk_i);
      start_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      start_i = 1'b0;
      if (!busy_o) $fatal(1, "processor did not start");
    end
  endtask

  task automatic soft_reset;
    begin
      @(negedge clk_i);
      soft_reset_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      soft_reset_i = 1'b0;
      if (busy_o || error_o || commands_retired_o != 0 || command_pc_o != 0)
        $fatal(1, "soft reset did not clear processor state");
    end
  endtask

  task automatic pulse_event(input logic [7:0] mask);
    begin
      @(negedge clk_i);
      event_set_i = mask;
      @(posedge clk_i);
      @(negedge clk_i);
      event_set_i = 8'h00;
    end
  endtask

  initial begin
    repeat (3) @(posedge clk_i);
    rst_ni = 1'b1;
    start_task();

    send_command(make_command(NPU_OP_NOP, 0, 16'h0001,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, 0));
    if (commands_retired_o != 1 || command_pc_o != 16)
      $fatal(1, "NOP retirement");

    send_command(make_command(NPU_OP_WAIT, 0, 16'h0002,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_EVENT_A0_READY));
    if (command_ready_o || commands_retired_o != 1 || command_pc_o != 32)
      $fatal(1, "WAIT did not block");
    pulse_event(NPU_EVENT_A0_READY);
    if (commands_retired_o != 2 || event_state_o != NPU_EVENT_A0_READY)
      $fatal(1, "WAIT completion");

    dma_command_ready_i = 1'b0;
    @(negedge clk_i);
    command_bits_i = make_command(
      NPU_OP_DMA_LOAD, NPU_FLAG_ASYNC, 16'h0003,
      16'h0000, 16'h0000, NPU_NONE_INDEX, 16'h0000, NPU_NONE_INDEX,
      NPU_EVENT_A0_READY);
    command_valid_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    command_valid_i = 1'b0;
    command_bits_i = '0;
    #1;
    if (!dma_command_valid_o || command_ready_o)
      $fatal(1, "DMA back-pressure");
    dma_command_ready_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    if (event_state_o != 0 || commands_retired_o != 3)
      $fatal(1, "DMA issue must clear stale event and retire");
    pulse_event(NPU_EVENT_A0_READY);

    compute_command_ready_i = 1'b0;
    @(negedge clk_i);
    command_bits_i = make_command(
      NPU_OP_CONV2D,
      NPU_FLAG_ASYNC | NPU_FLAG_SATURATE | NPU_FLAG_FUSED_POST_OP,
      16'h0004, 16'h0003, 16'h0000, 16'h0001, 16'h0002, 16'h0003,
      NPU_EVENT_C0_DONE);
    command_valid_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    command_valid_i = 1'b0;
    command_bits_i = '0;
    #1;
    if (!compute_command_valid_o || command_ready_o)
      $fatal(1, "compute back-pressure");
    compute_command_ready_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    if (commands_retired_o != 4) $fatal(1, "async CONV retirement");
    pulse_event(NPU_EVENT_C0_DONE);

    send_command(make_command(
      NPU_OP_VEC_ADD, NPU_FLAG_SATURATE, 16'h0005,
      16'h0005, 16'h0003, 16'h0000, 16'h0004, 16'h0005, 0));
    if (commands_retired_o != 4 || command_ready_o)
      $fatal(1, "synchronous vector command did not block");
    @(negedge clk_i);
    vector_sync_done_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    vector_sync_done_i = 1'b0;
    if (commands_retired_o != 5) $fatal(1, "vector completion");

    send_command(make_command(
      NPU_OP_UPSAMPLE2X, 0, 16'h0006,
      16'h0006, 16'h0005, NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX, 0));
    @(negedge clk_i);
    vector_sync_done_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    vector_sync_done_i = 1'b0;
    if (commands_retired_o != 6) $fatal(1, "upsample completion");

    units_idle_i = 1'b0;
    send_command(make_command(NPU_OP_END, NPU_FLAG_IRQ, 16'h0007,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, 0));
    if (!busy_o || done_pulse_o) $fatal(1, "END must wait for idle units");
    @(negedge clk_i);
    units_idle_i = 1'b1;
    @(posedge clk_i);
    #1;
    if (!done_pulse_o || !irq_pulse_o || busy_o || commands_retired_o != 7)
      $fatal(1, "END completion/IRQ");

    start_task();
    send_command(make_command(8'hff, 0, 16'hbeef,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, 0));
    if (!error_o || error_code_o != NPU_ERR_ILLEGAL_OPCODE
        || error_pc_o != 0 || error_inst_tag_o != 16'hbeef)
      $fatal(1, "illegal opcode first-error capture");

    soft_reset();
    start_task();
    send_command(make_command(NPU_OP_DMA_LOAD, 0, 16'h1001,
                              0, 0, NPU_NONE_INDEX, 0, NPU_NONE_INDEX, 0));
    if (!error_o || error_code_o != NPU_ERR_ILLEGAL_FLAGS)
      $fatal(1, "illegal flags");

    soft_reset();
    start_task();
    send_command(make_command(NPU_OP_WAIT, 0, 16'h1002,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, 0));
    if (!error_o || error_code_o != NPU_ERR_ILLEGAL_FIELDS)
      $fatal(1, "illegal fields");

    soft_reset();
    watchdog_limit_i = 32'd4;
    start_task();
    send_command(make_command(NPU_OP_WAIT, 0, 16'h1003,
                              NPU_NONE_INDEX, NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_NONE_INDEX, NPU_NONE_INDEX,
                              NPU_EVENT_S1_DONE));
    repeat (6) @(posedge clk_i);
    if (!error_o || error_code_o != NPU_ERR_WATCHDOG
        || error_pc_o != 0 || error_inst_tag_o != 16'h1003)
      $fatal(1, "watchdog first-error capture");

    soft_reset();
    watchdog_limit_i = 32'd100;
    start_task();
    @(negedge clk_i);
    execution_error_code_i = 16'h4321;
    execution_error_pc_i = 32'h0000_0080;
    execution_error_tag_i = 16'hcafe;
    execution_error_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    execution_error_i = 1'b0;
    if (!error_o || error_code_o != 16'h4321
        || error_pc_o != 32'h0000_0080 || error_inst_tag_o != 16'hcafe)
      $fatal(1, "execution error capture");

    $display("npu_command_processor: PASS");
    $finish;
  end
endmodule
