`timescale 1ns/1ps

module tb_audio_async_fifo;
  localparam integer DEPTH = 8192;

  logic wr_clk = 0;
  logic wr_rst_n = 0;
  logic wr_valid = 0;
  logic [63:0] wr_data = 0;
  logic wr_full;
  logic [13:0] wr_level;
  logic rd_clk = 0;
  logic rd_rst_n = 0;
  logic rd_ready = 0;
  logic rd_valid;
  logic [63:0] rd_data;
  logic rd_empty;
  logic [13:0] rd_level;
  integer seed;
  integer index;
  integer expected;
  integer accepted_writes = 0;
  integer accepted_reads = 0;

  always #5 wr_clk = ~wr_clk;
  always #8.5 rd_clk = ~rd_clk;

  audio_async_fifo dut (
    .wr_clk_i(wr_clk), .wr_rst_ni(wr_rst_n),
    .wr_valid_i(wr_valid), .wr_data_i(wr_data),
    .wr_full_o(wr_full), .wr_level_o(wr_level),
    .rd_clk_i(rd_clk), .rd_rst_ni(rd_rst_n),
    .rd_ready_i(rd_ready), .rd_valid_o(rd_valid),
    .rd_data_o(rd_data), .rd_empty_o(rd_empty), .rd_level_o(rd_level)
  );

  always_ff @(posedge wr_clk)
    if (wr_rst_n && wr_valid && !wr_full)
      accepted_writes <= accepted_writes + 1;

  always_ff @(posedge rd_clk)
    if (rd_rst_n && rd_ready && rd_valid)
      accepted_reads <= accepted_reads + 1;

  task automatic push(input logic [63:0] value);
    begin
      @(negedge wr_clk); wr_data = value; wr_valid = 1;
      do @(posedge wr_clk); while (wr_full);
      @(negedge wr_clk); wr_valid = 0;
    end
  endtask

  task automatic pop_and_expect(input logic [63:0] value);
    begin
      while (!rd_valid) @(posedge rd_clk);
      repeat ($urandom(seed) % 3) @(posedge rd_clk);
      if (!rd_valid || rd_data !== value)
        $fatal(1, "FIFO order mismatch got=%h expected=%h", rd_data, value);
      @(negedge rd_clk); rd_ready = 1;
      @(posedge rd_clk);
      @(negedge rd_clk); rd_ready = 0;
    end
  endtask

  initial begin
    if (!$value$plusargs("SEED=%d", seed))
      seed = 1;
    repeat (4) @(posedge wr_clk);
    wr_rst_n = 1;
    repeat (3) @(posedge rd_clk);
    rd_rst_n = 1;

    for (index = 0; index < DEPTH; index = index + 1)
      push(64'h1000_0000_0000_0000 + index);
    repeat (5) @(posedge wr_clk);
    if (!wr_full || wr_level != DEPTH || accepted_writes != DEPTH)
      $fatal(1, "full state mismatch full=%b level=%0d writes=%0d",
             wr_full, wr_level, accepted_writes);

    @(negedge wr_clk); wr_data = 64'hdead_beef_cafe_f00d; wr_valid = 1;
    repeat (3) @(posedge wr_clk);
    @(negedge wr_clk); wr_valid = 0;
    if (accepted_writes != DEPTH)
      $fatal(1, "write was accepted while full");

    for (expected = 0; expected < DEPTH/2; expected = expected + 1)
      pop_and_expect(64'h1000_0000_0000_0000 + expected);

    for (index = 0; index < DEPTH/2; index = index + 1)
      push(64'h2000_0000_0000_0000 + index);

    for (expected = DEPTH/2; expected < DEPTH; expected = expected + 1)
      pop_and_expect(64'h1000_0000_0000_0000 + expected);
    for (expected = 0; expected < DEPTH/2; expected = expected + 1)
      pop_and_expect(64'h2000_0000_0000_0000 + expected);

    repeat (6) @(posedge rd_clk);
    if (!rd_empty || rd_valid || rd_level != 0)
      $fatal(1, "empty state mismatch empty=%b valid=%b level=%0d",
             rd_empty, rd_valid, rd_level);
    @(negedge rd_clk); rd_ready = 1;
    repeat (3) @(posedge rd_clk);
    @(negedge rd_clk); rd_ready = 0;
    if (accepted_reads != DEPTH + DEPTH/2)
      $fatal(1, "read was accepted while empty");

    $display("audio_async_fifo: PASS seed=%0d", seed);
    $finish;
  end
endmodule
