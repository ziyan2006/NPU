`timescale 1ns/1ps

module tb_npu_scratchpad;
  import npu_dma_pkg::*;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic dma_write_valid_i = 1'b0;
  logic dma_write_ready_o;
  logic [1:0] dma_write_kind_i = '0;
  logic dma_write_bank_i = 1'b0;
  logic [31:0] dma_write_address_i = '0;
  logic [63:0] dma_write_data_i = '0;
  logic [7:0] dma_write_strobe_i = '0;
  logic dma_read_request_valid_i = 1'b0;
  logic dma_read_request_ready_o;
  logic [1:0] dma_read_kind_i = '0;
  logic dma_read_bank_i = 1'b0;
  logic [31:0] dma_read_address_i = '0;
  logic dma_read_response_valid_o;
  logic dma_read_response_ready_i = 1'b0;
  logic [63:0] dma_read_response_data_o;
  logic accelerator_write_valid_i = 1'b0;
  logic accelerator_write_ready_o;
  logic [1:0] accelerator_write_kind_i = '0;
  logic accelerator_write_bank_i = 1'b0;
  logic [31:0] accelerator_write_address_i = '0;
  logic [63:0] accelerator_write_data_i = '0;
  logic [7:0] accelerator_write_strobe_i = '0;
  logic accelerator_read_request_valid_i = 1'b0;
  logic accelerator_read_request_ready_o;
  logic [1:0] accelerator_read_kind_i = '0;
  logic accelerator_read_bank_i = 1'b0;
  logic [31:0] accelerator_read_address_i = '0;
  logic accelerator_read_response_valid_o;
  logic accelerator_read_response_ready_i = 1'b0;
  logic [63:0] accelerator_read_response_data_o;
  logic collision_stall_o;
  logic [63:0] held_data;

  always #5 clk_i = ~clk_i;
  npu_scratchpad dut (.*);

  task automatic dma_write(
    input logic [1:0] kind,
    input logic bank,
    input logic [31:0] address,
    input logic [63:0] data,
    input logic [7:0] strobe
  );
    begin
      @(negedge clk_i);
      dma_write_kind_i = kind;
      dma_write_bank_i = bank;
      dma_write_address_i = address;
      dma_write_data_i = data;
      dma_write_strobe_i = strobe;
      dma_write_valid_i = 1'b1;
      do @(posedge clk_i); while (!dma_write_ready_o);
      @(negedge clk_i);
      dma_write_valid_i = 1'b0;
    end
  endtask

  task automatic accelerator_write(
    input logic [1:0] kind,
    input logic bank,
    input logic [31:0] address,
    input logic [63:0] data
  );
    begin
      @(negedge clk_i);
      accelerator_write_kind_i = kind;
      accelerator_write_bank_i = bank;
      accelerator_write_address_i = address;
      accelerator_write_data_i = data;
      accelerator_write_strobe_i = 8'hff;
      accelerator_write_valid_i = 1'b1;
      do @(posedge clk_i); while (!accelerator_write_ready_o);
      @(negedge clk_i);
      accelerator_write_valid_i = 1'b0;
    end
  endtask

  task automatic dma_read_check(
    input logic [1:0] kind,
    input logic bank,
    input logic [31:0] address,
    input logic [63:0] expected
  );
    begin
      @(negedge clk_i);
      dma_read_kind_i = kind;
      dma_read_bank_i = bank;
      dma_read_address_i = address;
      dma_read_request_valid_i = 1'b1;
      do @(posedge clk_i); while (!dma_read_request_ready_o);
      @(negedge clk_i);
      dma_read_request_valid_i = 1'b0;
      do @(negedge clk_i); while (!dma_read_response_valid_o);
      if (dma_read_response_data_o !== expected)
        $fatal(1, "DMA read mismatch kind=%0d bank=%0d address=%0d",
               kind, bank, address);
      dma_read_response_ready_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      dma_read_response_ready_i = 1'b0;
    end
  endtask

  task automatic accelerator_read_check(
    input logic [1:0] kind,
    input logic bank,
    input logic [31:0] address,
    input logic [63:0] expected
  );
    begin
      @(negedge clk_i);
      accelerator_read_kind_i = kind;
      accelerator_read_bank_i = bank;
      accelerator_read_address_i = address;
      accelerator_read_request_valid_i = 1'b1;
      do @(posedge clk_i); while (!accelerator_read_request_ready_o);
      @(negedge clk_i);
      accelerator_read_request_valid_i = 1'b0;
      do @(negedge clk_i); while (!accelerator_read_response_valid_o);
      if (accelerator_read_response_data_o !== expected)
        $fatal(1, "accelerator read mismatch");
      accelerator_read_response_ready_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      accelerator_read_response_ready_i = 1'b0;
    end
  endtask

  initial begin
    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;

    // Byte strobes update only selected lanes, and the accelerator port sees
    // the same physical A0 memory through the second BRAM port.
    dma_write(NPU_SPAD_A, 1'b0, 32'd16,
              64'h1122_3344_5566_7788, 8'hff);
    dma_write(NPU_SPAD_A, 1'b0, 32'd16,
              64'haabb_ccdd_eeff_0011, 8'h0c);
    accelerator_read_check(NPU_SPAD_A, 1'b0, 32'd16,
                           64'h1122_3344_eeff_7788);

    // Each kind/bank is physically distinct.
    accelerator_write(NPU_SPAD_W, 1'b1, 32'd24,
                      64'hdead_beef_1234_5678);
    dma_read_check(NPU_SPAD_W, 1'b1, 32'd24,
                   64'hdead_beef_1234_5678);
    dma_write(NPU_SPAD_O, 1'b1, 32'd8,
              64'h0102_0304_0506_0708, 8'hff);
    dma_read_check(NPU_SPAD_O, 1'b1, 32'd8,
                   64'h0102_0304_0506_0708);

    // A simultaneous read/write on one physical port accepts the write and
    // holds the read request for the following cycle instead of deadlocking.
    @(negedge clk_i);
    dma_write_kind_i = NPU_SPAD_A;
    dma_write_bank_i = 1'b0;
    dma_write_address_i = 32'd192;
    dma_write_data_i = 64'h7654_3210_fedc_ba98;
    dma_write_strobe_i = 8'hff;
    dma_write_valid_i = 1'b1;
    dma_read_kind_i = NPU_SPAD_A;
    dma_read_bank_i = 1'b0;
    dma_read_address_i = 32'd192;
    dma_read_request_valid_i = 1'b1;
    @(posedge clk_i);
    if (!dma_write_ready_o || dma_read_request_ready_o)
      $fatal(1, "same-port write priority");
    @(negedge clk_i);
    dma_write_valid_i = 1'b0;
    @(posedge clk_i);
    if (!dma_read_request_ready_o)
      $fatal(1, "held read did not resume");
    @(negedge clk_i);
    dma_read_request_valid_i = 1'b0;
    do @(negedge clk_i); while (!dma_read_response_valid_o);
    if (dma_read_response_data_o !== 64'h7654_3210_fedc_ba98)
      $fatal(1, "write-priority readback");
    dma_read_response_ready_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    dma_read_response_ready_i = 1'b0;

    // A response must remain stable while its consumer applies back-pressure.
    @(negedge clk_i);
    dma_read_kind_i = NPU_SPAD_O;
    dma_read_bank_i = 1'b1;
    dma_read_address_i = 32'd8;
    dma_read_request_valid_i = 1'b1;
    do @(posedge clk_i); while (!dma_read_request_ready_o);
    @(negedge clk_i);
    dma_read_request_valid_i = 1'b0;
    do @(negedge clk_i); while (!dma_read_response_valid_o);
    held_data = dma_read_response_data_o;
    repeat (3) begin
      @(posedge clk_i);
      if (!dma_read_response_valid_o || dma_read_response_data_o !== held_data)
        $fatal(1, "DMA response changed under back-pressure");
    end
    @(negedge clk_i);
    dma_read_response_ready_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    dma_read_response_ready_i = 1'b0;

    // Different banks can be written in the same cycle.
    dma_write_kind_i = NPU_SPAD_A;
    dma_write_bank_i = 1'b0;
    dma_write_address_i = 32'd64;
    dma_write_data_i = 64'ha0a0_a0a0_a0a0_a0a0;
    dma_write_strobe_i = 8'hff;
    dma_write_valid_i = 1'b1;
    accelerator_write_kind_i = NPU_SPAD_A;
    accelerator_write_bank_i = 1'b1;
    accelerator_write_address_i = 32'd64;
    accelerator_write_data_i = 64'hb1b1_b1b1_b1b1_b1b1;
    accelerator_write_strobe_i = 8'hff;
    accelerator_write_valid_i = 1'b1;
    @(posedge clk_i);
    if (!dma_write_ready_o || !accelerator_write_ready_o || collision_stall_o)
      $fatal(1, "different-bank concurrency blocked");
    @(negedge clk_i);
    dma_write_valid_i = 1'b0;
    accelerator_write_valid_i = 1'b0;
    dma_read_check(NPU_SPAD_A, 1'b0, 32'd64,
                   64'ha0a0_a0a0_a0a0_a0a0);
    dma_read_check(NPU_SPAD_A, 1'b1, 32'd64,
                   64'hb1b1_b1b1_b1b1_b1b1);

    // Same-word writes prioritize DMA and hold the accelerator request until
    // the collision is gone.
    @(negedge clk_i);
    dma_write_kind_i = NPU_SPAD_O;
    dma_write_bank_i = 1'b0;
    dma_write_address_i = 32'd128;
    dma_write_data_i = 64'h1111_1111_1111_1111;
    dma_write_strobe_i = 8'hff;
    dma_write_valid_i = 1'b1;
    accelerator_write_kind_i = NPU_SPAD_O;
    accelerator_write_bank_i = 1'b0;
    accelerator_write_address_i = 32'd128;
    accelerator_write_data_i = 64'h2222_2222_2222_2222;
    accelerator_write_strobe_i = 8'hff;
    accelerator_write_valid_i = 1'b1;
    @(posedge clk_i);
    if (!dma_write_ready_o || accelerator_write_ready_o || !collision_stall_o)
      $fatal(1, "same-word collision arbitration");
    @(negedge clk_i);
    dma_write_valid_i = 1'b0;
    @(posedge clk_i);
    if (!accelerator_write_ready_o || collision_stall_o)
      $fatal(1, "accelerator did not resume after collision");
    @(negedge clk_i);
    accelerator_write_valid_i = 1'b0;
    dma_read_check(NPU_SPAD_O, 1'b0, 32'd128,
                   64'h2222_2222_2222_2222);

    $display("npu_scratchpad: PASS");
    $finish;
  end

endmodule
