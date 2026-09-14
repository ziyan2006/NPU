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
  logic [511:0] accelerator_write_data_i = '0;
  logic [63:0] accelerator_write_strobe_i = '0;
  logic accelerator_read_request_valid_i = 1'b0;
  logic accelerator_read_request_ready_o;
  logic [1:0] accelerator_read_kind_i = '0;
  logic accelerator_read_bank_i = 1'b0;
  logic [31:0] accelerator_read_address_i = '0;
  logic accelerator_read_response_valid_o;
  logic accelerator_read_response_ready_i = 1'b0;
  logic [511:0] accelerator_read_response_data_o;
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
    input logic [511:0] data,
    input logic [63:0] strobe
  );
    begin
      @(negedge clk_i);
      accelerator_write_kind_i = kind;
      accelerator_write_bank_i = bank;
      accelerator_write_address_i = address;
      accelerator_write_data_i = data;
      accelerator_write_strobe_i = strobe;
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
    input logic [511:0] expected
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

    // Byte strobes update only selected DMA lanes. A compute read returns both
    // 64-bit lanes of the 128-bit activation row in address order.
    dma_write(NPU_SPAD_A, 1'b0, 32'd16,
              64'h1122_3344_5566_7788, 8'hff);
    dma_write(NPU_SPAD_A, 1'b0, 32'd16,
              64'haabb_ccdd_eeff_0011, 8'h0c);
    dma_write(NPU_SPAD_A, 1'b0, 32'd24,
              64'h99aa_bbcc_ddee_ff00, 8'hff);
    accelerator_read_check(NPU_SPAD_A, 1'b0, 32'd16,
      {384'd0, 64'h99aa_bbcc_ddee_ff00, 64'h1122_3344_eeff_7788});

    // A 512-bit weight write distributes eight consecutive 64-bit lanes.
    accelerator_write(NPU_SPAD_W, 1'b1, 32'd64,
      {64'h7777_7777_7777_7777, 64'h6666_6666_6666_6666,
       64'h5555_5555_5555_5555, 64'h4444_4444_4444_4444,
       64'h3333_3333_3333_3333, 64'h2222_2222_2222_2222,
       64'h1111_1111_1111_1111, 64'hdead_beef_1234_5678},
      64'hffff_ffff_ffff_ffff);
    accelerator_read_check(NPU_SPAD_W, 1'b1, 32'd64,
      {64'h7777_7777_7777_7777, 64'h6666_6666_6666_6666,
       64'h5555_5555_5555_5555, 64'h4444_4444_4444_4444,
       64'h3333_3333_3333_3333, 64'h2222_2222_2222_2222,
       64'h1111_1111_1111_1111, 64'hdead_beef_1234_5678});
    dma_read_check(NPU_SPAD_W, 1'b1, 32'd64,
                   64'hdead_beef_1234_5678);
    dma_read_check(NPU_SPAD_W, 1'b1, 32'd120,
                   64'h7777_7777_7777_7777);
    // Each kind/bank remains physically distinct.
    dma_write(NPU_SPAD_O, 1'b1, 32'd8,
              64'h0102_0304_0506_0708, 8'hff);
    dma_read_check(NPU_SPAD_O, 1'b1, 32'd8,
                   64'h0102_0304_0506_0708);

    // Once the two-stage read path is full, aligned compute requests and
    // responses both sustain one transfer per cycle.
    dma_write(NPU_SPAD_A, 1'b0, 32'd32,
              64'h0000_0000_0000_0032, 8'hff);
    dma_write(NPU_SPAD_A, 1'b0, 32'd40,
              64'h0000_0000_0000_0040, 8'hff);
    dma_write(NPU_SPAD_A, 1'b0, 32'd48,
              64'h0000_0000_0000_0048, 8'hff);
    dma_write(NPU_SPAD_A, 1'b0, 32'd56,
              64'h0000_0000_0000_0056, 8'hff);
    @(negedge clk_i);
    accelerator_read_kind_i = NPU_SPAD_A;
    accelerator_read_bank_i = 1'b0;
    accelerator_read_address_i = 32'd32;
    accelerator_read_response_ready_i = 1'b1;
    accelerator_read_request_valid_i = 1'b1;
    @(posedge clk_i);
    if (!accelerator_read_request_ready_o)
      $fatal(1, "first pipelined read was not accepted");
    @(negedge clk_i);
    accelerator_read_address_i = 32'd48;
    @(posedge clk_i);
    if (!accelerator_read_request_ready_o)
      $fatal(1, "consecutive pipelined read was not accepted");
    @(negedge clk_i);
    accelerator_read_request_valid_i = 1'b0;
    if (!accelerator_read_response_valid_o
        || accelerator_read_response_data_o[127:0]
           !== {64'h0000_0000_0000_0040, 64'h0000_0000_0000_0032})
      $fatal(1, "first pipelined response mismatch");
    @(posedge clk_i);
    @(negedge clk_i);
    if (!accelerator_read_response_valid_o
        || accelerator_read_response_data_o[127:0]
           !== {64'h0000_0000_0000_0056, 64'h0000_0000_0000_0048})
      $fatal(1, "second pipelined response mismatch");
    @(posedge clk_i);
    @(negedge clk_i);
    accelerator_read_response_ready_i = 1'b0;

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
    accelerator_write_data_i = {384'd0,
      64'hb2b2_b2b2_b2b2_b2b2, 64'hb1b1_b1b1_b1b1_b1b1};
    accelerator_write_strobe_i = 64'h0000_0000_0000_ffff;
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
    accelerator_write_data_i = {384'd0,
      64'h3333_3333_3333_3333, 64'h2222_2222_2222_2222};
    accelerator_write_strobe_i = 64'h0000_0000_0000_ffff;
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
