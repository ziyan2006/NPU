`timescale 1ns/1ps

module tb_npu_descriptor_cache;
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic [63:0] tensor_base_i = 64'h0000_0000_0000_1000;
  logic [63:0] operator_base_i = 64'h0000_0000_0000_4000;
  logic [63:0] quant_base_i = 64'h0000_0000_0000_8000;
  logic [63:0] segment_base_i = 64'h0000_0000_0000_a000;
  logic request_valid_i = 1'b0;
  logic request_ready_o;
  logic [1:0] request_kind_i = '0;
  logic [15:0] request_index_i = '0;
  logic response_valid_o;
  logic response_ready_i = 1'b0;
  logic [511:0] response_data_o;
  logic [6:0] response_bytes_o;
  logic response_error_o;
  logic memory_request_valid_o;
  logic memory_request_ready_i = 1'b1;
  logic [63:0] memory_request_address_o;
  logic [6:0] memory_request_bytes_o;
  logic memory_response_valid_i = 1'b0;
  logic memory_response_ready_o;
  logic [511:0] memory_response_data_i = '0;
  logic memory_response_error_i = 1'b0;
  integer memory_read_count = 0;

  always #5 clk_i = ~clk_i;
  always @(posedge clk_i)
    if (memory_request_valid_o && memory_request_ready_i)
      memory_read_count <= memory_read_count + 1;

  npu_descriptor_cache dut (.*);

  task automatic begin_request(
    input logic [1:0] kind,
    input logic [15:0] index
  );
    begin
      @(negedge clk_i);
      request_kind_i = kind;
      request_index_i = index;
      request_valid_i = 1'b1;
      do @(posedge clk_i); while (!request_ready_o);
      @(negedge clk_i);
      request_valid_i = 1'b0;
    end
  endtask

  task automatic return_memory(
    input logic [511:0] data,
    input logic error
  );
    begin
      do @(negedge clk_i); while (!memory_response_ready_o);
      memory_response_data_i = data;
      memory_response_error_i = error;
      memory_response_valid_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      memory_response_valid_i = 1'b0;
      memory_response_error_i = 1'b0;
    end
  endtask

  task automatic consume_response(
    input logic [511:0] expected_data,
    input logic [6:0] expected_bytes,
    input logic expected_error
  );
    begin
      do @(negedge clk_i); while (!response_valid_o);
      if (response_data_o !== expected_data
          || response_bytes_o !== expected_bytes
          || response_error_o !== expected_error)
        $fatal(1, "descriptor response mismatch");
      response_ready_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      response_ready_i = 1'b0;
    end
  endtask

  initial begin
    repeat (3) @(posedge clk_i);
    rst_ni = 1'b1;

    begin_request(NPU_DESC_TENSOR, 16'd3);
    if (!memory_request_valid_o
        || memory_request_address_o != 64'h10c0
        || memory_request_bytes_o != 7'd64)
      $fatal(1, "tensor descriptor miss request");
    return_memory(512'h0123_4567_89ab_cdef, 1'b0);
    consume_response(512'h0123_4567_89ab_cdef, 7'd64, 1'b0);
    if (memory_read_count != 1) $fatal(1, "first miss count");

    // The same key must hit without another backend access.
    begin_request(NPU_DESC_TENSOR, 16'd3);
    if (memory_request_valid_o) $fatal(1, "cache hit issued memory read");
    consume_response(512'h0123_4567_89ab_cdef, 7'd64, 1'b0);
    if (memory_read_count != 1) $fatal(1, "hit changed miss count");

    begin_request(NPU_DESC_QUANT, 16'd2);
    if (!memory_request_valid_o
        || memory_request_address_o != 64'h8040
        || memory_request_bytes_o != 7'd32)
      $fatal(1, "quant descriptor address");
    return_memory(512'h55aa, 1'b0);
    consume_response(512'h55aa, 7'd32, 1'b0);

    begin_request(NPU_DESC_SEGMENT, 16'd5);
    if (!memory_request_valid_o
        || memory_request_address_o != 64'ha028
        || memory_request_bytes_o != 7'd8)
      $fatal(1, "segment descriptor address");
    return_memory(512'hdead_beef, 1'b1);
    consume_response(512'hdead_beef, 7'd8, 1'b1);

    begin_request(NPU_DESC_TENSOR, NPU_NONE_INDEX);
    if (memory_request_valid_o) $fatal(1, "NONE index reached memory");
    consume_response(512'h0, 7'd64, 1'b1);

    @(negedge clk_i);
    soft_reset_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    soft_reset_i = 1'b0;
    begin_request(NPU_DESC_TENSOR, 16'd1024);
    if (!memory_request_valid_o
        || memory_request_address_o != 64'h0000_0000_0001_1000)
      $fatal(1, "large descriptor index/address width");
    return_memory(512'h9876, 1'b0);
    consume_response(512'h9876, 7'd64, 1'b0);

    // Once the backend accepted a read, soft reset must drain its response
    // rather than abandoning an in-flight transaction.
    begin_request(NPU_DESC_OPERATOR, 16'd7);
    if (!memory_request_valid_o) $fatal(1, "drain setup miss");
    @(posedge clk_i);
    @(negedge clk_i);
    soft_reset_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    soft_reset_i = 1'b0;
    if (!memory_response_ready_o) $fatal(1, "reset did not drain backend");
    return_memory(512'hfeed, 1'b0);
    #1;
    if (response_valid_o || !request_ready_o)
      $fatal(1, "discarded response escaped after reset");

    $display("npu_descriptor_cache: PASS");
    $finish;
  end
endmodule
