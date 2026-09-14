`timescale 1ns/1ps

module tb_npu_dma_stream;
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  localparam integer MAX_REQUESTS = 2048;
  logic [127:0] commands [0:MAX_REQUESTS-1];
  logic [511:0] source_tensors [0:MAX_REQUESTS-1];
  logic [511:0] destination_tensors [0:MAX_REQUESTS-1];
  logic [511:0] operators [0:MAX_REQUESTS-1];
  logic [255:0] quant_descriptors [0:MAX_REQUESTS-1];
  logic [63:0] segments [0:MAX_REQUESTS-1];
  logic [NPU_DMA_REQUEST_BITS-1:0] expected [0:MAX_REQUESTS-1];

  logic command_valid_i = 1'b0;
  logic [127:0] command_bits_i = '0;
  logic [511:0] src_tensor_desc_bits_i = '0;
  logic [511:0] dst_tensor_desc_bits_i = '0;
  logic [511:0] operator_desc_bits_i = '0;
  logic [255:0] quant_desc_bits_i = '0;
  logic [63:0] segment_desc_bits_i = '0;
  logic [63:0] activation_base_i = '0;
  logic [63:0] weight_base_i = '0;
  logic [63:0] bias_base_i = '0;
  logic [63:0] quant_param_base_i = '0;
  logic request_valid_o;
  logic [NPU_DMA_REQUEST_BITS-1:0] request_bits_o;
  logic error_valid_o;
  logic [3:0] error_reason_o;

  string command_path;
  string source_path;
  string destination_path;
  string operator_path;
  string quant_path;
  string segment_path;
  string expected_path;
  integer request_count;
  integer index;

  npu_dma_agu dut (.*);

  initial begin
    if (!$value$plusargs("COMMAND_HEX=%s", command_path)
        || !$value$plusargs("SOURCE_HEX=%s", source_path)
        || !$value$plusargs("DESTINATION_HEX=%s", destination_path)
        || !$value$plusargs("OPERATOR_HEX=%s", operator_path)
        || !$value$plusargs("QUANT_HEX=%s", quant_path)
        || !$value$plusargs("SEGMENT_HEX=%s", segment_path)
        || !$value$plusargs("EXPECTED_HEX=%s", expected_path)
        || !$value$plusargs("REQUEST_COUNT=%d", request_count))
      $fatal(1, "missing DMA stream plusargs");
    if (request_count <= 0 || request_count > MAX_REQUESTS)
      $fatal(1, "invalid DMA request count %0d", request_count);

    $readmemh(command_path, commands);
    $readmemh(source_path, source_tensors);
    $readmemh(destination_path, destination_tensors);
    $readmemh(operator_path, operators);
    $readmemh(quant_path, quant_descriptors);
    $readmemh(segment_path, segments);
    $readmemh(expected_path, expected);

    command_valid_i = 1'b1;
    for (index = 0; index < request_count; index = index + 1) begin
      command_bits_i = commands[index];
      src_tensor_desc_bits_i = source_tensors[index];
      dst_tensor_desc_bits_i = destination_tensors[index];
      operator_desc_bits_i = operators[index];
      quant_desc_bits_i = quant_descriptors[index];
      segment_desc_bits_i = segments[index];
      #1;
      if (!request_valid_o || error_valid_o) begin
        $display("AGU rejected request %0d reason %0d command %032x",
                 index, error_reason_o, command_bits_i);
        $fatal(1, "DMA stream request rejected");
      end
      if (request_bits_o !== expected[index]) begin
        $display("request %0d expected %h", index, expected[index]);
        $display("request %0d actual   %h", index, request_bits_o);
        $fatal(1, "DMA stream request mismatch");
      end
    end
    $display("npu_dma_stream: PASS (%0d requests)", request_count);
    $finish;
  end
endmodule
