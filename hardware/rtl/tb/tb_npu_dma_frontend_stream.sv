`timescale 1ns/1ps

module tb_npu_dma_frontend_stream;
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  localparam integer MAX_COMMANDS = 2048;
  localparam integer MAX_TENSORS = 1024;
  localparam integer MAX_OPERATORS = 1024;
  localparam integer MAX_QUANTS = 1024;
  localparam integer MAX_SEGMENTS = 1024;
  localparam logic [63:0] TENSOR_BASE = 64'h0000_0000_0010_0000;
  localparam logic [63:0] OPERATOR_BASE = 64'h0000_0000_0020_0000;
  localparam logic [63:0] QUANT_BASE = 64'h0000_0000_0030_0000;
  localparam logic [63:0] SEGMENT_BASE = 64'h0000_0000_0040_0000;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic command_valid_i = 1'b0;
  logic command_ready_o;
  logic [127:0] command_bits_i = '0;
  logic descriptor_memory_request_valid_o;
  logic descriptor_memory_request_ready_i;
  logic [63:0] descriptor_memory_request_address_o;
  logic [6:0] descriptor_memory_request_bytes_o;
  logic descriptor_memory_response_valid_i = 1'b0;
  logic descriptor_memory_response_ready_o;
  logic [511:0] descriptor_memory_response_data_i = '0;
  logic descriptor_memory_response_error_i = 1'b0;
  logic request_valid_o;
  logic request_ready_i;
  logic [NPU_DMA_REQUEST_BITS-1:0] request_bits_o;
  logic busy_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;
  logic [3:0] error_detail_o;
  logic [15:0] error_tag_o;

  logic [127:0] commands [0:MAX_COMMANDS-1];
  logic [NPU_DMA_REQUEST_BITS-1:0] expected [0:MAX_COMMANDS-1];
  logic [511:0] tensor_descriptors [0:MAX_TENSORS-1];
  logic [511:0] operator_descriptors [0:MAX_OPERATORS-1];
  logic [255:0] quant_descriptors [0:MAX_QUANTS-1];
  logic [63:0] segment_descriptors [0:MAX_SEGMENTS-1];

  string command_path;
  string expected_path;
  string tensor_path;
  string operator_path;
  string quant_path;
  string segment_path;
  integer request_count;
  integer tensor_count;
  integer operator_count;
  integer quant_count;
  integer segment_count;
  integer accepted_count = 0;
  integer descriptor_read_count = 0;
  integer cycle_count = 0;
  integer index;

  logic backend_pending_q = 1'b0;
  logic [1:0] backend_delay_q = '0;
  logic [511:0] backend_data_q = '0;
  logic backend_error_q = 1'b0;
  integer descriptor_index;

  always #5 clk_i = ~clk_i;
  assign descriptor_memory_request_ready_i = !backend_pending_q
    && !descriptor_memory_response_valid_i && cycle_count[1:0] != 2'b00;
  assign request_ready_i = cycle_count[2:0] != 3'b000;

  npu_dma_frontend dut (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .command_valid_i(command_valid_i),
    .command_ready_o(command_ready_o),
    .command_bits_i(command_bits_i),
    .tensor_desc_base_i(TENSOR_BASE),
    .operator_desc_base_i(OPERATOR_BASE),
    .quant_desc_base_i(QUANT_BASE),
    .segment_desc_base_i(SEGMENT_BASE),
    .activation_base_i(64'd0),
    .weight_base_i(64'd0),
    .bias_base_i(64'd0),
    .quant_param_base_i(64'd0),
    .descriptor_memory_request_valid_o(descriptor_memory_request_valid_o),
    .descriptor_memory_request_ready_i(descriptor_memory_request_ready_i),
    .descriptor_memory_request_address_o(descriptor_memory_request_address_o),
    .descriptor_memory_request_bytes_o(descriptor_memory_request_bytes_o),
    .descriptor_memory_response_valid_i(descriptor_memory_response_valid_i),
    .descriptor_memory_response_ready_o(descriptor_memory_response_ready_o),
    .descriptor_memory_response_data_i(descriptor_memory_response_data_i),
    .descriptor_memory_response_error_i(descriptor_memory_response_error_i),
    .request_valid_o(request_valid_o),
    .request_ready_i(request_ready_i),
    .request_bits_o(request_bits_o),
    .busy_o(busy_o),
    .error_pulse_o(error_pulse_o),
    .error_reason_o(error_reason_o),
    .error_detail_o(error_detail_o),
    .error_tag_o(error_tag_o)
  );

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      cycle_count <= 0;
      descriptor_memory_response_valid_i <= 1'b0;
      descriptor_memory_response_error_i <= 1'b0;
      backend_pending_q <= 1'b0;
      backend_delay_q <= '0;
    end else begin
      cycle_count <= cycle_count + 1;

      if (descriptor_memory_response_valid_i
          && descriptor_memory_response_ready_o) begin
        descriptor_memory_response_valid_i <= 1'b0;
        descriptor_memory_response_error_i <= 1'b0;
      end

      if (descriptor_memory_request_valid_o
          && descriptor_memory_request_ready_i) begin
        descriptor_read_count <= descriptor_read_count + 1;
        backend_pending_q <= 1'b1;
        backend_delay_q <= 2;
        backend_data_q <= '0;
        backend_error_q <= 1'b0;
        if (descriptor_memory_request_address_o >= TENSOR_BASE
            && descriptor_memory_request_address_o
               < TENSOR_BASE + tensor_count * 64) begin
          descriptor_index = (descriptor_memory_request_address_o
                              - TENSOR_BASE) >> 6;
          backend_data_q <= tensor_descriptors[descriptor_index];
          if (descriptor_memory_request_bytes_o != 64)
            backend_error_q <= 1'b1;
        end else if (descriptor_memory_request_address_o >= OPERATOR_BASE
                     && descriptor_memory_request_address_o
                        < OPERATOR_BASE + operator_count * 64) begin
          descriptor_index = (descriptor_memory_request_address_o
                              - OPERATOR_BASE) >> 6;
          backend_data_q <= operator_descriptors[descriptor_index];
          if (descriptor_memory_request_bytes_o != 64)
            backend_error_q <= 1'b1;
        end else if (descriptor_memory_request_address_o >= QUANT_BASE
                     && descriptor_memory_request_address_o
                        < QUANT_BASE + quant_count * 32) begin
          descriptor_index = (descriptor_memory_request_address_o
                              - QUANT_BASE) >> 5;
          backend_data_q <= {256'd0, quant_descriptors[descriptor_index]};
          if (descriptor_memory_request_bytes_o != 32)
            backend_error_q <= 1'b1;
        end else if (descriptor_memory_request_address_o >= SEGMENT_BASE
                     && descriptor_memory_request_address_o
                        < SEGMENT_BASE + segment_count * 8) begin
          descriptor_index = (descriptor_memory_request_address_o
                              - SEGMENT_BASE) >> 3;
          backend_data_q <= {448'd0, segment_descriptors[descriptor_index]};
          if (descriptor_memory_request_bytes_o != 8)
            backend_error_q <= 1'b1;
        end else begin
          backend_error_q <= 1'b1;
        end
      end else if (backend_pending_q) begin
        if (backend_delay_q == 0) begin
          descriptor_memory_response_data_i <= backend_data_q;
          descriptor_memory_response_error_i <= backend_error_q;
          descriptor_memory_response_valid_i <= 1'b1;
          backend_pending_q <= 1'b0;
        end else begin
          backend_delay_q <= backend_delay_q - 1'b1;
        end
      end

      if (request_valid_o && request_ready_i) begin
        if (accepted_count >= request_count)
          $fatal(1, "too many requests");
        if (request_bits_o !== expected[accepted_count])
          $fatal(1, "request mismatch at %0d", accepted_count);
        accepted_count <= accepted_count + 1;
      end
      if (error_pulse_o)
        $fatal(1, "frontend error reason=%0d detail=%0d tag=%0d",
               error_reason_o, error_detail_o, error_tag_o);
      if (cycle_count > 1000000)
        $fatal(1, "frontend stream timeout at %0d/%0d",
               accepted_count, request_count);
    end
  end

  initial begin
    if (!$value$plusargs("COMMAND_HEX=%s", command_path)
        || !$value$plusargs("EXPECTED_HEX=%s", expected_path)
        || !$value$plusargs("TENSOR_HEX=%s", tensor_path)
        || !$value$plusargs("OPERATOR_HEX=%s", operator_path)
        || !$value$plusargs("QUANT_HEX=%s", quant_path)
        || !$value$plusargs("SEGMENT_HEX=%s", segment_path)
        || !$value$plusargs("REQUEST_COUNT=%d", request_count)
        || !$value$plusargs("TENSOR_COUNT=%d", tensor_count)
        || !$value$plusargs("OPERATOR_COUNT=%d", operator_count)
        || !$value$plusargs("QUANT_COUNT=%d", quant_count)
        || !$value$plusargs("SEGMENT_COUNT=%d", segment_count))
      $fatal(1, "missing stream plusargs");
    if (request_count > MAX_COMMANDS || tensor_count > MAX_TENSORS
        || operator_count > MAX_OPERATORS || quant_count > MAX_QUANTS
        || segment_count > MAX_SEGMENTS)
      $fatal(1, "test vector capacity");

    $readmemh(command_path, commands);
    $readmemh(expected_path, expected);
    $readmemh(tensor_path, tensor_descriptors);
    $readmemh(operator_path, operator_descriptors);
    $readmemh(quant_path, quant_descriptors);
    $readmemh(segment_path, segment_descriptors);

    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;
    for (index = 0; index < request_count; index = index + 1) begin
      @(negedge clk_i);
      command_bits_i = commands[index];
      command_valid_i = 1'b1;
      do @(posedge clk_i); while (!command_ready_o);
      @(negedge clk_i);
      command_valid_i = 1'b0;
    end

    while (accepted_count != request_count)
      @(posedge clk_i);
    @(negedge clk_i);
    if (busy_o) $fatal(1, "frontend remained busy");
    $display("npu_dma_frontend_stream: PASS (%0d requests, %0d descriptor reads)",
             accepted_count, descriptor_read_count);
    $finish;
  end

endmodule
