`timescale 1ns/1ps

module tb_npu_task_loader;
  localparam logic [63:0] TASK_BASE = 64'h0000_0000_1000_0000;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_valid_i = 1'b0;
  logic start_ready_o;
  logic [31:0] task_bytes_i;

  logic memory_request_valid;
  logic memory_request_ready;
  logic [63:0] memory_request_address;
  logic [6:0] memory_request_bytes;
  logic memory_response_valid;
  logic memory_response_ready;
  logic [511:0] memory_response_data;
  logic memory_response_error;

  logic [63:0] axi_araddr;
  logic [7:0] axi_arlen;
  logic [2:0] axi_arsize;
  logic [1:0] axi_arburst;
  logic axi_arvalid;
  logic axi_arready;
  logic [63:0] axi_rdata;
  logic [1:0] axi_rresp = 2'b00;
  logic axi_rlast;
  logic axi_rvalid;
  logic axi_rready;

  logic lut_write_valid;
  logic lut_write_ready;
  logic [11:0] lut_write_address;
  logic [15:0] lut_write_data;
  logic busy;
  logic done_pulse;
  logic error_pulse;
  logic [3:0] error_reason;
  logic [63:0] command_base;
  logic [31:0] command_bytes;
  logic [31:0] command_count;
  logic [63:0] tensor_desc_base;
  logic [63:0] operator_desc_base;
  logic [63:0] quant_desc_base;
  logic [63:0] segment_desc_base;
  logic [63:0] activation_base;
  logic [63:0] weight_base;
  logic [63:0] bias_base;
  logic [63:0] quant_param_base;
  logic [15:0] input_tensor;
  logic [15:0] output_tensor;

  logic [63:0] header_memory [0:15];
  logic [63:0] lut_memory [0:1023];
  logic [31:0] lfsr_q = 32'h14c3_7721;
  logic read_active_q = 1'b0;
  logic [63:0] read_base_q = '0;
  integer read_beat_q = 0;
  integer read_beats_q = 0;
  integer lut_write_count_q = 0;
  integer cycles_q = 0;

  integer expected_error;
  integer expected_error_reason;
  integer expected_total_bytes;
  integer command_offset;
  integer tensor_offset;
  integer operator_offset;
  integer quant_offset;
  integer segment_offset;
  integer activation_offset;
  integer weight_offset;
  integer bias_offset;
  integer quant_param_offset;
  integer lut_offset;
  string header_hex;
  string lut_hex;

  always #5 clk_i = ~clk_i;

  assign axi_arready = !read_active_q && lfsr_q[0];
  assign axi_rvalid = read_active_q && lfsr_q[1];
  assign axi_rlast = read_active_q && read_beat_q + 1 == read_beats_q;
  assign lut_write_ready = lfsr_q[2] || lfsr_q[3];
  always_comb begin
    axi_rdata = '0;
    if (read_base_q + read_beat_q * 8 < TASK_BASE + 128) begin
      axi_rdata = header_memory[
        (read_base_q - TASK_BASE) / 8 + read_beat_q];
    end else if (read_base_q + read_beat_q * 8
                 >= TASK_BASE + lut_offset
                 && read_base_q + read_beat_q * 8
                    < TASK_BASE + lut_offset + 8192) begin
      axi_rdata = lut_memory[
        (read_base_q - TASK_BASE - lut_offset) / 8 + read_beat_q];
    end
  end

  npu_axi_block_reader reader (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .request_valid_i(memory_request_valid),
    .request_ready_o(memory_request_ready),
    .request_address_i(memory_request_address),
    .request_bytes_i(memory_request_bytes),
    .response_valid_o(memory_response_valid),
    .response_ready_i(memory_response_ready),
    .response_data_o(memory_response_data),
    .response_error_o(memory_response_error),
    .m_axi_araddr_o(axi_araddr), .m_axi_arlen_o(axi_arlen),
    .m_axi_arsize_o(axi_arsize), .m_axi_arburst_o(axi_arburst),
    .m_axi_arvalid_o(axi_arvalid), .m_axi_arready_i(axi_arready),
    .m_axi_rdata_i(axi_rdata), .m_axi_rresp_i(axi_rresp),
    .m_axi_rlast_i(axi_rlast), .m_axi_rvalid_i(axi_rvalid),
    .m_axi_rready_o(axi_rready));

  npu_task_loader dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_valid_i(start_valid_i), .start_ready_o(start_ready_o),
    .task_base_i(TASK_BASE), .task_bytes_i(task_bytes_i),
    .memory_request_valid_o(memory_request_valid),
    .memory_request_ready_i(memory_request_ready),
    .memory_request_address_o(memory_request_address),
    .memory_request_bytes_o(memory_request_bytes),
    .memory_response_valid_i(memory_response_valid),
    .memory_response_ready_o(memory_response_ready),
    .memory_response_data_i(memory_response_data),
    .memory_response_error_i(memory_response_error),
    .lut_write_valid_o(lut_write_valid),
    .lut_write_ready_i(lut_write_ready),
    .lut_write_address_o(lut_write_address),
    .lut_write_data_o(lut_write_data), .busy_o(busy),
    .done_pulse_o(done_pulse), .error_pulse_o(error_pulse),
    .error_reason_o(error_reason), .command_base_o(command_base),
    .command_bytes_o(command_bytes), .command_count_o(command_count),
    .tensor_desc_base_o(tensor_desc_base),
    .operator_desc_base_o(operator_desc_base),
    .quant_desc_base_o(quant_desc_base),
    .segment_desc_base_o(segment_desc_base),
    .activation_base_o(activation_base), .weight_base_o(weight_base),
    .bias_base_o(bias_base), .quant_param_base_o(quant_param_base),
    .input_tensor_o(input_tensor), .output_tensor_o(output_tensor));

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      lfsr_q <= 32'h14c3_7721;
      read_active_q <= 1'b0;
      read_base_q <= '0;
      read_beat_q <= 0;
      read_beats_q <= 0;
      lut_write_count_q <= 0;
      cycles_q <= 0;
    end else begin
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      cycles_q <= cycles_q + 1;
      if (cycles_q > 100000)
        $fatal(1, "task loader timeout");

      if (axi_arvalid && axi_arready) begin
        if (axi_arsize != 3 || axi_arburst != 2'b01 || axi_arlen != 7)
          $fatal(1, "invalid block-reader AR attributes");
        if (!((axi_araddr >= TASK_BASE && axi_araddr < TASK_BASE + 128)
              || (axi_araddr >= TASK_BASE + lut_offset
                  && axi_araddr < TASK_BASE + lut_offset + 8192)))
          $fatal(1, "unexpected read address %h", axi_araddr);
        read_active_q <= 1'b1;
        read_base_q <= axi_araddr;
        read_beat_q <= 0;
        read_beats_q <= axi_arlen + 1;
      end
      if (axi_rvalid && axi_rready) begin
        if (axi_rlast) begin
          read_active_q <= 1'b0;
          read_beat_q <= 0;
        end else begin
          read_beat_q <= read_beat_q + 1;
        end
      end

      if (lut_write_valid && lut_write_ready) begin
        if (lut_write_address != lut_write_count_q)
          $fatal(1, "LUT address mismatch %0d != %0d",
                 lut_write_address, lut_write_count_q);
        if (lut_write_data
            != lut_memory[lut_write_address >> 2]
               [(lut_write_address & 3)*16 +: 16])
          $fatal(1, "LUT data mismatch at %0d", lut_write_address);
        lut_write_count_q <= lut_write_count_q + 1;
      end
    end
  end

  initial begin
    if (!$value$plusargs("HEADER=%s", header_hex)
        || !$value$plusargs("LUT=%s", lut_hex)
        || !$value$plusargs("EXPECT_ERROR=%d", expected_error)
        || !$value$plusargs("ERROR_REASON=%d", expected_error_reason)
        || !$value$plusargs("TOTAL_BYTES=%d", expected_total_bytes)
        || !$value$plusargs("COMMAND_OFFSET=%d", command_offset)
        || !$value$plusargs("TENSOR_OFFSET=%d", tensor_offset)
        || !$value$plusargs("OPERATOR_OFFSET=%d", operator_offset)
        || !$value$plusargs("QUANT_OFFSET=%d", quant_offset)
        || !$value$plusargs("SEGMENT_OFFSET=%d", segment_offset)
        || !$value$plusargs("ACTIVATION_OFFSET=%d", activation_offset)
        || !$value$plusargs("WEIGHT_OFFSET=%d", weight_offset)
        || !$value$plusargs("BIAS_OFFSET=%d", bias_offset)
        || !$value$plusargs("QUANT_PARAM_OFFSET=%d", quant_param_offset)
        || !$value$plusargs("LUT_OFFSET=%d", lut_offset))
      $fatal(1, "missing plusargs");
    $readmemh(header_hex, header_memory);
    $readmemh(lut_hex, lut_memory);
    task_bytes_i = expected_total_bytes;

    repeat (5) @(posedge clk_i);
    rst_ni <= 1'b1;
    @(posedge clk_i);
    start_valid_i <= 1'b1;
    do @(posedge clk_i); while (!start_ready_o);
    start_valid_i <= 1'b0;

    fork
      begin
        wait(done_pulse || error_pulse);
        if (expected_error) begin
          if (!error_pulse || error_reason != expected_error_reason)
            $fatal(1, "expected error %0d, got done=%0d error=%0d reason=%0d",
                   expected_error_reason, done_pulse, error_pulse, error_reason);
          if (lut_write_count_q != 0)
            $fatal(1, "invalid header must not load LUT");
        end else begin
          if (!done_pulse || error_pulse)
            $fatal(1, "valid image did not complete");
          if (lut_write_count_q != 4096)
            $fatal(1, "expected 4096 LUT writes, got %0d", lut_write_count_q);
          if (command_base != TASK_BASE + command_offset
              || command_bytes != 29904 || command_count != 1869
              || tensor_desc_base != TASK_BASE + tensor_offset
              || operator_desc_base != TASK_BASE + operator_offset
              || quant_desc_base != TASK_BASE + quant_offset
              || segment_desc_base != TASK_BASE + segment_offset
              || activation_base != TASK_BASE + activation_offset
              || weight_base != TASK_BASE + weight_offset
              || bias_base != TASK_BASE + bias_offset
              || quant_param_base != TASK_BASE + quant_param_offset
              || input_tensor != 0 || output_tensor != 41)
            $fatal(1, "parsed task configuration mismatch");
        end
        $display("task loader: PASS error=%0d cycles=%0d LUT writes=%0d",
                 expected_error, cycles_q, lut_write_count_q);
        $finish;
      end
    join
  end
endmodule
