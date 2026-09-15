`timescale 1ns/1ps

module tb_npu_vec_add;
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  localparam integer MAX_RESULTS = 256;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_valid_i = 1'b0;
  logic start_ready_o;
  logic [127:0] command_bits_i;
  logic [511:0] operator_desc_bits_i;
  logic read_request_valid_o;
  logic read_request_ready_i;
  logic [1:0] read_kind_o;
  logic read_bank_o;
  logic [31:0] read_address_o;
  logic read_response_valid_i = 1'b0;
  logic read_response_ready_o;
  logic [511:0] read_data_i = '0;
  logic write_valid_o;
  logic write_ready_i;
  logic [1:0] write_kind_o;
  logic write_bank_o;
  logic [31:0] write_address_o;
  logic [511:0] write_data_o;
  logic [63:0] write_strobe_o;
  logic busy_o;
  logic done_pulse_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;

  logic [127:0] command_memory [0:0];
  logic [511:0] descriptor_memory [0:0];
  logic [127:0] activation_memory [0:MAX_RESULTS-1];
  logic [127:0] output_memory [0:MAX_RESULTS-1];
  logic [127:0] expected_memory [0:MAX_RESULTS-1];
  logic [31:0] activation_address_memory [0:MAX_RESULTS-1];
  logic [127:0] quant_memory [0:0];
  logic [31:0] lfsr_q = 32'h6d5a_56e9;
  logic response_pending_q = 1'b0;
  logic [511:0] response_data_q;
  integer response_delay_q = 0;
  integer result_count;
  integer vector_quant_address;
  integer read_phase_q = 0;
  integer read_index_q = 0;
  integer write_index_q = 0;
  integer cycles_q = 0;
  string command_hex;
  string descriptor_hex;
  string activation_hex;
  string output_hex;
  string expected_hex;
  string activation_address_hex;
  string quant_hex;

  always #5 clk_i = ~clk_i;

  assign read_request_ready_i = !response_pending_q
    && !read_response_valid_i && lfsr_q[0];
  assign write_ready_i = lfsr_q[1];

  npu_vec_add dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_valid_i(start_valid_i), .start_ready_o(start_ready_o),
    .command_bits_i(command_bits_i),
    .operator_desc_bits_i(operator_desc_bits_i),
    .read_request_valid_o(read_request_valid_o),
    .read_request_ready_i(read_request_ready_i),
    .read_kind_o(read_kind_o), .read_bank_o(read_bank_o),
    .read_address_o(read_address_o),
    .read_response_valid_i(read_response_valid_i),
    .read_response_ready_o(read_response_ready_o),
    .read_data_i(read_data_i), .write_valid_o(write_valid_o),
    .write_ready_i(write_ready_i), .write_kind_o(write_kind_o),
    .write_bank_o(write_bank_o), .write_address_o(write_address_o),
    .write_data_o(write_data_o), .write_strobe_o(write_strobe_o),
    .busy_o(busy_o), .done_pulse_o(done_pulse_o),
    .error_pulse_o(error_pulse_o), .error_reason_o(error_reason_o));

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      lfsr_q <= 32'h6d5a_56e9;
      response_pending_q <= 1'b0;
      read_response_valid_i <= 1'b0;
      response_delay_q <= 0;
      read_phase_q <= 0;
      read_index_q <= 0;
      write_index_q <= 0;
      cycles_q <= 0;
    end else begin
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      cycles_q <= cycles_q + 1;
      if (cycles_q > 100000)
        $fatal(1, "VEC_ADD timeout");

      if (read_response_valid_i && read_response_ready_o) begin
        read_response_valid_i <= 1'b0;
        if (read_phase_q == 0)
          read_phase_q <= 1;
        else if (read_phase_q == 1)
          read_phase_q <= 2;
        else if (read_phase_q == 2) begin
          read_phase_q <= 1;
          read_index_q <= read_index_q + 1;
        end
      end
      if (response_pending_q) begin
        if (response_delay_q == 0) begin
          read_response_valid_i <= 1'b1;
          read_data_i <= response_data_q;
          response_pending_q <= 1'b0;
        end else begin
          response_delay_q <= response_delay_q - 1;
        end
      end

      if (read_request_valid_o && read_request_ready_i) begin
        response_pending_q <= 1'b1;
        response_delay_q <= lfsr_q[3:2];
        case (read_phase_q)
          0: begin
            if (read_kind_o != NPU_SPAD_W
                || read_address_o != vector_quant_address)
              $fatal(1, "quant request kind/address mismatch");
            if (read_bank_o
                != command_bits_i[112 + NPU_IMM_WEIGHT_BANK_BIT])
              $fatal(1, "quant bank mismatch");
            response_data_q <= {384'd0, quant_memory[0]};
          end
          1: begin
            if (read_kind_o != NPU_SPAD_O
                || read_address_o != read_index_q * 16)
              $fatal(1, "src0 request %0d mismatch", read_index_q);
            if (read_bank_o
                != command_bits_i[112 + NPU_IMM_OUTPUT_BANK_BIT])
              $fatal(1, "src0 bank mismatch");
            response_data_q <= {384'd0, output_memory[read_index_q]};
          end
          2: begin
            if (read_kind_o != NPU_SPAD_A
                || read_address_o
                   != activation_address_memory[read_index_q])
              $fatal(1, "src1 request %0d mismatch: got %0d expected %0d",
                     read_index_q, read_address_o,
                     activation_address_memory[read_index_q]);
            if (read_bank_o
                != command_bits_i[112 + NPU_IMM_ACTIVATION_BANK_BIT])
              $fatal(1, "src1 bank mismatch");
            response_data_q <= {384'd0, activation_memory[read_index_q]};
          end
          default: $fatal(1, "invalid read phase");
        endcase
      end

      if (write_valid_o && write_ready_i) begin
        if (write_index_q >= result_count)
          $fatal(1, "too many VEC_ADD writes");
        if (write_kind_o != NPU_SPAD_O
            || write_address_o != write_index_q * 16)
          $fatal(1, "write %0d kind/address mismatch", write_index_q);
        if (write_bank_o
            != command_bits_i[112 + NPU_IMM_OUTPUT_BANK_BIT])
          $fatal(1, "write bank mismatch");
        if (write_strobe_o[15:0] != 16'hffff
            || write_strobe_o[63:16] != 0)
          $fatal(1, "write strobe mismatch");
        if (write_data_o[127:0] !== expected_memory[write_index_q])
          $fatal(1, "result %0d mismatch: got %h expected %h",
                 write_index_q, write_data_o[127:0],
                 expected_memory[write_index_q]);
        write_index_q <= write_index_q + 1;
      end

      if (error_pulse_o)
        $fatal(1, "unexpected VEC_ADD error %0d", error_reason_o);
    end
  end

  initial begin
    if (!$value$plusargs("COMMAND_HEX=%s", command_hex)
        || !$value$plusargs("DESCRIPTOR_HEX=%s", descriptor_hex)
        || !$value$plusargs("ACTIVATION_HEX=%s", activation_hex)
        || !$value$plusargs("OUTPUT_HEX=%s", output_hex)
        || !$value$plusargs("EXPECTED_HEX=%s", expected_hex)
        || !$value$plusargs("ACTIVATION_ADDRESS_HEX=%s",
                            activation_address_hex)
        || !$value$plusargs("QUANT_HEX=%s", quant_hex)
        || !$value$plusargs("RESULT_COUNT=%d", result_count)
        || !$value$plusargs("VECTOR_QUANT_ADDRESS=%d",
                            vector_quant_address))
      $fatal(1, "missing VEC_ADD plusargs");
    if (result_count <= 0 || result_count > MAX_RESULTS)
      $fatal(1, "invalid result count");

    $readmemh(command_hex, command_memory);
    $readmemh(descriptor_hex, descriptor_memory);
    $readmemh(activation_hex, activation_memory, 0, result_count - 1);
    $readmemh(output_hex, output_memory, 0, result_count - 1);
    $readmemh(expected_hex, expected_memory, 0, result_count - 1);
    $readmemh(activation_address_hex, activation_address_memory,
              0, result_count - 1);
    $readmemh(quant_hex, quant_memory);
    command_bits_i = command_memory[0];
    operator_desc_bits_i = descriptor_memory[0];

    repeat (5) @(posedge clk_i);
    rst_ni <= 1'b1;
    do @(posedge clk_i); while (!start_ready_o);
    start_valid_i <= 1'b1;
    @(posedge clk_i);
    start_valid_i <= 1'b0;
    do @(posedge clk_i); while (!done_pulse_o);
    #1;
    if (write_index_q != result_count)
      $fatal(1, "missing writes: got %0d expected %0d",
             write_index_q, result_count);
    if (read_index_q != result_count)
      $fatal(1, "missing reads: got %0d expected %0d",
             read_index_q, result_count);
    $display("npu_vec_add: %0d vectors PASS in %0d cycles",
             result_count, cycles_q);
    $finish;
  end

endmodule
