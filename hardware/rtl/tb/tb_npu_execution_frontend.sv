`timescale 1ns/1ps

module tb_npu_execution_frontend;
  import npu_isa_pkg::*;

  localparam integer MAX_COMMANDS = 300;
  localparam integer MAX_OPERATORS = 256;
  localparam integer MAX_TENSORS = 64;
  localparam logic [63:0] TENSOR_BASE = 64'h0000_0000_4000_0000;
  localparam logic [63:0] OPERATOR_BASE = 64'h0000_0000_3000_0000;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic compute_valid;
  logic compute_ready;
  logic [127:0] compute_bits;
  logic [31:0] compute_pc;
  logic vector_valid;
  logic vector_ready;
  logic [127:0] vector_bits;
  logic [31:0] vector_pc;

  logic memory_request_valid;
  logic memory_request_ready;
  logic [63:0] memory_request_address;
  logic [6:0] memory_request_bytes;
  logic memory_response_valid = 1'b0;
  logic memory_response_ready;
  logic [511:0] memory_response_data;
  logic memory_response_error = 1'b0;

  logic conv_start_valid;
  logic conv_start_ready;
  logic [127:0] conv_command;
  logic [511:0] conv_desc;
  logic conv_activation_bank;
  logic conv_weight_bank;
  logic conv_output_bank;
  logic [7:0] conv_event;
  logic [31:0] conv_pc;
  logic [15:0] conv_tag;
  logic vec_start_valid;
  logic vec_start_ready;
  logic [127:0] vec_command;
  logic [511:0] vec_desc;
  logic vec_done = 1'b0;
  logic vec_error = 1'b0;
  logic [3:0] vec_error_reason = 0;
  logic up_start_valid;
  logic up_start_ready;
  logic [127:0] up_command;
  logic [511:0] up_source_desc;
  logic [511:0] up_destination_desc;
  logic up_done = 1'b0;
  logic up_error = 1'b0;
  logic [3:0] up_error_reason = 0;
  logic vector_sync_done;
  logic busy;
  logic error_pulse;
  logic [15:0] error_code;
  logic [31:0] error_pc;
  logic [15:0] error_tag;

  logic [127:0] command_memory [0:MAX_COMMANDS-1];
  logic [31:0] pc_memory [0:MAX_COMMANDS-1];
  logic [511:0] operator_memory [0:MAX_OPERATORS-1];
  logic [511:0] tensor_memory [0:MAX_TENSORS-1];
  logic [31:0] lfsr_q = 32'h8a55_291d;
  logic memory_pending_q = 1'b0;
  logic [511:0] memory_pending_data_q = '0;
  integer memory_delay_q = 0;
  logic unit_pending_q = 1'b0;
  logic unit_is_upsample_q = 1'b0;
  integer unit_delay_q = 0;
  integer issue_q = 0;
  integer compute_issue_q = 0;
  integer compute_tag_sum_q = 0;
  integer conv_tag_sum_q = 0;
  integer conv_count_q = 0;
  integer vec_count_q = 0;
  integer up_count_q = 0;
  integer sync_count_q = 0;
  integer cycles_q = 0;
  integer command_count;
  string command_hex;
  string pc_hex;
  string operator_hex;
  string tensor_hex;

  npu_command_t current_command;
  npu_command_t dispatched_vec_command;
  npu_command_t dispatched_up_command;
  npu_command_t dispatched_conv_command;

  always #5 clk_i = ~clk_i;
  assign current_command = npu_command_t'(command_memory[issue_q]);
  assign dispatched_vec_command = npu_command_t'(vec_command);
  assign dispatched_up_command = npu_command_t'(up_command);
  assign dispatched_conv_command = npu_command_t'(conv_command);
  assign compute_valid = issue_q < command_count
    && current_command.opcode == NPU_OP_CONV2D;
  assign vector_valid = issue_q < command_count
    && current_command.opcode != NPU_OP_CONV2D;
  assign compute_bits = command_memory[issue_q];
  assign vector_bits = command_memory[issue_q];
  assign compute_pc = pc_memory[issue_q];
  assign vector_pc = pc_memory[issue_q];
  assign memory_request_ready = !memory_pending_q
    && !memory_response_valid && lfsr_q[0];
  assign memory_response_data = memory_pending_data_q;
  assign conv_start_ready = lfsr_q[1] || lfsr_q[2];
  assign vec_start_ready = !unit_pending_q && (lfsr_q[3] || lfsr_q[4]);
  assign up_start_ready = !unit_pending_q && (lfsr_q[5] || lfsr_q[6]);

  npu_execution_frontend dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .compute_command_valid_i(compute_valid),
    .compute_command_ready_o(compute_ready),
    .compute_command_bits_i(compute_bits),
    .compute_command_pc_i(compute_pc),
    .vector_command_valid_i(vector_valid),
    .vector_command_ready_o(vector_ready),
    .vector_command_bits_i(vector_bits), .vector_command_pc_i(vector_pc),
    .tensor_desc_base_i(TENSOR_BASE),
    .operator_desc_base_i(OPERATOR_BASE),
    .quant_desc_base_i(64'h5000_0000),
    .segment_desc_base_i(64'h6000_0000),
    .memory_request_valid_o(memory_request_valid),
    .memory_request_ready_i(memory_request_ready),
    .memory_request_address_o(memory_request_address),
    .memory_request_bytes_o(memory_request_bytes),
    .memory_response_valid_i(memory_response_valid),
    .memory_response_ready_o(memory_response_ready),
    .memory_response_data_i(memory_response_data),
    .memory_response_error_i(memory_response_error),
    .conv_start_valid_o(conv_start_valid),
    .conv_start_ready_i(conv_start_ready),
    .conv_command_bits_o(conv_command),
    .conv_operator_desc_bits_o(conv_desc),
    .conv_activation_bank_o(conv_activation_bank),
    .conv_weight_bank_o(conv_weight_bank),
    .conv_output_bank_o(conv_output_bank),
    .conv_completion_event_o(conv_event),
    .conv_command_pc_o(conv_pc), .conv_command_tag_o(conv_tag),
    .vec_start_valid_o(vec_start_valid), .vec_start_ready_i(vec_start_ready),
    .vec_command_bits_o(vec_command),
    .vec_operator_desc_bits_o(vec_desc), .vec_done_pulse_i(vec_done),
    .vec_error_pulse_i(vec_error), .vec_error_reason_i(vec_error_reason),
    .upsample_start_valid_o(up_start_valid),
    .upsample_start_ready_i(up_start_ready),
    .upsample_command_bits_o(up_command),
    .upsample_source_desc_bits_o(up_source_desc),
    .upsample_destination_desc_bits_o(up_destination_desc),
    .upsample_done_pulse_i(up_done), .upsample_error_pulse_i(up_error),
    .upsample_error_reason_i(up_error_reason),
    .vector_sync_done_o(vector_sync_done), .busy_o(busy),
    .error_pulse_o(error_pulse), .error_code_o(error_code),
    .error_pc_o(error_pc), .error_tag_o(error_tag));

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      lfsr_q <= 32'h8a55_291d;
      memory_pending_q <= 1'b0;
      memory_response_valid <= 1'b0;
      unit_pending_q <= 1'b0;
      vec_done <= 1'b0;
      up_done <= 1'b0;
      issue_q <= 0;
      compute_issue_q <= 0;
      compute_tag_sum_q <= 0;
      conv_tag_sum_q <= 0;
      conv_count_q <= 0;
      vec_count_q <= 0;
      up_count_q <= 0;
      sync_count_q <= 0;
      cycles_q <= 0;
    end else begin
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      vec_done <= 1'b0;
      up_done <= 1'b0;
      cycles_q <= cycles_q + 1;
      if (cycles_q > 200000)
        $fatal(1, "execution frontend timeout");
      if (error_pulse)
        $fatal(1, "unexpected execution error %h pc=%0d tag=%0d",
               error_code, error_pc, error_tag);

      if ((compute_valid && compute_ready) || (vector_valid && vector_ready))
        issue_q <= issue_q + 1;
      if (compute_valid && compute_ready) begin
        compute_issue_q <= compute_issue_q + 1;
        compute_tag_sum_q <= compute_tag_sum_q + current_command.tag;
      end

      if (memory_request_valid && memory_request_ready) begin
        if (memory_request_bytes != 64)
          $fatal(1, "unexpected descriptor bytes %0d", memory_request_bytes);
        memory_pending_q <= 1'b1;
        memory_delay_q <= lfsr_q[9:7];
        if (memory_request_address >= OPERATOR_BASE
            && memory_request_address < OPERATOR_BASE + MAX_OPERATORS * 64)
          memory_pending_data_q <= operator_memory[
            (memory_request_address - OPERATOR_BASE) >> 6];
        else if (memory_request_address >= TENSOR_BASE
                 && memory_request_address < TENSOR_BASE + MAX_TENSORS * 64)
          memory_pending_data_q <= tensor_memory[
            (memory_request_address - TENSOR_BASE) >> 6];
        else
          $fatal(1, "unexpected descriptor address %h", memory_request_address);
      end
      if (memory_pending_q && !memory_response_valid) begin
        if (memory_delay_q == 0) begin
          memory_pending_q <= 1'b0;
          memory_response_valid <= 1'b1;
        end else begin
          memory_delay_q <= memory_delay_q - 1;
        end
      end
      if (memory_response_valid && memory_response_ready)
        memory_response_valid <= 1'b0;

      if (conv_start_valid && conv_start_ready) begin
        if (conv_desc != operator_memory[dispatched_conv_command.op_desc])
          $fatal(1, "CONV2D descriptor mismatch");
        if (conv_activation_bank != dispatched_conv_command.imm[8]
            || conv_weight_bank != dispatched_conv_command.imm[9]
            || conv_output_bank != dispatched_conv_command.imm[11]
            || conv_event != dispatched_conv_command.imm[7:0])
          $fatal(1, "CONV2D control mismatch");
        conv_count_q <= conv_count_q + 1;
        conv_tag_sum_q <= conv_tag_sum_q + dispatched_conv_command.tag;
      end
      if (vec_start_valid && vec_start_ready) begin
        if (vec_desc != operator_memory[dispatched_vec_command.op_desc])
          $fatal(1, "VEC_ADD descriptor mismatch");
        vec_count_q <= vec_count_q + 1;
        unit_pending_q <= 1'b1;
        unit_is_upsample_q <= 1'b0;
        unit_delay_q <= lfsr_q[11:9] + 1;
      end
      if (up_start_valid && up_start_ready) begin
        if (up_source_desc != tensor_memory[dispatched_up_command.src0_td]
            || up_destination_desc != tensor_memory[dispatched_up_command.dst_td])
          $fatal(1, "UPSAMPLE descriptor mismatch");
        up_count_q <= up_count_q + 1;
        unit_pending_q <= 1'b1;
        unit_is_upsample_q <= 1'b1;
        unit_delay_q <= lfsr_q[11:9] + 1;
      end
      if (unit_pending_q) begin
        if (unit_delay_q == 0) begin
          unit_pending_q <= 1'b0;
          if (unit_is_upsample_q)
            up_done <= 1'b1;
          else
            vec_done <= 1'b1;
        end else begin
          unit_delay_q <= unit_delay_q - 1;
        end
      end
      if (vector_sync_done)
        sync_count_q <= sync_count_q + 1;
    end
  end

  initial begin
    if (!$value$plusargs("COMMANDS=%s", command_hex)
        || !$value$plusargs("PCS=%s", pc_hex)
        || !$value$plusargs("OPERATORS=%s", operator_hex)
        || !$value$plusargs("TENSORS=%s", tensor_hex)
        || !$value$plusargs("COMMAND_COUNT=%d", command_count))
      $fatal(1, "missing plusargs");
    $readmemh(command_hex, command_memory);
    $readmemh(pc_hex, pc_memory);
    $readmemh(operator_hex, operator_memory);
    $readmemh(tensor_hex, tensor_memory);
    repeat (5) @(posedge clk_i);
    rst_ni <= 1'b1;
    wait(issue_q == command_count && conv_count_q == 248
         && vec_count_q == 32 && up_count_q == 3 && sync_count_q == 35);
    repeat (2) @(posedge clk_i);
    if (conv_count_q != 248 || vec_count_q != 32 || up_count_q != 3
        || sync_count_q != 35)
      $fatal(1, "dispatch counts issue=%0d conv_issue=%0d conv=%0d vec=%0d up=%0d sync=%0d tag sums=%0d/%0d",
             issue_q, compute_issue_q, conv_count_q, vec_count_q, up_count_q,
             sync_count_q, compute_tag_sum_q, conv_tag_sum_q);
    $display("execution frontend: PASS commands=%0d cycles=%0d", issue_q, cycles_q);
    $finish;
  end
endmodule
