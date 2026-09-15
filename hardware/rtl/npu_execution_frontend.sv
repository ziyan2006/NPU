`timescale 1ns/1ps

// Fetches the descriptors required by compute/vector commands and dispatches
// them to the corresponding execution unit.  CONV2D is asynchronous; VEC_ADD
// and UPSAMPLE2X hold the command processor in its vector wait state.
module npu_execution_frontend (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         compute_command_valid_i,
  output logic         compute_command_ready_o,
  input  logic [127:0] compute_command_bits_i,
  input  logic [31:0]  compute_command_pc_i,
  input  logic         vector_command_valid_i,
  output logic         vector_command_ready_o,
  input  logic [127:0] vector_command_bits_i,
  input  logic [31:0]  vector_command_pc_i,

  input  logic [63:0]  tensor_desc_base_i,
  input  logic [63:0]  operator_desc_base_i,
  input  logic [63:0]  quant_desc_base_i,
  input  logic [63:0]  segment_desc_base_i,

  output logic         memory_request_valid_o,
  input  logic         memory_request_ready_i,
  output logic [63:0]  memory_request_address_o,
  output logic [6:0]   memory_request_bytes_o,
  input  logic         memory_response_valid_i,
  output logic         memory_response_ready_o,
  input  logic [511:0] memory_response_data_i,
  input  logic         memory_response_error_i,

  output logic         conv_start_valid_o,
  input  logic         conv_start_ready_i,
  output logic [127:0] conv_command_bits_o,
  output logic [511:0] conv_operator_desc_bits_o,
  output logic         conv_activation_bank_o,
  output logic         conv_weight_bank_o,
  output logic         conv_output_bank_o,
  output logic [7:0]   conv_completion_event_o,
  output logic [31:0]  conv_command_pc_o,
  output logic [15:0]  conv_command_tag_o,

  output logic         vec_start_valid_o,
  input  logic         vec_start_ready_i,
  output logic [127:0] vec_command_bits_o,
  output logic [511:0] vec_operator_desc_bits_o,
  input  logic         vec_done_pulse_i,
  input  logic         vec_error_pulse_i,
  input  logic [3:0]   vec_error_reason_i,

  output logic         upsample_start_valid_o,
  input  logic         upsample_start_ready_i,
  output logic [127:0] upsample_command_bits_o,
  output logic [511:0] upsample_source_desc_bits_o,
  output logic [511:0] upsample_destination_desc_bits_o,
  input  logic         upsample_done_pulse_i,
  input  logic         upsample_error_pulse_i,
  input  logic [3:0]   upsample_error_reason_i,

  output logic         vector_sync_done_o,
  output logic         busy_o,
  output logic         error_pulse_o,
  output logic [15:0]  error_code_o,
  output logic [31:0]  error_pc_o,
  output logic [15:0]  error_tag_o
);
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  localparam logic [15:0] EXEC_ERROR_DESCRIPTOR = 16'h0101;
  localparam logic [15:0] EXEC_ERROR_UNSUPPORTED = 16'h0102;
  localparam logic [15:0] EXEC_ERROR_VEC_BASE = 16'h0200;
  localparam logic [15:0] EXEC_ERROR_UPSAMPLE_BASE = 16'h0300;

  typedef enum logic [3:0] {
    EF_IDLE,
    EF_DESC_REQUEST,
    EF_DESC_RESPONSE,
    EF_CONV_START,
    EF_VEC_START,
    EF_UPSAMPLE_SOURCE_REQUEST,
    EF_UPSAMPLE_SOURCE_RESPONSE,
    EF_UPSAMPLE_DEST_REQUEST,
    EF_UPSAMPLE_DEST_RESPONSE,
    EF_UPSAMPLE_START,
    EF_WAIT_VEC,
    EF_WAIT_UPSAMPLE,
    EF_ERROR
  } execution_frontend_state_e;

  execution_frontend_state_e state_q;
  npu_command_t command_q;
  npu_command_t compute_command;
  npu_command_t vector_command;
  logic [31:0] command_pc_q;
  logic [511:0] operator_desc_q;
  logic [511:0] source_desc_q;
  logic [511:0] destination_desc_q;
  logic [15:0] error_code_q;

  logic cache_request_valid;
  logic cache_request_ready;
  logic [1:0] cache_request_kind;
  logic [15:0] cache_request_index;
  logic cache_response_valid;
  logic cache_response_ready;
  logic [511:0] cache_response_data;
  logic [6:0] cache_response_bytes;
  logic cache_response_error;

  assign compute_command = npu_command_t'(compute_command_bits_i);
  assign vector_command = npu_command_t'(vector_command_bits_i);
  assign compute_command_ready_o = state_q == EF_IDLE;
  assign vector_command_ready_o = state_q == EF_IDLE
    && !compute_command_valid_i;
  assign busy_o = state_q != EF_IDLE && state_q != EF_ERROR;

  assign cache_request_valid = state_q == EF_DESC_REQUEST
    || state_q == EF_UPSAMPLE_SOURCE_REQUEST
    || state_q == EF_UPSAMPLE_DEST_REQUEST;
  assign cache_request_kind = state_q == EF_DESC_REQUEST
    ? NPU_DESC_OPERATOR : NPU_DESC_TENSOR;
  assign cache_request_index = state_q == EF_UPSAMPLE_SOURCE_REQUEST
    ? command_q.src0_td
    : state_q == EF_UPSAMPLE_DEST_REQUEST
      ? command_q.dst_td : command_q.op_desc;
  assign cache_response_ready = state_q == EF_DESC_RESPONSE
    || state_q == EF_UPSAMPLE_SOURCE_RESPONSE
    || state_q == EF_UPSAMPLE_DEST_RESPONSE;

  assign conv_start_valid_o = state_q == EF_CONV_START;
  assign conv_command_bits_o = command_q;
  assign conv_operator_desc_bits_o = operator_desc_q;
  assign conv_activation_bank_o = command_q.imm[NPU_IMM_ACTIVATION_BANK_BIT];
  assign conv_weight_bank_o = command_q.imm[NPU_IMM_WEIGHT_BANK_BIT];
  assign conv_output_bank_o = command_q.imm[NPU_IMM_OUTPUT_BANK_BIT];
  assign conv_completion_event_o = command_q.imm[7:0];
  assign vec_start_valid_o = state_q == EF_VEC_START;
  assign vec_command_bits_o = command_q;
  assign vec_operator_desc_bits_o = operator_desc_q;
  assign upsample_start_valid_o = state_q == EF_UPSAMPLE_START;
  assign upsample_command_bits_o = command_q;
  assign upsample_source_desc_bits_o = source_desc_q;
  assign upsample_destination_desc_bits_o = destination_desc_q;

  assign error_pulse_o = state_q == EF_ERROR;
  assign error_code_o = error_code_q;
  assign error_pc_o = command_pc_q;
  assign error_tag_o = command_q.tag;

  npu_descriptor_cache #(.LINE_COUNT(4)) descriptor_cache (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .tensor_base_i(tensor_desc_base_i),
    .operator_base_i(operator_desc_base_i),
    .quant_base_i(quant_desc_base_i), .segment_base_i(segment_desc_base_i),
    .request_valid_i(cache_request_valid),
    .request_ready_o(cache_request_ready),
    .request_kind_i(cache_request_kind),
    .request_index_i(cache_request_index),
    .response_valid_o(cache_response_valid),
    .response_ready_i(cache_response_ready),
    .response_data_o(cache_response_data),
    .response_bytes_o(cache_response_bytes),
    .response_error_o(cache_response_error),
    .memory_request_valid_o(memory_request_valid_o),
    .memory_request_ready_i(memory_request_ready_i),
    .memory_request_address_o(memory_request_address_o),
    .memory_request_bytes_o(memory_request_bytes_o),
    .memory_response_valid_i(memory_response_valid_i),
    .memory_response_ready_o(memory_response_ready_o),
    .memory_response_data_i(memory_response_data_i),
    .memory_response_error_i(memory_response_error_i));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= EF_IDLE;
      command_q <= '0;
      command_pc_q <= '0;
      operator_desc_q <= '0;
      source_desc_q <= '0;
      destination_desc_q <= '0;
      error_code_q <= '0;
      vector_sync_done_o <= 1'b0;
      conv_command_pc_o <= '0;
      conv_command_tag_o <= '0;
    end else if (soft_reset_i) begin
      state_q <= EF_IDLE;
      error_code_q <= '0;
      vector_sync_done_o <= 1'b0;
    end else begin
      vector_sync_done_o <= 1'b0;
      case (state_q)
        EF_IDLE: begin
          error_code_q <= '0;
          if (compute_command_valid_i) begin
            command_q <= compute_command;
            command_pc_q <= compute_command_pc_i;
            if (compute_command.opcode != NPU_OP_CONV2D) begin
              error_code_q <= EXEC_ERROR_UNSUPPORTED;
              state_q <= EF_ERROR;
            end else begin
              state_q <= EF_DESC_REQUEST;
            end
          end else if (vector_command_valid_i) begin
            command_q <= vector_command;
            command_pc_q <= vector_command_pc_i;
            case (vector_command.opcode)
              NPU_OP_VEC_ADD: state_q <= EF_DESC_REQUEST;
              NPU_OP_UPSAMPLE2X: state_q <= EF_UPSAMPLE_SOURCE_REQUEST;
              default: begin
                error_code_q <= EXEC_ERROR_UNSUPPORTED;
                state_q <= EF_ERROR;
              end
            endcase
          end
        end

        EF_DESC_REQUEST: begin
          if (cache_request_ready)
            state_q <= EF_DESC_RESPONSE;
        end
        EF_DESC_RESPONSE: begin
          if (cache_response_valid) begin
            if (cache_response_error || cache_response_bytes != 64) begin
              error_code_q <= EXEC_ERROR_DESCRIPTOR;
              state_q <= EF_ERROR;
            end else begin
              operator_desc_q <= cache_response_data;
              state_q <= command_q.opcode == NPU_OP_CONV2D
                ? EF_CONV_START : EF_VEC_START;
            end
          end
        end

        EF_CONV_START: begin
          if (conv_start_ready_i) begin
            conv_command_pc_o <= command_pc_q;
            conv_command_tag_o <= command_q.tag;
            state_q <= EF_IDLE;
          end
        end
        EF_VEC_START: begin
          if (vec_start_ready_i)
            state_q <= EF_WAIT_VEC;
        end

        EF_UPSAMPLE_SOURCE_REQUEST: begin
          if (cache_request_ready)
            state_q <= EF_UPSAMPLE_SOURCE_RESPONSE;
        end
        EF_UPSAMPLE_SOURCE_RESPONSE: begin
          if (cache_response_valid) begin
            if (cache_response_error || cache_response_bytes != 64) begin
              error_code_q <= EXEC_ERROR_DESCRIPTOR;
              state_q <= EF_ERROR;
            end else begin
              source_desc_q <= cache_response_data;
              state_q <= EF_UPSAMPLE_DEST_REQUEST;
            end
          end
        end
        EF_UPSAMPLE_DEST_REQUEST: begin
          if (cache_request_ready)
            state_q <= EF_UPSAMPLE_DEST_RESPONSE;
        end
        EF_UPSAMPLE_DEST_RESPONSE: begin
          if (cache_response_valid) begin
            if (cache_response_error || cache_response_bytes != 64) begin
              error_code_q <= EXEC_ERROR_DESCRIPTOR;
              state_q <= EF_ERROR;
            end else begin
              destination_desc_q <= cache_response_data;
              state_q <= EF_UPSAMPLE_START;
            end
          end
        end
        EF_UPSAMPLE_START: begin
          if (upsample_start_ready_i)
            state_q <= EF_WAIT_UPSAMPLE;
        end

        EF_WAIT_VEC: begin
          if (vec_error_pulse_i) begin
            error_code_q <= EXEC_ERROR_VEC_BASE | {12'd0, vec_error_reason_i};
            state_q <= EF_ERROR;
          end else if (vec_done_pulse_i) begin
            vector_sync_done_o <= 1'b1;
            state_q <= EF_IDLE;
          end
        end
        EF_WAIT_UPSAMPLE: begin
          if (upsample_error_pulse_i) begin
            error_code_q <= EXEC_ERROR_UPSAMPLE_BASE
              | {12'd0, upsample_error_reason_i};
            state_q <= EF_ERROR;
          end else if (upsample_done_pulse_i) begin
            vector_sync_done_o <= 1'b1;
            state_q <= EF_IDLE;
          end
        end

        EF_ERROR: state_q <= EF_IDLE;
        default: state_q <= EF_IDLE;
      endcase
    end
  end
endmodule
