`timescale 1ns/1ps

module npu_command_fetch (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         start_valid_i,
  output logic         start_ready_o,
  input  logic [63:0]  command_base_i,
  input  logic [31:0]  command_bytes_i,

  output logic         memory_request_valid_o,
  input  logic         memory_request_ready_i,
  output logic [63:0]  memory_request_address_o,
  output logic [6:0]   memory_request_bytes_o,
  input  logic         memory_response_valid_i,
  output logic         memory_response_ready_o,
  input  logic [511:0] memory_response_data_i,
  input  logic         memory_response_error_i,

  output logic         command_valid_o,
  input  logic         command_ready_i,
  output logic [127:0] command_bits_o,
  output logic [31:0]  command_pc_o,

  output logic         busy_o,
  output logic         error_pulse_o,
  output logic [3:0]   error_reason_o,
  output logic [31:0]  error_pc_o,
  output logic [15:0]  error_tag_o
);
  localparam logic [7:0] OP_END = 8'h03;
  localparam logic [3:0] FETCH_ERROR_NONE = 4'h0;
  localparam logic [3:0] FETCH_ERROR_FORMAT = 4'h1;
  localparam logic [3:0] FETCH_ERROR_MEMORY = 4'h2;
  localparam logic [3:0] FETCH_ERROR_EARLY_END = 4'h3;
  localparam logic [3:0] FETCH_ERROR_MISSING_END = 4'h4;

  typedef enum logic [2:0] {
    CF_IDLE,
    CF_REQUEST,
    CF_RESPONSE,
    CF_COMMAND,
    CF_ERROR
  } command_fetch_state_e;

  command_fetch_state_e state_q;
  logic [63:0] base_q;
  logic [31:0] bytes_q;
  logic [31:0] pc_q;
  logic [127:0] command_q;
  logic [3:0] error_reason_q;
  logic [31:0] error_pc_q;
  logic [15:0] error_tag_q;
  logic command_is_end;
  logic command_is_last;

  assign start_ready_o = state_q == CF_IDLE;
  assign memory_request_valid_o = state_q == CF_REQUEST;
  assign memory_request_address_o = base_q + pc_q;
  assign memory_request_bytes_o = 7'd16;
  assign memory_response_ready_o = state_q == CF_RESPONSE;
  assign command_valid_o = state_q == CF_COMMAND;
  assign command_bits_o = command_q;
  assign command_pc_o = pc_q;
  assign busy_o = state_q != CF_IDLE && state_q != CF_ERROR;
  assign error_pulse_o = state_q == CF_ERROR;
  assign error_reason_o = error_reason_q;
  assign error_pc_o = error_pc_q;
  assign error_tag_o = error_tag_q;
  assign command_is_end = command_q[7:0] == OP_END;
  assign command_is_last = pc_q + 16 == bytes_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= CF_IDLE;
      base_q <= '0;
      bytes_q <= '0;
      pc_q <= '0;
      command_q <= '0;
      error_reason_q <= FETCH_ERROR_NONE;
      error_pc_q <= '0;
      error_tag_q <= '0;
    end else if (soft_reset_i) begin
      state_q <= CF_IDLE;
      pc_q <= '0;
      error_reason_q <= FETCH_ERROR_NONE;
    end else begin
      case (state_q)
        CF_IDLE: begin
          if (start_valid_i) begin
            base_q <= command_base_i;
            bytes_q <= command_bytes_i;
            pc_q <= 0;
            error_reason_q <= FETCH_ERROR_NONE;
            if (command_base_i[3:0] != 0 || command_bytes_i == 0
                || command_bytes_i[3:0] != 0) begin
              error_reason_q <= FETCH_ERROR_FORMAT;
              error_pc_q <= 0;
              error_tag_q <= 0;
              state_q <= CF_ERROR;
            end else begin
              state_q <= CF_REQUEST;
            end
          end
        end
        CF_REQUEST: begin
          if (memory_request_ready_i)
            state_q <= CF_RESPONSE;
        end
        CF_RESPONSE: begin
          if (memory_response_valid_i) begin
            if (memory_response_error_i) begin
              error_reason_q <= FETCH_ERROR_MEMORY;
              error_pc_q <= pc_q;
              error_tag_q <= 0;
              state_q <= CF_ERROR;
            end else begin
              command_q <= memory_response_data_i[127:0];
              state_q <= CF_COMMAND;
            end
          end
        end
        CF_COMMAND: begin
          if (command_ready_i) begin
            if (command_is_end && !command_is_last) begin
              error_reason_q <= FETCH_ERROR_EARLY_END;
              error_pc_q <= pc_q;
              error_tag_q <= command_q[31:16];
              state_q <= CF_ERROR;
            end else if (!command_is_end && command_is_last) begin
              error_reason_q <= FETCH_ERROR_MISSING_END;
              error_pc_q <= pc_q;
              error_tag_q <= command_q[31:16];
              state_q <= CF_ERROR;
            end else if (command_is_last) begin
              state_q <= CF_IDLE;
            end else begin
              pc_q <= pc_q + 16;
              state_q <= CF_REQUEST;
            end
          end
        end
        CF_ERROR: state_q <= CF_IDLE;
        default: state_q <= CF_IDLE;
      endcase
    end
  end
endmodule
