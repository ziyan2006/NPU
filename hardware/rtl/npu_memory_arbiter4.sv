`timescale 1ns/1ps

// Round-robin, single-outstanding arbiter for four block-memory clients.
module npu_memory_arbiter4 (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  soft_reset_i,

  input  logic [3:0]            request_valid_i,
  output logic [3:0]            request_ready_o,
  input  logic [3:0][63:0]      request_address_i,
  input  logic [3:0][6:0]       request_bytes_i,
  output logic [3:0]            response_valid_o,
  input  logic [3:0]            response_ready_i,
  output logic [3:0][511:0]     response_data_o,
  output logic [3:0]            response_error_o,

  output logic                  memory_request_valid_o,
  input  logic                  memory_request_ready_i,
  output logic [63:0]           memory_request_address_o,
  output logic [6:0]            memory_request_bytes_o,
  input  logic                  memory_response_valid_i,
  output logic                  memory_response_ready_o,
  input  logic [511:0]          memory_response_data_i,
  input  logic                  memory_response_error_i
);
  typedef enum logic [1:0] {MA_IDLE, MA_REQUEST, MA_RESPONSE} memory_arb_state_e;
  memory_arb_state_e state_q;
  logic [1:0] owner_q;
  logic [1:0] round_robin_q;

  integer offset;
  integer candidate;
  logic found;
  logic [1:0] selected;
  always_comb begin
    found = 1'b0;
    selected = round_robin_q;
    for (offset = 0; offset < 4; offset = offset + 1) begin
      candidate = (round_robin_q + offset) & 3;
      if (!found && request_valid_i[candidate]) begin
        found = 1'b1;
        selected = candidate[1:0];
      end
    end
  end

  always_comb begin
    request_ready_o = 4'b0000;
    response_valid_o = 4'b0000;
    response_error_o = 4'b0000;
    response_data_o = '0;
    memory_request_valid_o = state_q == MA_REQUEST;
    memory_request_address_o = request_address_i[owner_q];
    memory_request_bytes_o = request_bytes_i[owner_q];
    memory_response_ready_o = state_q == MA_RESPONSE
      && response_ready_i[owner_q];
    if (state_q == MA_REQUEST)
      request_ready_o[owner_q] = memory_request_ready_i;
    if (state_q == MA_RESPONSE) begin
      response_valid_o[owner_q] = memory_response_valid_i;
      response_data_o[owner_q] = memory_response_data_i;
      response_error_o[owner_q] = memory_response_error_i;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= MA_IDLE;
      owner_q <= 0;
      round_robin_q <= 0;
    end else if (soft_reset_i) begin
      state_q <= MA_IDLE;
      owner_q <= 0;
      round_robin_q <= 0;
    end else begin
      case (state_q)
        MA_IDLE: begin
          if (found) begin
            owner_q <= selected;
            state_q <= MA_REQUEST;
          end
        end
        MA_REQUEST: begin
          if (memory_request_ready_i) begin
            round_robin_q <= owner_q + 1'b1;
            state_q <= MA_RESPONSE;
          end
        end
        MA_RESPONSE: begin
          if (memory_response_valid_i && response_ready_i[owner_q])
            state_q <= MA_IDLE;
        end
        default: state_q <= MA_IDLE;
      endcase
    end
  end
endmodule
