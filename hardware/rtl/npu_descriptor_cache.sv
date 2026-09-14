`timescale 1ns/1ps

module npu_descriptor_cache #(
  parameter integer LINE_COUNT = 4
) (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic [63:0]  tensor_base_i,
  input  logic [63:0]  operator_base_i,
  input  logic [63:0]  quant_base_i,
  input  logic [63:0]  segment_base_i,

  input  logic         request_valid_i,
  output logic         request_ready_o,
  input  logic [1:0]   request_kind_i,
  input  logic [15:0]  request_index_i,

  output logic         response_valid_o,
  input  logic         response_ready_i,
  output logic [511:0] response_data_o,
  output logic [6:0]   response_bytes_o,
  output logic         response_error_o,

  output logic         memory_request_valid_o,
  input  logic         memory_request_ready_i,
  output logic [63:0]  memory_request_address_o,
  output logic [6:0]   memory_request_bytes_o,
  input  logic         memory_response_valid_i,
  output logic         memory_response_ready_o,
  input  logic [511:0] memory_response_data_i,
  input  logic         memory_response_error_i
);
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  typedef enum logic [2:0] {
    DC_IDLE,
    DC_MEMORY_REQUEST,
    DC_MEMORY_RESPONSE,
    DC_RETURN
  } dc_state_e;

  dc_state_e state_q;
  logic [LINE_COUNT-1:0] valid_q;
  logic [1:0] kind_q [0:LINE_COUNT-1];
  logic [15:0] index_q [0:LINE_COUNT-1];
  logic [511:0] data_q [0:LINE_COUNT-1];
  logic [6:0] bytes_q [0:LINE_COUNT-1];

  logic [1:0] pending_kind_q;
  logic [15:0] pending_index_q;
  integer pending_line_q;
  logic [63:0] pending_address_q;
  logic [6:0] pending_bytes_q;
  logic [511:0] response_data_q;
  logic [6:0] response_bytes_q;
  logic response_error_q;
  logic discard_response_q;

  integer request_line;
  logic request_kind_valid;
  logic request_hit;
  logic [63:0] request_base;
  logic [6:0] request_bytes;
  logic [63:0] request_address;
  integer line_index;

  always_comb begin
    request_kind_valid = 1'b1;
    request_base = 64'h0;
    request_bytes = 7'd0;
    case (request_kind_i)
      NPU_DESC_TENSOR: begin
        request_base = tensor_base_i;
        request_bytes = NPU_TENSOR_DESC_BYTES;
      end
      NPU_DESC_OPERATOR: begin
        request_base = operator_base_i;
        request_bytes = NPU_OPERATOR_DESC_BYTES;
      end
      NPU_DESC_QUANT: begin
        request_base = quant_base_i;
        request_bytes = NPU_QUANT_DESC_BYTES;
      end
      NPU_DESC_SEGMENT: begin
        request_base = segment_base_i;
        request_bytes = NPU_SEGMENT_DESC_BYTES;
      end
      default: request_kind_valid = 1'b0;
    endcase
    request_line = request_index_i % LINE_COUNT;
    request_hit = request_kind_valid && (request_index_i != NPU_NONE_INDEX)
      && valid_q[request_line]
      && (kind_q[request_line] == request_kind_i)
      && (index_q[request_line] == request_index_i);
    case (request_kind_i)
      NPU_DESC_TENSOR,
      NPU_DESC_OPERATOR: request_address = request_base
        + ({48'd0, request_index_i} << 6);
      NPU_DESC_QUANT: request_address = request_base
        + ({48'd0, request_index_i} << 5);
      default: request_address = request_base
        + ({48'd0, request_index_i} << 3);
    endcase
  end

  assign request_ready_o = (state_q == DC_IDLE);
  assign response_valid_o = (state_q == DC_RETURN);
  assign response_data_o = response_data_q;
  assign response_bytes_o = response_bytes_q;
  assign response_error_o = response_error_q;
  assign memory_request_valid_o = (state_q == DC_MEMORY_REQUEST);
  assign memory_request_address_o = pending_address_q;
  assign memory_request_bytes_o = pending_bytes_q;
  assign memory_response_ready_o = (state_q == DC_MEMORY_RESPONSE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= DC_IDLE;
      valid_q <= '0;
      pending_kind_q <= '0;
      pending_index_q <= '0;
      pending_line_q <= 0;
      pending_address_q <= '0;
      pending_bytes_q <= '0;
      response_data_q <= '0;
      response_bytes_q <= '0;
      response_error_q <= 1'b0;
      discard_response_q <= 1'b0;
      for (line_index = 0; line_index < LINE_COUNT; line_index = line_index + 1) begin
        kind_q[line_index] <= '0;
        index_q[line_index] <= '0;
        data_q[line_index] <= '0;
        bytes_q[line_index] <= '0;
      end
    end else if (soft_reset_i) begin
      valid_q <= '0;
      response_error_q <= 1'b0;
      // A descriptor read that has reached MEMORY_RESPONSE already crossed
      // the ready/valid boundary and must be drained before becoming idle.
      if (state_q == DC_MEMORY_RESPONSE) begin
        if (memory_response_valid_i) begin
          state_q <= DC_IDLE;
          discard_response_q <= 1'b0;
        end else begin
          state_q <= DC_MEMORY_RESPONSE;
          discard_response_q <= 1'b1;
        end
      end else if (state_q == DC_MEMORY_REQUEST
                   && memory_request_ready_i) begin
        state_q <= DC_MEMORY_RESPONSE;
        discard_response_q <= 1'b1;
      end else begin
        state_q <= DC_IDLE;
        discard_response_q <= 1'b0;
      end
    end else begin
      case (state_q)
        DC_IDLE: begin
          if (request_valid_i) begin
            if (!request_kind_valid || request_index_i == NPU_NONE_INDEX) begin
              response_data_q <= '0;
              response_bytes_q <= request_bytes;
              response_error_q <= 1'b1;
              state_q <= DC_RETURN;
            end else if (request_hit) begin
              response_data_q <= data_q[request_line];
              response_bytes_q <= bytes_q[request_line];
              response_error_q <= 1'b0;
              state_q <= DC_RETURN;
            end else begin
              pending_kind_q <= request_kind_i;
              pending_index_q <= request_index_i;
              pending_line_q <= request_line;
              pending_address_q <= request_address;
              pending_bytes_q <= request_bytes;
              discard_response_q <= 1'b0;
              state_q <= DC_MEMORY_REQUEST;
            end
          end
        end

        DC_MEMORY_REQUEST: begin
          if (memory_request_ready_i)
            state_q <= DC_MEMORY_RESPONSE;
        end

        DC_MEMORY_RESPONSE: begin
          if (memory_response_valid_i) begin
            if (discard_response_q) begin
              discard_response_q <= 1'b0;
              state_q <= DC_IDLE;
            end else begin
              response_data_q <= memory_response_data_i;
              response_bytes_q <= pending_bytes_q;
              response_error_q <= memory_response_error_i;
              if (!memory_response_error_i) begin
                valid_q[pending_line_q] <= 1'b1;
                kind_q[pending_line_q] <= pending_kind_q;
                index_q[pending_line_q] <= pending_index_q;
                data_q[pending_line_q] <= memory_response_data_i;
                bytes_q[pending_line_q] <= pending_bytes_q;
              end
              state_q <= DC_RETURN;
            end
          end
        end

        DC_RETURN: begin
          if (response_ready_i)
            state_q <= DC_IDLE;
        end

        default: state_q <= DC_IDLE;
      endcase
    end
  end

endmodule
