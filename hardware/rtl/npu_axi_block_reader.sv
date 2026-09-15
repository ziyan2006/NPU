`timescale 1ns/1ps

// Single-outstanding adapter for command, descriptor, and task-header reads.
// Requests are 8-byte aligned, contain 1..8 complete 64-bit beats, and never
// cross a 4 KiB boundary.
module npu_axi_block_reader (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         request_valid_i,
  output logic         request_ready_o,
  input  logic [63:0]  request_address_i,
  input  logic [6:0]   request_bytes_i,
  output logic         response_valid_o,
  input  logic         response_ready_i,
  output logic [511:0] response_data_o,
  output logic         response_error_o,

  output logic [63:0]  m_axi_araddr_o,
  output logic [7:0]   m_axi_arlen_o,
  output logic [2:0]   m_axi_arsize_o,
  output logic [1:0]   m_axi_arburst_o,
  output logic         m_axi_arvalid_o,
  input  logic         m_axi_arready_i,
  input  logic [63:0]  m_axi_rdata_i,
  input  logic [1:0]   m_axi_rresp_i,
  input  logic         m_axi_rlast_i,
  input  logic         m_axi_rvalid_i,
  output logic         m_axi_rready_o
);
  typedef enum logic [2:0] {
    BR_IDLE,
    BR_ADDRESS,
    BR_DATA,
    BR_RESPONSE
  } block_reader_state_e;

  block_reader_state_e state_q;
  logic [63:0] address_q;
  logic [3:0] beat_count_q;
  logic [3:0] beat_index_q;
  logic [511:0] data_q;
  logic error_q;

  logic request_format_valid;
  logic expected_last;

  assign request_format_valid = request_bytes_i >= 8
    && request_bytes_i <= 64
    && request_bytes_i[2:0] == 3'b000
    && request_address_i[2:0] == 3'b000
    && ({1'b0, request_address_i[11:0]} + request_bytes_i) <= 13'd4096;

  assign request_ready_o = state_q == BR_IDLE;
  assign response_valid_o = state_q == BR_RESPONSE;
  assign response_data_o = data_q;
  assign response_error_o = error_q;

  assign m_axi_araddr_o = address_q;
  assign m_axi_arlen_o = {4'd0, beat_count_q} - 1'b1;
  assign m_axi_arsize_o = 3'd3;
  assign m_axi_arburst_o = 2'b01;
  assign m_axi_arvalid_o = state_q == BR_ADDRESS;
  assign m_axi_rready_o = state_q == BR_DATA;
  assign expected_last = beat_index_q + 1'b1 == beat_count_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= BR_IDLE;
      address_q <= '0;
      beat_count_q <= '0;
      beat_index_q <= '0;
      data_q <= '0;
      error_q <= 1'b0;
    end else if (soft_reset_i) begin
      state_q <= BR_IDLE;
      beat_index_q <= '0;
      data_q <= '0;
      error_q <= 1'b0;
    end else begin
      case (state_q)
        BR_IDLE: begin
          if (request_valid_i) begin
            address_q <= request_address_i;
            beat_count_q <= {1'b0, request_bytes_i[6:3]};
            beat_index_q <= '0;
            data_q <= '0;
            error_q <= 1'b0;
            state_q <= request_format_valid ? BR_ADDRESS : BR_RESPONSE;
            if (!request_format_valid)
              error_q <= 1'b1;
          end
        end

        BR_ADDRESS: begin
          if (m_axi_arready_i)
            state_q <= BR_DATA;
        end

        BR_DATA: begin
          if (m_axi_rvalid_i) begin
            data_q[beat_index_q*64 +: 64] <= m_axi_rdata_i;
            if (m_axi_rresp_i != 2'b00 || m_axi_rlast_i != expected_last)
              error_q <= 1'b1;
            if (m_axi_rlast_i || expected_last) begin
              state_q <= BR_RESPONSE;
            end else begin
              beat_index_q <= beat_index_q + 1'b1;
            end
          end
        end

        BR_RESPONSE: begin
          if (response_ready_i)
            state_q <= BR_IDLE;
        end

        default: state_q <= BR_IDLE;
      endcase
    end
  end
endmodule
