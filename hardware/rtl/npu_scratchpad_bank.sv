`timescale 1ns/1ps

module npu_scratchpad_bank #(
  parameter integer WORD_COUNT = 2048,
  parameter integer ADDRESS_BITS = $clog2(WORD_COUNT)
) (
  input  logic         clk_i,
  input  logic         rst_ni,

  input  logic         port_a_write_valid_i,
  output logic         port_a_write_ready_o,
  input  logic [31:0]  port_a_write_address_i,
  input  logic [63:0]  port_a_write_data_i,
  input  logic [7:0]   port_a_write_strobe_i,
  input  logic         port_a_read_request_valid_i,
  output logic         port_a_read_request_ready_o,
  input  logic [31:0]  port_a_read_address_i,
  output logic         port_a_read_response_valid_o,
  input  logic         port_a_read_response_ready_i,
  output logic [63:0]  port_a_read_response_data_o,

  input  logic         port_b_write_valid_i,
  output logic         port_b_write_ready_o,
  input  logic [31:0]  port_b_write_address_i,
  input  logic [63:0]  port_b_write_data_i,
  input  logic [7:0]   port_b_write_strobe_i,
  input  logic         port_b_read_request_valid_i,
  output logic         port_b_read_request_ready_o,
  input  logic [31:0]  port_b_read_address_i,
  output logic         port_b_read_response_valid_o,
  input  logic         port_b_read_response_ready_i,
  output logic [63:0]  port_b_read_response_data_o
);
  (* ram_style = "block" *) logic [63:0] memory [0:WORD_COUNT-1];
  logic port_a_response_valid_q;
  logic [63:0] port_a_response_data_q;
  logic port_b_response_valid_q;
  logic [63:0] port_b_response_data_q;
  logic [ADDRESS_BITS-1:0] port_a_word_address;
  logic [ADDRESS_BITS-1:0] port_b_word_address;
  logic port_a_enable;
  logic port_b_enable;
  integer byte_lane_a;
  integer byte_lane_b;

  // One operation is accepted per physical port per cycle. A simultaneous
  // read remains asserted while the write takes priority, avoiding a mutual
  // ready dependency between the two request channels.
  assign port_a_write_ready_o = 1'b1;
  assign port_a_read_request_ready_o = !port_a_write_valid_i
    && (!port_a_response_valid_q || port_a_read_response_ready_i);
  assign port_a_read_response_valid_o = port_a_response_valid_q;
  assign port_a_read_response_data_o = port_a_response_data_q;

  assign port_b_write_ready_o = 1'b1;
  assign port_b_read_request_ready_o = !port_b_write_valid_i
    && (!port_b_response_valid_q || port_b_read_response_ready_i);
  assign port_b_read_response_valid_o = port_b_response_valid_q;
  assign port_b_read_response_data_o = port_b_response_data_q;
  assign port_a_word_address = port_a_write_valid_i
    ? port_a_write_address_i[ADDRESS_BITS+2:3]
    : port_a_read_address_i[ADDRESS_BITS+2:3];
  assign port_b_word_address = port_b_write_valid_i
    ? port_b_write_address_i[ADDRESS_BITS+2:3]
    : port_b_read_address_i[ADDRESS_BITS+2:3];
  assign port_a_enable = (port_a_write_valid_i && port_a_write_ready_o)
    || (port_a_read_request_valid_i && port_a_read_request_ready_o);
  assign port_b_enable = (port_b_write_valid_i && port_b_write_ready_o)
    || (port_b_read_request_valid_i && port_b_read_request_ready_o);

  // Independent clocked processes match the standard true-dual-port BRAM
  // inference template.  Same-word cross-port write collisions are prevented
  // by npu_scratchpad's port-B arbitration.
  always_ff @(posedge clk_i) begin
    if (port_a_enable) begin
      for (byte_lane_a = 0; byte_lane_a < 8; byte_lane_a = byte_lane_a + 1)
        if (port_a_write_valid_i && port_a_write_strobe_i[byte_lane_a])
          memory[port_a_word_address][byte_lane_a*8 +: 8]
            <= port_a_write_data_i[byte_lane_a*8 +: 8];
      port_a_response_data_q <= memory[port_a_word_address];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      port_a_response_valid_q <= 1'b0;
    end else begin
      if (port_a_read_request_valid_i && port_a_read_request_ready_o)
        port_a_response_valid_q <= 1'b1;
      else if (port_a_response_valid_q && port_a_read_response_ready_i)
        port_a_response_valid_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (port_b_enable) begin
      for (byte_lane_b = 0; byte_lane_b < 8; byte_lane_b = byte_lane_b + 1)
        if (port_b_write_valid_i && port_b_write_strobe_i[byte_lane_b])
          memory[port_b_word_address][byte_lane_b*8 +: 8]
            <= port_b_write_data_i[byte_lane_b*8 +: 8];
      port_b_response_data_q <= memory[port_b_word_address];
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      port_b_response_valid_q <= 1'b0;
    end else begin
      if (port_b_read_request_valid_i && port_b_read_request_ready_o)
        port_b_response_valid_q <= 1'b1;
      else if (port_b_response_valid_q && port_b_read_response_ready_i)
        port_b_response_valid_q <= 1'b0;
    end
  end

endmodule
