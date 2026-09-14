`timescale 1ns/1ps

module npu_scratchpad_lane #(
  parameter integer ROW_COUNT = 1024,
  parameter integer ROW_ADDRESS_BITS = $clog2(ROW_COUNT)
) (
  input  logic                         clk_i,
  input  logic                         write_enable_i,
  input  logic [ROW_ADDRESS_BITS-1:0]  write_address_i,
  input  logic [63:0]                  write_data_i,
  input  logic [7:0]                   write_strobe_i,
  input  logic                         read_enable_i,
  input  logic [ROW_ADDRESS_BITS-1:0]  read_address_i,
  output logic [63:0]                  read_data_o
);
  (* ram_style = "block" *) logic [63:0] memory [0:ROW_COUNT-1];
  integer byte_index;

  always_ff @(posedge clk_i) begin
    if (write_enable_i) begin
      for (byte_index = 0; byte_index < 8; byte_index = byte_index + 1)
        if (write_strobe_i[byte_index])
          memory[write_address_i][byte_index*8 +: 8]
            <= write_data_i[byte_index*8 +: 8];
    end
  end

  always_ff @(posedge clk_i) begin
    if (read_enable_i)
      read_data_o <= memory[read_address_i];
  end
endmodule

module npu_scratchpad_bank #(
  parameter integer WORD_COUNT = 2048,
  parameter integer COMPUTE_LANES = 2,
  parameter integer LANE_BITS = $clog2(COMPUTE_LANES),
  parameter integer ROW_COUNT = WORD_COUNT / COMPUTE_LANES,
  parameter integer ROW_ADDRESS_BITS = $clog2(ROW_COUNT)
) (
  input  logic                           clk_i,
  input  logic                           rst_ni,

  input  logic                           port_a_write_valid_i,
  output logic                           port_a_write_ready_o,
  input  logic [31:0]                    port_a_write_address_i,
  input  logic [63:0]                    port_a_write_data_i,
  input  logic [7:0]                     port_a_write_strobe_i,
  input  logic                           port_a_read_request_valid_i,
  output logic                           port_a_read_request_ready_o,
  input  logic [31:0]                    port_a_read_address_i,
  output logic                           port_a_read_response_valid_o,
  input  logic                           port_a_read_response_ready_i,
  output logic [63:0]                    port_a_read_response_data_o,

  input  logic                           port_b_write_valid_i,
  output logic                           port_b_write_ready_o,
  input  logic [31:0]                    port_b_write_address_i,
  input  logic [COMPUTE_LANES*64-1:0]    port_b_write_data_i,
  input  logic [COMPUTE_LANES*8-1:0]     port_b_write_strobe_i,
  input  logic                           port_b_read_request_valid_i,
  output logic                           port_b_read_request_ready_o,
  input  logic [31:0]                    port_b_read_address_i,
  output logic                           port_b_read_response_valid_o,
  input  logic                           port_b_read_response_ready_i,
  output logic [COMPUTE_LANES*64-1:0]    port_b_read_response_data_o
);
  initial begin
    if (COMPUTE_LANES < 2 || (COMPUTE_LANES & (COMPUTE_LANES - 1)) != 0)
      $error("COMPUTE_LANES must be a power of two >= 2");
    if (WORD_COUNT % COMPUTE_LANES != 0)
      $error("WORD_COUNT must be divisible by COMPUTE_LANES");
  end

  logic [LANE_BITS-1:0] port_a_write_lane;
  logic [LANE_BITS-1:0] port_a_read_lane;
  logic [LANE_BITS-1:0] port_a_read_lane_q;
  logic [ROW_ADDRESS_BITS-1:0] port_a_write_row;
  logic [ROW_ADDRESS_BITS-1:0] port_a_read_row;
  logic [ROW_ADDRESS_BITS-1:0] port_b_write_row;
  logic [ROW_ADDRESS_BITS-1:0] port_b_read_row;
  logic [ROW_ADDRESS_BITS-1:0] selected_write_row;
  logic [ROW_ADDRESS_BITS-1:0] selected_read_row;
  logic [COMPUTE_LANES-1:0] write_lane_enable;
  logic [COMPUTE_LANES-1:0][63:0] selected_write_data;
  logic [COMPUTE_LANES-1:0][7:0] selected_write_strobe;
  logic [COMPUTE_LANES-1:0][63:0] read_data_q;
  logic port_a_response_valid_q;
  logic port_b_response_valid_q;
  logic accept_port_a_read;
  logic accept_port_b_read;
  logic accept_read;

  assign port_a_write_lane = port_a_write_address_i[LANE_BITS+2:3];
  assign port_a_read_lane = port_a_read_address_i[LANE_BITS+2:3];
  assign port_a_write_row
    = port_a_write_address_i[ROW_ADDRESS_BITS+LANE_BITS+2:LANE_BITS+3];
  assign port_a_read_row
    = port_a_read_address_i[ROW_ADDRESS_BITS+LANE_BITS+2:LANE_BITS+3];
  assign port_b_write_row
    = port_b_write_address_i[ROW_ADDRESS_BITS+LANE_BITS+2:LANE_BITS+3];
  assign port_b_read_row
    = port_b_read_address_i[ROW_ADDRESS_BITS+LANE_BITS+2:LANE_BITS+3];

  // DMA has priority when both clients need one physical direction. A
  // client-local simultaneous read/write keeps the original write priority.
  // The parent owns response credits, so ready here contains no return-path
  // dependency and can drive the BRAM enables at full clock rate.
  assign port_a_write_ready_o = 1'b1;
  assign port_b_write_ready_o = !port_a_write_valid_i;
  assign port_a_read_request_ready_o = !port_a_write_valid_i;
  assign port_b_read_request_ready_o = !port_b_write_valid_i
    && !port_a_read_request_valid_i;

  assign accept_port_a_read = port_a_read_request_valid_i
    && port_a_read_request_ready_o;
  assign accept_port_b_read = port_b_read_request_valid_i
    && port_b_read_request_ready_o;
  assign accept_read = accept_port_a_read || accept_port_b_read;
  assign selected_write_row = port_a_write_valid_i
    ? port_a_write_row : port_b_write_row;
  assign selected_read_row = accept_port_a_read
    ? port_a_read_row : port_b_read_row;

  always_comb begin
    write_lane_enable = '0;
    selected_write_data = '0;
    selected_write_strobe = '0;
    if (port_a_write_valid_i) begin
      write_lane_enable[port_a_write_lane] = 1'b1;
      selected_write_data[port_a_write_lane] = port_a_write_data_i;
      selected_write_strobe[port_a_write_lane] = port_a_write_strobe_i;
    end else if (port_b_write_valid_i && port_b_write_ready_o) begin
      write_lane_enable = '1;
      selected_write_data = port_b_write_data_i;
      selected_write_strobe = port_b_write_strobe_i;
    end
  end

  // A module boundary per lane keeps the inferred RAM one-dimensional and
  // matches Vivado's simple-dual-port template exactly.
  genvar memory_lane;
  generate
    for (memory_lane = 0; memory_lane < COMPUTE_LANES;
         memory_lane = memory_lane + 1) begin : g_memory_lane
      npu_scratchpad_lane #(
        .ROW_COUNT(ROW_COUNT), .ROW_ADDRESS_BITS(ROW_ADDRESS_BITS)
      ) lane (
        .clk_i(clk_i),
        .write_enable_i(write_lane_enable[memory_lane]),
        .write_address_i(selected_write_row),
        .write_data_i(selected_write_data[memory_lane]),
        .write_strobe_i(selected_write_strobe[memory_lane]),
        .read_enable_i(accept_read),
        .read_address_i(selected_read_row),
        .read_data_o(read_data_q[memory_lane]));
    end
  endgenerate

  always_ff @(posedge clk_i) begin
    if (accept_port_a_read)
      port_a_read_lane_q <= port_a_read_lane;
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      port_a_response_valid_q <= 1'b0;
      port_b_response_valid_q <= 1'b0;
    end else begin
      if (accept_port_a_read)
        port_a_response_valid_q <= 1'b1;
      else if (port_a_response_valid_q && port_a_read_response_ready_i)
        port_a_response_valid_q <= 1'b0;

      if (accept_port_b_read)
        port_b_response_valid_q <= 1'b1;
      else if (port_b_response_valid_q && port_b_read_response_ready_i)
        port_b_response_valid_q <= 1'b0;
    end
  end

  assign port_a_read_response_valid_o = port_a_response_valid_q;
  assign port_a_read_response_data_o = read_data_q[port_a_read_lane_q];
  assign port_b_read_response_valid_o = port_b_response_valid_q;
  assign port_b_read_response_data_o = read_data_q;

endmodule
