`timescale 1ns/1ps

module audio_async_fifo #(
  parameter integer DEPTH = 8192,
  parameter integer ADDR_WIDTH = 13
) (
  input  logic                  wr_clk_i,
  input  logic                  wr_rst_ni,
  input  logic                  wr_valid_i,
  input  logic [63:0]           wr_data_i,
  output logic                  wr_full_o,
  output logic [ADDR_WIDTH:0]   wr_level_o,

  input  logic                  rd_clk_i,
  input  logic                  rd_rst_ni,
  input  logic                  rd_ready_i,
  output logic                  rd_valid_o,
  output logic [63:0]           rd_data_o,
  output logic                  rd_empty_o,
  output logic [ADDR_WIDTH:0]   rd_level_o
);
  localparam integer PTR_WIDTH = ADDR_WIDTH + 1;

  (* ram_style = "block" *) logic [63:0] memory [0:DEPTH-1];

  logic [PTR_WIDTH-1:0] wr_bin_q;
  logic [PTR_WIDTH-1:0] wr_gray_q;
  logic [PTR_WIDTH-1:0] wr_bin_next;
  logic [PTR_WIDTH-1:0] wr_gray_next;
  logic wr_full_next;
  (* async_reg = "true" *) logic [PTR_WIDTH-1:0] rd_gray_wr_sync1_q;
  (* async_reg = "true" *) logic [PTR_WIDTH-1:0] rd_gray_wr_sync2_q;

  logic [PTR_WIDTH-1:0] rd_bin_q;
  logic [PTR_WIDTH-1:0] rd_gray_q;
  logic rd_valid_q;
  logic [63:0] rd_data_q;
  logic rd_fetch;
  logic [ADDR_WIDTH-1:0] rd_fetch_address;
  logic [PTR_WIDTH-1:0] rd_consumed_bin;
  (* async_reg = "true" *) logic [PTR_WIDTH-1:0] wr_gray_rd_sync1_q;
  (* async_reg = "true" *) logic [PTR_WIDTH-1:0] wr_gray_rd_sync2_q;

  logic [PTR_WIDTH-1:0] rd_bin_wr_sync;
  logic [PTR_WIDTH-1:0] wr_bin_rd_sync;

  initial begin
    if (DEPTH != (1 << ADDR_WIDTH))
      $error("audio_async_fifo DEPTH must equal 2**ADDR_WIDTH");
  end

  function automatic logic [PTR_WIDTH-1:0] binary_to_gray(
    input logic [PTR_WIDTH-1:0] value
  );
    binary_to_gray = (value >> 1) ^ value;
  endfunction

  function automatic logic [PTR_WIDTH-1:0] gray_to_binary(
    input logic [PTR_WIDTH-1:0] value
  );
    integer bit_index;
    begin
      gray_to_binary[PTR_WIDTH-1] = value[PTR_WIDTH-1];
      for (bit_index = PTR_WIDTH-2; bit_index >= 0; bit_index = bit_index-1)
        gray_to_binary[bit_index] = gray_to_binary[bit_index+1]
          ^ value[bit_index];
    end
  endfunction

  always_comb begin
    wr_bin_next = wr_bin_q;
    if (wr_valid_i && !wr_full_o)
      wr_bin_next = wr_bin_q + 1'b1;
    wr_gray_next = binary_to_gray(wr_bin_next);
    wr_full_next = wr_gray_next == {
      ~rd_gray_wr_sync2_q[PTR_WIDTH-1:PTR_WIDTH-2],
      rd_gray_wr_sync2_q[PTR_WIDTH-3:0]
    };
    rd_bin_wr_sync = gray_to_binary(rd_gray_wr_sync2_q);
    wr_bin_rd_sync = gray_to_binary(wr_gray_rd_sync2_q);
    wr_level_o = wr_bin_q - rd_bin_wr_sync;
    rd_level_o = wr_bin_rd_sync - rd_bin_q;
    rd_consumed_bin = rd_bin_q + 1'b1;
    rd_fetch = (!rd_valid_q && rd_bin_q != wr_bin_rd_sync)
      || (rd_valid_q && rd_ready_i && rd_consumed_bin != wr_bin_rd_sync);
    rd_fetch_address = (!rd_valid_q)
      ? rd_bin_q[ADDR_WIDTH-1:0]
      : rd_consumed_bin[ADDR_WIDTH-1:0];
  end

  assign rd_valid_o = rd_valid_q;
  assign rd_data_o = rd_data_q;
  assign rd_empty_o = rd_bin_q == wr_bin_rd_sync;

  always_ff @(posedge wr_clk_i or negedge wr_rst_ni) begin
    if (!wr_rst_ni) begin
      wr_bin_q <= '0;
      wr_gray_q <= '0;
      wr_full_o <= 1'b0;
      rd_gray_wr_sync1_q <= '0;
      rd_gray_wr_sync2_q <= '0;
    end else begin
      rd_gray_wr_sync1_q <= rd_gray_q;
      rd_gray_wr_sync2_q <= rd_gray_wr_sync1_q;
      wr_bin_q <= wr_bin_next;
      wr_gray_q <= wr_gray_next;
      wr_full_o <= wr_full_next;
    end
  end

  always_ff @(posedge wr_clk_i) begin
    if (wr_valid_i && !wr_full_o)
      memory[wr_bin_q[ADDR_WIDTH-1:0]] <= wr_data_i;
  end

  always_ff @(posedge rd_clk_i or negedge rd_rst_ni) begin
    if (!rd_rst_ni) begin
      rd_bin_q <= '0;
      rd_gray_q <= '0;
      rd_valid_q <= 1'b0;
      wr_gray_rd_sync1_q <= '0;
      wr_gray_rd_sync2_q <= '0;
    end else begin
      wr_gray_rd_sync1_q <= wr_gray_q;
      wr_gray_rd_sync2_q <= wr_gray_rd_sync1_q;
      if (!rd_valid_q && rd_fetch) begin
        rd_valid_q <= 1'b1;
      end else if (rd_valid_q && rd_ready_i) begin
        rd_bin_q <= rd_consumed_bin;
        rd_gray_q <= binary_to_gray(rd_consumed_bin);
        rd_valid_q <= rd_consumed_bin != wr_bin_rd_sync;
      end
    end
  end

  // Keep the memory read in a reset-free synchronous process so Vivado can
  // infer the asynchronous FIFO storage as simple dual-port block RAM.
  always_ff @(posedge rd_clk_i) begin
    if (rd_fetch)
      rd_data_q <= memory[rd_fetch_address];
  end
endmodule
