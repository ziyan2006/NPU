`timescale 1ns/1ps

module npu_scratchpad (
  input  logic         clk_i,
  input  logic         rst_ni,

  input  logic         dma_write_valid_i,
  output logic         dma_write_ready_o,
  input  logic [1:0]   dma_write_kind_i,
  input  logic         dma_write_bank_i,
  input  logic [31:0]  dma_write_address_i,
  input  logic [63:0]  dma_write_data_i,
  input  logic [7:0]   dma_write_strobe_i,
  input  logic         dma_read_request_valid_i,
  output logic         dma_read_request_ready_o,
  input  logic [1:0]   dma_read_kind_i,
  input  logic         dma_read_bank_i,
  input  logic [31:0]  dma_read_address_i,
  output logic         dma_read_response_valid_o,
  input  logic         dma_read_response_ready_i,
  output logic [63:0]  dma_read_response_data_o,

  input  logic         accelerator_write_valid_i,
  output logic         accelerator_write_ready_o,
  input  logic [1:0]   accelerator_write_kind_i,
  input  logic         accelerator_write_bank_i,
  input  logic [31:0]  accelerator_write_address_i,
  input  logic [63:0]  accelerator_write_data_i,
  input  logic [7:0]   accelerator_write_strobe_i,
  input  logic         accelerator_read_request_valid_i,
  output logic         accelerator_read_request_ready_o,
  input  logic [1:0]   accelerator_read_kind_i,
  input  logic         accelerator_read_bank_i,
  input  logic [31:0]  accelerator_read_address_i,
  output logic         accelerator_read_response_valid_o,
  input  logic         accelerator_read_response_ready_i,
  output logic [63:0]  accelerator_read_response_data_o,

  output logic         collision_stall_o
);
  import npu_dma_pkg::*;

  logic [2:0] dma_write_select;
  logic [2:0] dma_read_select;
  logic [2:0] accelerator_write_select;
  logic [2:0] accelerator_read_select;
  logic dma_write_select_valid;
  logic dma_read_select_valid;
  logic accelerator_write_select_valid;
  logic accelerator_read_select_valid;
  logic accelerator_write_collision;
  logic accelerator_read_collision;

  logic [5:0] a_write_valid;
  logic [5:0] a_write_ready;
  logic [5:0] a_read_valid;
  logic [5:0] a_read_ready;
  logic [5:0] a_response_valid;
  logic [5:0] a_response_ready;
  logic [5:0][63:0] a_response_data;
  logic [5:0] b_write_valid;
  logic [5:0] b_write_ready;
  logic [5:0] b_read_valid;
  logic [5:0] b_read_ready;
  logic [5:0] b_response_valid;
  logic [5:0] b_response_ready;
  logic [5:0][63:0] b_response_data;
  logic any_a_response;
  logic any_b_response;
  integer route_index;

  function automatic logic [2:0] bank_select(
    input logic [1:0] kind,
    input logic bank
  );
    case (kind)
      NPU_SPAD_A: bank_select = {2'd0, bank};
      NPU_SPAD_W: bank_select = 3'd2 + bank;
      NPU_SPAD_O: bank_select = 3'd4 + bank;
      default: bank_select = 3'd7;
    endcase
  endfunction

  assign dma_write_select = bank_select(dma_write_kind_i, dma_write_bank_i);
  assign dma_read_select = bank_select(dma_read_kind_i, dma_read_bank_i);
  assign accelerator_write_select = bank_select(
    accelerator_write_kind_i, accelerator_write_bank_i);
  assign accelerator_read_select = bank_select(
    accelerator_read_kind_i, accelerator_read_bank_i);
  assign dma_write_select_valid = dma_write_select < 6;
  assign dma_read_select_valid = dma_read_select < 6;
  assign accelerator_write_select_valid = accelerator_write_select < 6;
  assign accelerator_read_select_valid = accelerator_read_select < 6;
  assign any_a_response = |a_response_valid;
  assign any_b_response = |b_response_valid;

  assign accelerator_write_collision = accelerator_write_valid_i
    && accelerator_write_select_valid
    && ((dma_write_valid_i && dma_write_select_valid
          && dma_write_select == accelerator_write_select
          && dma_write_address_i[31:3]
             == accelerator_write_address_i[31:3])
        || (dma_read_request_valid_i && dma_read_select_valid
            && dma_read_select == accelerator_write_select
             && dma_read_address_i[31:3]
                == accelerator_write_address_i[31:3]));
  assign accelerator_read_collision = accelerator_read_request_valid_i
    && accelerator_read_select_valid
    && dma_write_valid_i && dma_write_select_valid
    && dma_write_select == accelerator_read_select
    && dma_write_address_i[31:3] == accelerator_read_address_i[31:3];
  assign collision_stall_o = accelerator_write_collision
    || accelerator_read_collision;

  always @* begin
    a_write_valid = '0;
    a_read_valid = '0;
    a_response_ready = {6{dma_read_response_ready_i}};
    b_write_valid = '0;
    b_read_valid = '0;
    b_response_ready = {6{accelerator_read_response_ready_i}};
    dma_write_ready_o = 1'b0;
    dma_read_request_ready_o = 1'b0;
    accelerator_write_ready_o = 1'b0;
    accelerator_read_request_ready_o = 1'b0;
    dma_read_response_valid_o = any_a_response;
    dma_read_response_data_o = '0;
    accelerator_read_response_valid_o = any_b_response;
    accelerator_read_response_data_o = '0;

    if (dma_write_select_valid) begin
      a_write_valid[dma_write_select] = dma_write_valid_i;
      dma_write_ready_o = a_write_ready[dma_write_select];
    end
    if (dma_read_select_valid && (!any_a_response
                                  || dma_read_response_ready_i)) begin
      a_read_valid[dma_read_select] = dma_read_request_valid_i;
      dma_read_request_ready_o = a_read_ready[dma_read_select];
    end
    if (accelerator_write_select_valid && !accelerator_write_collision) begin
      b_write_valid[accelerator_write_select] = accelerator_write_valid_i;
      accelerator_write_ready_o = b_write_ready[accelerator_write_select];
    end
    if (accelerator_read_select_valid && !accelerator_read_collision
        && (!any_b_response || accelerator_read_response_ready_i)) begin
      b_read_valid[accelerator_read_select]
        = accelerator_read_request_valid_i;
      accelerator_read_request_ready_o
        = b_read_ready[accelerator_read_select];
    end
    for (route_index = 0; route_index < 6; route_index = route_index + 1) begin
      if (a_response_valid[route_index])
        dma_read_response_data_o = a_response_data[route_index];
      if (b_response_valid[route_index])
        accelerator_read_response_data_o = b_response_data[route_index];
    end
  end

  npu_scratchpad_bank #(.WORD_COUNT(8192)) activation_bank0 (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .port_a_write_valid_i(a_write_valid[0]),
    .port_a_write_ready_o(a_write_ready[0]),
    .port_a_write_address_i(dma_write_address_i),
    .port_a_write_data_i(dma_write_data_i),
    .port_a_write_strobe_i(dma_write_strobe_i),
    .port_a_read_request_valid_i(a_read_valid[0]),
    .port_a_read_request_ready_o(a_read_ready[0]),
    .port_a_read_address_i(dma_read_address_i),
    .port_a_read_response_valid_o(a_response_valid[0]),
    .port_a_read_response_ready_i(a_response_ready[0]),
    .port_a_read_response_data_o(a_response_data[0]),
    .port_b_write_valid_i(b_write_valid[0]),
    .port_b_write_ready_o(b_write_ready[0]),
    .port_b_write_address_i(accelerator_write_address_i),
    .port_b_write_data_i(accelerator_write_data_i),
    .port_b_write_strobe_i(accelerator_write_strobe_i),
    .port_b_read_request_valid_i(b_read_valid[0]),
    .port_b_read_request_ready_o(b_read_ready[0]),
    .port_b_read_address_i(accelerator_read_address_i),
    .port_b_read_response_valid_o(b_response_valid[0]),
    .port_b_read_response_ready_i(b_response_ready[0]),
    .port_b_read_response_data_o(b_response_data[0]));

  npu_scratchpad_bank #(.WORD_COUNT(8192)) activation_bank1 (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .port_a_write_valid_i(a_write_valid[1]), .port_a_write_ready_o(a_write_ready[1]),
    .port_a_write_address_i(dma_write_address_i), .port_a_write_data_i(dma_write_data_i),
    .port_a_write_strobe_i(dma_write_strobe_i),
    .port_a_read_request_valid_i(a_read_valid[1]), .port_a_read_request_ready_o(a_read_ready[1]),
    .port_a_read_address_i(dma_read_address_i), .port_a_read_response_valid_o(a_response_valid[1]),
    .port_a_read_response_ready_i(a_response_ready[1]), .port_a_read_response_data_o(a_response_data[1]),
    .port_b_write_valid_i(b_write_valid[1]), .port_b_write_ready_o(b_write_ready[1]),
    .port_b_write_address_i(accelerator_write_address_i), .port_b_write_data_i(accelerator_write_data_i),
    .port_b_write_strobe_i(accelerator_write_strobe_i),
    .port_b_read_request_valid_i(b_read_valid[1]), .port_b_read_request_ready_o(b_read_ready[1]),
    .port_b_read_address_i(accelerator_read_address_i), .port_b_read_response_valid_o(b_response_valid[1]),
    .port_b_read_response_ready_i(b_response_ready[1]), .port_b_read_response_data_o(b_response_data[1]));

  npu_scratchpad_bank #(.WORD_COUNT(4096)) weight_bank0 (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .port_a_write_valid_i(a_write_valid[2]), .port_a_write_ready_o(a_write_ready[2]),
    .port_a_write_address_i(dma_write_address_i), .port_a_write_data_i(dma_write_data_i),
    .port_a_write_strobe_i(dma_write_strobe_i),
    .port_a_read_request_valid_i(a_read_valid[2]), .port_a_read_request_ready_o(a_read_ready[2]),
    .port_a_read_address_i(dma_read_address_i), .port_a_read_response_valid_o(a_response_valid[2]),
    .port_a_read_response_ready_i(a_response_ready[2]), .port_a_read_response_data_o(a_response_data[2]),
    .port_b_write_valid_i(b_write_valid[2]), .port_b_write_ready_o(b_write_ready[2]),
    .port_b_write_address_i(accelerator_write_address_i), .port_b_write_data_i(accelerator_write_data_i),
    .port_b_write_strobe_i(accelerator_write_strobe_i),
    .port_b_read_request_valid_i(b_read_valid[2]), .port_b_read_request_ready_o(b_read_ready[2]),
    .port_b_read_address_i(accelerator_read_address_i), .port_b_read_response_valid_o(b_response_valid[2]),
    .port_b_read_response_ready_i(b_response_ready[2]), .port_b_read_response_data_o(b_response_data[2]));

  npu_scratchpad_bank #(.WORD_COUNT(4096)) weight_bank1 (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .port_a_write_valid_i(a_write_valid[3]), .port_a_write_ready_o(a_write_ready[3]),
    .port_a_write_address_i(dma_write_address_i), .port_a_write_data_i(dma_write_data_i),
    .port_a_write_strobe_i(dma_write_strobe_i),
    .port_a_read_request_valid_i(a_read_valid[3]), .port_a_read_request_ready_o(a_read_ready[3]),
    .port_a_read_address_i(dma_read_address_i), .port_a_read_response_valid_o(a_response_valid[3]),
    .port_a_read_response_ready_i(a_response_ready[3]), .port_a_read_response_data_o(a_response_data[3]),
    .port_b_write_valid_i(b_write_valid[3]), .port_b_write_ready_o(b_write_ready[3]),
    .port_b_write_address_i(accelerator_write_address_i), .port_b_write_data_i(accelerator_write_data_i),
    .port_b_write_strobe_i(accelerator_write_strobe_i),
    .port_b_read_request_valid_i(b_read_valid[3]), .port_b_read_request_ready_o(b_read_ready[3]),
    .port_b_read_address_i(accelerator_read_address_i), .port_b_read_response_valid_o(b_response_valid[3]),
    .port_b_read_response_ready_i(b_response_ready[3]), .port_b_read_response_data_o(b_response_data[3]));

  npu_scratchpad_bank #(.WORD_COUNT(2048)) output_bank0 (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .port_a_write_valid_i(a_write_valid[4]), .port_a_write_ready_o(a_write_ready[4]),
    .port_a_write_address_i(dma_write_address_i), .port_a_write_data_i(dma_write_data_i),
    .port_a_write_strobe_i(dma_write_strobe_i),
    .port_a_read_request_valid_i(a_read_valid[4]), .port_a_read_request_ready_o(a_read_ready[4]),
    .port_a_read_address_i(dma_read_address_i), .port_a_read_response_valid_o(a_response_valid[4]),
    .port_a_read_response_ready_i(a_response_ready[4]), .port_a_read_response_data_o(a_response_data[4]),
    .port_b_write_valid_i(b_write_valid[4]), .port_b_write_ready_o(b_write_ready[4]),
    .port_b_write_address_i(accelerator_write_address_i), .port_b_write_data_i(accelerator_write_data_i),
    .port_b_write_strobe_i(accelerator_write_strobe_i),
    .port_b_read_request_valid_i(b_read_valid[4]), .port_b_read_request_ready_o(b_read_ready[4]),
    .port_b_read_address_i(accelerator_read_address_i), .port_b_read_response_valid_o(b_response_valid[4]),
    .port_b_read_response_ready_i(b_response_ready[4]), .port_b_read_response_data_o(b_response_data[4]));

  npu_scratchpad_bank #(.WORD_COUNT(2048)) output_bank1 (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .port_a_write_valid_i(a_write_valid[5]), .port_a_write_ready_o(a_write_ready[5]),
    .port_a_write_address_i(dma_write_address_i), .port_a_write_data_i(dma_write_data_i),
    .port_a_write_strobe_i(dma_write_strobe_i),
    .port_a_read_request_valid_i(a_read_valid[5]), .port_a_read_request_ready_o(a_read_ready[5]),
    .port_a_read_address_i(dma_read_address_i), .port_a_read_response_valid_o(a_response_valid[5]),
    .port_a_read_response_ready_i(a_response_ready[5]), .port_a_read_response_data_o(a_response_data[5]),
    .port_b_write_valid_i(b_write_valid[5]), .port_b_write_ready_o(b_write_ready[5]),
    .port_b_write_address_i(accelerator_write_address_i), .port_b_write_data_i(accelerator_write_data_i),
    .port_b_write_strobe_i(accelerator_write_strobe_i),
    .port_b_read_request_valid_i(b_read_valid[5]), .port_b_read_request_ready_o(b_read_ready[5]),
    .port_b_read_address_i(accelerator_read_address_i), .port_b_read_response_valid_o(b_response_valid[5]),
    .port_b_read_response_ready_i(b_response_ready[5]), .port_b_read_response_data_o(b_response_data[5]));

endmodule
