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
  input  logic [511:0] accelerator_write_data_i,
  input  logic [63:0]  accelerator_write_strobe_i,
  input  logic         accelerator_read_request_valid_i,
  output logic         accelerator_read_request_ready_o,
  input  logic [1:0]   accelerator_read_kind_i,
  input  logic         accelerator_read_bank_i,
  input  logic [31:0]  accelerator_read_address_i,
  output logic         accelerator_read_response_valid_o,
  input  logic         accelerator_read_response_ready_i,
  output logic [511:0] accelerator_read_response_data_o,

  input  logic         compute_activation_read_request_valid_i,
  output logic         compute_activation_read_request_ready_o,
  input  logic         compute_activation_read_bank_i,
  input  logic [31:0]  compute_activation_read_address_i,
  output logic         compute_activation_read_response_valid_o,
  input  logic         compute_activation_read_response_ready_i,
  output logic [127:0] compute_activation_read_response_data_o,
  input  logic         compute_weight_read_request_valid_i,
  output logic         compute_weight_read_request_ready_o,
  input  logic         compute_weight_read_bank_i,
  input  logic [31:0]  compute_weight_read_address_i,
  output logic         compute_weight_read_response_valid_o,
  input  logic         compute_weight_read_response_ready_i,
  output logic [511:0] compute_weight_read_response_data_o,
  input  logic         compute_output_write_valid_i,
  output logic         compute_output_write_ready_o,
  input  logic         compute_output_write_bank_i,
  input  logic [31:0]  compute_output_write_address_i,
  input  logic [255:0] compute_output_write_data_i,
  input  logic [31:0]  compute_output_write_strobe_i,

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
  logic dma_read_write_conflict;
  logic accelerator_read_write_conflict;

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
  logic [5:0][511:0] b_response_data;
  logic [1:0][127:0] activation_response_data;
  logic [1:0][511:0] weight_response_data;
  logic [1:0][255:0] output_response_data;
  logic [5:0][31:0] b_write_address;
  logic [5:0][511:0] b_write_data;
  logic [5:0][63:0] b_write_strobe;
  logic [5:0][31:0] b_read_address;
  logic dma_read_pending_q;
  logic [2:0] dma_response_select_q;
  logic dma_output_valid_q;
  logic [63:0] dma_output_data_q;
  logic accelerator_read_pending_q;
  logic [2:0] accelerator_response_select_q;
  logic accelerator_output_valid_q;
  logic [511:0] accelerator_output_data_q;
  logic dma_capture_allowed;
  logic accelerator_capture_allowed;
  logic dma_read_accept;
  logic accelerator_read_accept;
  logic compute_activation_read_pending_q;
  logic compute_activation_response_bank_q;
  logic compute_activation_output_valid_q;
  logic [127:0] compute_activation_output_data_q;
  logic compute_weight_read_pending_q;
  logic compute_weight_response_bank_q;
  logic compute_weight_output_valid_q;
  logic [511:0] compute_weight_output_data_q;
  logic compute_activation_capture_allowed;
  logic compute_weight_capture_allowed;
  logic compute_activation_read_accept;
  logic compute_weight_read_accept;
  logic compute_activation_read_collision;
  logic compute_weight_read_collision;
  logic compute_output_write_collision;
  logic [2:0] compute_activation_select;
  logic [2:0] compute_weight_select;
  logic [2:0] compute_output_select;

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

  function automatic logic same_compute_row(
    input logic [1:0] kind,
    input logic [31:0] narrow_address,
    input logic [31:0] wide_address
  );
    case (kind)
      NPU_SPAD_W:
        same_compute_row = narrow_address[31:6] == wide_address[31:6];
      NPU_SPAD_A:
        same_compute_row = narrow_address[31:4] == wide_address[31:4];
      NPU_SPAD_O:
        same_compute_row = narrow_address[31:5] == wide_address[31:5];
      default:
        same_compute_row = 1'b0;
    endcase
  endfunction

  assign dma_write_select = bank_select(dma_write_kind_i, dma_write_bank_i);
  assign dma_read_select = bank_select(dma_read_kind_i, dma_read_bank_i);
  assign accelerator_write_select = bank_select(
    accelerator_write_kind_i, accelerator_write_bank_i);
  assign accelerator_read_select = bank_select(
    accelerator_read_kind_i, accelerator_read_bank_i);
  assign compute_activation_select = {2'd0, compute_activation_read_bank_i};
  assign compute_weight_select = 3'd2 + compute_weight_read_bank_i;
  assign compute_output_select = 3'd4 + compute_output_write_bank_i;
  assign dma_write_select_valid = dma_write_select < 6;
  assign dma_read_select_valid = dma_read_select < 6;
  assign accelerator_write_select_valid = accelerator_write_select < 6;
  assign accelerator_read_select_valid = accelerator_read_select < 6;
  assign dma_read_accept = dma_read_request_valid_i
    && dma_read_request_ready_o;
  assign accelerator_read_accept = accelerator_read_request_valid_i
    && accelerator_read_request_ready_o;
  assign compute_activation_read_accept
    = compute_activation_read_request_valid_i
      && compute_activation_read_request_ready_o;
  assign compute_weight_read_accept = compute_weight_read_request_valid_i
    && compute_weight_read_request_ready_o;
  assign dma_capture_allowed = !dma_output_valid_q
    || dma_read_response_ready_i;
  assign accelerator_capture_allowed = !accelerator_output_valid_q
    || accelerator_read_response_ready_i;
  assign compute_activation_capture_allowed
    = !compute_activation_output_valid_q
      || compute_activation_read_response_ready_i;
  assign compute_weight_capture_allowed = !compute_weight_output_valid_q
    || compute_weight_read_response_ready_i;

  // One pending selector follows the synchronous BRAM read, then a top-level
  // response register isolates bank routing and consumer back-pressure from
  // the next BRAM enable path. Both stages can advance every cycle.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      dma_read_pending_q <= 1'b0;
      dma_response_select_q <= '0;
      dma_output_valid_q <= 1'b0;
      accelerator_read_pending_q <= 1'b0;
      accelerator_response_select_q <= '0;
      accelerator_output_valid_q <= 1'b0;
      compute_activation_read_pending_q <= 1'b0;
      compute_activation_response_bank_q <= 1'b0;
      compute_activation_output_valid_q <= 1'b0;
      compute_weight_read_pending_q <= 1'b0;
      compute_weight_response_bank_q <= 1'b0;
      compute_weight_output_valid_q <= 1'b0;
    end else begin
      if (dma_output_valid_q && dma_read_response_ready_i)
        dma_output_valid_q <= 1'b0;
      if (dma_read_pending_q && dma_capture_allowed) begin
        dma_read_pending_q <= 1'b0;
        dma_output_valid_q <= 1'b1;
        dma_output_data_q <= a_response_data[dma_response_select_q];
      end
      if (dma_read_accept) begin
        dma_read_pending_q <= 1'b1;
        dma_response_select_q <= dma_read_select;
      end

      if (accelerator_output_valid_q && accelerator_read_response_ready_i)
        accelerator_output_valid_q <= 1'b0;
      if (accelerator_read_pending_q && accelerator_capture_allowed) begin
        accelerator_read_pending_q <= 1'b0;
        accelerator_output_valid_q <= 1'b1;
        accelerator_output_data_q
          <= b_response_data[accelerator_response_select_q];
      end
      if (accelerator_read_accept) begin
        accelerator_read_pending_q <= 1'b1;
        accelerator_response_select_q <= accelerator_read_select;
      end

      if (compute_activation_output_valid_q
          && compute_activation_read_response_ready_i)
        compute_activation_output_valid_q <= 1'b0;
      if (compute_activation_read_pending_q
          && compute_activation_capture_allowed) begin
        compute_activation_read_pending_q <= 1'b0;
        compute_activation_output_valid_q <= 1'b1;
        compute_activation_output_data_q
          <= activation_response_data[compute_activation_response_bank_q];
      end
      if (compute_activation_read_accept) begin
        compute_activation_read_pending_q <= 1'b1;
        compute_activation_response_bank_q
          <= compute_activation_read_bank_i;
      end

      if (compute_weight_output_valid_q
          && compute_weight_read_response_ready_i)
        compute_weight_output_valid_q <= 1'b0;
      if (compute_weight_read_pending_q
          && compute_weight_capture_allowed) begin
        compute_weight_read_pending_q <= 1'b0;
        compute_weight_output_valid_q <= 1'b1;
        compute_weight_output_data_q
          <= weight_response_data[compute_weight_response_bank_q];
      end
      if (compute_weight_read_accept) begin
        compute_weight_read_pending_q <= 1'b1;
        compute_weight_response_bank_q <= compute_weight_read_bank_i;
      end
    end
  end

  // A lane-striped bank has one physical write direction and one read
  // direction. DMA keeps priority when two clients request the same direction.
  // Opposite-direction operations may proceed together unless their rows are
  // equal, in which case the accelerator waits to avoid BRAM collision modes.
  assign accelerator_write_collision = accelerator_write_valid_i
    && accelerator_write_select_valid
    && ((dma_write_valid_i && dma_write_select_valid
         && dma_write_select == accelerator_write_select)
        || (dma_read_request_valid_i && dma_read_select_valid
            && dma_read_select == accelerator_write_select
            && same_compute_row(accelerator_write_kind_i,
                                dma_read_address_i,
                                accelerator_write_address_i))
        || (compute_output_write_valid_i
            && accelerator_write_select == compute_output_select));
  assign accelerator_read_collision = accelerator_read_request_valid_i
    && accelerator_read_select_valid
    && ((dma_read_request_valid_i && dma_read_select_valid
         && dma_read_select == accelerator_read_select)
        || (dma_write_valid_i && dma_write_select_valid
            && dma_write_select == accelerator_read_select
            && same_compute_row(accelerator_read_kind_i,
                                dma_write_address_i,
                                accelerator_read_address_i))
        || (compute_activation_read_request_valid_i
            && accelerator_read_select == compute_activation_select)
        || (compute_weight_read_request_valid_i
            && accelerator_read_select == compute_weight_select)
        || (compute_output_write_valid_i
            && accelerator_read_select == compute_output_select));
  assign compute_activation_read_collision
    = compute_activation_read_request_valid_i
      && ((dma_read_request_valid_i && dma_read_select_valid
           && dma_read_select == compute_activation_select)
          || (dma_write_valid_i && dma_write_select_valid
              && dma_write_select == compute_activation_select
              && same_compute_row(NPU_SPAD_A, dma_write_address_i,
                                  compute_activation_read_address_i)));
  assign compute_weight_read_collision
    = compute_weight_read_request_valid_i
      && ((dma_read_request_valid_i && dma_read_select_valid
           && dma_read_select == compute_weight_select)
          || (dma_write_valid_i && dma_write_select_valid
              && dma_write_select == compute_weight_select
              && same_compute_row(NPU_SPAD_W, dma_write_address_i,
                                  compute_weight_read_address_i)));
  assign compute_output_write_collision = compute_output_write_valid_i
    && ((dma_write_valid_i && dma_write_select_valid
         && dma_write_select == compute_output_select)
        || (dma_read_request_valid_i && dma_read_select_valid
            && dma_read_select == compute_output_select
            && same_compute_row(NPU_SPAD_O, dma_read_address_i,
                                compute_output_write_address_i)));
  assign collision_stall_o = accelerator_write_collision
    || accelerator_read_collision || compute_activation_read_collision
    || compute_weight_read_collision || compute_output_write_collision;
  assign dma_read_write_conflict = dma_read_request_valid_i
    && dma_read_select_valid && dma_write_valid_i
    && dma_write_select_valid && dma_read_select == dma_write_select;
  assign accelerator_read_write_conflict
    = accelerator_read_request_valid_i
      && accelerator_read_select_valid && accelerator_write_valid_i
      && accelerator_write_select_valid
      && accelerator_read_select == accelerator_write_select;

  always_comb begin
    a_write_valid = '0;
    a_read_valid = '0;
    a_response_ready = '0;
    b_write_valid = '0;
    b_read_valid = '0;
    b_response_ready = '0;
    b_write_address = '0;
    b_write_data = '0;
    b_write_strobe = '0;
    b_read_address = '0;
    dma_write_ready_o = 1'b0;
    dma_read_request_ready_o = 1'b0;
    accelerator_write_ready_o = 1'b0;
    accelerator_read_request_ready_o = 1'b0;
    dma_read_response_valid_o = dma_output_valid_q;
    dma_read_response_data_o = dma_output_data_q;
    accelerator_read_response_valid_o = accelerator_output_valid_q;
    accelerator_read_response_data_o = accelerator_output_data_q;
    compute_activation_read_request_ready_o = 1'b0;
    compute_activation_read_response_valid_o
      = compute_activation_output_valid_q;
    compute_activation_read_response_data_o
      = compute_activation_output_data_q;
    compute_weight_read_request_ready_o = 1'b0;
    compute_weight_read_response_valid_o = compute_weight_output_valid_q;
    compute_weight_read_response_data_o = compute_weight_output_data_q;
    compute_output_write_ready_o = 1'b0;

    if (dma_write_select_valid) begin
      a_write_valid[dma_write_select] = dma_write_valid_i;
      dma_write_ready_o = a_write_ready[dma_write_select];
    end
    if (dma_read_pending_q && dma_capture_allowed) begin
      a_response_ready[dma_response_select_q]
        = 1'b1;
    end
    if (dma_read_select_valid
        && (!dma_read_pending_q || dma_capture_allowed)
        && !dma_read_write_conflict) begin
      a_read_valid[dma_read_select] = dma_read_request_valid_i;
      dma_read_request_ready_o = 1'b1;
    end
    if (accelerator_write_select_valid && !accelerator_write_collision) begin
      b_write_valid[accelerator_write_select] = accelerator_write_valid_i;
      accelerator_write_ready_o = b_write_ready[accelerator_write_select];
      b_write_address[accelerator_write_select]
        = accelerator_write_address_i;
      b_write_data[accelerator_write_select] = accelerator_write_data_i;
      b_write_strobe[accelerator_write_select]
        = accelerator_write_strobe_i;
    end
    if (accelerator_read_pending_q && accelerator_capture_allowed) begin
      b_response_ready[accelerator_response_select_q]
        = 1'b1;
    end
    if (accelerator_read_select_valid && !accelerator_read_collision
        && (!accelerator_read_pending_q
            || accelerator_capture_allowed)
        && !accelerator_read_write_conflict) begin
      b_read_valid[accelerator_read_select]
        = accelerator_read_request_valid_i;
      b_read_address[accelerator_read_select] = accelerator_read_address_i;
      accelerator_read_request_ready_o
        = b_read_ready[accelerator_read_select];
    end

    if (compute_activation_read_pending_q
        && compute_activation_capture_allowed)
      b_response_ready[{2'd0, compute_activation_response_bank_q}] = 1'b1;
    if (!compute_activation_read_collision
        && (!compute_activation_read_pending_q
            || compute_activation_capture_allowed)) begin
      compute_activation_read_request_ready_o
        = b_read_ready[compute_activation_select];
      if (compute_activation_read_request_valid_i) begin
        b_read_valid[compute_activation_select] = 1'b1;
        b_read_address[compute_activation_select]
          = compute_activation_read_address_i;
      end
    end

    if (compute_weight_read_pending_q && compute_weight_capture_allowed)
      b_response_ready[3'd2 + compute_weight_response_bank_q] = 1'b1;
    if (!compute_weight_read_collision
        && (!compute_weight_read_pending_q
            || compute_weight_capture_allowed)) begin
      compute_weight_read_request_ready_o = b_read_ready[compute_weight_select];
      if (compute_weight_read_request_valid_i) begin
        b_read_valid[compute_weight_select] = 1'b1;
        b_read_address[compute_weight_select] = compute_weight_read_address_i;
      end
    end

    if (!compute_output_write_collision) begin
      compute_output_write_ready_o = b_write_ready[compute_output_select];
      if (compute_output_write_valid_i) begin
        b_write_valid[compute_output_select] = 1'b1;
        b_write_address[compute_output_select]
          = compute_output_write_address_i;
        b_write_data[compute_output_select]
          = {256'd0, compute_output_write_data_i};
        b_write_strobe[compute_output_select]
          = {32'd0, compute_output_write_strobe_i};
      end
    end
  end

  always_comb begin
    b_response_data = '0;
    b_response_data[0][127:0] = activation_response_data[0];
    b_response_data[1][127:0] = activation_response_data[1];
    b_response_data[2] = weight_response_data[0];
    b_response_data[3] = weight_response_data[1];
    b_response_data[4][255:0] = output_response_data[0];
    b_response_data[5][255:0] = output_response_data[1];
  end

  genvar bank_index;
  generate
    for (bank_index = 0; bank_index < 2; bank_index = bank_index + 1) begin : g_activation
      npu_scratchpad_bank #(
        .WORD_COUNT(8192), .COMPUTE_LANES(2)
      ) bank (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .port_a_write_valid_i(a_write_valid[bank_index]),
        .port_a_write_ready_o(a_write_ready[bank_index]),
        .port_a_write_address_i(dma_write_address_i),
        .port_a_write_data_i(dma_write_data_i),
        .port_a_write_strobe_i(dma_write_strobe_i),
        .port_a_read_request_valid_i(a_read_valid[bank_index]),
        .port_a_read_request_ready_o(a_read_ready[bank_index]),
        .port_a_read_address_i(dma_read_address_i),
        .port_a_read_response_valid_o(a_response_valid[bank_index]),
        .port_a_read_response_ready_i(a_response_ready[bank_index]),
        .port_a_read_response_data_o(a_response_data[bank_index]),
        .port_b_write_valid_i(b_write_valid[bank_index]),
        .port_b_write_ready_o(b_write_ready[bank_index]),
        .port_b_write_address_i(b_write_address[bank_index]),
        .port_b_write_data_i(b_write_data[bank_index][127:0]),
        .port_b_write_strobe_i(b_write_strobe[bank_index][15:0]),
        .port_b_read_request_valid_i(b_read_valid[bank_index]),
        .port_b_read_request_ready_o(b_read_ready[bank_index]),
        .port_b_read_address_i(b_read_address[bank_index]),
        .port_b_read_response_valid_o(b_response_valid[bank_index]),
        .port_b_read_response_ready_i(b_response_ready[bank_index]),
        .port_b_read_response_data_o(activation_response_data[bank_index]));
    end
    for (bank_index = 0; bank_index < 2; bank_index = bank_index + 1) begin : g_weight
      npu_scratchpad_bank #(
        .WORD_COUNT(4096), .COMPUTE_LANES(8)
      ) bank (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .port_a_write_valid_i(a_write_valid[bank_index+2]),
        .port_a_write_ready_o(a_write_ready[bank_index+2]),
        .port_a_write_address_i(dma_write_address_i),
        .port_a_write_data_i(dma_write_data_i),
        .port_a_write_strobe_i(dma_write_strobe_i),
        .port_a_read_request_valid_i(a_read_valid[bank_index+2]),
        .port_a_read_request_ready_o(a_read_ready[bank_index+2]),
        .port_a_read_address_i(dma_read_address_i),
        .port_a_read_response_valid_o(a_response_valid[bank_index+2]),
        .port_a_read_response_ready_i(a_response_ready[bank_index+2]),
        .port_a_read_response_data_o(a_response_data[bank_index+2]),
        .port_b_write_valid_i(b_write_valid[bank_index+2]),
        .port_b_write_ready_o(b_write_ready[bank_index+2]),
        .port_b_write_address_i(b_write_address[bank_index+2]),
        .port_b_write_data_i(b_write_data[bank_index+2]),
        .port_b_write_strobe_i(b_write_strobe[bank_index+2]),
        .port_b_read_request_valid_i(b_read_valid[bank_index+2]),
        .port_b_read_request_ready_o(b_read_ready[bank_index+2]),
        .port_b_read_address_i(b_read_address[bank_index+2]),
        .port_b_read_response_valid_o(b_response_valid[bank_index+2]),
        .port_b_read_response_ready_i(b_response_ready[bank_index+2]),
        .port_b_read_response_data_o(weight_response_data[bank_index]));
    end
    for (bank_index = 0; bank_index < 2; bank_index = bank_index + 1) begin : g_output
      npu_scratchpad_bank #(
        .WORD_COUNT(2048), .COMPUTE_LANES(4)
      ) bank (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .port_a_write_valid_i(a_write_valid[bank_index+4]),
        .port_a_write_ready_o(a_write_ready[bank_index+4]),
        .port_a_write_address_i(dma_write_address_i),
        .port_a_write_data_i(dma_write_data_i),
        .port_a_write_strobe_i(dma_write_strobe_i),
        .port_a_read_request_valid_i(a_read_valid[bank_index+4]),
        .port_a_read_request_ready_o(a_read_ready[bank_index+4]),
        .port_a_read_address_i(dma_read_address_i),
        .port_a_read_response_valid_o(a_response_valid[bank_index+4]),
        .port_a_read_response_ready_i(a_response_ready[bank_index+4]),
        .port_a_read_response_data_o(a_response_data[bank_index+4]),
        .port_b_write_valid_i(b_write_valid[bank_index+4]),
        .port_b_write_ready_o(b_write_ready[bank_index+4]),
        .port_b_write_address_i(b_write_address[bank_index+4]),
        .port_b_write_data_i(b_write_data[bank_index+4][255:0]),
        .port_b_write_strobe_i(b_write_strobe[bank_index+4][31:0]),
        .port_b_read_request_valid_i(b_read_valid[bank_index+4]),
        .port_b_read_request_ready_o(b_read_ready[bank_index+4]),
        .port_b_read_address_i(b_read_address[bank_index+4]),
        .port_b_read_response_valid_o(b_response_valid[bank_index+4]),
        .port_b_read_response_ready_i(b_response_ready[bank_index+4]),
        .port_b_read_response_data_o(output_response_data[bank_index]));
    end
  endgenerate

endmodule
