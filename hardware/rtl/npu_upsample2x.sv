`timescale 1ns/1ps

module npu_upsample2x (
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          soft_reset_i,

  input  logic          start_valid_i,
  output logic          start_ready_o,
  input  logic [127:0]  command_bits_i,
  input  logic [511:0]  source_desc_bits_i,
  input  logic [511:0]  destination_desc_bits_i,
  input  logic [63:0]   activation_base_i,

  output logic [63:0]   m_axi_araddr_o,
  output logic [7:0]    m_axi_arlen_o,
  output logic [2:0]    m_axi_arsize_o,
  output logic [1:0]    m_axi_arburst_o,
  output logic          m_axi_arvalid_o,
  input  logic          m_axi_arready_i,
  input  logic [63:0]   m_axi_rdata_i,
  input  logic [1:0]    m_axi_rresp_i,
  input  logic          m_axi_rlast_i,
  input  logic          m_axi_rvalid_i,
  output logic          m_axi_rready_o,

  output logic [63:0]   m_axi_awaddr_o,
  output logic [7:0]    m_axi_awlen_o,
  output logic [2:0]    m_axi_awsize_o,
  output logic [1:0]    m_axi_awburst_o,
  output logic          m_axi_awvalid_o,
  input  logic          m_axi_awready_i,
  output logic [63:0]   m_axi_wdata_o,
  output logic [7:0]    m_axi_wstrb_o,
  output logic          m_axi_wlast_o,
  output logic          m_axi_wvalid_o,
  input  logic          m_axi_wready_i,
  input  logic [1:0]    m_axi_bresp_i,
  input  logic          m_axi_bvalid_i,
  output logic          m_axi_bready_o,

  output logic          busy_o,
  output logic          done_pulse_o,
  output logic          error_pulse_o,
  output logic [3:0]    error_reason_o,
  output logic [31:0]   bytes_read_o,
  output logic [31:0]   bytes_written_o
);
  import npu_isa_pkg::*;

  localparam logic [3:0] UPSAMPLE_ERROR_NONE = 4'h0;
  localparam logic [3:0] UPSAMPLE_ERROR_COMMAND = 4'h1;
  localparam logic [3:0] UPSAMPLE_ERROR_DESCRIPTOR = 4'h2;
  localparam logic [3:0] UPSAMPLE_ERROR_GEOMETRY = 4'h3;
  localparam logic [3:0] UPSAMPLE_ERROR_ALIGNMENT = 4'h4;
  localparam logic [3:0] UPSAMPLE_ERROR_BOUNDS = 4'h5;
  localparam logic [3:0] UPSAMPLE_ERROR_READ = 4'h6;
  localparam logic [3:0] UPSAMPLE_ERROR_WRITE = 4'h7;
  localparam logic [3:0] UPSAMPLE_ERROR_PROTOCOL = 4'h8;

  typedef enum logic [3:0] {
    UPSAMPLE_IDLE,
    UPSAMPLE_SETUP_ROW,
    UPSAMPLE_PLAN_READ,
    UPSAMPLE_READ_ADDRESS,
    UPSAMPLE_READ_DATA,
    UPSAMPLE_PLAN_WRITE,
    UPSAMPLE_WRITE_ADDRESS,
    UPSAMPLE_WRITE_PREFETCH,
    UPSAMPLE_WRITE_DATA,
    UPSAMPLE_WRITE_RESPONSE,
    UPSAMPLE_DONE,
    UPSAMPLE_ERROR
  } upsample_state_e;

  upsample_state_e state_q;
  npu_command_t start_command;
  npu_tensor_desc_t start_source_desc;
  npu_tensor_desc_t start_destination_desc;
  npu_tensor_desc_t source_desc_q;
  npu_tensor_desc_t destination_desc_q;
  logic [63:0] source_base_q;
  logic [63:0] destination_base_q;
  logic [63:0] source_row_address_q;
  logic [63:0] destination_row_address_q;
  logic [31:0] source_row_bytes_q;
  logic [31:0] destination_row_bytes_q;
  logic [8:0] source_row_beats_q;
  logic [8:0] destination_row_beats_q;
  logic [31:0] pixel_bytes_q;
  logic [6:0] pixel_beats_q;
  logic [15:0] source_row_q;
  logic output_row_copy_q;
  logic [8:0] read_beat_q;
  logic [8:0] write_beat_q;
  logic [8:0] read_burst_beats_q;
  logic [8:0] read_burst_beat_q;
  logic [8:0] write_burst_beats_q;
  logic [8:0] write_burst_beat_q;
  logic [7:0] buffer_index_q;
  logic [6:0] beat_in_pixel_q;
  logic horizontal_copy_q;
  logic read_error_q;
  (* ram_style = "block" *) logic [63:0] row_buffer [0:255];
  logic [63:0] buffer_read_data_q;
  logic row_buffer_read_enable;
  logic [7:0] row_buffer_read_address;
  logic [7:0] next_buffer_index;
  logic [6:0] next_beat_in_pixel;
  logic next_horizontal_copy;

  logic start_command_valid;
  logic start_descriptor_valid;
  logic start_geometry_valid;
  logic start_alignment_valid;
  logic start_address_valid;
  logic [16:0] start_padded_channels;
  logic [31:0] start_pixel_bytes;
  logic [64:0] start_source_base;
  logic [64:0] start_destination_base;
  logic start_fire;

  logic mul_start;
  logic mul_busy;
  logic mul_done;
  logic [63:0] mul_result;
  logic read_fire;
  logic write_fire;
  logic expected_read_last;
  logic expected_write_last;
  logic [63:0] read_burst_address_q;
  logic [63:0] write_burst_address_q;
  logic [63:0] next_read_burst_address;
  logic [63:0] next_write_burst_address;
  logic [8:0] read_remaining_beats;
  logic [8:0] write_remaining_beats;
  logic [9:0] read_boundary_beats;
  logic [9:0] write_boundary_beats;
  logic [8:0] planned_read_burst_beats;
  logic [8:0] planned_write_burst_beats;

  assign start_command = npu_command_t'(command_bits_i);
  assign start_source_desc = npu_tensor_desc_t'(source_desc_bits_i);
  assign start_destination_desc
    = npu_tensor_desc_t'(destination_desc_bits_i);
  assign start_padded_channels = (start_source_desc.shape_c + 7) & 17'h1fff8;
  assign start_pixel_bytes = start_padded_channels << 1;
  assign start_source_base
    = {1'b0, activation_base_i} + start_source_desc.base_offset;
  assign start_destination_base
    = {1'b0, activation_base_i} + start_destination_desc.base_offset;

  assign start_command_valid = start_command.opcode == NPU_OP_UPSAMPLE2X
    && start_command.flags == 0 && start_command.dst_td != NPU_NONE_INDEX
    && start_command.src0_td != NPU_NONE_INDEX
    && start_command.src1_td == NPU_NONE_INDEX
    && start_command.op_desc == NPU_NONE_INDEX
    && start_command.quant_desc == NPU_NONE_INDEX && start_command.imm == 0;
  assign start_descriptor_valid = start_source_desc.shape_n == 1
    && start_destination_desc.shape_n == 1
    && start_source_desc.dtype == NPU_DTYPE_INT12_IN_INT16
    && start_destination_desc.dtype == NPU_DTYPE_INT12_IN_INT16
    && start_source_desc.layout == NPU_LAYOUT_NHWC8
    && start_destination_desc.layout == NPU_LAYOUT_NHWC8
    && start_source_desc.stride_c == 2
    && start_destination_desc.stride_c == 2
    && start_source_desc.stride_w == start_pixel_bytes
    && start_destination_desc.stride_w == start_pixel_bytes;
  assign start_geometry_valid = start_source_desc.shape_h != 0
    && start_source_desc.shape_w != 0 && start_source_desc.shape_c != 0
    && start_source_desc.shape_h <= 128
    && start_source_desc.shape_w <= 128
    && start_source_desc.shape_c <= 224
    && start_destination_desc.shape_h
       == (start_source_desc.shape_h << 1)
    && start_destination_desc.shape_w
       == (start_source_desc.shape_w << 1)
    && start_destination_desc.shape_c == start_source_desc.shape_c;
  assign start_alignment_valid = !start_source_base[64]
    && !start_destination_base[64]
    && start_source_base[2:0] == 0 && start_destination_base[2:0] == 0
    && start_source_desc.stride_h[2:0] == 0
    && start_destination_desc.stride_h[2:0] == 0;
  assign start_address_valid = start_source_desc.allocation_bytes != 0
    && start_destination_desc.allocation_bytes != 0
    && start_source_desc.stride_n <= start_source_desc.allocation_bytes
    && start_destination_desc.stride_n
       <= start_destination_desc.allocation_bytes;
  assign start_ready_o = rst_ni && !soft_reset_i
    && state_q == UPSAMPLE_IDLE;
  assign start_fire = start_valid_i && start_ready_o;
  assign busy_o = state_q != UPSAMPLE_IDLE;

  assign mul_start = state_q == UPSAMPLE_SETUP_ROW && !mul_busy && !mul_done;
  npu_u32_mul_iter row_multiplier (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_i(mul_start), .operand_a_i({16'd0, source_desc_q.shape_w}),
    .operand_b_i(pixel_bytes_q), .busy_o(mul_busy), .done_o(mul_done),
    .result_o(mul_result));

  assign next_read_burst_address
    = source_row_address_q + ({55'd0, read_beat_q} << 3);
  assign next_write_burst_address
    = destination_row_address_q + ({55'd0, write_beat_q} << 3);
  assign read_remaining_beats = source_row_beats_q - read_beat_q;
  assign write_remaining_beats = destination_row_beats_q - write_beat_q;
  assign read_boundary_beats
    = (13'd4096 - {1'b0, next_read_burst_address[11:0]}) >> 3;
  assign write_boundary_beats
    = (13'd4096 - {1'b0, next_write_burst_address[11:0]}) >> 3;

  always_comb begin
    planned_read_burst_beats = read_remaining_beats;
    if (planned_read_burst_beats > read_boundary_beats)
      planned_read_burst_beats = read_boundary_beats[8:0];
    if (planned_read_burst_beats > 256)
      planned_read_burst_beats = 256;
    planned_write_burst_beats = write_remaining_beats;
    if (planned_write_burst_beats > write_boundary_beats)
      planned_write_burst_beats = write_boundary_beats[8:0];
    if (planned_write_burst_beats > 256)
      planned_write_burst_beats = 256;
  end

  assign m_axi_araddr_o = read_burst_address_q;
  assign m_axi_arlen_o = read_burst_beats_q[7:0] - 1'b1;
  assign m_axi_arsize_o = 3'd3;
  assign m_axi_arburst_o = 2'b01;
  assign m_axi_arvalid_o = state_q == UPSAMPLE_READ_ADDRESS;
  assign m_axi_rready_o = state_q == UPSAMPLE_READ_DATA;
  assign read_fire = m_axi_rvalid_i && m_axi_rready_o;
  assign expected_read_last = read_burst_beat_q + 1 == read_burst_beats_q;

  assign m_axi_awaddr_o = write_burst_address_q;
  assign m_axi_awlen_o = write_burst_beats_q[7:0] - 1'b1;
  assign m_axi_awsize_o = 3'd3;
  assign m_axi_awburst_o = 2'b01;
  assign m_axi_awvalid_o = state_q == UPSAMPLE_WRITE_ADDRESS;
  assign m_axi_wdata_o = buffer_read_data_q;
  assign m_axi_wstrb_o = 8'hff;
  assign m_axi_wlast_o = write_burst_beat_q + 1 == write_burst_beats_q;
  assign m_axi_wvalid_o = state_q == UPSAMPLE_WRITE_DATA;
  assign write_fire = m_axi_wvalid_o && m_axi_wready_i;
  assign m_axi_bready_o = state_q == UPSAMPLE_WRITE_RESPONSE;
  assign row_buffer_read_enable = state_q == UPSAMPLE_WRITE_PREFETCH
    || (write_fire && !m_axi_wlast_o);
  assign expected_write_last = write_burst_beat_q + 1
    == write_burst_beats_q;
  assign row_buffer_read_address = state_q == UPSAMPLE_WRITE_PREFETCH
    ? buffer_index_q : next_buffer_index;

  always_comb begin
    next_buffer_index = buffer_index_q + 1'b1;
    next_beat_in_pixel = beat_in_pixel_q + 1'b1;
    next_horizontal_copy = horizontal_copy_q;
    if (beat_in_pixel_q + 1 == pixel_beats_q) begin
      next_beat_in_pixel = '0;
      if (!horizontal_copy_q) begin
        next_buffer_index = buffer_index_q - pixel_beats_q + 1'b1;
        next_horizontal_copy = 1'b1;
      end else begin
        next_buffer_index = buffer_index_q + 1'b1;
        next_horizontal_copy = 1'b0;
      end
    end
  end

  // Separate write/read processes match Vivado's simple-dual-port BRAM
  // template. The read is synchronous and prefetched one cycle before WVALID.
  always_ff @(posedge clk_i) begin
    if (read_fire && m_axi_rresp_i == 2'b00)
      row_buffer[read_beat_q[7:0]] <= m_axi_rdata_i;
  end

  always_ff @(posedge clk_i) begin
    if (row_buffer_read_enable)
      buffer_read_data_q <= row_buffer[row_buffer_read_address];
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= UPSAMPLE_IDLE;
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      error_reason_o <= UPSAMPLE_ERROR_NONE;
      bytes_read_o <= '0;
      bytes_written_o <= '0;
    end else if (soft_reset_i && state_q == UPSAMPLE_IDLE) begin
      state_q <= UPSAMPLE_IDLE;
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      error_reason_o <= UPSAMPLE_ERROR_NONE;
      bytes_read_o <= '0;
      bytes_written_o <= '0;
    end else begin
      done_pulse_o <= state_q == UPSAMPLE_DONE;
      error_pulse_o <= state_q == UPSAMPLE_ERROR;

      case (state_q)
        UPSAMPLE_IDLE: begin
          if (start_fire) begin
            error_reason_o <= UPSAMPLE_ERROR_NONE;
            bytes_read_o <= '0;
            bytes_written_o <= '0;
            if (!start_command_valid) begin
              error_reason_o <= UPSAMPLE_ERROR_COMMAND;
              state_q <= UPSAMPLE_ERROR;
            end else if (!start_descriptor_valid) begin
              error_reason_o <= UPSAMPLE_ERROR_DESCRIPTOR;
              state_q <= UPSAMPLE_ERROR;
            end else if (!start_geometry_valid) begin
              error_reason_o <= UPSAMPLE_ERROR_GEOMETRY;
              state_q <= UPSAMPLE_ERROR;
            end else if (!start_alignment_valid) begin
              error_reason_o <= UPSAMPLE_ERROR_ALIGNMENT;
              state_q <= UPSAMPLE_ERROR;
            end else if (!start_address_valid) begin
              error_reason_o <= UPSAMPLE_ERROR_BOUNDS;
              state_q <= UPSAMPLE_ERROR;
            end else begin
              source_desc_q <= start_source_desc;
              destination_desc_q <= start_destination_desc;
              source_base_q <= start_source_base[63:0];
              destination_base_q <= start_destination_base[63:0];
              source_row_address_q <= start_source_base[63:0];
              destination_row_address_q <= start_destination_base[63:0];
              pixel_beats_q <= start_pixel_bytes[9:3];
              pixel_bytes_q <= start_pixel_bytes;
              source_row_q <= '0;
              state_q <= UPSAMPLE_SETUP_ROW;
            end
          end
        end

        UPSAMPLE_SETUP_ROW: begin
          if (mul_done) begin
            if (mul_result[63:0] == 0 || mul_result[63:0] > 1024
                || mul_result[2:0] != 0
                || source_desc_q.stride_h != mul_result[31:0]
                || destination_desc_q.stride_h
                   != (mul_result[31:0] << 1)) begin
              error_reason_o <= UPSAMPLE_ERROR_DESCRIPTOR;
              state_q <= UPSAMPLE_ERROR;
            end else begin
              source_row_bytes_q <= mul_result[31:0];
              destination_row_bytes_q <= mul_result[31:0] << 1;
              source_row_beats_q <= mul_result[11:3];
              destination_row_beats_q <= mul_result[10:2];
              read_beat_q <= '0;
              state_q <= UPSAMPLE_PLAN_READ;
            end
          end
        end

        // Compute and register each AXI burst before asserting ARVALID.  This
        // cuts the 4 KiB boundary/minimum calculation out of the address
        // handshake and arbiter-control paths.
        UPSAMPLE_PLAN_READ: begin
          if (source_row_address_q - source_base_q + source_row_bytes_q
              > source_desc_q.allocation_bytes) begin
            error_reason_o <= UPSAMPLE_ERROR_BOUNDS;
            state_q <= UPSAMPLE_ERROR;
          end else begin
            read_burst_address_q <= next_read_burst_address;
            read_burst_beats_q <= planned_read_burst_beats;
            state_q <= UPSAMPLE_READ_ADDRESS;
          end
        end

        UPSAMPLE_READ_ADDRESS: begin
          if (m_axi_arready_i) begin
            read_burst_beat_q <= '0;
            read_error_q <= 1'b0;
            state_q <= UPSAMPLE_READ_DATA;
          end
        end

        UPSAMPLE_READ_DATA: begin
          if (read_fire) begin
            bytes_read_o <= bytes_read_o + 8;
            if (m_axi_rresp_i != 2'b00)
              read_error_q <= 1'b1;
            if (m_axi_rlast_i != expected_read_last) begin
              error_reason_o <= UPSAMPLE_ERROR_PROTOCOL;
              state_q <= UPSAMPLE_ERROR;
            end else if (expected_read_last) begin
              if (read_error_q || m_axi_rresp_i != 2'b00) begin
                error_reason_o <= UPSAMPLE_ERROR_READ;
                state_q <= UPSAMPLE_ERROR;
              end else if (read_beat_q + 1 != source_row_beats_q) begin
                read_beat_q <= read_beat_q + 1'b1;
                state_q <= UPSAMPLE_PLAN_READ;
              end else begin
                output_row_copy_q <= 1'b0;
                write_beat_q <= '0;
                buffer_index_q <= '0;
                beat_in_pixel_q <= '0;
                horizontal_copy_q <= 1'b0;
                state_q <= UPSAMPLE_PLAN_WRITE;
              end
            end else begin
              read_beat_q <= read_beat_q + 1'b1;
              read_burst_beat_q <= read_burst_beat_q + 1'b1;
            end
          end
        end

        // Likewise, register AWADDR/AWLEN before the write arbiter observes
        // AWVALID.  Besides improving timing, this keeps all AXI address
        // fields stable independently of downstream back-pressure.
        UPSAMPLE_PLAN_WRITE: begin
          if (destination_row_address_q - destination_base_q
              + destination_row_bytes_q
              > destination_desc_q.allocation_bytes) begin
            error_reason_o <= UPSAMPLE_ERROR_BOUNDS;
            state_q <= UPSAMPLE_ERROR;
          end else begin
            write_burst_address_q <= next_write_burst_address;
            write_burst_beats_q <= planned_write_burst_beats;
            state_q <= UPSAMPLE_WRITE_ADDRESS;
          end
        end

        UPSAMPLE_WRITE_ADDRESS: begin
          if (m_axi_awready_i) begin
            write_burst_beat_q <= '0;
            state_q <= UPSAMPLE_WRITE_PREFETCH;
          end
        end

        UPSAMPLE_WRITE_PREFETCH: state_q <= UPSAMPLE_WRITE_DATA;

        UPSAMPLE_WRITE_DATA: begin
          if (write_fire) begin
            bytes_written_o <= bytes_written_o + 8;
            if (m_axi_wlast_o != expected_write_last) begin
              error_reason_o <= UPSAMPLE_ERROR_PROTOCOL;
              state_q <= UPSAMPLE_ERROR;
            end else if (m_axi_wlast_o) begin
              if (write_beat_q + 1 != destination_row_beats_q) begin
                write_beat_q <= write_beat_q + 1'b1;
                buffer_index_q <= next_buffer_index;
                beat_in_pixel_q <= next_beat_in_pixel;
                horizontal_copy_q <= next_horizontal_copy;
              end
              state_q <= UPSAMPLE_WRITE_RESPONSE;
            end else begin
              write_beat_q <= write_beat_q + 1'b1;
              write_burst_beat_q <= write_burst_beat_q + 1'b1;
              buffer_index_q <= next_buffer_index;
              beat_in_pixel_q <= next_beat_in_pixel;
              horizontal_copy_q <= next_horizontal_copy;
            end
          end
        end

        UPSAMPLE_WRITE_RESPONSE: begin
          if (m_axi_bvalid_i) begin
            if (m_axi_bresp_i != 2'b00) begin
              error_reason_o <= UPSAMPLE_ERROR_WRITE;
              state_q <= UPSAMPLE_ERROR;
            end else if (write_beat_q + 1 != destination_row_beats_q) begin
              state_q <= UPSAMPLE_PLAN_WRITE;
            end else if (!output_row_copy_q) begin
              output_row_copy_q <= 1'b1;
              destination_row_address_q
                <= destination_row_address_q + destination_desc_q.stride_h;
              write_beat_q <= '0;
              buffer_index_q <= '0;
              beat_in_pixel_q <= '0;
              horizontal_copy_q <= 1'b0;
              state_q <= UPSAMPLE_PLAN_WRITE;
            end else if (source_row_q + 1 == source_desc_q.shape_h) begin
              state_q <= UPSAMPLE_DONE;
            end else begin
              source_row_q <= source_row_q + 1'b1;
              source_row_address_q
                <= source_row_address_q + source_desc_q.stride_h;
              destination_row_address_q
                <= destination_row_address_q + destination_desc_q.stride_h;
              read_beat_q <= '0;
              state_q <= UPSAMPLE_PLAN_READ;
            end
          end
        end

        UPSAMPLE_DONE: state_q <= UPSAMPLE_IDLE;
        UPSAMPLE_ERROR: state_q <= UPSAMPLE_IDLE;
        default: begin
          error_reason_o <= UPSAMPLE_ERROR_PROTOCOL;
          state_q <= UPSAMPLE_ERROR;
        end
      endcase
    end
  end

endmodule
