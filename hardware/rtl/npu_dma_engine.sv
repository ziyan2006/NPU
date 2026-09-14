`timescale 1ns/1ps

module npu_dma_engine (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         request_valid_i,
  output logic         request_ready_o,
  input  logic [npu_dma_pkg::NPU_DMA_REQUEST_BITS-1:0] request_bits_i,

  output logic         scratchpad_write_valid_o,
  input  logic         scratchpad_write_ready_i,
  output logic [1:0]   scratchpad_write_kind_o,
  output logic         scratchpad_write_bank_o,
  output logic [31:0]  scratchpad_write_address_o,
  output logic [63:0]  scratchpad_write_data_o,
  output logic [7:0]   scratchpad_write_strobe_o,

  output logic         scratchpad_read_request_valid_o,
  input  logic         scratchpad_read_request_ready_i,
  output logic [1:0]   scratchpad_read_kind_o,
  output logic         scratchpad_read_bank_o,
  output logic [31:0]  scratchpad_read_address_o,
  input  logic         scratchpad_read_response_valid_i,
  output logic         scratchpad_read_response_ready_o,
  input  logic [63:0]  scratchpad_read_response_data_i,

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
  output logic         m_axi_rready_o,

  output logic [63:0]  m_axi_awaddr_o,
  output logic [7:0]   m_axi_awlen_o,
  output logic [2:0]   m_axi_awsize_o,
  output logic [1:0]   m_axi_awburst_o,
  output logic         m_axi_awvalid_o,
  input  logic         m_axi_awready_i,
  output logic [63:0]  m_axi_wdata_o,
  output logic [7:0]   m_axi_wstrb_o,
  output logic         m_axi_wlast_o,
  output logic         m_axi_wvalid_o,
  input  logic         m_axi_wready_i,
  input  logic [1:0]   m_axi_bresp_i,
  input  logic         m_axi_bvalid_i,
  output logic         m_axi_bready_o,

  output logic         busy_o,
  output logic         done_pulse_o,
  output logic [7:0]   event_set_o,
  output logic         error_pulse_o,
  output logic [3:0]   error_reason_o,
  output logic [15:0]  error_tag_o,
  output logic [31:0]  bytes_read_o,
  output logic [31:0]  bytes_written_o
);
  import npu_dma_pkg::*;

  typedef enum logic [3:0] {
    DE_IDLE,
    DE_CLEAR,
    DE_PLAN_SIZE,
    DE_PLAN_BURST,
    DE_READ_ADDRESS,
    DE_READ_DATA,
    DE_WRITE_ADDRESS,
    DE_WRITE_SP_REQUEST,
    DE_WRITE_SP_RESPONSE,
    DE_WRITE_DATA,
    DE_WRITE_RESPONSE,
    DE_DONE,
    DE_ERROR
  } dma_engine_state_e;

  dma_engine_state_e state_q;
  npu_dma_request_t incoming_request;
  npu_dma_request_t request_q;

  logic [15:0] y_index_q;
  logic [15:0] z_index_q;
  logic [63:0] external_plane_q;
  logic [63:0] external_row_q;
  logic [63:0] external_cursor_q;
  logic [31:0] scratchpad_plane_q;
  logic [31:0] scratchpad_row_q;
  logic [31:0] scratchpad_cursor_q;
  logic [31:0] row_bytes_remaining_q;
  logic [31:0] clear_cursor_q;
  logic [31:0] clear_remaining_q;

  logic [2:0] beat_size_q;
  logic [3:0] beat_bytes_q;
  logic [8:0] beats_remaining_q;
  logic burst_finishes_row_q;
  logic [2:0] planned_beat_size_q;
  logic [3:0] planned_beat_bytes_q;
  logic [31:0] planned_beats_by_row_q;
  logic [31:0] planned_row_bytes_q;
  logic [31:0] planned_boundary_bytes_q;
  logic [8:0] planned_burst_beats_q;
  logic planned_finishes_row_q;
  logic read_error_q;
  logic read_drain_q;
  logic [3:0] pending_error_q;
  logic [63:0] write_data_q;

  logic [2:0] next_beat_size;
  logic [3:0] next_beat_bytes;
  logic [31:0] beats_by_row;
  logic [31:0] boundary_bytes;
  logic [31:0] planned_beats_by_boundary;
  logic [31:0] selected_burst_beats;
  logic selected_finishes_row;
  logic [7:0] byte_strobe;
  logic request_format_valid;
  logic request_alignment_valid;
  logic read_beat_accepted;
  logic write_beat_accepted;
  logic clear_beat_accepted;
  logic expected_read_last;

  assign incoming_request = npu_dma_request_t'(request_bits_i);
  assign request_ready_o = state_q == DE_IDLE;
  assign busy_o = state_q != DE_IDLE;
  assign done_pulse_o = state_q == DE_DONE;
  assign event_set_o = state_q == DE_DONE ? request_q.event_mask : 8'h00;
  assign error_pulse_o = state_q == DE_ERROR;
  assign error_reason_o = pending_error_q;
  assign error_tag_o = request_q.tag;

  assign scratchpad_write_kind_o = request_q.scratchpad;
  assign scratchpad_write_bank_o = request_q.bank;
  assign scratchpad_read_kind_o = request_q.scratchpad;
  assign scratchpad_read_bank_o = request_q.bank;

  // This v1 engine accepts aligned row bases.  Narrow tail transfers remain
  // legal because both cursors advance together and use matching byte lanes.
  always @* begin
    request_format_valid = incoming_request.x_bytes != 0
      && incoming_request.y_count != 0
      && incoming_request.z_count != 0
      && incoming_request.scratchpad != 2'b11
      && (!incoming_request.store || !incoming_request.clear_before);
    request_alignment_valid = incoming_request.external_address[2:0] == 0
      && incoming_request.scratchpad_offset[2:0] == 0
      && (incoming_request.y_count == 1
          || (incoming_request.external_y_stride[2:0] == 0
              && incoming_request.scratchpad_y_stride[2:0] == 0))
      && (incoming_request.z_count == 1
          || (incoming_request.external_z_stride[2:0] == 0
              && incoming_request.scratchpad_z_stride[2:0] == 0))
      && (!incoming_request.clear_before
          || incoming_request.clear_bytes[2:0] == 0);
  end

  // Select the largest exact AXI transfer size.  Rows use 8-byte bursts and
  // at most three smaller one-beat tails (4/2/1 B), so DDR is never over-read.
  always @* begin
    if (row_bytes_remaining_q >= 8) begin
      next_beat_size = 3'd3;
      next_beat_bytes = 4'd8;
      beats_by_row = row_bytes_remaining_q >> 3;
    end else if (row_bytes_remaining_q >= 4) begin
      next_beat_size = 3'd2;
      next_beat_bytes = 4'd4;
      beats_by_row = 1;
    end else if (row_bytes_remaining_q >= 2) begin
      next_beat_size = 3'd1;
      next_beat_bytes = 4'd2;
      beats_by_row = 1;
    end else begin
      next_beat_size = 3'd0;
      next_beat_bytes = 4'd1;
      beats_by_row = 1;
    end

    boundary_bytes = 32'd4096 - {20'd0, external_cursor_q[11:0]};
  end

  always @* begin
    planned_beats_by_boundary =
      planned_boundary_bytes_q >> planned_beat_size_q;
    selected_burst_beats = planned_beats_by_row_q;
    if (selected_burst_beats > planned_beats_by_boundary)
      selected_burst_beats = planned_beats_by_boundary;
    if (selected_burst_beats > 256)
      selected_burst_beats = 256;
    selected_finishes_row =
      planned_beats_by_row_q <= planned_beats_by_boundary
      && planned_beats_by_row_q <= 256
      && planned_row_bytes_q
         == (planned_beats_by_row_q << planned_beat_size_q);
  end

  always @* begin
    case (beat_bytes_q)
      4'd8: byte_strobe = 8'hff;
      4'd4: byte_strobe = 8'h0f << external_cursor_q[2:0];
      4'd2: byte_strobe = 8'h03 << external_cursor_q[2:0];
      default: byte_strobe = 8'h01 << external_cursor_q[2:0];
    endcase
  end

  assign m_axi_araddr_o = external_cursor_q;
  assign m_axi_arlen_o = planned_burst_beats_q[7:0] - 1'b1;
  assign m_axi_arsize_o = planned_beat_size_q;
  assign m_axi_arburst_o = 2'b01;
  assign m_axi_arvalid_o = state_q == DE_READ_ADDRESS;

  assign scratchpad_write_valid_o =
      (state_q == DE_CLEAR)
      || (state_q == DE_READ_DATA && m_axi_rvalid_i
          && m_axi_rresp_i == 2'b00 && !read_error_q && !read_drain_q);
  assign scratchpad_write_address_o = state_q == DE_CLEAR
    ? {clear_cursor_q[31:3], 3'b000}
    : {scratchpad_cursor_q[31:3], 3'b000};
  assign scratchpad_write_data_o = state_q == DE_CLEAR ? 64'h0 : m_axi_rdata_i;
  assign scratchpad_write_strobe_o = state_q == DE_CLEAR
    ? (clear_remaining_q >= 8 ? 8'hff
       : ((8'h01 << clear_remaining_q) - 1'b1))
    : byte_strobe;
  assign m_axi_rready_o = state_q == DE_READ_DATA
    && ((m_axi_rresp_i != 2'b00 || read_error_q || read_drain_q)
        ? 1'b1 : scratchpad_write_ready_i);

  assign m_axi_awaddr_o = external_cursor_q;
  assign m_axi_awlen_o = planned_burst_beats_q[7:0] - 1'b1;
  assign m_axi_awsize_o = planned_beat_size_q;
  assign m_axi_awburst_o = 2'b01;
  assign m_axi_awvalid_o = state_q == DE_WRITE_ADDRESS;
  assign scratchpad_read_request_valid_o = state_q == DE_WRITE_SP_REQUEST;
  assign scratchpad_read_address_o = {scratchpad_cursor_q[31:3], 3'b000};
  assign scratchpad_read_response_ready_o = state_q == DE_WRITE_SP_RESPONSE;
  assign m_axi_wdata_o = write_data_q;
  assign m_axi_wstrb_o = byte_strobe;
  assign m_axi_wlast_o = beats_remaining_q == 1;
  assign m_axi_wvalid_o = state_q == DE_WRITE_DATA;
  assign m_axi_bready_o = state_q == DE_WRITE_RESPONSE;

  assign read_beat_accepted = m_axi_rvalid_i && m_axi_rready_o;
  assign write_beat_accepted = m_axi_wvalid_o && m_axi_wready_i;
  assign clear_beat_accepted = scratchpad_write_valid_o
    && scratchpad_write_ready_i && state_q == DE_CLEAR;
  assign expected_read_last = beats_remaining_q == 1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= DE_IDLE;
      request_q <= '0;
      y_index_q <= '0;
      z_index_q <= '0;
      external_plane_q <= '0;
      external_row_q <= '0;
      external_cursor_q <= '0;
      scratchpad_plane_q <= '0;
      scratchpad_row_q <= '0;
      scratchpad_cursor_q <= '0;
      row_bytes_remaining_q <= '0;
      clear_cursor_q <= '0;
      clear_remaining_q <= '0;
      beat_size_q <= '0;
      beat_bytes_q <= '0;
      beats_remaining_q <= '0;
      burst_finishes_row_q <= 1'b0;
      planned_beat_size_q <= '0;
      planned_beat_bytes_q <= '0;
      planned_beats_by_row_q <= '0;
      planned_row_bytes_q <= '0;
      planned_boundary_bytes_q <= '0;
      planned_burst_beats_q <= '0;
      planned_finishes_row_q <= 1'b0;
      read_error_q <= 1'b0;
      read_drain_q <= 1'b0;
      pending_error_q <= NPU_DMA_EXEC_OK;
      write_data_q <= '0;
      bytes_read_o <= '0;
      bytes_written_o <= '0;
    end else if (soft_reset_i && state_q == DE_IDLE) begin
      // A soft reset never abandons an accepted request.  If asserted while
      // busy it is ignored here; the outer reset controller keeps new commands
      // stopped, waits for idle, then asserts/holds soft reset again.
      state_q <= DE_IDLE;
      pending_error_q <= NPU_DMA_EXEC_OK;
      read_error_q <= 1'b0;
      read_drain_q <= 1'b0;
      bytes_read_o <= '0;
      bytes_written_o <= '0;
    end else begin
      case (state_q)
        DE_IDLE: begin
          if (request_valid_i) begin
            request_q <= incoming_request;
            pending_error_q <= NPU_DMA_EXEC_OK;
            bytes_read_o <= '0;
            bytes_written_o <= '0;
            if (!request_format_valid) begin
              pending_error_q <= NPU_DMA_EXEC_BAD_REQUEST;
              state_q <= DE_ERROR;
            end else if (!request_alignment_valid) begin
              pending_error_q <= NPU_DMA_EXEC_UNSUPPORTED_ALIGNMENT;
              state_q <= DE_ERROR;
            end else begin
              y_index_q <= 0;
              z_index_q <= 0;
              external_plane_q <= incoming_request.external_address;
              external_row_q <= incoming_request.external_address;
              external_cursor_q <= incoming_request.external_address;
              scratchpad_plane_q <= incoming_request.scratchpad_offset;
              scratchpad_row_q <= incoming_request.scratchpad_offset;
              scratchpad_cursor_q <= incoming_request.scratchpad_offset;
              row_bytes_remaining_q <= incoming_request.x_bytes;
              clear_cursor_q <= 0;
              clear_remaining_q <= incoming_request.clear_bytes;
              if (incoming_request.clear_before
                  && incoming_request.clear_bytes != 0)
                state_q <= DE_CLEAR;
              else
                state_q <= DE_PLAN_SIZE;
            end
          end
        end

        DE_CLEAR: begin
          if (clear_beat_accepted) begin
            if (clear_remaining_q <= 8) begin
              clear_cursor_q <= 0;
              clear_remaining_q <= 0;
              state_q <= DE_PLAN_SIZE;
            end else begin
              clear_cursor_q <= clear_cursor_q + 8;
              clear_remaining_q <= clear_remaining_q - 8;
            end
          end
        end

        DE_PLAN_SIZE: begin
          planned_beat_size_q <= next_beat_size;
          planned_beat_bytes_q <= next_beat_bytes;
          planned_beats_by_row_q <= beats_by_row;
          planned_row_bytes_q <= row_bytes_remaining_q;
          planned_boundary_bytes_q <= boundary_bytes;
          state_q <= DE_PLAN_BURST;
        end

        DE_PLAN_BURST: begin
          planned_burst_beats_q <= selected_burst_beats[8:0];
          planned_finishes_row_q <= selected_finishes_row;
          state_q <= request_q.store ? DE_WRITE_ADDRESS : DE_READ_ADDRESS;
        end

        DE_READ_ADDRESS: begin
          if (m_axi_arready_i) begin
            beat_size_q <= planned_beat_size_q;
            beat_bytes_q <= planned_beat_bytes_q;
            beats_remaining_q <= planned_burst_beats_q;
            burst_finishes_row_q <= planned_finishes_row_q;
            read_error_q <= 1'b0;
            read_drain_q <= 1'b0;
            state_q <= DE_READ_DATA;
          end
        end

        DE_READ_DATA: begin
          if (read_beat_accepted) begin
            external_cursor_q <= external_cursor_q + beat_bytes_q;
            scratchpad_cursor_q <= scratchpad_cursor_q + beat_bytes_q;
            row_bytes_remaining_q <= row_bytes_remaining_q - beat_bytes_q;
            bytes_read_o <= bytes_read_o + beat_bytes_q;
            if (m_axi_rresp_i != 2'b00) begin
              read_error_q <= 1'b1;
              pending_error_q <= NPU_DMA_EXEC_AXI_READ;
            end
            if (m_axi_rlast_i != expected_read_last) begin
              read_error_q <= 1'b1;
              pending_error_q <= NPU_DMA_EXEC_AXI_PROTOCOL;
              if (!m_axi_rlast_i)
                read_drain_q <= 1'b1;
            end

            if (m_axi_rlast_i) begin
              if (read_error_q || m_axi_rresp_i != 2'b00
                  || m_axi_rlast_i != expected_read_last) begin
                state_q <= DE_ERROR;
              end else if (burst_finishes_row_q) begin
                if (y_index_q + 1 < request_q.y_count) begin
                  y_index_q <= y_index_q + 1'b1;
                  external_row_q <= external_row_q
                    + request_q.external_y_stride;
                  external_cursor_q <= external_row_q
                    + request_q.external_y_stride;
                  scratchpad_row_q <= scratchpad_row_q
                    + request_q.scratchpad_y_stride;
                  scratchpad_cursor_q <= scratchpad_row_q
                    + request_q.scratchpad_y_stride;
                  row_bytes_remaining_q <= request_q.x_bytes;
                  state_q <= DE_PLAN_SIZE;
                end else if (z_index_q + 1 < request_q.z_count) begin
                  z_index_q <= z_index_q + 1'b1;
                  y_index_q <= 0;
                  external_plane_q <= external_plane_q
                    + request_q.external_z_stride;
                  external_row_q <= external_plane_q
                    + request_q.external_z_stride;
                  external_cursor_q <= external_plane_q
                    + request_q.external_z_stride;
                  scratchpad_plane_q <= scratchpad_plane_q
                    + request_q.scratchpad_z_stride;
                  scratchpad_row_q <= scratchpad_plane_q
                    + request_q.scratchpad_z_stride;
                  scratchpad_cursor_q <= scratchpad_plane_q
                    + request_q.scratchpad_z_stride;
                  row_bytes_remaining_q <= request_q.x_bytes;
                  state_q <= DE_PLAN_SIZE;
                end else begin
                  state_q <= DE_DONE;
                end
              end else begin
                state_q <= DE_PLAN_SIZE;
              end
            end else if (!read_drain_q) begin
              beats_remaining_q <= beats_remaining_q - 1'b1;
            end
          end
        end

        DE_WRITE_ADDRESS: begin
          if (m_axi_awready_i) begin
            beat_size_q <= planned_beat_size_q;
            beat_bytes_q <= planned_beat_bytes_q;
            beats_remaining_q <= planned_burst_beats_q;
            burst_finishes_row_q <= planned_finishes_row_q;
            state_q <= DE_WRITE_SP_REQUEST;
          end
        end

        DE_WRITE_SP_REQUEST: begin
          if (scratchpad_read_request_ready_i)
            state_q <= DE_WRITE_SP_RESPONSE;
        end

        DE_WRITE_SP_RESPONSE: begin
          if (scratchpad_read_response_valid_i) begin
            write_data_q <= scratchpad_read_response_data_i;
            state_q <= DE_WRITE_DATA;
          end
        end

        DE_WRITE_DATA: begin
          if (write_beat_accepted) begin
            external_cursor_q <= external_cursor_q + beat_bytes_q;
            scratchpad_cursor_q <= scratchpad_cursor_q + beat_bytes_q;
            row_bytes_remaining_q <= row_bytes_remaining_q - beat_bytes_q;
            bytes_written_o <= bytes_written_o + beat_bytes_q;
            beats_remaining_q <= beats_remaining_q - 1'b1;
            if (beats_remaining_q == 1)
              state_q <= DE_WRITE_RESPONSE;
            else
              state_q <= DE_WRITE_SP_REQUEST;
          end
        end

        DE_WRITE_RESPONSE: begin
          if (m_axi_bvalid_i) begin
            if (m_axi_bresp_i != 2'b00) begin
              pending_error_q <= NPU_DMA_EXEC_AXI_WRITE;
              state_q <= DE_ERROR;
            end else if (burst_finishes_row_q) begin
              if (y_index_q + 1 < request_q.y_count) begin
                y_index_q <= y_index_q + 1'b1;
                external_row_q <= external_row_q
                  + request_q.external_y_stride;
                external_cursor_q <= external_row_q
                  + request_q.external_y_stride;
                scratchpad_row_q <= scratchpad_row_q
                  + request_q.scratchpad_y_stride;
                scratchpad_cursor_q <= scratchpad_row_q
                  + request_q.scratchpad_y_stride;
                row_bytes_remaining_q <= request_q.x_bytes;
                state_q <= DE_PLAN_SIZE;
              end else if (z_index_q + 1 < request_q.z_count) begin
                z_index_q <= z_index_q + 1'b1;
                y_index_q <= 0;
                external_plane_q <= external_plane_q
                  + request_q.external_z_stride;
                external_row_q <= external_plane_q
                  + request_q.external_z_stride;
                external_cursor_q <= external_plane_q
                  + request_q.external_z_stride;
                scratchpad_plane_q <= scratchpad_plane_q
                  + request_q.scratchpad_z_stride;
                scratchpad_row_q <= scratchpad_plane_q
                  + request_q.scratchpad_z_stride;
                scratchpad_cursor_q <= scratchpad_plane_q
                  + request_q.scratchpad_z_stride;
                row_bytes_remaining_q <= request_q.x_bytes;
                state_q <= DE_PLAN_SIZE;
              end else begin
                state_q <= DE_DONE;
              end
            end else begin
              state_q <= DE_PLAN_SIZE;
            end
          end
        end

        DE_DONE: state_q <= DE_IDLE;
        DE_ERROR: state_q <= DE_IDLE;
        default: begin
          pending_error_q <= NPU_DMA_EXEC_BAD_REQUEST;
          state_q <= DE_ERROR;
        end
      endcase
    end
  end

endmodule
