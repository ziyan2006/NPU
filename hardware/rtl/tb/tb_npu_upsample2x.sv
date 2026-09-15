`timescale 1ns/1ps

module tb_npu_upsample2x;
  import npu_isa_pkg::*;

  localparam integer MAX_SOURCE_BEATS = 8192;
  localparam integer MAX_DESTINATION_BEATS = 32768;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic start_valid_i = 1'b0;
  logic start_ready_o;
  logic [127:0] command_bits_i;
  logic [511:0] source_desc_bits_i;
  logic [511:0] destination_desc_bits_i;
  logic [63:0] activation_base_i;
  logic [63:0] m_axi_araddr_o;
  logic [7:0] m_axi_arlen_o;
  logic [2:0] m_axi_arsize_o;
  logic [1:0] m_axi_arburst_o;
  logic m_axi_arvalid_o;
  logic m_axi_arready_i;
  logic [63:0] m_axi_rdata_i;
  logic [1:0] m_axi_rresp_i;
  logic m_axi_rlast_i;
  logic m_axi_rvalid_i;
  logic m_axi_rready_o;
  logic [63:0] m_axi_awaddr_o;
  logic [7:0] m_axi_awlen_o;
  logic [2:0] m_axi_awsize_o;
  logic [1:0] m_axi_awburst_o;
  logic m_axi_awvalid_o;
  logic m_axi_awready_i;
  logic [63:0] m_axi_wdata_o;
  logic [7:0] m_axi_wstrb_o;
  logic m_axi_wlast_o;
  logic m_axi_wvalid_o;
  logic m_axi_wready_i;
  logic [1:0] m_axi_bresp_i = 2'b00;
  logic m_axi_bvalid_i = 1'b0;
  logic m_axi_bready_o;
  logic busy_o;
  logic done_pulse_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;
  logic [31:0] bytes_read_o;
  logic [31:0] bytes_written_o;

  logic [127:0] command_memory [0:0];
  logic [511:0] source_desc_memory [0:0];
  logic [511:0] destination_desc_memory [0:0];
  logic [63:0] source_memory [0:MAX_SOURCE_BEATS-1];
  logic [63:0] expected_memory [0:MAX_DESTINATION_BEATS-1];
  npu_tensor_desc_t source_desc;
  npu_tensor_desc_t destination_desc;
  logic [31:0] lfsr_q = 32'h758b_31d4;
  logic read_active_q = 1'b0;
  logic write_active_q = 1'b0;
  integer read_row_q = 0;
  integer read_beat_q = 0;
  integer read_burst_beat_q = 0;
  integer read_burst_beats_q = 0;
  integer write_row_q = 0;
  integer write_beat_q = 0;
  integer write_burst_beat_q = 0;
  integer write_burst_beats_q = 0;
  logic write_burst_completed_row_q = 1'b0;
  integer source_row_beats;
  integer destination_row_beats;
  integer source_total_beats;
  integer destination_total_beats;
  integer b_delay_q = 0;
  logic b_pending_q = 1'b0;
  integer cycles_q = 0;
  integer expected_burst_beats;
  integer boundary_beats;
  string command_hex;
  string source_desc_hex;
  string destination_desc_hex;
  string source_hex;
  string expected_hex;

  always #5 clk_i = ~clk_i;

  assign source_desc = npu_tensor_desc_t'(source_desc_bits_i);
  assign destination_desc = npu_tensor_desc_t'(destination_desc_bits_i);
  assign m_axi_arready_i = !read_active_q && lfsr_q[0];
  assign m_axi_rvalid_i = read_active_q && lfsr_q[1];
  assign m_axi_rdata_i
    = source_memory[read_row_q * source_row_beats + read_beat_q];
  assign m_axi_rresp_i = 2'b00;
  assign m_axi_rlast_i = read_active_q
    && read_burst_beat_q + 1 == read_burst_beats_q;
  assign m_axi_awready_i = !write_active_q && !m_axi_bvalid_i && lfsr_q[2];
  assign m_axi_wready_i = write_active_q && lfsr_q[3];

  npu_upsample2x dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_valid_i(start_valid_i), .start_ready_o(start_ready_o),
    .command_bits_i(command_bits_i), .source_desc_bits_i(source_desc_bits_i),
    .destination_desc_bits_i(destination_desc_bits_i),
    .activation_base_i(activation_base_i), .m_axi_araddr_o(m_axi_araddr_o),
    .m_axi_arlen_o(m_axi_arlen_o), .m_axi_arsize_o(m_axi_arsize_o),
    .m_axi_arburst_o(m_axi_arburst_o), .m_axi_arvalid_o(m_axi_arvalid_o),
    .m_axi_arready_i(m_axi_arready_i), .m_axi_rdata_i(m_axi_rdata_i),
    .m_axi_rresp_i(m_axi_rresp_i), .m_axi_rlast_i(m_axi_rlast_i),
    .m_axi_rvalid_i(m_axi_rvalid_i), .m_axi_rready_o(m_axi_rready_o),
    .m_axi_awaddr_o(m_axi_awaddr_o), .m_axi_awlen_o(m_axi_awlen_o),
    .m_axi_awsize_o(m_axi_awsize_o), .m_axi_awburst_o(m_axi_awburst_o),
    .m_axi_awvalid_o(m_axi_awvalid_o), .m_axi_awready_i(m_axi_awready_i),
    .m_axi_wdata_o(m_axi_wdata_o), .m_axi_wstrb_o(m_axi_wstrb_o),
    .m_axi_wlast_o(m_axi_wlast_o), .m_axi_wvalid_o(m_axi_wvalid_o),
    .m_axi_wready_i(m_axi_wready_i), .m_axi_bresp_i(m_axi_bresp_i),
    .m_axi_bvalid_i(m_axi_bvalid_i), .m_axi_bready_o(m_axi_bready_o),
    .busy_o(busy_o), .done_pulse_o(done_pulse_o),
    .error_pulse_o(error_pulse_o), .error_reason_o(error_reason_o),
    .bytes_read_o(bytes_read_o), .bytes_written_o(bytes_written_o));

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      lfsr_q <= 32'h758b_31d4;
      read_active_q <= 1'b0;
      write_active_q <= 1'b0;
      read_row_q <= 0;
      read_beat_q <= 0;
      read_burst_beat_q <= 0;
      read_burst_beats_q <= 0;
      write_row_q <= 0;
      write_beat_q <= 0;
      write_burst_beat_q <= 0;
      write_burst_beats_q <= 0;
      write_burst_completed_row_q <= 1'b0;
      m_axi_bvalid_i <= 1'b0;
      b_pending_q <= 1'b0;
      b_delay_q <= 0;
      cycles_q <= 0;
    end else begin
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      cycles_q <= cycles_q + 1;
      if (cycles_q > 500000)
        $fatal(1, "UPSAMPLE2X timeout");

      if (m_axi_arvalid_o && m_axi_arready_i) begin
        if (m_axi_araddr_o != activation_base_i + source_desc.base_offset
            + read_row_q * source_desc.stride_h + read_beat_q * 8)
          $fatal(1, "AR address mismatch at row %0d", read_row_q);
        boundary_beats = (4096 - (m_axi_araddr_o & 4095)) / 8;
        expected_burst_beats = source_row_beats - read_beat_q;
        if (expected_burst_beats > boundary_beats)
          expected_burst_beats = boundary_beats;
        if (expected_burst_beats > 256)
          expected_burst_beats = 256;
        if (m_axi_arlen_o != expected_burst_beats - 1
            || m_axi_arsize_o != 3 || m_axi_arburst_o != 2'b01)
          $fatal(1, "AR attributes mismatch");
        read_active_q <= 1'b1;
        read_burst_beat_q <= 0;
        read_burst_beats_q <= expected_burst_beats;
      end
      if (m_axi_rvalid_i && m_axi_rready_o) begin
        if (m_axi_rlast_i && read_beat_q + 1 == source_row_beats) begin
          read_active_q <= 1'b0;
          read_beat_q <= 0;
          read_burst_beat_q <= 0;
          read_row_q <= read_row_q + 1;
        end else if (m_axi_rlast_i) begin
          read_active_q <= 1'b0;
          read_beat_q <= read_beat_q + 1;
          read_burst_beat_q <= 0;
        end else begin
          read_beat_q <= read_beat_q + 1;
          read_burst_beat_q <= read_burst_beat_q + 1;
        end
      end

      if (m_axi_awvalid_o && m_axi_awready_i) begin
        if (m_axi_awaddr_o != activation_base_i
            + destination_desc.base_offset
            + write_row_q * destination_desc.stride_h + write_beat_q * 8)
          $fatal(1, "AW address mismatch at row %0d", write_row_q);
        boundary_beats = (4096 - (m_axi_awaddr_o & 4095)) / 8;
        expected_burst_beats = destination_row_beats - write_beat_q;
        if (expected_burst_beats > boundary_beats)
          expected_burst_beats = boundary_beats;
        if (expected_burst_beats > 256)
          expected_burst_beats = 256;
        if (m_axi_awlen_o != expected_burst_beats - 1
            || m_axi_awsize_o != 3 || m_axi_awburst_o != 2'b01)
          $fatal(1, "AW attributes mismatch");
        write_active_q <= 1'b1;
        write_burst_beat_q <= 0;
        write_burst_beats_q <= expected_burst_beats;
      end
      if (m_axi_wvalid_o && m_axi_wready_i) begin
        if (!write_active_q)
          $fatal(1, "W without AW");
        if (m_axi_wdata_o !== expected_memory[
              write_row_q * destination_row_beats + write_beat_q])
          $fatal(1, "W data mismatch at row %0d beat %0d",
                 write_row_q, write_beat_q);
        if (m_axi_wstrb_o != 8'hff
            || m_axi_wlast_o != (write_burst_beat_q + 1
                                 == write_burst_beats_q))
          $fatal(1, "W attributes mismatch at row %0d beat %0d",
                 write_row_q, write_beat_q);
        if (m_axi_wlast_o) begin
          write_active_q <= 1'b0;
          write_burst_completed_row_q
            <= write_beat_q + 1 == destination_row_beats;
          if (write_beat_q + 1 != destination_row_beats)
            write_beat_q <= write_beat_q + 1;
          write_burst_beat_q <= 0;
          b_delay_q <= lfsr_q[5:4];
          b_pending_q <= 1'b1;
        end else begin
          write_beat_q <= write_beat_q + 1;
          write_burst_beat_q <= write_burst_beat_q + 1;
        end
      end

      if (b_pending_q && !m_axi_bvalid_i && b_delay_q != 0)
        b_delay_q <= b_delay_q - 1;
      if (b_pending_q && !m_axi_bvalid_i && b_delay_q == 0)
        m_axi_bvalid_i <= 1'b1;
      if (m_axi_bvalid_i && m_axi_bready_o) begin
        m_axi_bvalid_i <= 1'b0;
        b_pending_q <= 1'b0;
        if (write_burst_completed_row_q) begin
          write_row_q <= write_row_q + 1;
          write_beat_q <= 0;
        end
      end

      if (error_pulse_o)
        $fatal(1, "unexpected UPSAMPLE2X error %0d", error_reason_o);
    end
  end

  initial begin
    if (!$value$plusargs("COMMAND_HEX=%s", command_hex)
        || !$value$plusargs("SOURCE_DESC_HEX=%s", source_desc_hex)
        || !$value$plusargs("DESTINATION_DESC_HEX=%s", destination_desc_hex)
        || !$value$plusargs("SOURCE_HEX=%s", source_hex)
        || !$value$plusargs("EXPECTED_HEX=%s", expected_hex)
        || !$value$plusargs("ACTIVATION_BASE=%h", activation_base_i)
        || !$value$plusargs("SOURCE_ROW_BEATS=%d", source_row_beats)
        || !$value$plusargs("DESTINATION_ROW_BEATS=%d",
                            destination_row_beats)
        || !$value$plusargs("SOURCE_TOTAL_BEATS=%d", source_total_beats)
        || !$value$plusargs("DESTINATION_TOTAL_BEATS=%d",
                            destination_total_beats))
      $fatal(1, "missing UPSAMPLE2X plusargs");
    if (source_total_beats <= 0 || source_total_beats > MAX_SOURCE_BEATS
        || destination_total_beats <= 0
        || destination_total_beats > MAX_DESTINATION_BEATS)
      $fatal(1, "invalid UPSAMPLE2X vector count");

    $readmemh(command_hex, command_memory);
    $readmemh(source_desc_hex, source_desc_memory);
    $readmemh(destination_desc_hex, destination_desc_memory);
    $readmemh(source_hex, source_memory, 0, source_total_beats - 1);
    $readmemh(expected_hex, expected_memory, 0,
              destination_total_beats - 1);
    command_bits_i = command_memory[0];
    source_desc_bits_i = source_desc_memory[0];
    destination_desc_bits_i = destination_desc_memory[0];

    repeat (5) @(posedge clk_i);
    rst_ni <= 1'b1;
    do @(posedge clk_i); while (!start_ready_o);
    start_valid_i <= 1'b1;
    @(posedge clk_i);
    start_valid_i <= 1'b0;
    do @(posedge clk_i); while (!done_pulse_o);
    #1;
    if (read_row_q != source_desc.shape_h)
      $fatal(1, "source row count mismatch");
    if (write_row_q != destination_desc.shape_h)
      $fatal(1, "destination row count mismatch");
    if (bytes_read_o != source_total_beats * 8
        || bytes_written_o != destination_total_beats * 8)
      $fatal(1, "byte counters mismatch");
    $display("npu_upsample2x: %0d -> %0d bytes PASS in %0d cycles",
             source_total_beats * 8, destination_total_beats * 8, cycles_q);
    $finish;
  end

endmodule
