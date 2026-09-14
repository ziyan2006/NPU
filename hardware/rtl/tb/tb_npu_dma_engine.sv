`timescale 1ns/1ps

module tb_npu_dma_engine;
  import npu_dma_pkg::*;

  localparam integer MEMORY_BYTES = 16384;
  localparam integer SCRATCHPAD_BYTES = 65536;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic request_valid_i = 1'b0;
  logic request_ready_o;
  logic [NPU_DMA_REQUEST_BITS-1:0] request_bits_i;
  npu_dma_request_t request;

  logic scratchpad_write_valid_o;
  logic scratchpad_write_ready_i;
  logic [1:0] scratchpad_write_kind_o;
  logic scratchpad_write_bank_o;
  logic [31:0] scratchpad_write_address_o;
  logic [63:0] scratchpad_write_data_o;
  logic [7:0] scratchpad_write_strobe_o;
  logic scratchpad_read_request_valid_o;
  logic scratchpad_read_request_ready_i;
  logic [1:0] scratchpad_read_kind_o;
  logic scratchpad_read_bank_o;
  logic [31:0] scratchpad_read_address_o;
  logic scratchpad_read_response_valid_i = 1'b0;
  logic scratchpad_read_response_ready_o;
  logic [63:0] scratchpad_read_response_data_i = '0;

  logic [63:0] m_axi_araddr_o;
  logic [7:0] m_axi_arlen_o;
  logic [2:0] m_axi_arsize_o;
  logic [1:0] m_axi_arburst_o;
  logic m_axi_arvalid_o;
  logic m_axi_arready_i;
  logic [63:0] m_axi_rdata_i = '0;
  logic [1:0] m_axi_rresp_i = '0;
  logic m_axi_rlast_i = 1'b0;
  logic m_axi_rvalid_i = 1'b0;
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
  logic [1:0] m_axi_bresp_i = '0;
  logic m_axi_bvalid_i = 1'b0;
  logic m_axi_bready_o;
  logic busy_o;
  logic done_pulse_o;
  logic [7:0] event_set_o;
  logic error_pulse_o;
  logic [3:0] error_reason_o;
  logic [15:0] error_tag_o;
  logic [31:0] bytes_read_o;
  logic [31:0] bytes_written_o;

  logic [7:0] external_memory [0:MEMORY_BYTES-1];
  logic [7:0] scratchpad_memory [0:SCRATCHPAD_BYTES-1];
  integer cycle_q = 0;
  integer ar_count = 0;
  integer aw_count = 0;
  logic [63:0] ar_address_log [0:63];
  logic [7:0] ar_length_log [0:63];
  logic [2:0] ar_size_log [0:63];
  logic [63:0] aw_address_log [0:63];
  logic [7:0] aw_length_log [0:63];
  logic [2:0] aw_size_log [0:63];

  logic read_active_q = 1'b0;
  logic [63:0] read_address_q = '0;
  logic [8:0] read_beats_q = '0;
  logic [2:0] read_size_q = '0;
  logic [8:0] read_beat_index_q = '0;
  logic inject_read_error = 1'b0;
  logic inject_early_rlast = 1'b0;

  logic write_active_q = 1'b0;
  logic [63:0] write_address_q = '0;
  logic [8:0] write_beats_q = '0;
  logic [2:0] write_size_q = '0;
  logic write_response_pending_q = 1'b0;
  logic inject_write_error = 1'b0;

  integer lane;
  integer index;
  integer row;
  integer start_count;
  integer timeout;
  logic saw_done;
  logic saw_error;
  logic ar_stalled_q = 1'b0;
  logic [76:0] ar_payload_q = '0;
  logic aw_stalled_q = 1'b0;
  logic [76:0] aw_payload_q = '0;
  logic w_stalled_q = 1'b0;
  logic [72:0] w_payload_q = '0;

  assign request_bits_i = request;
  always #5 clk_i = ~clk_i;

  npu_dma_engine dut (.*);

  function automatic logic [63:0] external_word(input logic [63:0] address);
    integer byte_lane;
    integer word_base;
    begin
      external_word = '0;
      word_base = address & 64'hffff_ffff_ffff_fff8;
      for (byte_lane = 0; byte_lane < 8; byte_lane = byte_lane + 1)
        external_word[byte_lane*8 +: 8] = external_memory[word_base + byte_lane];
    end
  endfunction

  function automatic logic [63:0] scratchpad_word(input logic [31:0] address);
    integer byte_lane;
    integer word_base;
    begin
      scratchpad_word = '0;
      word_base = address & 32'hffff_fff8;
      for (byte_lane = 0; byte_lane < 8; byte_lane = byte_lane + 1)
        scratchpad_word[byte_lane*8 +: 8] = scratchpad_memory[word_base + byte_lane];
    end
  endfunction

  // Deterministic stalls exercise ready/valid stability without random seeds.
  always @* begin
    m_axi_arready_i = !read_active_q && !m_axi_rvalid_i && cycle_q[0];
    m_axi_awready_i = !write_active_q && !write_response_pending_q
      && !m_axi_bvalid_i && cycle_q[1];
    m_axi_wready_i = cycle_q[0] || cycle_q[2];
    scratchpad_write_ready_i = cycle_q[0] || !cycle_q[1];
    scratchpad_read_request_ready_i = !scratchpad_read_response_valid_i
      && (cycle_q[0] || cycle_q[1]);
  end

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      cycle_q <= 0;
      ar_count <= 0;
      aw_count <= 0;
      read_active_q <= 1'b0;
      m_axi_rvalid_i <= 1'b0;
      write_active_q <= 1'b0;
      write_response_pending_q <= 1'b0;
      m_axi_bvalid_i <= 1'b0;
      scratchpad_read_response_valid_i <= 1'b0;
      ar_stalled_q <= 1'b0;
      aw_stalled_q <= 1'b0;
      w_stalled_q <= 1'b0;
    end else begin
      cycle_q <= cycle_q + 1;

      if (ar_stalled_q
          && (!m_axi_arvalid_o
              || {m_axi_araddr_o, m_axi_arlen_o, m_axi_arsize_o,
                  m_axi_arburst_o} !== ar_payload_q))
        $fatal(1, "AR payload changed while stalled");
      if (aw_stalled_q
          && (!m_axi_awvalid_o
              || {m_axi_awaddr_o, m_axi_awlen_o, m_axi_awsize_o,
                  m_axi_awburst_o} !== aw_payload_q))
        $fatal(1, "AW payload changed while stalled");
      if (w_stalled_q
          && (!m_axi_wvalid_o
              || {m_axi_wdata_o, m_axi_wstrb_o, m_axi_wlast_o}
                 !== w_payload_q))
        $fatal(1, "W payload changed while stalled");
      ar_stalled_q <= m_axi_arvalid_o && !m_axi_arready_i;
      ar_payload_q <= {m_axi_araddr_o, m_axi_arlen_o, m_axi_arsize_o,
                       m_axi_arburst_o};
      aw_stalled_q <= m_axi_awvalid_o && !m_axi_awready_i;
      aw_payload_q <= {m_axi_awaddr_o, m_axi_awlen_o, m_axi_awsize_o,
                       m_axi_awburst_o};
      w_stalled_q <= m_axi_wvalid_o && !m_axi_wready_i;
      w_payload_q <= {m_axi_wdata_o, m_axi_wstrb_o, m_axi_wlast_o};

      if (m_axi_arvalid_o && m_axi_arready_i) begin
        if (m_axi_arburst_o != 2'b01)
          $fatal(1, "read burst type");
        if (m_axi_araddr_o[11:0]
            + ((m_axi_arlen_o + 1) << m_axi_arsize_o) > 4096)
          $fatal(1, "read burst crossed 4 KiB");
        ar_address_log[ar_count] <= m_axi_araddr_o;
        ar_length_log[ar_count] <= m_axi_arlen_o;
        ar_size_log[ar_count] <= m_axi_arsize_o;
        ar_count <= ar_count + 1;
        read_active_q <= 1'b1;
        read_address_q <= m_axi_araddr_o;
        read_beats_q <= m_axi_arlen_o + 1'b1;
        read_size_q <= m_axi_arsize_o;
        read_beat_index_q <= 0;
      end

      if (read_active_q && !m_axi_rvalid_i) begin
        m_axi_rdata_i <= external_word(read_address_q);
        m_axi_rresp_i <= (inject_read_error && read_beat_index_q == 0)
          ? 2'b10 : 2'b00;
        m_axi_rlast_i <= (inject_early_rlast && read_beat_index_q == 0)
          || read_beats_q == 1;
        m_axi_rvalid_i <= 1'b1;
      end
      if (m_axi_rvalid_i && m_axi_rready_o) begin
        m_axi_rvalid_i <= 1'b0;
        if (m_axi_rlast_i) begin
          read_active_q <= 1'b0;
        end else begin
          read_address_q <= read_address_q + (1 << read_size_q);
          read_beats_q <= read_beats_q - 1'b1;
          read_beat_index_q <= read_beat_index_q + 1'b1;
        end
      end

      if (scratchpad_write_valid_o && scratchpad_write_ready_i) begin
        if (scratchpad_write_kind_o != request.scratchpad
            || scratchpad_write_bank_o != request.bank)
          $fatal(1, "scratchpad write route");
        for (lane = 0; lane < 8; lane = lane + 1)
          if (scratchpad_write_strobe_o[lane])
            scratchpad_memory[scratchpad_write_address_o + lane]
              <= scratchpad_write_data_o[lane*8 +: 8];
      end

      if (scratchpad_read_request_valid_o
          && scratchpad_read_request_ready_i) begin
        if (scratchpad_read_kind_o != request.scratchpad
            || scratchpad_read_bank_o != request.bank)
          $fatal(1, "scratchpad read route");
        scratchpad_read_response_data_i
          <= scratchpad_word(scratchpad_read_address_o);
        scratchpad_read_response_valid_i <= 1'b1;
      end else if (scratchpad_read_response_valid_i
                   && scratchpad_read_response_ready_o) begin
        scratchpad_read_response_valid_i <= 1'b0;
      end

      if (m_axi_awvalid_o && m_axi_awready_i) begin
        if (m_axi_awburst_o != 2'b01)
          $fatal(1, "write burst type");
        if (m_axi_awaddr_o[11:0]
            + ((m_axi_awlen_o + 1) << m_axi_awsize_o) > 4096)
          $fatal(1, "write burst crossed 4 KiB");
        aw_address_log[aw_count] <= m_axi_awaddr_o;
        aw_length_log[aw_count] <= m_axi_awlen_o;
        aw_size_log[aw_count] <= m_axi_awsize_o;
        aw_count <= aw_count + 1;
        write_active_q <= 1'b1;
        write_address_q <= m_axi_awaddr_o;
        write_beats_q <= m_axi_awlen_o + 1'b1;
        write_size_q <= m_axi_awsize_o;
      end

      if (m_axi_wvalid_o && m_axi_wready_i) begin
        if (!write_active_q || m_axi_wlast_o != (write_beats_q == 1))
          $fatal(1, "write beat/last protocol");
        for (lane = 0; lane < 8; lane = lane + 1)
          if (m_axi_wstrb_o[lane])
            external_memory[(write_address_q & 64'hffff_ffff_ffff_fff8) + lane]
              <= m_axi_wdata_o[lane*8 +: 8];
        if (write_beats_q == 1) begin
          write_active_q <= 1'b0;
          write_response_pending_q <= 1'b1;
        end else begin
          write_address_q <= write_address_q + (1 << write_size_q);
          write_beats_q <= write_beats_q - 1'b1;
        end
      end

      if (write_response_pending_q && !m_axi_bvalid_i) begin
        m_axi_bresp_i <= inject_write_error ? 2'b10 : 2'b00;
        m_axi_bvalid_i <= 1'b1;
        write_response_pending_q <= 1'b0;
      end else if (m_axi_bvalid_i && m_axi_bready_o) begin
        m_axi_bvalid_i <= 1'b0;
      end
    end
  end

  task automatic submit_request(input npu_dma_request_t value);
    begin
      do @(negedge clk_i); while (!request_ready_o);
      request = value;
      request_valid_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      request_valid_i = 1'b0;
    end
  endtask

  task automatic wait_result(
    input logic expect_error,
    input logic [3:0] expected_reason,
    input logic [31:0] expected_bytes
  );
    begin
      saw_done = 1'b0;
      saw_error = 1'b0;
      timeout = 0;
      while (timeout < 10000 && !saw_done && !saw_error) begin
        #1;
        if (done_pulse_o) begin
          saw_done = 1'b1;
          if (event_set_o != request.event_mask)
            $fatal(1, "completion event");
          if (request.store && bytes_written_o != expected_bytes)
            $fatal(1, "write byte count");
          if (!request.store && bytes_read_o != expected_bytes)
            $fatal(1, "read byte count");
        end
        if (error_pulse_o) begin
          saw_error = 1'b1;
          if (error_reason_o != expected_reason
              || error_tag_o != request.tag)
            $fatal(1, "error report");
        end
        if (!saw_done && !saw_error)
          @(negedge clk_i);
        timeout = timeout + 1;
      end
      if (expect_error ? (!saw_error || saw_done) : (!saw_done || saw_error)) begin
        $display("timeout state=%0d busy=%0b AR=%0b/%0b R=%0b/%0b AW=%0b/%0b W=%0b/%0b B=%0b/%0b SPW=%0b/%0b SPR=%0b/%0b",
                 dut.state_q, busy_o, m_axi_arvalid_o, m_axi_arready_i,
                 m_axi_rvalid_i, m_axi_rready_o, m_axi_awvalid_o,
                 m_axi_awready_i, m_axi_wvalid_o, m_axi_wready_i,
                 m_axi_bvalid_i, m_axi_bready_o, scratchpad_write_valid_o,
                 scratchpad_write_ready_i, scratchpad_read_request_valid_o,
                 scratchpad_read_request_ready_i);
        $fatal(1, "DMA completion kind or timeout");
      end
      @(posedge clk_i);
    end
  endtask

  task automatic initialize_memories;
    begin
      for (index = 0; index < MEMORY_BYTES; index = index + 1)
        external_memory[index] = (index * 13 + 7) & 8'hff;
      for (index = 0; index < SCRATCHPAD_BYTES; index = index + 1)
        scratchpad_memory[index] = 8'haa;
    end
  endtask

  task automatic check_load_row(
    input integer external_base,
    input integer scratchpad_base,
    input integer byte_count
  );
    begin
      for (index = 0; index < byte_count; index = index + 1)
        if (scratchpad_memory[scratchpad_base + index]
            !== external_memory[external_base + index])
          $fatal(1, "load data mismatch at byte %0d", index);
    end
  endtask

  initial begin
    request = '0;
    initialize_memories();
    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;

    // 2x2 rows, with the first row beginning eight bytes before a 4 KiB
    // boundary.  Also verifies explicit local clear and all back-pressure.
    request = '0;
    request.tag = 16'h1001;
    request.event_mask = 8'h5a;
    request.memory_space = NPU_DMA_SPACE_ACTIVATION;
    request.scratchpad = NPU_SPAD_A;
    request.clear_before = 1'b1;
    request.external_address = 64'd4088;
    request.x_bytes = 40;
    request.y_count = 2;
    request.z_count = 2;
    request.external_y_stride = 64;
    request.external_z_stride = 256;
    request.scratchpad_y_stride = 64;
    request.scratchpad_z_stride = 256;
    request.clear_bytes = 64;
    start_count = ar_count;
    submit_request(request);
    wait_result(1'b0, NPU_DMA_EXEC_OK, 160);
    if (ar_count - start_count != 5
        || ar_address_log[start_count] != 4088
        || ar_length_log[start_count] != 0
        || ar_address_log[start_count + 1] != 4096
        || ar_length_log[start_count + 1] != 3)
      $fatal(1, "4 KiB read split");
    check_load_row(4088, 0, 40);
    check_load_row(4152, 64, 40);
    check_load_row(4344, 256, 40);
    check_load_row(4408, 320, 40);
    for (index = 40; index < 64; index = index + 1)
      if (scratchpad_memory[index] != 0)
        $fatal(1, "activation clear gap");

    // A six-byte row becomes one 4-byte and one 2-byte narrow transfer.
    request = '0;
    request.tag = 16'h1002;
    request.event_mask = 8'h01;
    request.memory_space = NPU_DMA_SPACE_ACTIVATION;
    request.scratchpad = NPU_SPAD_A;
    request.bank = 1;
    request.external_address = 8192;
    request.scratchpad_offset = 512;
    request.x_bytes = 6;
    request.y_count = 1;
    request.z_count = 1;
    start_count = ar_count;
    submit_request(request);
    wait_result(1'b0, NPU_DMA_EXEC_OK, 6);
    if (ar_count - start_count != 2
        || ar_size_log[start_count] != 2
        || ar_size_log[start_count + 1] != 1
        || ar_address_log[start_count + 1] != 8196)
      $fatal(1, "narrow read split");
    check_load_row(8192, 512, 6);
    if (scratchpad_memory[518] != 8'haa)
      $fatal(1, "narrow read overwrote tail");

    // A long row is capped at the AXI maximum of 256 beats.
    request = '0;
    request.tag = 16'h1004;
    request.event_mask = 8'h02;
    request.memory_space = NPU_DMA_SPACE_WEIGHT;
    request.scratchpad = NPU_SPAD_W;
    request.external_address = 0;
    request.scratchpad_offset = 4096;
    request.x_bytes = 3000;
    request.y_count = 1;
    request.z_count = 1;
    start_count = ar_count;
    submit_request(request);
    wait_result(1'b0, NPU_DMA_EXEC_OK, 3000);
    if (ar_count - start_count != 2
        || ar_length_log[start_count] != 8'hff
        || ar_length_log[start_count + 1] != 8'd118
        || ar_size_log[start_count] != 3)
      $fatal(1, "256-beat read limit");
    check_load_row(0, 4096, 3000);

    // Store four strided rows of ten bytes; padded bytes must remain intact.
    for (row = 0; row < 2; row = row + 1)
      for (index = 0; index < 10; index = index + 1) begin
        scratchpad_memory[1024 + row*16 + index] = 8'h20 + row*16 + index;
        scratchpad_memory[1088 + row*16 + index] = 8'h80 + row*16 + index;
      end
    for (index = 12288; index < 12464; index = index + 1)
      external_memory[index] = 8'hee;
    request = '0;
    request.tag = 16'h1003;
    request.event_mask = 8'h80;
    request.store = 1'b1;
    request.memory_space = NPU_DMA_SPACE_ACTIVATION;
    request.scratchpad = NPU_SPAD_O;
    request.external_address = 12288;
    request.scratchpad_offset = 1024;
    request.x_bytes = 10;
    request.y_count = 2;
    request.z_count = 2;
    request.external_y_stride = 32;
    request.external_z_stride = 128;
    request.scratchpad_y_stride = 16;
    request.scratchpad_z_stride = 64;
    start_count = aw_count;
    submit_request(request);
    wait_result(1'b0, NPU_DMA_EXEC_OK, 40);
    if (aw_count - start_count != 8)
      $fatal(1, "narrow write burst count");
    for (row = 0; row < 2; row = row + 1)
      for (index = 0; index < 10; index = index + 1) begin
        if (external_memory[12288 + row*32 + index]
            !== scratchpad_memory[1024 + row*16 + index])
          $fatal(1, "store plane 0 mismatch");
        if (external_memory[12416 + row*32 + index]
            !== scratchpad_memory[1088 + row*16 + index])
          $fatal(1, "store plane 1 mismatch");
      end
    if (external_memory[12298] != 8'hee
        || external_memory[12426] != 8'hee)
      $fatal(1, "store overwrote row padding");

    // Write splitting uses the same 4 KiB rule and preserves a 2-byte tail
    // in lanes 4/5 of the final 64-bit word.
    for (index = 0; index < 14; index = index + 1)
      scratchpad_memory[8192 + index] = 8'hc0 + index;
    for (index = 4088; index < 4112; index = index + 1)
      external_memory[index] = 8'hee;
    request = '0;
    request.tag = 16'h1005;
    request.event_mask = 8'h40;
    request.store = 1'b1;
    request.memory_space = NPU_DMA_SPACE_ACTIVATION;
    request.scratchpad = NPU_SPAD_O;
    request.external_address = 4088;
    request.scratchpad_offset = 8192;
    request.x_bytes = 14;
    request.y_count = 1;
    request.z_count = 1;
    start_count = aw_count;
    submit_request(request);
    wait_result(1'b0, NPU_DMA_EXEC_OK, 14);
    if (aw_count - start_count != 3
        || aw_address_log[start_count] != 4088
        || aw_size_log[start_count] != 3
        || aw_address_log[start_count + 1] != 4096
        || aw_size_log[start_count + 1] != 2
        || aw_address_log[start_count + 2] != 4100
        || aw_size_log[start_count + 2] != 1)
      $fatal(1, "4 KiB/narrow write split");
    for (index = 0; index < 14; index = index + 1)
      if (external_memory[4088 + index] != scratchpad_memory[8192 + index])
        $fatal(1, "cross-boundary store data");
    if (external_memory[4102] != 8'hee)
      $fatal(1, "narrow write overwrote tail");

    // A busy engine does not abandon an accepted AXI request on soft reset.
    request = '0;
    request.tag = 16'h1006;
    request.event_mask = 8'h04;
    request.external_address = 5000;
    request.scratchpad_offset = 12000;
    request.x_bytes = 32;
    request.y_count = 1;
    request.z_count = 1;
    submit_request(request);
    do @(negedge clk_i); while (!m_axi_rvalid_i);
    soft_reset_i = 1'b1;
    @(posedge clk_i);
    @(negedge clk_i);
    soft_reset_i = 1'b0;
    wait_result(1'b0, NPU_DMA_EXEC_OK, 32);
    check_load_row(5000, 12000, 32);

    // Invalid alignment is rejected before any AXI transaction.
    request = '0;
    request.tag = 16'h2001;
    request.external_address = 64'd2;
    request.x_bytes = 8;
    request.y_count = 1;
    request.z_count = 1;
    start_count = ar_count;
    submit_request(request);
    wait_result(1'b1, NPU_DMA_EXEC_UNSUPPORTED_ALIGNMENT, 0);
    if (ar_count != start_count) $fatal(1, "invalid request reached AXI");

    // RRESP error is drained through RLAST before reporting failure.
    request = '0;
    request.tag = 16'h2002;
    request.external_address = 6000;
    request.scratchpad_offset = 2048;
    request.x_bytes = 16;
    request.y_count = 1;
    request.z_count = 1;
    inject_read_error = 1'b1;
    submit_request(request);
    wait_result(1'b1, NPU_DMA_EXEC_AXI_READ, 0);
    inject_read_error = 1'b0;
    if (read_active_q || m_axi_rvalid_i)
      $fatal(1, "read error was not drained");

    // Early RLAST is detected as a protocol error.
    request.tag = 16'h2003;
    request.external_address = 8000;
    inject_early_rlast = 1'b1;
    submit_request(request);
    wait_result(1'b1, NPU_DMA_EXEC_AXI_PROTOCOL, 0);
    inject_early_rlast = 1'b0;

    // BRESP error is reported only after the accepted write is complete.
    request = '0;
    request.tag = 16'h2004;
    request.store = 1'b1;
    request.scratchpad = NPU_SPAD_O;
    request.external_address = 9000;
    request.scratchpad_offset = 2048;
    request.x_bytes = 8;
    request.y_count = 1;
    request.z_count = 1;
    inject_write_error = 1'b1;
    submit_request(request);
    wait_result(1'b1, NPU_DMA_EXEC_AXI_WRITE, 0);
    inject_write_error = 1'b0;

    $display("npu_dma_engine: PASS");
    $finish;
  end
endmodule
