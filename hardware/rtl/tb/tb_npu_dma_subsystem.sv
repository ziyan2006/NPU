`timescale 1ns/1ps

module tb_npu_dma_subsystem;
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic soft_reset_i = 1'b0;
  logic command_valid_i = 1'b0;
  logic command_ready_o;
  logic [127:0] command_bits_i = '0;
  logic [31:0] command_pc_i = '0;
  logic [63:0] tensor_desc_base_i = 64'h1000;
  logic [63:0] operator_desc_base_i = 64'h2000;
  logic [63:0] quant_desc_base_i = 64'h3000;
  logic [63:0] segment_desc_base_i = 64'h4000;
  logic [63:0] activation_base_i = 64'h0010_0000;
  logic [63:0] weight_base_i = 64'h0020_0000;
  logic [63:0] bias_base_i = 64'h0030_0000;
  logic [63:0] quant_param_base_i = 64'h0040_0000;
  logic descriptor_memory_request_valid_o;
  logic descriptor_memory_request_ready_i;
  logic [63:0] descriptor_memory_request_address_o;
  logic [6:0] descriptor_memory_request_bytes_o;
  logic descriptor_memory_response_valid_i = 1'b0;
  logic descriptor_memory_response_ready_o;
  logic [511:0] descriptor_memory_response_data_i = '0;
  logic descriptor_memory_response_error_i = 1'b0;
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
  logic m_axi_awready_i = 1'b1;
  logic [63:0] m_axi_wdata_o;
  logic [7:0] m_axi_wstrb_o;
  logic m_axi_wlast_o;
  logic m_axi_wvalid_o;
  logic m_axi_wready_i = 1'b1;
  logic [1:0] m_axi_bresp_i = '0;
  logic m_axi_bvalid_i = 1'b0;
  logic m_axi_bready_o;
  logic accelerator_write_valid_i = 1'b0;
  logic accelerator_write_ready_o;
  logic [1:0] accelerator_write_kind_i = '0;
  logic accelerator_write_bank_i = 1'b0;
  logic [31:0] accelerator_write_address_i = '0;
  logic [63:0] accelerator_write_data_i = '0;
  logic [7:0] accelerator_write_strobe_i = '0;
  logic accelerator_read_request_valid_i = 1'b0;
  logic accelerator_read_request_ready_o;
  logic [1:0] accelerator_read_kind_i = '0;
  logic accelerator_read_bank_i = 1'b0;
  logic [31:0] accelerator_read_address_i = '0;
  logic accelerator_read_response_valid_o;
  logic accelerator_read_response_ready_i = 1'b0;
  logic [63:0] accelerator_read_response_data_o;
  logic busy_o;
  logic [7:0] event_set_o;
  logic error_pulse_o;
  logic [1:0] error_source_o;
  logic [3:0] error_reason_o;
  logic [3:0] error_detail_o;
  logic [31:0] error_pc_o;
  logic [15:0] error_tag_o;
  logic [31:0] bytes_read_o;
  logic [31:0] bytes_written_o;
  logic scratchpad_collision_stall_o;

  npu_command_t command;
  npu_tensor_desc_t tensor_desc;
  npu_operator_desc_t operator_desc;
  logic descriptor_pending_q = 1'b0;
  logic [511:0] descriptor_pending_data_q = '0;
  logic read_active_q = 1'b0;
  logic [63:0] read_cursor_q = '0;
  logic [8:0] read_beats_q = '0;
  logic [3:0] read_beat_bytes_q = '0;
  logic inject_read_error = 1'b0;
  integer timeout;

  always #5 clk_i = ~clk_i;
  assign descriptor_memory_request_ready_i = !descriptor_pending_q
    && !descriptor_memory_response_valid_i;
  assign m_axi_arready_i = !read_active_q && !m_axi_rvalid_i;

  function automatic logic [63:0] external_word(input logic [63:0] address);
    external_word = 64'hc0de_0000_0000_0000 ^ address;
  endfunction

  npu_dma_subsystem dut (.*);

  always @(posedge clk_i) begin
    if (!rst_ni) begin
      descriptor_pending_q <= 1'b0;
      descriptor_memory_response_valid_i <= 1'b0;
      read_active_q <= 1'b0;
      m_axi_rvalid_i <= 1'b0;
    end else begin
      if (descriptor_memory_response_valid_i
          && descriptor_memory_response_ready_o)
        descriptor_memory_response_valid_i <= 1'b0;
      if (descriptor_memory_request_valid_o
          && descriptor_memory_request_ready_i) begin
        descriptor_pending_q <= 1'b1;
        if (descriptor_memory_request_address_o == operator_desc_base_i
            && descriptor_memory_request_bytes_o == 64)
          descriptor_pending_data_q <= operator_desc;
        else if (descriptor_memory_request_address_o == tensor_desc_base_i
                 && descriptor_memory_request_bytes_o == 64)
          descriptor_pending_data_q <= tensor_desc;
        else
          $fatal(1, "unexpected descriptor request %h/%0d",
                 descriptor_memory_request_address_o,
                 descriptor_memory_request_bytes_o);
      end else if (descriptor_pending_q) begin
        descriptor_memory_response_data_i <= descriptor_pending_data_q;
        descriptor_memory_response_valid_i <= 1'b1;
        descriptor_pending_q <= 1'b0;
      end

      if (m_axi_arvalid_o && m_axi_arready_i) begin
        if (m_axi_arburst_o != 2'b01)
          $fatal(1, "read was not INCR");
        read_cursor_q <= m_axi_araddr_o;
        read_beats_q <= m_axi_arlen_o + 1'b1;
        read_beat_bytes_q <= 4'd1 << m_axi_arsize_o;
        read_active_q <= 1'b1;
      end
      if (m_axi_rvalid_i) begin
        if (m_axi_rready_o) begin
          m_axi_rvalid_i <= 1'b0;
          if (m_axi_rlast_i) begin
            read_active_q <= 1'b0;
          end else begin
            read_cursor_q <= read_cursor_q + read_beat_bytes_q;
            read_beats_q <= read_beats_q - 1'b1;
          end
        end
      end else if (read_active_q) begin
        m_axi_rdata_i <= external_word(read_cursor_q);
        m_axi_rresp_i <= inject_read_error ? 2'b10 : 2'b00;
        m_axi_rlast_i <= read_beats_q == 1;
        m_axi_rvalid_i <= 1'b1;
      end

      if (m_axi_awvalid_o || m_axi_wvalid_o || m_axi_bready_o)
        $fatal(1, "load unexpectedly drove write AXI");
    end
  end

  task automatic issue_load(input logic [31:0] pc);
    begin
      @(negedge clk_i);
      command_pc_i = pc;
      command_bits_i = command;
      command_valid_i = 1'b1;
      do @(posedge clk_i); while (!command_ready_o);
      @(negedge clk_i);
      command_valid_i = 1'b0;
    end
  endtask

  task automatic read_a0_check(
    input logic [31:0] address,
    input logic [63:0] expected
  );
    begin
      @(negedge clk_i);
      accelerator_read_kind_i = NPU_SPAD_A;
      accelerator_read_bank_i = 1'b0;
      accelerator_read_address_i = address;
      accelerator_read_request_valid_i = 1'b1;
      do @(posedge clk_i); while (!accelerator_read_request_ready_o);
      @(negedge clk_i);
      accelerator_read_request_valid_i = 1'b0;
      do @(negedge clk_i); while (!accelerator_read_response_valid_o);
      if (accelerator_read_response_data_o !== expected)
        $fatal(1, "end-to-end scratchpad data mismatch at %0d", address);
      accelerator_read_response_ready_i = 1'b1;
      @(posedge clk_i);
      @(negedge clk_i);
      accelerator_read_response_ready_i = 1'b0;
    end
  endtask

  initial begin
    command = '0;
    command.opcode = NPU_OP_DMA_LOAD;
    command.flags = NPU_FLAG_ASYNC;
    command.tag = 16'h1234;
    command.dst_td = 0;
    command.src0_td = 0;
    command.src1_td = NPU_NONE_INDEX;
    command.op_desc = 0;
    command.quant_desc = NPU_NONE_INDEX;
    command.imm = 16'h0001;

    tensor_desc = '0;
    tensor_desc.base_offset = 64'h100;
    tensor_desc.allocation_bytes = 32;
    tensor_desc.shape_n = 1;
    tensor_desc.shape_h = 1;
    tensor_desc.shape_w = 2;
    tensor_desc.shape_c = 8;
    tensor_desc.stride_n = 32;
    tensor_desc.stride_h = 32;
    tensor_desc.stride_w = 16;
    tensor_desc.stride_c = 2;
    tensor_desc.dtype = NPU_DTYPE_INT12_IN_INT16;
    tensor_desc.layout = NPU_LAYOUT_NHWC8;

    operator_desc = '0;
    operator_desc.src_td = 0;
    operator_desc.dst_td = 1;
    operator_desc.weight_td = 2;
    operator_desc.bias_td = 3;
    operator_desc.kh = 1;
    operator_desc.kw = 1;
    operator_desc.stride_h = 1;
    operator_desc.stride_w = 1;
    operator_desc.dilation_h = 1;
    operator_desc.dilation_w = 1;
    operator_desc.groups = 1;
    operator_desc.tile_h = 1;
    operator_desc.tile_w = 2;
    operator_desc.tile_cout = 8;
    operator_desc.input_channel_count = 8;

    repeat (4) @(posedge clk_i);
    rst_ni = 1'b1;
    issue_load(32'h40);
    while (!read_active_q)
      @(negedge clk_i);
    soft_reset_i = 1'b1;
    repeat (2) begin
      @(posedge clk_i);
      if (command_ready_o)
        $fatal(1, "soft reset allowed a new command while busy");
    end
    @(negedge clk_i);
    soft_reset_i = 1'b0;
    timeout = 0;
    while (event_set_o != 8'h01 && timeout < 1000) begin
      @(posedge clk_i);
      timeout = timeout + 1;
    end
    if (timeout == 1000) $fatal(1, "end-to-end load timeout");
    if (bytes_read_o != 32 || bytes_written_o != 0 || error_pulse_o)
      $fatal(1, "end-to-end counters/error");
    read_a0_check(0, external_word(activation_base_i + 64'h100));
    read_a0_check(8, external_word(activation_base_i + 64'h108));
    read_a0_check(16, external_word(activation_base_i + 64'h110));
    read_a0_check(24, external_word(activation_base_i + 64'h118));

    // The subsystem retains the issuing PC while the asynchronous engine is
    // active and reports it with a later AXI error.
    inject_read_error = 1'b1;
    issue_load(32'h80);
    timeout = 0;
    while (!error_pulse_o && timeout < 1000) begin
      @(negedge clk_i);
      timeout = timeout + 1;
    end
    if (timeout == 1000 || error_source_o != 2
        || error_reason_o != NPU_DMA_EXEC_AXI_READ
        || error_pc_o != 32'h80 || error_tag_o != 16'h1234)
      $fatal(1, "asynchronous error context");

    $display("npu_dma_subsystem: PASS");
    $finish;
  end

endmodule
