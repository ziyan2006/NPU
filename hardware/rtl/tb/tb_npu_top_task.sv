`timescale 1ns/1ps

module tb_npu_top_task;
  localparam integer MAX_WORDS = 262144;
  localparam logic [63:0] TASK_BASE = 64'h0000_0000_1000_0000;

  logic aclk = 0;
  logic aresetn = 0;
  logic [11:0] s_axi_ctrl_awaddr = 0;
  logic [2:0] s_axi_ctrl_awprot = 0;
  logic s_axi_ctrl_awvalid = 0;
  logic s_axi_ctrl_awready;
  logic [31:0] s_axi_ctrl_wdata = 0;
  logic [3:0] s_axi_ctrl_wstrb = 0;
  logic s_axi_ctrl_wvalid = 0;
  logic s_axi_ctrl_wready;
  logic [1:0] s_axi_ctrl_bresp;
  logic s_axi_ctrl_bvalid;
  logic s_axi_ctrl_bready = 1;
  logic [11:0] s_axi_ctrl_araddr = 0;
  logic [2:0] s_axi_ctrl_arprot = 0;
  logic s_axi_ctrl_arvalid = 0;
  logic s_axi_ctrl_arready;
  logic [31:0] s_axi_ctrl_rdata;
  logic [1:0] s_axi_ctrl_rresp;
  logic s_axi_ctrl_rvalid;
  logic s_axi_ctrl_rready = 0;

  logic [3:0] m_axi_mem_arid;
  logic [63:0] m_axi_mem_araddr;
  logic [7:0] m_axi_mem_arlen;
  logic [2:0] m_axi_mem_arsize;
  logic [1:0] m_axi_mem_arburst;
  logic m_axi_mem_arlock;
  logic [3:0] m_axi_mem_arcache;
  logic [2:0] m_axi_mem_arprot;
  logic [3:0] m_axi_mem_arqos;
  logic m_axi_mem_arvalid;
  logic m_axi_mem_arready;
  logic [3:0] m_axi_mem_rid = 0;
  logic [63:0] m_axi_mem_rdata = 0;
  logic [1:0] m_axi_mem_rresp = 0;
  logic m_axi_mem_rlast = 0;
  logic m_axi_mem_rvalid = 0;
  logic m_axi_mem_rready;
  logic [3:0] m_axi_mem_awid;
  logic [63:0] m_axi_mem_awaddr;
  logic [7:0] m_axi_mem_awlen;
  logic [2:0] m_axi_mem_awsize;
  logic [1:0] m_axi_mem_awburst;
  logic m_axi_mem_awlock;
  logic [3:0] m_axi_mem_awcache;
  logic [2:0] m_axi_mem_awprot;
  logic [3:0] m_axi_mem_awqos;
  logic m_axi_mem_awvalid;
  logic m_axi_mem_awready;
  logic [63:0] m_axi_mem_wdata;
  logic [7:0] m_axi_mem_wstrb;
  logic m_axi_mem_wlast;
  logic m_axi_mem_wvalid;
  logic m_axi_mem_wready;
  logic [3:0] m_axi_mem_bid = 0;
  logic [1:0] m_axi_mem_bresp = 0;
  logic m_axi_mem_bvalid = 0;
  logic m_axi_mem_bready;
  logic irq_o;

  logic [63:0] memory [0:MAX_WORDS-1];
  logic read_active_q = 0;
  logic [63:0] read_address_q = 0;
  logic [8:0] read_beats_q = 0;
  logic [3:0] read_beat_bytes_q = 0;
  logic write_active_q = 0;
  logic [63:0] write_address_q = 0;
  logic [8:0] write_beats_q = 0;
  logic [3:0] write_beat_bytes_q = 0;
  logic write_response_pending_q = 0;
  integer cycles_q = 0;
  integer lane;
  integer image_bytes;
  integer output_offset;
  integer output_bytes;
  string image_hex;
  string output_hex;
  logic [31:0] read_value;

  always #5 aclk = ~aclk;
  assign m_axi_mem_arready = !read_active_q && !m_axi_mem_rvalid;
  assign m_axi_mem_awready = !write_active_q
    && !write_response_pending_q && !m_axi_mem_bvalid;
  assign m_axi_mem_wready = write_active_q;

  npu_top dut (.*);

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      read_active_q <= 0;
      m_axi_mem_rvalid <= 0;
      write_active_q <= 0;
      write_response_pending_q <= 0;
      m_axi_mem_bvalid <= 0;
      cycles_q <= 0;
    end else begin
      cycles_q <= cycles_q + 1;
      if (cycles_q != 0 && cycles_q % 1_000_000 == 0)
        $display("npu_top progress: %0d cycles", cycles_q);
      if (cycles_q > 10_000_000)
        $fatal(1, "full task timeout");

      if (m_axi_mem_arvalid && m_axi_mem_arready) begin
        if (m_axi_mem_arburst != 2'b01 || m_axi_mem_arsize > 3
            || m_axi_mem_araddr < TASK_BASE
            || m_axi_mem_araddr - TASK_BASE
               + ((m_axi_mem_arlen + 1) << m_axi_mem_arsize) > image_bytes
            || m_axi_mem_araddr[11:0]
               + ((m_axi_mem_arlen + 1) << m_axi_mem_arsize) > 4096)
          $fatal(1, "illegal AXI read %h len=%0d size=%0d",
                 m_axi_mem_araddr, m_axi_mem_arlen, m_axi_mem_arsize);
        read_active_q <= 1;
        read_address_q <= m_axi_mem_araddr;
        read_beats_q <= m_axi_mem_arlen + 1;
        read_beat_bytes_q <= 4'b0001 << m_axi_mem_arsize;
      end
      if (m_axi_mem_rvalid) begin
        if (m_axi_mem_rready) begin
          m_axi_mem_rvalid <= 0;
          if (m_axi_mem_rlast) begin
            read_active_q <= 0;
          end else begin
            read_address_q <= read_address_q + read_beat_bytes_q;
            read_beats_q <= read_beats_q - 1;
          end
        end
      end else if (read_active_q) begin
        m_axi_mem_rdata <= memory[
          ((read_address_q & 64'hffff_ffff_ffff_fff8) - TASK_BASE) >> 3];
        m_axi_mem_rresp <= 0;
        m_axi_mem_rlast <= read_beats_q == 1;
        m_axi_mem_rvalid <= 1;
      end

      if (m_axi_mem_awvalid && m_axi_mem_awready) begin
        if (m_axi_mem_awburst != 2'b01 || m_axi_mem_awsize > 3
            || m_axi_mem_awaddr < TASK_BASE
            || m_axi_mem_awaddr - TASK_BASE
               + ((m_axi_mem_awlen + 1) << m_axi_mem_awsize) > image_bytes
            || m_axi_mem_awaddr[11:0]
               + ((m_axi_mem_awlen + 1) << m_axi_mem_awsize) > 4096)
          $fatal(1, "illegal AXI write %h len=%0d size=%0d",
                 m_axi_mem_awaddr, m_axi_mem_awlen, m_axi_mem_awsize);
        write_active_q <= 1;
        write_address_q <= m_axi_mem_awaddr;
        write_beats_q <= m_axi_mem_awlen + 1;
        write_beat_bytes_q <= 4'b0001 << m_axi_mem_awsize;
      end
      if (m_axi_mem_wvalid && m_axi_mem_wready) begin
        if (!write_active_q || m_axi_mem_wlast != (write_beats_q == 1))
          $fatal(1, "illegal AXI WLAST");
        for (lane = 0; lane < 8; lane = lane + 1)
          if (m_axi_mem_wstrb[lane])
            memory[((write_address_q & 64'hffff_ffff_ffff_fff8)
                    - TASK_BASE) >> 3][lane*8 +: 8]
              <= m_axi_mem_wdata[lane*8 +: 8];
        if (m_axi_mem_wlast) begin
          write_active_q <= 0;
          write_response_pending_q <= 1;
        end else begin
          write_address_q <= write_address_q + write_beat_bytes_q;
          write_beats_q <= write_beats_q - 1;
        end
      end
      if (write_response_pending_q && !m_axi_mem_bvalid) begin
        write_response_pending_q <= 0;
        m_axi_mem_bresp <= 0;
        m_axi_mem_bvalid <= 1;
      end
      if (m_axi_mem_bvalid && m_axi_mem_bready)
        m_axi_mem_bvalid <= 0;
    end
  end

  task automatic csr_write(input logic [11:0] address,
                           input logic [31:0] data);
    begin
      @(negedge aclk);
      s_axi_ctrl_awaddr = address;
      s_axi_ctrl_awvalid = 1;
      s_axi_ctrl_wdata = data;
      s_axi_ctrl_wstrb = 4'hf;
      s_axi_ctrl_wvalid = 1;
      do @(posedge aclk); while (!(s_axi_ctrl_awready && s_axi_ctrl_wready));
      @(negedge aclk);
      s_axi_ctrl_awvalid = 0;
      s_axi_ctrl_wvalid = 0;
      wait(s_axi_ctrl_bvalid);
      if (s_axi_ctrl_bresp != 0)
        $fatal(1, "CSR write failed %h", address);
      @(posedge aclk);
    end
  endtask

  task automatic csr_read(input logic [11:0] address,
                          output logic [31:0] data);
    begin
      @(negedge aclk);
      s_axi_ctrl_araddr = address;
      s_axi_ctrl_arvalid = 1;
      do @(posedge aclk); while (!s_axi_ctrl_arready);
      @(negedge aclk);
      s_axi_ctrl_arvalid = 0;
      s_axi_ctrl_rready = 1;
      wait(s_axi_ctrl_rvalid);
      if (s_axi_ctrl_rresp != 0)
        $fatal(1, "CSR read failed %h", address);
      data = s_axi_ctrl_rdata;
      @(posedge aclk);
      @(negedge aclk);
      s_axi_ctrl_rready = 0;
    end
  endtask

  integer expected_commands;
  integer max_sim_ns;
  initial begin
    if ($value$plusargs("MAX_SIM_NS=%d", max_sim_ns)) begin
      #(max_sim_ns);
      $fatal(1, "npu_top task timed out after %0d ns", max_sim_ns);
    end
  end
  initial begin
    if (!$value$plusargs("IMAGE=%s", image_hex)
        || !$value$plusargs("IMAGE_BYTES=%d", image_bytes)
        || !$value$plusargs("OUTPUT=%s", output_hex)
        || !$value$plusargs("OUTPUT_OFFSET=%d", output_offset)
        || !$value$plusargs("OUTPUT_BYTES=%d", output_bytes))
      $fatal(1, "missing plusargs");
    if (!$value$plusargs("EXPECTED_COMMANDS=%d", expected_commands))
      expected_commands = 1869;
    $readmemh(image_hex, memory, 0, image_bytes / 8 - 1);
    repeat (10) @(posedge aclk);
    aresetn <= 1;
    repeat (5) @(posedge aclk);
    csr_write(12'h024, TASK_BASE[31:0]);
    csr_write(12'h028, TASK_BASE[63:32]);
    csr_write(12'h02c, image_bytes);
    csr_write(12'h030, 32'h1357_2468);
    csr_write(12'h048, 32'd9_000_000);
    csr_write(12'h020, 32'h0000_0007);
    csr_write(12'h014, 32'h0000_0002);
    csr_write(12'h034, 32'h0000_0001);
    wait(irq_o);
    csr_read(12'h018, read_value);
    if (read_value[4]) begin
      csr_read(12'h03c, read_value);
      $fatal(1, "NPU task error code %h", read_value);
    end
    if (!read_value[3] || !read_value[0])
      $fatal(1, "NPU task missing done/idle status %h", read_value);
    csr_read(12'h038, read_value);
    if (read_value != 32'h1357_2468)
      $fatal(1, "completed tag mismatch");
    csr_read(12'h04c, read_value);
    if (read_value != expected_commands)
      $fatal(1, "retired commands mismatch %0d", read_value);
    csr_read(12'h050, read_value);
    $display("npu_top task: PASS cycles=%0d CSR_cycles_low=%0d",
             cycles_q, read_value);
    $writememh(output_hex, memory,
               (output_offset >> 3), ((output_offset + output_bytes) >> 3) - 1);
    $finish;
  end
endmodule
