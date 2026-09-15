`timescale 1ns/1ps

module tb_npu_axi_arbiters;
  logic clk_i = 0;
  logic rst_ni = 0;
  logic soft_reset_i = 0;
  logic [31:0] lfsr_q = 32'hb159_4a2d;
  integer cycles_q = 0;

  logic [2:0][63:0] s_araddr = '0;
  logic [2:0][7:0] s_arlen = '0;
  logic [2:0][2:0] s_arsize = '0;
  logic [2:0][1:0] s_arburst = '0;
  logic [2:0] s_arvalid = '0;
  logic [2:0] s_arready;
  logic [2:0][63:0] s_rdata;
  logic [2:0][1:0] s_rresp;
  logic [2:0] s_rlast;
  logic [2:0] s_rvalid;
  logic [2:0] s_rready = '0;
  logic [63:0] m_araddr;
  logic [7:0] m_arlen;
  logic [2:0] m_arsize;
  logic [1:0] m_arburst;
  logic m_arvalid;
  logic m_arready;
  logic [63:0] m_rdata;
  logic [1:0] m_rresp = 0;
  logic m_rlast;
  logic m_rvalid;
  logic m_rready;
  logic read_busy;
  logic read_active_q = 0;
  logic [63:0] read_base_q = 0;
  integer read_beat_q = 0;
  integer read_beats_q = 0;

  logic [1:0][63:0] s_awaddr = '0;
  logic [1:0][7:0] s_awlen = '0;
  logic [1:0][2:0] s_awsize = '0;
  logic [1:0][1:0] s_awburst = '0;
  logic [1:0] s_awvalid = '0;
  logic [1:0] s_awready;
  logic [1:0][63:0] s_wdata = '0;
  logic [1:0][7:0] s_wstrb = '0;
  logic [1:0] s_wlast = '0;
  logic [1:0] s_wvalid = '0;
  logic [1:0] s_wready;
  logic [1:0][1:0] s_bresp;
  logic [1:0] s_bvalid;
  logic [1:0] s_bready = 2'b11;
  logic [63:0] m_awaddr;
  logic [7:0] m_awlen;
  logic [2:0] m_awsize;
  logic [1:0] m_awburst;
  logic m_awvalid;
  logic m_awready;
  logic [63:0] m_wdata;
  logic [7:0] m_wstrb;
  logic m_wlast;
  logic m_wvalid;
  logic m_wready;
  logic [1:0] m_bresp = 0;
  logic m_bvalid = 0;
  logic m_bready;
  logic write_busy;
  logic write_active_q = 0;
  logic [63:0] write_base_q = 0;
  integer write_beat_q = 0;
  integer write_beats_q = 0;
  integer b_delay_q = 0;
  logic b_pending_q = 0;

  always #5 clk_i = ~clk_i;
  assign m_arready = !read_active_q && lfsr_q[0];
  assign m_rvalid = read_active_q && lfsr_q[1];
  assign m_rdata = read_base_q + read_beat_q;
  assign m_rlast = read_active_q && read_beat_q + 1 == read_beats_q;
  assign m_awready = !write_active_q && !m_bvalid && !b_pending_q && lfsr_q[2];
  assign m_wready = write_active_q && lfsr_q[3];

  npu_axi_read_arbiter3 read_arb (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .s_axi_araddr_i(s_araddr), .s_axi_arlen_i(s_arlen),
    .s_axi_arsize_i(s_arsize), .s_axi_arburst_i(s_arburst),
    .s_axi_arvalid_i(s_arvalid), .s_axi_arready_o(s_arready),
    .s_axi_rdata_o(s_rdata), .s_axi_rresp_o(s_rresp),
    .s_axi_rlast_o(s_rlast), .s_axi_rvalid_o(s_rvalid),
    .s_axi_rready_i(s_rready), .m_axi_araddr_o(m_araddr),
    .m_axi_arlen_o(m_arlen), .m_axi_arsize_o(m_arsize),
    .m_axi_arburst_o(m_arburst), .m_axi_arvalid_o(m_arvalid),
    .m_axi_arready_i(m_arready), .m_axi_rdata_i(m_rdata),
    .m_axi_rresp_i(m_rresp), .m_axi_rlast_i(m_rlast),
    .m_axi_rvalid_i(m_rvalid), .m_axi_rready_o(m_rready), .busy_o(read_busy));

  npu_axi_write_arbiter2 write_arb (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .s_axi_awaddr_i(s_awaddr), .s_axi_awlen_i(s_awlen),
    .s_axi_awsize_i(s_awsize), .s_axi_awburst_i(s_awburst),
    .s_axi_awvalid_i(s_awvalid), .s_axi_awready_o(s_awready),
    .s_axi_wdata_i(s_wdata), .s_axi_wstrb_i(s_wstrb),
    .s_axi_wlast_i(s_wlast), .s_axi_wvalid_i(s_wvalid),
    .s_axi_wready_o(s_wready), .s_axi_bresp_o(s_bresp),
    .s_axi_bvalid_o(s_bvalid), .s_axi_bready_i(s_bready),
    .m_axi_awaddr_o(m_awaddr), .m_axi_awlen_o(m_awlen),
    .m_axi_awsize_o(m_awsize), .m_axi_awburst_o(m_awburst),
    .m_axi_awvalid_o(m_awvalid), .m_axi_awready_i(m_awready),
    .m_axi_wdata_o(m_wdata), .m_axi_wstrb_o(m_wstrb),
    .m_axi_wlast_o(m_wlast), .m_axi_wvalid_o(m_wvalid),
    .m_axi_wready_i(m_wready), .m_axi_bresp_i(m_bresp),
    .m_axi_bvalid_i(m_bvalid), .m_axi_bready_o(m_bready), .busy_o(write_busy));

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      lfsr_q <= 32'hb159_4a2d;
      cycles_q <= 0;
      read_active_q <= 0;
      write_active_q <= 0;
      m_bvalid <= 0;
      b_pending_q <= 0;
    end else begin
      lfsr_q <= {lfsr_q[30:0],
                 lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
      cycles_q <= cycles_q + 1;
      if (cycles_q > 100000)
        $fatal(1, "AXI arbiter timeout");
      if (!$onehot0(s_rvalid) || !$onehot0(s_bvalid))
        $fatal(1, "response routed to multiple clients");

      if (m_arvalid && m_arready) begin
        if (m_arsize != 3 || m_arburst != 2'b01)
          $fatal(1, "read attributes corrupted");
        read_active_q <= 1;
        read_base_q <= m_araddr;
        read_beat_q <= 0;
        read_beats_q <= m_arlen + 1;
      end
      if (m_rvalid && m_rready) begin
        if (m_rlast) begin
          read_active_q <= 0;
          read_beat_q <= 0;
        end else begin
          read_beat_q <= read_beat_q + 1;
        end
      end

      if (m_awvalid && m_awready) begin
        if (m_awsize != 3 || m_awburst != 2'b01)
          $fatal(1, "write attributes corrupted");
        write_active_q <= 1;
        write_base_q <= m_awaddr;
        write_beat_q <= 0;
        write_beats_q <= m_awlen + 1;
      end
      if (m_wvalid && m_wready) begin
        if (m_wdata != write_base_q + write_beat_q || m_wstrb != 8'hff)
          $fatal(1, "write payload corrupted");
        if (m_wlast != (write_beat_q + 1 == write_beats_q))
          $fatal(1, "write last corrupted");
        if (m_wlast) begin
          write_active_q <= 0;
          b_pending_q <= 1;
          b_delay_q <= lfsr_q[6:4];
        end else begin
          write_beat_q <= write_beat_q + 1;
        end
      end
      if (b_pending_q && !m_bvalid) begin
        if (b_delay_q == 0) begin
          b_pending_q <= 0;
          m_bvalid <= 1;
        end else begin
          b_delay_q <= b_delay_q - 1;
        end
      end
      if (m_bvalid && m_bready)
        m_bvalid <= 0;
    end
  end

  task automatic read_client(input integer id);
    integer round;
    integer beat;
    logic [63:0] base;
    begin
      s_rready[id] = 1;
      for (round = 0; round < 8; round = round + 1) begin
        base = 64'h1000_0000 + id * 64'h0010_0000 + round * 64'h100;
        @(negedge clk_i);
        s_araddr[id] = base;
        s_arlen[id] = id + round % 4;
        s_arsize[id] = 3;
        s_arburst[id] = 2'b01;
        s_arvalid[id] = 1;
        do @(posedge clk_i); while (!s_arready[id]);
        @(negedge clk_i); s_arvalid[id] = 0;
        for (beat = 0; beat <= id + round % 4; beat = beat + 1) begin
          do @(posedge clk_i); while (!s_rvalid[id]);
          if (s_rdata[id] != base + beat || s_rresp[id] != 0
              || s_rlast[id] != (beat == id + round % 4))
            $fatal(1, "read client %0d mismatch round=%0d beat=%0d",
                   id, round, beat);
        end
      end
    end
  endtask

  task automatic write_client(input integer id);
    integer round;
    integer beat;
    integer beats;
    logic [63:0] base;
    begin
      for (round = 0; round < 8; round = round + 1) begin
        base = 64'h8000_0000 + id * 64'h0010_0000 + round * 64'h100;
        beats = id + round % 4 + 1;
        @(negedge clk_i);
        s_awaddr[id] = base;
        s_awlen[id] = beats - 1;
        s_awsize[id] = 3;
        s_awburst[id] = 2'b01;
        s_awvalid[id] = 1;
        do @(posedge clk_i); while (!s_awready[id]);
        @(negedge clk_i); s_awvalid[id] = 0;
        for (beat = 0; beat < beats; beat = beat + 1) begin
          s_wdata[id] = base + beat;
          s_wstrb[id] = 8'hff;
          s_wlast[id] = beat + 1 == beats;
          s_wvalid[id] = 1;
          do @(posedge clk_i); while (!s_wready[id]);
          @(negedge clk_i); s_wvalid[id] = 0;
        end
        do @(posedge clk_i); while (!s_bvalid[id]);
        if (s_bresp[id] != 0)
          $fatal(1, "write response error client %0d", id);
      end
    end
  endtask

  initial begin
    repeat (5) @(posedge clk_i);
    rst_ni <= 1;
    repeat (2) @(posedge clk_i);
    fork
      read_client(0);
      read_client(1);
      read_client(2);
      write_client(0);
      write_client(1);
    join
    wait(!read_busy && !write_busy);
    $display("NPU AXI arbiters: PASS cycles=%0d", cycles_q);
    $finish;
  end
endmodule
