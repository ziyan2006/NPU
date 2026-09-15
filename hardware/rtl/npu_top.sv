`timescale 1ns/1ps

// Software-facing NPU IP top.  The memory master is AXI4 and uses one fixed
// transaction ID; a Vivado SmartConnect/protocol converter adapts it to the
// Zynq-7000 PS HP AXI3 port in the board-specific block design.
module npu_top (
  input  logic         aclk,
  input  logic         aresetn,

  input  logic [11:0]  s_axi_ctrl_awaddr,
  input  logic [2:0]   s_axi_ctrl_awprot,
  input  logic         s_axi_ctrl_awvalid,
  output logic         s_axi_ctrl_awready,
  input  logic [31:0]  s_axi_ctrl_wdata,
  input  logic [3:0]   s_axi_ctrl_wstrb,
  input  logic         s_axi_ctrl_wvalid,
  output logic         s_axi_ctrl_wready,
  output logic [1:0]   s_axi_ctrl_bresp,
  output logic         s_axi_ctrl_bvalid,
  input  logic         s_axi_ctrl_bready,
  input  logic [11:0]  s_axi_ctrl_araddr,
  input  logic [2:0]   s_axi_ctrl_arprot,
  input  logic         s_axi_ctrl_arvalid,
  output logic         s_axi_ctrl_arready,
  output logic [31:0]  s_axi_ctrl_rdata,
  output logic [1:0]   s_axi_ctrl_rresp,
  output logic         s_axi_ctrl_rvalid,
  input  logic         s_axi_ctrl_rready,

  output logic [3:0]   m_axi_mem_arid,
  output logic [63:0]  m_axi_mem_araddr,
  output logic [7:0]   m_axi_mem_arlen,
  output logic [2:0]   m_axi_mem_arsize,
  output logic [1:0]   m_axi_mem_arburst,
  output logic         m_axi_mem_arlock,
  output logic [3:0]   m_axi_mem_arcache,
  output logic [2:0]   m_axi_mem_arprot,
  output logic [3:0]   m_axi_mem_arqos,
  output logic         m_axi_mem_arvalid,
  input  logic         m_axi_mem_arready,
  input  logic [3:0]   m_axi_mem_rid,
  input  logic [63:0]  m_axi_mem_rdata,
  input  logic [1:0]   m_axi_mem_rresp,
  input  logic         m_axi_mem_rlast,
  input  logic         m_axi_mem_rvalid,
  output logic         m_axi_mem_rready,

  output logic [3:0]   m_axi_mem_awid,
  output logic [63:0]  m_axi_mem_awaddr,
  output logic [7:0]   m_axi_mem_awlen,
  output logic [2:0]   m_axi_mem_awsize,
  output logic [1:0]   m_axi_mem_awburst,
  output logic         m_axi_mem_awlock,
  output logic [3:0]   m_axi_mem_awcache,
  output logic [2:0]   m_axi_mem_awprot,
  output logic [3:0]   m_axi_mem_awqos,
  output logic         m_axi_mem_awvalid,
  input  logic         m_axi_mem_awready,
  output logic [63:0]  m_axi_mem_wdata,
  output logic [7:0]   m_axi_mem_wstrb,
  output logic         m_axi_mem_wlast,
  output logic         m_axi_mem_wvalid,
  input  logic         m_axi_mem_wready,
  input  logic [3:0]   m_axi_mem_bid,
  input  logic [1:0]   m_axi_mem_bresp,
  input  logic         m_axi_mem_bvalid,
  output logic         m_axi_mem_bready,

  output logic         irq_o
);
  logic soft_reset_pulse;
  logic task_start_pulse;
  logic [63:0] task_base;
  logic [31:0] task_bytes;
  logic [31:0] task_tag;
  logic [31:0] watchdog_limit;
  logic core_busy;
  logic core_done;
  logic core_error;
  logic [15:0] core_error_code;
  logic [31:0] core_error_pc;
  logic [15:0] core_error_tag;
  logic [31:0] completed_tag;
  logic [31:0] commands_retired;
  logic [63:0] cycles_total;
  logic [63:0] compute_busy_cycles;
  logic [63:0] read_wait_cycles;
  logic [63:0] write_wait_cycles;
  logic [63:0] bank_stall_cycles;
  logic [63:0] bytes_read;
  logic [63:0] bytes_written;
  logic [63:0] read_highwater;
  logic [63:0] write_highwater;

  assign m_axi_mem_arid = 0;
  assign m_axi_mem_arlock = 1'b0;
  assign m_axi_mem_arcache = 4'b0011;
  assign m_axi_mem_arprot = 3'b000;
  assign m_axi_mem_arqos = 4'b0000;
  assign m_axi_mem_awid = 0;
  assign m_axi_mem_awlock = 1'b0;
  assign m_axi_mem_awcache = 4'b0011;
  assign m_axi_mem_awprot = 3'b000;
  assign m_axi_mem_awqos = 4'b0000;

  npu_csr csr (
    .clk_i(aclk), .rst_ni(aresetn), .s_axi_awaddr_i(s_axi_ctrl_awaddr),
    .s_axi_awvalid_i(s_axi_ctrl_awvalid),
    .s_axi_awready_o(s_axi_ctrl_awready), .s_axi_wdata_i(s_axi_ctrl_wdata),
    .s_axi_wstrb_i(s_axi_ctrl_wstrb), .s_axi_wvalid_i(s_axi_ctrl_wvalid),
    .s_axi_wready_o(s_axi_ctrl_wready), .s_axi_bresp_o(s_axi_ctrl_bresp),
    .s_axi_bvalid_o(s_axi_ctrl_bvalid), .s_axi_bready_i(s_axi_ctrl_bready),
    .s_axi_araddr_i(s_axi_ctrl_araddr),
    .s_axi_arvalid_i(s_axi_ctrl_arvalid),
    .s_axi_arready_o(s_axi_ctrl_arready), .s_axi_rdata_o(s_axi_ctrl_rdata),
    .s_axi_rresp_o(s_axi_ctrl_rresp), .s_axi_rvalid_o(s_axi_ctrl_rvalid),
    .s_axi_rready_i(s_axi_ctrl_rready),
    .soft_reset_pulse_o(soft_reset_pulse),
    .task_start_pulse_o(task_start_pulse), .task_base_o(task_base),
    .task_bytes_o(task_bytes), .task_tag_o(task_tag),
    .watchdog_limit_o(watchdog_limit), .irq_o(irq_o),
    .core_busy_i(core_busy), .core_done_pulse_i(core_done),
    .core_error_pulse_i(core_error), .core_error_code_i(core_error_code),
    .core_error_pc_i(core_error_pc), .core_error_tag_i(core_error_tag),
    .completed_tag_i(completed_tag), .commands_retired_i(commands_retired),
    .cycles_total_i(cycles_total),
    .compute_busy_cycles_i(compute_busy_cycles),
    .read_wait_cycles_i(read_wait_cycles),
    .write_wait_cycles_i(write_wait_cycles),
    .bank_stall_cycles_i(bank_stall_cycles), .bytes_read_i(bytes_read),
    .bytes_written_i(bytes_written), .read_highwater_i(read_highwater),
    .write_highwater_i(write_highwater));

  npu_core core (
    .clk_i(aclk), .rst_ni(aresetn), .soft_reset_i(soft_reset_pulse),
    .task_start_i(task_start_pulse), .task_base_i(task_base),
    .task_bytes_i(task_bytes), .task_tag_i(task_tag),
    .watchdog_limit_i(watchdog_limit), .m_axi_araddr_o(m_axi_mem_araddr),
    .m_axi_arlen_o(m_axi_mem_arlen), .m_axi_arsize_o(m_axi_mem_arsize),
    .m_axi_arburst_o(m_axi_mem_arburst), .m_axi_arvalid_o(m_axi_mem_arvalid),
    .m_axi_arready_i(m_axi_mem_arready), .m_axi_rdata_i(m_axi_mem_rdata),
    .m_axi_rresp_i(m_axi_mem_rresp), .m_axi_rlast_i(m_axi_mem_rlast),
    .m_axi_rvalid_i(m_axi_mem_rvalid), .m_axi_rready_o(m_axi_mem_rready),
    .m_axi_awaddr_o(m_axi_mem_awaddr), .m_axi_awlen_o(m_axi_mem_awlen),
    .m_axi_awsize_o(m_axi_mem_awsize), .m_axi_awburst_o(m_axi_mem_awburst),
    .m_axi_awvalid_o(m_axi_mem_awvalid), .m_axi_awready_i(m_axi_mem_awready),
    .m_axi_wdata_o(m_axi_mem_wdata), .m_axi_wstrb_o(m_axi_mem_wstrb),
    .m_axi_wlast_o(m_axi_mem_wlast), .m_axi_wvalid_o(m_axi_mem_wvalid),
    .m_axi_wready_i(m_axi_mem_wready), .m_axi_bresp_i(m_axi_mem_bresp),
    .m_axi_bvalid_i(m_axi_mem_bvalid), .m_axi_bready_o(m_axi_mem_bready),
    .busy_o(core_busy), .done_pulse_o(core_done),
    .error_pulse_o(core_error), .error_code_o(core_error_code),
    .error_pc_o(core_error_pc), .error_tag_o(core_error_tag),
    .completed_tag_o(completed_tag), .commands_retired_o(commands_retired),
    .cycles_total_o(cycles_total),
    .compute_busy_cycles_o(compute_busy_cycles),
    .read_wait_cycles_o(read_wait_cycles),
    .write_wait_cycles_o(write_wait_cycles),
    .bank_stall_cycles_o(bank_stall_cycles), .bytes_read_o(bytes_read),
    .bytes_written_o(bytes_written), .read_highwater_o(read_highwater),
    .write_highwater_o(write_highwater));

  // Protection fields are accepted for AXI4-Lite interface completeness.
  wire unused_ctrl_prot = ^{s_axi_ctrl_awprot, s_axi_ctrl_arprot,
                            m_axi_mem_rid, m_axi_mem_bid};
endmodule
