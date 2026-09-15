`timescale 1ns/1ps

// Two-client AXI4 write arbiter.  The owner is held for AW, every W beat, and
// B, preventing write data and responses from different clients interleaving.
module npu_axi_write_arbiter2 (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  soft_reset_i,

  input  logic [1:0][63:0]      s_axi_awaddr_i,
  input  logic [1:0][7:0]       s_axi_awlen_i,
  input  logic [1:0][2:0]       s_axi_awsize_i,
  input  logic [1:0][1:0]       s_axi_awburst_i,
  input  logic [1:0]            s_axi_awvalid_i,
  output logic [1:0]            s_axi_awready_o,
  input  logic [1:0][63:0]      s_axi_wdata_i,
  input  logic [1:0][7:0]       s_axi_wstrb_i,
  input  logic [1:0]            s_axi_wlast_i,
  input  logic [1:0]            s_axi_wvalid_i,
  output logic [1:0]            s_axi_wready_o,
  output logic [1:0][1:0]       s_axi_bresp_o,
  output logic [1:0]            s_axi_bvalid_o,
  input  logic [1:0]            s_axi_bready_i,

  output logic [63:0]           m_axi_awaddr_o,
  output logic [7:0]            m_axi_awlen_o,
  output logic [2:0]            m_axi_awsize_o,
  output logic [1:0]            m_axi_awburst_o,
  output logic                  m_axi_awvalid_o,
  input  logic                  m_axi_awready_i,
  output logic [63:0]           m_axi_wdata_o,
  output logic [7:0]            m_axi_wstrb_o,
  output logic                  m_axi_wlast_o,
  output logic                  m_axi_wvalid_o,
  input  logic                  m_axi_wready_i,
  input  logic [1:0]            m_axi_bresp_i,
  input  logic                  m_axi_bvalid_i,
  output logic                  m_axi_bready_o,
  output logic                  busy_o
);
  typedef enum logic [2:0] {WA_IDLE, WA_ADDRESS, WA_DATA, WA_RESPONSE}
    write_arb_state_e;
  write_arb_state_e state_q;
  logic owner_q;
  logic round_robin_q;
  logic selected;
  logic found;

  always_comb begin
    found = 1'b0;
    selected = round_robin_q;
    if (s_axi_awvalid_i[round_robin_q]) begin
      found = 1'b1;
      selected = round_robin_q;
    end else if (s_axi_awvalid_i[~round_robin_q]) begin
      found = 1'b1;
      selected = ~round_robin_q;
    end
  end

  always_comb begin
    s_axi_awready_o = '0;
    s_axi_wready_o = '0;
    s_axi_bresp_o = '0;
    s_axi_bvalid_o = '0;
    m_axi_awaddr_o = s_axi_awaddr_i[owner_q];
    m_axi_awlen_o = s_axi_awlen_i[owner_q];
    m_axi_awsize_o = s_axi_awsize_i[owner_q];
    m_axi_awburst_o = s_axi_awburst_i[owner_q];
    m_axi_awvalid_o = state_q == WA_ADDRESS
      && s_axi_awvalid_i[owner_q];
    m_axi_wdata_o = s_axi_wdata_i[owner_q];
    m_axi_wstrb_o = s_axi_wstrb_i[owner_q];
    m_axi_wlast_o = s_axi_wlast_i[owner_q];
    m_axi_wvalid_o = state_q == WA_DATA && s_axi_wvalid_i[owner_q];
    m_axi_bready_o = state_q == WA_RESPONSE && s_axi_bready_i[owner_q];
    if (state_q == WA_ADDRESS)
      s_axi_awready_o[owner_q] = m_axi_awready_i;
    if (state_q == WA_DATA)
      s_axi_wready_o[owner_q] = m_axi_wready_i;
    if (state_q == WA_RESPONSE) begin
      s_axi_bresp_o[owner_q] = m_axi_bresp_i;
      s_axi_bvalid_o[owner_q] = m_axi_bvalid_i;
    end
  end
  assign busy_o = state_q != WA_IDLE;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= WA_IDLE;
      owner_q <= 0;
      round_robin_q <= 0;
    end else if (soft_reset_i) begin
      state_q <= WA_IDLE;
      owner_q <= 0;
      round_robin_q <= 0;
    end else begin
      case (state_q)
        WA_IDLE: begin
          if (found) begin
            owner_q <= selected;
            state_q <= WA_ADDRESS;
          end
        end
        WA_ADDRESS: begin
          if (s_axi_awvalid_i[owner_q] && m_axi_awready_i)
            state_q <= WA_DATA;
        end
        WA_DATA: begin
          if (s_axi_wvalid_i[owner_q] && m_axi_wready_i
              && s_axi_wlast_i[owner_q])
            state_q <= WA_RESPONSE;
        end
        WA_RESPONSE: begin
          if (m_axi_bvalid_i && s_axi_bready_i[owner_q]) begin
            round_robin_q <= ~owner_q;
            state_q <= WA_IDLE;
          end
        end
        default: state_q <= WA_IDLE;
      endcase
    end
  end
endmodule
