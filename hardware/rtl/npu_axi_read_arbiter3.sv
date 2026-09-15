`timescale 1ns/1ps

// Three-client AXI4 read arbiter.  A grant is held from AR handshake through
// the accepted RLAST beat, so response data can never be routed to the wrong
// client.  The shared port permits one outstanding read burst.
module npu_axi_read_arbiter3 (
  input  logic                  clk_i,
  input  logic                  rst_ni,
  input  logic                  soft_reset_i,

  input  logic [2:0][63:0]      s_axi_araddr_i,
  input  logic [2:0][7:0]       s_axi_arlen_i,
  input  logic [2:0][2:0]       s_axi_arsize_i,
  input  logic [2:0][1:0]       s_axi_arburst_i,
  input  logic [2:0]            s_axi_arvalid_i,
  output logic [2:0]            s_axi_arready_o,
  output logic [2:0][63:0]      s_axi_rdata_o,
  output logic [2:0][1:0]       s_axi_rresp_o,
  output logic [2:0]            s_axi_rlast_o,
  output logic [2:0]            s_axi_rvalid_o,
  input  logic [2:0]            s_axi_rready_i,

  output logic [63:0]           m_axi_araddr_o,
  output logic [7:0]            m_axi_arlen_o,
  output logic [2:0]            m_axi_arsize_o,
  output logic [1:0]            m_axi_arburst_o,
  output logic                  m_axi_arvalid_o,
  input  logic                  m_axi_arready_i,
  input  logic [63:0]           m_axi_rdata_i,
  input  logic [1:0]            m_axi_rresp_i,
  input  logic                  m_axi_rlast_i,
  input  logic                  m_axi_rvalid_i,
  output logic                  m_axi_rready_o,
  output logic                  busy_o
);
  typedef enum logic [1:0] {RA_IDLE, RA_ADDRESS, RA_DATA} read_arb_state_e;
  read_arb_state_e state_q;
  logic [1:0] owner_q;
  logic [1:0] round_robin_q;
  logic [1:0] selected;
  logic found;
  integer offset;
  integer candidate;

  always_comb begin
    found = 1'b0;
    selected = round_robin_q;
    for (offset = 0; offset < 3; offset = offset + 1) begin
      candidate = round_robin_q + offset;
      if (candidate >= 3)
        candidate = candidate - 3;
      if (!found && s_axi_arvalid_i[candidate]) begin
        found = 1'b1;
        selected = candidate[1:0];
      end
    end
  end

  always_comb begin
    s_axi_arready_o = '0;
    s_axi_rdata_o = '0;
    s_axi_rresp_o = '0;
    s_axi_rlast_o = '0;
    s_axi_rvalid_o = '0;
    m_axi_araddr_o = s_axi_araddr_i[owner_q];
    m_axi_arlen_o = s_axi_arlen_i[owner_q];
    m_axi_arsize_o = s_axi_arsize_i[owner_q];
    m_axi_arburst_o = s_axi_arburst_i[owner_q];
    m_axi_arvalid_o = state_q == RA_ADDRESS
      && s_axi_arvalid_i[owner_q];
    m_axi_rready_o = state_q == RA_DATA && s_axi_rready_i[owner_q];
    if (state_q == RA_ADDRESS)
      s_axi_arready_o[owner_q] = m_axi_arready_i;
    if (state_q == RA_DATA) begin
      s_axi_rdata_o[owner_q] = m_axi_rdata_i;
      s_axi_rresp_o[owner_q] = m_axi_rresp_i;
      s_axi_rlast_o[owner_q] = m_axi_rlast_i;
      s_axi_rvalid_o[owner_q] = m_axi_rvalid_i;
    end
  end
  assign busy_o = state_q != RA_IDLE;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= RA_IDLE;
      owner_q <= 0;
      round_robin_q <= 0;
    end else if (soft_reset_i) begin
      state_q <= RA_IDLE;
      owner_q <= 0;
      round_robin_q <= 0;
    end else begin
      case (state_q)
        RA_IDLE: begin
          if (found) begin
            owner_q <= selected;
            state_q <= RA_ADDRESS;
          end
        end
        RA_ADDRESS: begin
          if (s_axi_arvalid_i[owner_q] && m_axi_arready_i) begin
            round_robin_q <= owner_q == 2 ? 0 : owner_q + 1'b1;
            state_q <= RA_DATA;
          end
        end
        RA_DATA: begin
          if (m_axi_rvalid_i && s_axi_rready_i[owner_q] && m_axi_rlast_i)
            state_q <= RA_IDLE;
        end
        default: state_q <= RA_IDLE;
      endcase
    end
  end
endmodule
