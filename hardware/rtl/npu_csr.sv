`timescale 1ns/1ps

module npu_csr (
  input  logic         clk_i,
  input  logic         rst_ni,

  input  logic [11:0]  s_axi_awaddr_i,
  input  logic         s_axi_awvalid_i,
  output logic         s_axi_awready_o,
  input  logic [31:0]  s_axi_wdata_i,
  input  logic [3:0]   s_axi_wstrb_i,
  input  logic         s_axi_wvalid_i,
  output logic         s_axi_wready_o,
  output logic [1:0]   s_axi_bresp_o,
  output logic         s_axi_bvalid_o,
  input  logic         s_axi_bready_i,
  input  logic [11:0]  s_axi_araddr_i,
  input  logic         s_axi_arvalid_i,
  output logic         s_axi_arready_o,
  output logic [31:0]  s_axi_rdata_o,
  output logic [1:0]   s_axi_rresp_o,
  output logic         s_axi_rvalid_o,
  input  logic         s_axi_rready_i,

  output logic         soft_reset_pulse_o,
  output logic         task_start_pulse_o,
  output logic [63:0]  task_base_o,
  output logic [31:0]  task_bytes_o,
  output logic [31:0]  task_tag_o,
  output logic [31:0]  watchdog_limit_o,
  output logic         irq_o,

  input  logic         core_busy_i,
  input  logic         core_done_pulse_i,
  input  logic         core_error_pulse_i,
  input  logic [15:0]  core_error_code_i,
  input  logic [31:0]  core_error_pc_i,
  input  logic [15:0]  core_error_tag_i,
  input  logic [31:0]  completed_tag_i,
  input  logic [31:0]  commands_retired_i,
  input  logic [63:0]  cycles_total_i,
  input  logic [63:0]  compute_busy_cycles_i,
  input  logic [63:0]  read_wait_cycles_i,
  input  logic [63:0]  write_wait_cycles_i,
  input  logic [63:0]  bank_stall_cycles_i,
  input  logic [63:0]  bytes_read_i,
  input  logic [63:0]  bytes_written_i,
  input  logic [63:0]  read_highwater_i,
  input  logic [63:0]  write_highwater_i
);
  import npu_isa_pkg::*;

  localparam logic [31:0] IP_ID = 32'h3155_504e; // "NPU1"
  localparam logic [31:0] VERSION = 32'h0001_0000;
  localparam logic [31:0] ISA_VERSION = 32'h0001_0000;
  localparam logic [31:0] CAPABILITY0 = 32'h0000_007f;
  localparam logic [31:0] CAPABILITY1 = 32'h0040_0400; // 64 lanes, 1024-B row
  localparam logic [15:0] SUBMISSION_BUSY_ERROR = 16'hf001;
  localparam logic [15:0] RESET_BUSY_ERROR = 16'hf002;

  logic aw_hold_q;
  logic [11:0] awaddr_q;
  logic w_hold_q;
  logic [31:0] wdata_q;
  logic [3:0] wstrb_q;
  logic bvalid_q;
  logic [1:0] bresp_q;
  logic rvalid_q;
  logic [31:0] read_data_q;
  logic [1:0] read_resp_q;
  logic [31:0] read_decode_data;
  logic [1:0] read_decode_resp;

  logic irq_global_enable_q;
  logic [2:0] irq_status_q;
  logic [2:0] irq_enable_q;
  logic done_sticky_q;
  logic error_sticky_q;
  logic [15:0] error_code_q;
  logic [31:0] error_pc_q;
  logic [15:0] error_inst_tag_q;
  logic [31:0] error_count_q;

  function automatic logic [31:0] merge_strobe(
    input logic [31:0] old_value,
    input logic [31:0] new_value,
    input logic [3:0] strobe
  );
    integer lane;
    begin
      merge_strobe = old_value;
      for (lane = 0; lane < 4; lane = lane + 1)
        if (strobe[lane])
          merge_strobe[lane*8 +: 8] = new_value[lane*8 +: 8];
    end
  endfunction

  function automatic logic write_address_valid(input logic [11:0] address);
    begin
      case (address)
        12'h014, 12'h01c, 12'h020, 12'h024, 12'h028,
        12'h02c, 12'h030, 12'h034, 12'h048: write_address_valid = 1'b1;
        default: write_address_valid = 1'b0;
      endcase
    end
  endfunction

  always_comb begin
    case (s_axi_araddr_i)
      12'h000: read_decode_data = IP_ID;
      12'h004: read_decode_data = VERSION;
      12'h008: read_decode_data = ISA_VERSION;
      12'h00c: read_decode_data = CAPABILITY0;
      12'h010: read_decode_data = CAPABILITY1;
      12'h014: read_decode_data = {30'd0, irq_global_enable_q, 1'b0};
      12'h018: read_decode_data = {27'd0, error_sticky_q, done_sticky_q,
                        1'b0, core_busy_i, !core_busy_i};
      12'h01c: read_decode_data = {29'd0, irq_status_q};
      12'h020: read_decode_data = {29'd0, irq_enable_q};
      12'h024: read_decode_data = task_base_o[31:0];
      12'h028: read_decode_data = task_base_o[63:32];
      12'h02c: read_decode_data = task_bytes_o;
      12'h030: read_decode_data = task_tag_o;
      12'h034: read_decode_data = 32'd0;
      12'h038: read_decode_data = completed_tag_i;
      12'h03c: read_decode_data = {16'd0, error_code_q};
      12'h040: read_decode_data = error_pc_q;
      12'h044: read_decode_data = {16'd0, error_inst_tag_q};
      12'h048: read_decode_data = watchdog_limit_o;
      12'h04c: read_decode_data = commands_retired_i;
      12'h050: read_decode_data = cycles_total_i[31:0];
      12'h054: read_decode_data = cycles_total_i[63:32];
      12'h058: read_decode_data = compute_busy_cycles_i[31:0];
      12'h05c: read_decode_data = compute_busy_cycles_i[63:32];
      12'h060: read_decode_data = read_wait_cycles_i[31:0];
      12'h064: read_decode_data = read_wait_cycles_i[63:32];
      12'h068: read_decode_data = write_wait_cycles_i[31:0];
      12'h06c: read_decode_data = write_wait_cycles_i[63:32];
      12'h070: read_decode_data = bank_stall_cycles_i[31:0];
      12'h074: read_decode_data = bank_stall_cycles_i[63:32];
      12'h078: read_decode_data = bytes_read_i[31:0];
      12'h07c: read_decode_data = bytes_read_i[63:32];
      12'h080: read_decode_data = bytes_written_i[31:0];
      12'h084: read_decode_data = bytes_written_i[63:32];
      12'h088: read_decode_data = read_highwater_i[31:0];
      12'h08c: read_decode_data = read_highwater_i[63:32];
      12'h090: read_decode_data = error_count_q;
      12'h094: read_decode_data = write_highwater_i[31:0];
      12'h098: read_decode_data = write_highwater_i[63:32];
      default: read_decode_data = 32'd0;
    endcase
    read_decode_resp = (s_axi_araddr_i[1:0] != 0 || s_axi_araddr_i > 12'h098)
      ? 2'b10 : 2'b00;
  end

  assign s_axi_awready_o = !aw_hold_q && !bvalid_q;
  assign s_axi_wready_o = !w_hold_q && !bvalid_q;
  assign s_axi_bvalid_o = bvalid_q;
  assign s_axi_bresp_o = bresp_q;
  assign s_axi_arready_o = !rvalid_q;
  assign s_axi_rvalid_o = rvalid_q;
  assign s_axi_rdata_o = read_data_q;
  assign s_axi_rresp_o = read_resp_q;
  assign irq_o = irq_global_enable_q && |(irq_status_q & irq_enable_q);

  logic [31:0] merged_write;
  always_comb begin
    merged_write = merge_strobe(32'd0, wdata_q, wstrb_q);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      aw_hold_q <= 1'b0;
      awaddr_q <= '0;
      w_hold_q <= 1'b0;
      wdata_q <= '0;
      wstrb_q <= '0;
      bvalid_q <= 1'b0;
      bresp_q <= 2'b00;
      rvalid_q <= 1'b0;
      read_data_q <= '0;
      read_resp_q <= 2'b00;
      task_base_o <= '0;
      task_bytes_o <= '0;
      task_tag_o <= '0;
      watchdog_limit_o <= 32'd4_000_000;
      irq_global_enable_q <= 1'b0;
      irq_status_q <= '0;
      irq_enable_q <= '0;
      done_sticky_q <= 1'b0;
      error_sticky_q <= 1'b0;
      error_code_q <= '0;
      error_pc_q <= '0;
      error_inst_tag_q <= '0;
      error_count_q <= '0;
      soft_reset_pulse_o <= 1'b0;
      task_start_pulse_o <= 1'b0;
    end else begin
      soft_reset_pulse_o <= 1'b0;
      task_start_pulse_o <= 1'b0;

      if (s_axi_awvalid_i && s_axi_awready_o) begin
        aw_hold_q <= 1'b1;
        awaddr_q <= s_axi_awaddr_i;
      end
      if (s_axi_wvalid_i && s_axi_wready_o) begin
        w_hold_q <= 1'b1;
        wdata_q <= s_axi_wdata_i;
        wstrb_q <= s_axi_wstrb_i;
      end
      if (bvalid_q && s_axi_bready_i)
        bvalid_q <= 1'b0;

      if (aw_hold_q && w_hold_q && !bvalid_q) begin
        aw_hold_q <= 1'b0;
        w_hold_q <= 1'b0;
        bvalid_q <= 1'b1;
        bresp_q <= (awaddr_q[1:0] == 0 && write_address_valid(awaddr_q))
          ? 2'b00 : 2'b10;
        if (awaddr_q[1:0] == 0 && write_address_valid(awaddr_q)) begin
          case (awaddr_q)
            12'h014: begin
              if (wstrb_q[0])
                irq_global_enable_q <= wdata_q[1];
              if (merged_write[0]) begin
                if (core_busy_i) begin
                  error_sticky_q <= 1'b1;
                  error_code_q <= RESET_BUSY_ERROR;
                  error_pc_q <= 0;
                  error_inst_tag_q <= task_tag_o[15:0];
                  error_count_q <= error_count_q + 1'b1;
                  irq_status_q[1] <= 1'b1;
                end else begin
                  soft_reset_pulse_o <= 1'b1;
                  done_sticky_q <= 1'b0;
                  error_sticky_q <= 1'b0;
                  irq_status_q <= '0;
                  error_code_q <= '0;
                  error_pc_q <= '0;
                  error_inst_tag_q <= '0;
                end
              end
            end
            12'h01c: irq_status_q <= irq_status_q
              & ~(merged_write[2:0]);
            12'h020: begin
              if (wstrb_q[0])
                irq_enable_q <= wdata_q[2:0];
            end
            12'h024: task_base_o[31:0] <= merge_strobe(
              task_base_o[31:0], wdata_q, wstrb_q);
            12'h028: task_base_o[63:32] <= merge_strobe(
              task_base_o[63:32], wdata_q, wstrb_q);
            12'h02c: task_bytes_o <= merge_strobe(
              task_bytes_o, wdata_q, wstrb_q);
            12'h030: task_tag_o <= merge_strobe(task_tag_o, wdata_q, wstrb_q);
            12'h034: begin
              if (merged_write[0]) begin
                if (core_busy_i) begin
                  error_sticky_q <= 1'b1;
                  error_code_q <= SUBMISSION_BUSY_ERROR;
                  error_pc_q <= 0;
                  error_inst_tag_q <= task_tag_o[15:0];
                  error_count_q <= error_count_q + 1'b1;
                  irq_status_q[1] <= 1'b1;
                end else begin
                  task_start_pulse_o <= 1'b1;
                  done_sticky_q <= 1'b0;
                  error_sticky_q <= 1'b0;
                  error_code_q <= 0;
                  error_pc_q <= 0;
                  error_inst_tag_q <= 0;
                  irq_status_q <= 0;
                end
              end
            end
            12'h048: watchdog_limit_o <= merge_strobe(
              watchdog_limit_o, wdata_q, wstrb_q);
            default: begin end
          endcase
        end
      end

      if (s_axi_arvalid_i && s_axi_arready_o) begin
        rvalid_q <= 1'b1;
        read_data_q <= read_decode_data;
        read_resp_q <= read_decode_resp;
      end else if (rvalid_q && s_axi_rready_i)
        rvalid_q <= 1'b0;

      if (core_done_pulse_i) begin
        done_sticky_q <= 1'b1;
        irq_status_q[0] <= 1'b1;
      end
      if (core_error_pulse_i) begin
        error_sticky_q <= 1'b1;
        error_code_q <= core_error_code_i;
        error_pc_q <= core_error_pc_i;
        error_inst_tag_q <= core_error_tag_i;
        error_count_q <= error_count_q + 1'b1;
        irq_status_q[1] <= 1'b1;
        if (core_error_code_i == NPU_ERR_WATCHDOG)
          irq_status_q[2] <= 1'b1;
      end
    end
  end
endmodule
