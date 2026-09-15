`timescale 1ns/1ps

module npu_core (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,
  input  logic         task_start_i,
  input  logic [63:0]  task_base_i,
  input  logic [31:0]  task_bytes_i,
  input  logic [31:0]  task_tag_i,
  input  logic [31:0]  watchdog_limit_i,

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
  output logic         error_pulse_o,
  output logic [15:0]  error_code_o,
  output logic [31:0]  error_pc_o,
  output logic [15:0]  error_tag_o,
  output logic [31:0]  completed_tag_o,
  output logic [31:0]  commands_retired_o,
  output logic [63:0]  cycles_total_o,
  output logic [63:0]  compute_busy_cycles_o,
  output logic [63:0]  read_wait_cycles_o,
  output logic [63:0]  write_wait_cycles_o,
  output logic [63:0]  bank_stall_cycles_o,
  output logic [63:0]  bytes_read_o,
  output logic [63:0]  bytes_written_o,
  output logic [63:0]  read_highwater_o,
  output logic [63:0]  write_highwater_o
);
  import npu_isa_pkg::*;

  typedef enum logic [2:0] {
    CORE_IDLE,
    CORE_PREPARE,
    CORE_LOAD,
    CORE_RUN,
    CORE_RECOVER
  } core_state_e;
  core_state_e state_q;
  logic [63:0] task_base_q;
  logic [31:0] task_bytes_q;
  logic [31:0] task_tag_q;
  logic unit_soft_reset;

  logic loader_start_ready;
  logic loader_busy;
  logic loader_done;
  logic loader_error;
  logic [3:0] loader_error_reason;
  logic loader_lut_valid;
  logic loader_lut_ready;
  logic [11:0] loader_lut_address;
  logic [15:0] loader_lut_data;
  logic [63:0] command_base;
  logic [31:0] command_bytes;
  logic [31:0] command_count;
  logic [63:0] tensor_desc_base;
  logic [63:0] operator_desc_base;
  logic [63:0] quant_desc_base;
  logic [63:0] segment_desc_base;
  logic [63:0] activation_base;
  logic [63:0] weight_base;
  logic [63:0] bias_base;
  logic [63:0] quant_param_base;
  logic [15:0] input_tensor;
  logic [15:0] output_tensor;

  logic fetch_start_ready;
  logic fetch_raw_command_valid;
  logic fetch_raw_command_ready;
  logic [127:0] fetch_raw_command_bits;
  logic [31:0] fetch_raw_command_pc;
  logic fetch_command_valid;
  logic fetch_command_ready;
  logic [127:0] fetch_command_bits;
  logic [31:0] fetch_command_pc;
  logic fetch_busy;
  logic fetch_error;
  logic [3:0] fetch_error_reason;
  logic [31:0] fetch_error_pc;
  logic [15:0] fetch_error_tag;

  logic cp_dma_valid;
  logic cp_dma_ready;
  logic [127:0] cp_dma_bits;
  logic cp_compute_valid;
  logic cp_compute_ready;
  logic [127:0] cp_compute_bits;
  logic cp_vector_valid;
  logic cp_vector_ready;
  logic [127:0] cp_vector_bits;
  logic cp_busy;
  logic cp_done;
  logic cp_irq;
  logic cp_error;
  logic [15:0] cp_error_code;
  logic [31:0] cp_error_pc;
  logic [15:0] cp_error_tag;
  logic [31:0] cp_pc;
  logic [63:0] cp_cycles;
  logic [7:0] cp_events;
  logic [7:0] event_set;
  logic units_idle;
  logic vector_sync_done;

  logic exec_busy;
  logic exec_error;
  logic [15:0] exec_error_code;
  logic [31:0] exec_error_pc;
  logic [15:0] exec_error_tag;
  logic exec_conv_start_valid;
  logic exec_conv_start_ready;
  logic [127:0] exec_conv_command;
  logic [511:0] exec_conv_desc;
  logic exec_conv_activation_bank;
  logic exec_conv_weight_bank;
  logic exec_conv_output_bank;
  logic [7:0] exec_conv_event;
  logic [31:0] exec_conv_pc;
  logic [15:0] exec_conv_tag;
  logic exec_vec_start_valid;
  logic exec_vec_start_ready;
  logic [127:0] exec_vec_command;
  logic [511:0] exec_vec_desc;
  logic exec_up_start_valid;
  logic exec_up_start_ready;
  logic [127:0] exec_up_command;
  logic [511:0] exec_up_source_desc;
  logic [511:0] exec_up_destination_desc;

  logic dma_busy;
  logic [7:0] dma_event_set;
  logic dma_error;
  logic [1:0] dma_error_source;
  logic [3:0] dma_error_reason;
  logic [3:0] dma_error_detail;
  logic [31:0] dma_error_pc;
  logic [15:0] dma_error_tag;
  logic [31:0] dma_bytes_read;
  logic [31:0] dma_bytes_written;
  logic bank_collision_stall;
  logic conv_busy;
  logic conv_done;
  logic [7:0] conv_event_set;
  logic conv_error;
  logic [3:0] conv_error_reason;

  logic vec_busy;
  logic vec_done;
  logic vec_error;
  logic [3:0] vec_error_reason;
  logic vec_read_request_valid;
  logic vec_read_request_ready;
  logic [1:0] vec_read_kind;
  logic vec_read_bank;
  logic [31:0] vec_read_address;
  logic vec_read_response_valid;
  logic vec_read_response_ready;
  logic [511:0] vec_read_data;
  logic vec_write_valid;
  logic vec_write_ready;
  logic [1:0] vec_write_kind;
  logic vec_write_bank;
  logic [31:0] vec_write_address;
  logic [511:0] vec_write_data;
  logic [63:0] vec_write_strobe;

  logic up_busy;
  logic up_done;
  logic up_error;
  logic [3:0] up_error_reason;
  logic [31:0] up_bytes_read;
  logic [31:0] up_bytes_written;

  logic execution_error;
  logic [15:0] execution_error_code;
  logic [31:0] execution_error_pc;
  logic [15:0] execution_error_tag;

  logic [3:0] memory_request_valid;
  logic [3:0] memory_request_ready;
  logic [3:0][63:0] memory_request_address;
  logic [3:0][6:0] memory_request_bytes;
  logic [3:0] memory_response_valid;
  logic [3:0] memory_response_ready;
  logic [3:0][511:0] memory_response_data;
  logic [3:0] memory_response_error;
  logic block_request_valid;
  logic block_request_ready;
  logic [63:0] block_request_address;
  logic [6:0] block_request_bytes;
  logic block_response_valid;
  logic block_response_ready;
  logic [511:0] block_response_data;
  logic block_response_error;

  logic [2:0][63:0] read_araddr;
  logic [2:0][7:0] read_arlen;
  logic [2:0][2:0] read_arsize;
  logic [2:0][1:0] read_arburst;
  logic [2:0] read_arvalid;
  logic [2:0] read_arready;
  logic [2:0][63:0] read_rdata;
  logic [2:0][1:0] read_rresp;
  logic [2:0] read_rlast;
  logic [2:0] read_rvalid;
  logic [2:0] read_rready;
  logic read_arb_busy;

  logic [1:0][63:0] write_awaddr;
  logic [1:0][7:0] write_awlen;
  logic [1:0][2:0] write_awsize;
  logic [1:0][1:0] write_awburst;
  logic [1:0] write_awvalid;
  logic [1:0] write_awready;
  logic [1:0][63:0] write_wdata;
  logic [1:0][7:0] write_wstrb;
  logic [1:0] write_wlast;
  logic [1:0] write_wvalid;
  logic [1:0] write_wready;
  logic [1:0][1:0] write_bresp;
  logic [1:0] write_bvalid;
  logic [1:0] write_bready;
  logic write_arb_busy;

  logic [3:0] read_beat_bytes_q;
  integer strobe_lane;
  logic [3:0] write_accepted_bytes;
  always_comb begin
    write_accepted_bytes = 0;
    for (strobe_lane = 0; strobe_lane < 8; strobe_lane = strobe_lane + 1)
      write_accepted_bytes = write_accepted_bytes + m_axi_wstrb_o[strobe_lane];
  end

  assign unit_soft_reset = soft_reset_i || state_q == CORE_PREPARE
    || state_q == CORE_RECOVER;
  assign busy_o = state_q != CORE_IDLE;
  assign units_idle = !dma_busy && !conv_busy && !exec_busy
    && !vec_busy && !up_busy && !write_arb_busy;
  assign event_set = dma_event_set | conv_event_set;

  // A one-entry, fully registered command stage breaks the long combinational
  // ready path from an execution unit through decode back into the fetch PC.
  // Fetch is much slower than one command/cycle, so disallowing fall-through
  // does not reduce task throughput.
  assign fetch_raw_command_ready = !fetch_command_valid;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      fetch_command_valid <= 1'b0;
      fetch_command_bits <= '0;
      fetch_command_pc <= '0;
    end else if (unit_soft_reset) begin
      fetch_command_valid <= 1'b0;
      fetch_command_bits <= '0;
      fetch_command_pc <= '0;
    end else begin
      if (fetch_command_valid && fetch_command_ready)
        fetch_command_valid <= 1'b0;
      if (fetch_raw_command_valid && fetch_raw_command_ready) begin
        fetch_command_valid <= 1'b1;
        fetch_command_bits <= fetch_raw_command_bits;
        fetch_command_pc <= fetch_raw_command_pc;
      end
    end
  end

  always_comb begin
    execution_error = 1'b0;
    execution_error_code = NPU_ERR_NONE;
    execution_error_pc = cp_pc;
    execution_error_tag = 0;
    if (fetch_error) begin
      execution_error = 1'b1;
      execution_error_code = 16'h1100 | {12'd0, fetch_error_reason};
      execution_error_pc = fetch_error_pc;
      execution_error_tag = fetch_error_tag;
    end else if (exec_error) begin
      execution_error = 1'b1;
      execution_error_code = exec_error_code;
      execution_error_pc = exec_error_pc;
      execution_error_tag = exec_error_tag;
    end else if (dma_error) begin
      execution_error = 1'b1;
      execution_error_code = 16'h2000 | {6'd0, dma_error_source,
                                        dma_error_reason, dma_error_detail};
      execution_error_pc = dma_error_pc;
      execution_error_tag = dma_error_tag;
    end else if (conv_error) begin
      execution_error = 1'b1;
      execution_error_code = 16'h3000 | {12'd0, conv_error_reason};
      execution_error_pc = exec_conv_pc;
      execution_error_tag = exec_conv_tag;
    end
  end

  npu_task_loader loader (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .start_valid_i(state_q == CORE_LOAD), .start_ready_o(loader_start_ready),
    .task_base_i(task_base_q), .task_bytes_i(task_bytes_q),
    .memory_request_valid_o(memory_request_valid[0]),
    .memory_request_ready_i(memory_request_ready[0]),
    .memory_request_address_o(memory_request_address[0]),
    .memory_request_bytes_o(memory_request_bytes[0]),
    .memory_response_valid_i(memory_response_valid[0]),
    .memory_response_ready_o(memory_response_ready[0]),
    .memory_response_data_i(memory_response_data[0]),
    .memory_response_error_i(memory_response_error[0]),
    .lut_write_valid_o(loader_lut_valid),
    .lut_write_ready_i(loader_lut_ready),
    .lut_write_address_o(loader_lut_address),
    .lut_write_data_o(loader_lut_data), .busy_o(loader_busy),
    .done_pulse_o(loader_done), .error_pulse_o(loader_error),
    .error_reason_o(loader_error_reason), .command_base_o(command_base),
    .command_bytes_o(command_bytes), .command_count_o(command_count),
    .tensor_desc_base_o(tensor_desc_base),
    .operator_desc_base_o(operator_desc_base),
    .quant_desc_base_o(quant_desc_base),
    .segment_desc_base_o(segment_desc_base),
    .activation_base_o(activation_base), .weight_base_o(weight_base),
    .bias_base_o(bias_base), .quant_param_base_o(quant_param_base),
    .input_tensor_o(input_tensor), .output_tensor_o(output_tensor));

  npu_command_fetch command_fetch (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .start_valid_i(loader_done), .start_ready_o(fetch_start_ready),
    .command_base_i(command_base), .command_bytes_i(command_bytes),
    .memory_request_valid_o(memory_request_valid[1]),
    .memory_request_ready_i(memory_request_ready[1]),
    .memory_request_address_o(memory_request_address[1]),
    .memory_request_bytes_o(memory_request_bytes[1]),
    .memory_response_valid_i(memory_response_valid[1]),
    .memory_response_ready_o(memory_response_ready[1]),
    .memory_response_data_i(memory_response_data[1]),
    .memory_response_error_i(memory_response_error[1]),
    .command_valid_o(fetch_raw_command_valid),
    .command_ready_i(fetch_raw_command_ready),
    .command_bits_o(fetch_raw_command_bits),
    .command_pc_o(fetch_raw_command_pc),
    .busy_o(fetch_busy), .error_pulse_o(fetch_error),
    .error_reason_o(fetch_error_reason), .error_pc_o(fetch_error_pc),
    .error_tag_o(fetch_error_tag));

  npu_command_processor command_processor (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .start_i(loader_done), .command_valid_i(fetch_command_valid),
    .command_ready_o(fetch_command_ready), .command_bits_i(fetch_command_bits),
    .dma_command_valid_o(cp_dma_valid), .dma_command_ready_i(cp_dma_ready),
    .dma_command_bits_o(cp_dma_bits),
    .compute_command_valid_o(cp_compute_valid),
    .compute_command_ready_i(cp_compute_ready),
    .compute_command_bits_o(cp_compute_bits),
    .vector_command_valid_o(cp_vector_valid),
    .vector_command_ready_i(cp_vector_ready),
    .vector_command_bits_o(cp_vector_bits), .event_set_i(event_set),
    .vector_sync_done_i(vector_sync_done), .units_idle_i(units_idle),
    .watchdog_limit_i(watchdog_limit_i),
    .execution_error_i(execution_error),
    .execution_error_code_i(execution_error_code),
    .execution_error_pc_i(execution_error_pc),
    .execution_error_tag_i(execution_error_tag), .busy_o(cp_busy),
    .done_pulse_o(cp_done), .irq_pulse_o(cp_irq), .error_o(cp_error),
    .error_code_o(cp_error_code), .error_pc_o(cp_error_pc),
    .error_inst_tag_o(cp_error_tag), .command_pc_o(cp_pc),
    .commands_retired_o(commands_retired_o), .cycles_total_o(cp_cycles),
    .event_state_o(cp_events));

  npu_execution_frontend execution_frontend (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .compute_command_valid_i(cp_compute_valid),
    .compute_command_ready_o(cp_compute_ready),
    .compute_command_bits_i(cp_compute_bits),
    .compute_command_pc_i(fetch_command_pc),
    .vector_command_valid_i(cp_vector_valid),
    .vector_command_ready_o(cp_vector_ready),
    .vector_command_bits_i(cp_vector_bits),
    .vector_command_pc_i(fetch_command_pc),
    .tensor_desc_base_i(tensor_desc_base),
    .operator_desc_base_i(operator_desc_base),
    .quant_desc_base_i(quant_desc_base), .segment_desc_base_i(segment_desc_base),
    .memory_request_valid_o(memory_request_valid[3]),
    .memory_request_ready_i(memory_request_ready[3]),
    .memory_request_address_o(memory_request_address[3]),
    .memory_request_bytes_o(memory_request_bytes[3]),
    .memory_response_valid_i(memory_response_valid[3]),
    .memory_response_ready_o(memory_response_ready[3]),
    .memory_response_data_i(memory_response_data[3]),
    .memory_response_error_i(memory_response_error[3]),
    .conv_start_valid_o(exec_conv_start_valid),
    .conv_start_ready_i(exec_conv_start_ready),
    .conv_command_bits_o(exec_conv_command),
    .conv_operator_desc_bits_o(exec_conv_desc),
    .conv_activation_bank_o(exec_conv_activation_bank),
    .conv_weight_bank_o(exec_conv_weight_bank),
    .conv_output_bank_o(exec_conv_output_bank),
    .conv_completion_event_o(exec_conv_event),
    .conv_command_pc_o(exec_conv_pc), .conv_command_tag_o(exec_conv_tag),
    .vec_start_valid_o(exec_vec_start_valid),
    .vec_start_ready_i(exec_vec_start_ready),
    .vec_command_bits_o(exec_vec_command),
    .vec_operator_desc_bits_o(exec_vec_desc), .vec_done_pulse_i(vec_done),
    .vec_error_pulse_i(vec_error), .vec_error_reason_i(vec_error_reason),
    .upsample_start_valid_o(exec_up_start_valid),
    .upsample_start_ready_i(exec_up_start_ready),
    .upsample_command_bits_o(exec_up_command),
    .upsample_source_desc_bits_o(exec_up_source_desc),
    .upsample_destination_desc_bits_o(exec_up_destination_desc),
    .upsample_done_pulse_i(up_done), .upsample_error_pulse_i(up_error),
    .upsample_error_reason_i(up_error_reason),
    .vector_sync_done_o(vector_sync_done), .busy_o(exec_busy),
    .error_pulse_o(exec_error), .error_code_o(exec_error_code),
    .error_pc_o(exec_error_pc), .error_tag_o(exec_error_tag));

  npu_dma_subsystem data_subsystem (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .command_valid_i(cp_dma_valid), .command_ready_o(cp_dma_ready),
    .command_bits_i(cp_dma_bits), .command_pc_i(fetch_command_pc),
    .tensor_desc_base_i(tensor_desc_base),
    .operator_desc_base_i(operator_desc_base),
    .quant_desc_base_i(quant_desc_base),
    .segment_desc_base_i(segment_desc_base),
    .activation_base_i(activation_base), .weight_base_i(weight_base),
    .bias_base_i(bias_base), .quant_param_base_i(quant_param_base),
    .descriptor_memory_request_valid_o(memory_request_valid[2]),
    .descriptor_memory_request_ready_i(memory_request_ready[2]),
    .descriptor_memory_request_address_o(memory_request_address[2]),
    .descriptor_memory_request_bytes_o(memory_request_bytes[2]),
    .descriptor_memory_response_valid_i(memory_response_valid[2]),
    .descriptor_memory_response_ready_o(memory_response_ready[2]),
    .descriptor_memory_response_data_i(memory_response_data[2]),
    .descriptor_memory_response_error_i(memory_response_error[2]),
    .m_axi_araddr_o(read_araddr[1]), .m_axi_arlen_o(read_arlen[1]),
    .m_axi_arsize_o(read_arsize[1]), .m_axi_arburst_o(read_arburst[1]),
    .m_axi_arvalid_o(read_arvalid[1]), .m_axi_arready_i(read_arready[1]),
    .m_axi_rdata_i(read_rdata[1]), .m_axi_rresp_i(read_rresp[1]),
    .m_axi_rlast_i(read_rlast[1]), .m_axi_rvalid_i(read_rvalid[1]),
    .m_axi_rready_o(read_rready[1]), .m_axi_awaddr_o(write_awaddr[0]),
    .m_axi_awlen_o(write_awlen[0]), .m_axi_awsize_o(write_awsize[0]),
    .m_axi_awburst_o(write_awburst[0]), .m_axi_awvalid_o(write_awvalid[0]),
    .m_axi_awready_i(write_awready[0]), .m_axi_wdata_o(write_wdata[0]),
    .m_axi_wstrb_o(write_wstrb[0]), .m_axi_wlast_o(write_wlast[0]),
    .m_axi_wvalid_o(write_wvalid[0]), .m_axi_wready_i(write_wready[0]),
    .m_axi_bresp_i(write_bresp[0]), .m_axi_bvalid_i(write_bvalid[0]),
    .m_axi_bready_o(write_bready[0]),
    .accelerator_write_valid_i(vec_write_valid),
    .accelerator_write_ready_o(vec_write_ready),
    .accelerator_write_kind_i(vec_write_kind),
    .accelerator_write_bank_i(vec_write_bank),
    .accelerator_write_address_i(vec_write_address),
    .accelerator_write_data_i(vec_write_data),
    .accelerator_write_strobe_i(vec_write_strobe),
    .accelerator_read_request_valid_i(vec_read_request_valid),
    .accelerator_read_request_ready_o(vec_read_request_ready),
    .accelerator_read_kind_i(vec_read_kind),
    .accelerator_read_bank_i(vec_read_bank),
    .accelerator_read_address_i(vec_read_address),
    .accelerator_read_response_valid_o(vec_read_response_valid),
    .accelerator_read_response_ready_i(vec_read_response_ready),
    .accelerator_read_response_data_o(vec_read_data),
    .conv_start_valid_i(exec_conv_start_valid),
    .conv_start_ready_o(exec_conv_start_ready),
    .conv_operator_desc_bits_i(exec_conv_desc),
    .conv_activation_bank_i(exec_conv_activation_bank),
    .conv_weight_bank_i(exec_conv_weight_bank),
    .conv_output_bank_i(exec_conv_output_bank),
    .conv_completion_event_i(exec_conv_event),
    .conv_lut_write_valid_i(loader_lut_valid),
    .conv_lut_write_ready_o(loader_lut_ready),
    .conv_lut_write_address_i(loader_lut_address),
    .conv_lut_write_data_i(loader_lut_data), .conv_busy_o(conv_busy),
    .conv_done_pulse_o(conv_done), .conv_event_set_o(conv_event_set),
    .conv_error_pulse_o(conv_error), .conv_error_reason_o(conv_error_reason),
    .busy_o(dma_busy), .event_set_o(dma_event_set),
    .error_pulse_o(dma_error), .error_source_o(dma_error_source),
    .error_reason_o(dma_error_reason), .error_detail_o(dma_error_detail),
    .error_pc_o(dma_error_pc), .error_tag_o(dma_error_tag),
    .bytes_read_o(dma_bytes_read), .bytes_written_o(dma_bytes_written),
    .scratchpad_collision_stall_o(bank_collision_stall));

  npu_vec_add vec_add (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .start_valid_i(exec_vec_start_valid), .start_ready_o(exec_vec_start_ready),
    .command_bits_i(exec_vec_command), .operator_desc_bits_i(exec_vec_desc),
    .read_request_valid_o(vec_read_request_valid),
    .read_request_ready_i(vec_read_request_ready), .read_kind_o(vec_read_kind),
    .read_bank_o(vec_read_bank), .read_address_o(vec_read_address),
    .read_response_valid_i(vec_read_response_valid),
    .read_response_ready_o(vec_read_response_ready), .read_data_i(vec_read_data),
    .write_valid_o(vec_write_valid), .write_ready_i(vec_write_ready),
    .write_kind_o(vec_write_kind), .write_bank_o(vec_write_bank),
    .write_address_o(vec_write_address), .write_data_o(vec_write_data),
    .write_strobe_o(vec_write_strobe), .busy_o(vec_busy),
    .done_pulse_o(vec_done), .error_pulse_o(vec_error),
    .error_reason_o(vec_error_reason));

  npu_upsample2x upsample (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .start_valid_i(exec_up_start_valid), .start_ready_o(exec_up_start_ready),
    .command_bits_i(exec_up_command),
    .source_desc_bits_i(exec_up_source_desc),
    .destination_desc_bits_i(exec_up_destination_desc),
    .activation_base_i(activation_base), .m_axi_araddr_o(read_araddr[2]),
    .m_axi_arlen_o(read_arlen[2]), .m_axi_arsize_o(read_arsize[2]),
    .m_axi_arburst_o(read_arburst[2]), .m_axi_arvalid_o(read_arvalid[2]),
    .m_axi_arready_i(read_arready[2]), .m_axi_rdata_i(read_rdata[2]),
    .m_axi_rresp_i(read_rresp[2]), .m_axi_rlast_i(read_rlast[2]),
    .m_axi_rvalid_i(read_rvalid[2]), .m_axi_rready_o(read_rready[2]),
    .m_axi_awaddr_o(write_awaddr[1]), .m_axi_awlen_o(write_awlen[1]),
    .m_axi_awsize_o(write_awsize[1]), .m_axi_awburst_o(write_awburst[1]),
    .m_axi_awvalid_o(write_awvalid[1]), .m_axi_awready_i(write_awready[1]),
    .m_axi_wdata_o(write_wdata[1]), .m_axi_wstrb_o(write_wstrb[1]),
    .m_axi_wlast_o(write_wlast[1]), .m_axi_wvalid_o(write_wvalid[1]),
    .m_axi_wready_i(write_wready[1]), .m_axi_bresp_i(write_bresp[1]),
    .m_axi_bvalid_i(write_bvalid[1]), .m_axi_bready_o(write_bready[1]),
    .busy_o(up_busy), .done_pulse_o(up_done), .error_pulse_o(up_error),
    .error_reason_o(up_error_reason), .bytes_read_o(up_bytes_read),
    .bytes_written_o(up_bytes_written));

  npu_memory_arbiter4 memory_arbiter (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .request_valid_i(memory_request_valid),
    .request_ready_o(memory_request_ready),
    .request_address_i(memory_request_address),
    .request_bytes_i(memory_request_bytes),
    .response_valid_o(memory_response_valid),
    .response_ready_i(memory_response_ready),
    .response_data_o(memory_response_data),
    .response_error_o(memory_response_error),
    .memory_request_valid_o(block_request_valid),
    .memory_request_ready_i(block_request_ready),
    .memory_request_address_o(block_request_address),
    .memory_request_bytes_o(block_request_bytes),
    .memory_response_valid_i(block_response_valid),
    .memory_response_ready_o(block_response_ready),
    .memory_response_data_i(block_response_data),
    .memory_response_error_i(block_response_error));

  npu_axi_block_reader block_reader (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .request_valid_i(block_request_valid), .request_ready_o(block_request_ready),
    .request_address_i(block_request_address),
    .request_bytes_i(block_request_bytes), .response_valid_o(block_response_valid),
    .response_ready_i(block_response_ready), .response_data_o(block_response_data),
    .response_error_o(block_response_error), .m_axi_araddr_o(read_araddr[0]),
    .m_axi_arlen_o(read_arlen[0]), .m_axi_arsize_o(read_arsize[0]),
    .m_axi_arburst_o(read_arburst[0]), .m_axi_arvalid_o(read_arvalid[0]),
    .m_axi_arready_i(read_arready[0]), .m_axi_rdata_i(read_rdata[0]),
    .m_axi_rresp_i(read_rresp[0]), .m_axi_rlast_i(read_rlast[0]),
    .m_axi_rvalid_i(read_rvalid[0]), .m_axi_rready_o(read_rready[0]));

  npu_axi_read_arbiter3 read_arbiter (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .s_axi_araddr_i(read_araddr), .s_axi_arlen_i(read_arlen),
    .s_axi_arsize_i(read_arsize), .s_axi_arburst_i(read_arburst),
    .s_axi_arvalid_i(read_arvalid), .s_axi_arready_o(read_arready),
    .s_axi_rdata_o(read_rdata), .s_axi_rresp_o(read_rresp),
    .s_axi_rlast_o(read_rlast), .s_axi_rvalid_o(read_rvalid),
    .s_axi_rready_i(read_rready), .m_axi_araddr_o(m_axi_araddr_o),
    .m_axi_arlen_o(m_axi_arlen_o), .m_axi_arsize_o(m_axi_arsize_o),
    .m_axi_arburst_o(m_axi_arburst_o), .m_axi_arvalid_o(m_axi_arvalid_o),
    .m_axi_arready_i(m_axi_arready_i), .m_axi_rdata_i(m_axi_rdata_i),
    .m_axi_rresp_i(m_axi_rresp_i), .m_axi_rlast_i(m_axi_rlast_i),
    .m_axi_rvalid_i(m_axi_rvalid_i), .m_axi_rready_o(m_axi_rready_o),
    .busy_o(read_arb_busy));

  npu_axi_write_arbiter2 write_arbiter (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(unit_soft_reset),
    .s_axi_awaddr_i(write_awaddr), .s_axi_awlen_i(write_awlen),
    .s_axi_awsize_i(write_awsize), .s_axi_awburst_i(write_awburst),
    .s_axi_awvalid_i(write_awvalid), .s_axi_awready_o(write_awready),
    .s_axi_wdata_i(write_wdata), .s_axi_wstrb_i(write_wstrb),
    .s_axi_wlast_i(write_wlast), .s_axi_wvalid_i(write_wvalid),
    .s_axi_wready_o(write_wready), .s_axi_bresp_o(write_bresp),
    .s_axi_bvalid_o(write_bvalid), .s_axi_bready_i(write_bready),
    .m_axi_awaddr_o(m_axi_awaddr_o), .m_axi_awlen_o(m_axi_awlen_o),
    .m_axi_awsize_o(m_axi_awsize_o), .m_axi_awburst_o(m_axi_awburst_o),
    .m_axi_awvalid_o(m_axi_awvalid_o), .m_axi_awready_i(m_axi_awready_i),
    .m_axi_wdata_o(m_axi_wdata_o), .m_axi_wstrb_o(m_axi_wstrb_o),
    .m_axi_wlast_o(m_axi_wlast_o), .m_axi_wvalid_o(m_axi_wvalid_o),
    .m_axi_wready_i(m_axi_wready_i), .m_axi_bresp_i(m_axi_bresp_i),
    .m_axi_bvalid_i(m_axi_bvalid_i), .m_axi_bready_o(m_axi_bready_o),
    .busy_o(write_arb_busy));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= CORE_IDLE;
      task_base_q <= '0;
      task_bytes_q <= '0;
      task_tag_q <= '0;
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      error_code_o <= '0;
      error_pc_o <= '0;
      error_tag_o <= '0;
      completed_tag_o <= '0;
      cycles_total_o <= '0;
      compute_busy_cycles_o <= '0;
      read_wait_cycles_o <= '0;
      write_wait_cycles_o <= '0;
      bank_stall_cycles_o <= '0;
      bytes_read_o <= '0;
      bytes_written_o <= '0;
      read_highwater_o <= '0;
      write_highwater_o <= '0;
      read_beat_bytes_q <= 0;
    end else if (soft_reset_i) begin
      state_q <= CORE_IDLE;
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      error_code_o <= '0;
      error_pc_o <= '0;
      error_tag_o <= '0;
    end else begin
      done_pulse_o <= 1'b0;
      error_pulse_o <= 1'b0;
      if (state_q != CORE_IDLE)
        cycles_total_o <= cycles_total_o + 1'b1;
      if (state_q == CORE_RUN && (conv_busy || vec_busy || up_busy))
        compute_busy_cycles_o <= compute_busy_cycles_o + 1'b1;
      if ((m_axi_arvalid_o && !m_axi_arready_i)
          || (m_axi_rvalid_i && !m_axi_rready_o))
        read_wait_cycles_o <= read_wait_cycles_o + 1'b1;
      if ((m_axi_awvalid_o && !m_axi_awready_i)
          || (m_axi_wvalid_o && !m_axi_wready_i)
          || (m_axi_bvalid_i && !m_axi_bready_o))
        write_wait_cycles_o <= write_wait_cycles_o + 1'b1;
      if (bank_collision_stall)
        bank_stall_cycles_o <= bank_stall_cycles_o + 1'b1;
      if (m_axi_arvalid_o && m_axi_arready_i) begin
        read_beat_bytes_q <= 4'b0001 << m_axi_arsize_o;
        read_highwater_o <= 1;
      end
      if (m_axi_rvalid_i && m_axi_rready_o)
        bytes_read_o <= bytes_read_o + read_beat_bytes_q;
      if (m_axi_awvalid_o && m_axi_awready_i)
        write_highwater_o <= 1;
      if (m_axi_wvalid_o && m_axi_wready_i)
        bytes_written_o <= bytes_written_o + write_accepted_bytes;

      case (state_q)
        CORE_IDLE: begin
          if (task_start_i) begin
            task_base_q <= task_base_i;
            task_bytes_q <= task_bytes_i;
            task_tag_q <= task_tag_i;
            error_code_o <= 0;
            error_pc_o <= 0;
            error_tag_o <= 0;
            cycles_total_o <= 0;
            compute_busy_cycles_o <= 0;
            read_wait_cycles_o <= 0;
            write_wait_cycles_o <= 0;
            bank_stall_cycles_o <= 0;
            bytes_read_o <= 0;
            bytes_written_o <= 0;
            read_highwater_o <= 0;
            write_highwater_o <= 0;
            state_q <= CORE_PREPARE;
          end
        end
        CORE_PREPARE: state_q <= CORE_LOAD;
        CORE_LOAD: begin
          if (loader_error) begin
            error_code_o <= 16'h1000 | {12'd0, loader_error_reason};
            error_pc_o <= 0;
            error_tag_o <= task_tag_q[15:0];
            error_pulse_o <= 1'b1;
            state_q <= CORE_RECOVER;
          end else if (loader_done) begin
            state_q <= CORE_RUN;
          end
        end
        CORE_RUN: begin
          if (cp_error) begin
            error_code_o <= cp_error_code;
            error_pc_o <= cp_error_pc;
            error_tag_o <= cp_error_tag;
            error_pulse_o <= 1'b1;
            state_q <= CORE_RECOVER;
          end else if (cp_done) begin
            completed_tag_o <= task_tag_q;
            done_pulse_o <= 1'b1;
            state_q <= CORE_IDLE;
          end
        end
        CORE_RECOVER: state_q <= CORE_IDLE;
        default: state_q <= CORE_IDLE;
      endcase
    end
  end
endmodule
