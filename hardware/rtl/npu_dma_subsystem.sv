`timescale 1ns/1ps

module npu_dma_subsystem (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         command_valid_i,
  output logic         command_ready_o,
  input  logic [127:0] command_bits_i,
  input  logic [31:0]  command_pc_i,

  input  logic [63:0]  tensor_desc_base_i,
  input  logic [63:0]  operator_desc_base_i,
  input  logic [63:0]  quant_desc_base_i,
  input  logic [63:0]  segment_desc_base_i,
  input  logic [63:0]  activation_base_i,
  input  logic [63:0]  weight_base_i,
  input  logic [63:0]  bias_base_i,
  input  logic [63:0]  quant_param_base_i,

  output logic         descriptor_memory_request_valid_o,
  input  logic         descriptor_memory_request_ready_i,
  output logic [63:0]  descriptor_memory_request_address_o,
  output logic [6:0]   descriptor_memory_request_bytes_o,
  input  logic         descriptor_memory_response_valid_i,
  output logic         descriptor_memory_response_ready_o,
  input  logic [511:0] descriptor_memory_response_data_i,
  input  logic         descriptor_memory_response_error_i,

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

  input  logic         accelerator_write_valid_i,
  output logic         accelerator_write_ready_o,
  input  logic [1:0]   accelerator_write_kind_i,
  input  logic         accelerator_write_bank_i,
  input  logic [31:0]  accelerator_write_address_i,
  input  logic [511:0] accelerator_write_data_i,
  input  logic [63:0]  accelerator_write_strobe_i,
  input  logic         accelerator_read_request_valid_i,
  output logic         accelerator_read_request_ready_o,
  input  logic [1:0]   accelerator_read_kind_i,
  input  logic         accelerator_read_bank_i,
  input  logic [31:0]  accelerator_read_address_i,
  output logic         accelerator_read_response_valid_o,
  input  logic         accelerator_read_response_ready_i,
  output logic [511:0] accelerator_read_response_data_o,

  input  logic         conv_start_valid_i,
  output logic         conv_start_ready_o,
  input  logic [511:0] conv_operator_desc_bits_i,
  input  logic         conv_activation_bank_i,
  input  logic         conv_weight_bank_i,
  input  logic         conv_output_bank_i,
  input  logic [7:0]   conv_completion_event_i,
  output logic         conv_busy_o,
  output logic         conv_done_pulse_o,
  output logic [7:0]   conv_event_set_o,
  output logic         conv_error_pulse_o,
  output logic [3:0]   conv_error_reason_o,

  output logic         busy_o,
  output logic [7:0]   event_set_o,
  output logic         error_pulse_o,
  output logic [1:0]   error_source_o,
  output logic [3:0]   error_reason_o,
  output logic [3:0]   error_detail_o,
  output logic [31:0]  error_pc_o,
  output logic [15:0]  error_tag_o,
  output logic [31:0]  bytes_read_o,
  output logic [31:0]  bytes_written_o,
  output logic         scratchpad_collision_stall_o
);
  import npu_dma_pkg::*;

  logic frontend_busy;
  logic frontend_command_ready;
  logic frontend_request_valid;
  logic frontend_request_ready;
  logic [NPU_DMA_REQUEST_BITS-1:0] frontend_request_bits;
  logic frontend_error;
  logic [3:0] frontend_error_reason;
  logic [3:0] frontend_error_detail;
  logic [15:0] frontend_error_tag;
  logic engine_busy;
  logic engine_error;
  logic [3:0] engine_error_reason;
  logic [15:0] engine_error_tag;
  logic [31:0] frontend_pc_q;
  logic [31:0] engine_pc_q;
  logic internal_soft_reset;

  logic spad_write_valid;
  logic spad_write_ready;
  logic [1:0] spad_write_kind;
  logic spad_write_bank;
  logic [31:0] spad_write_address;
  logic [63:0] spad_write_data;
  logic [7:0] spad_write_strobe;
  logic spad_read_request_valid;
  logic spad_read_request_ready;
  logic [1:0] spad_read_kind;
  logic spad_read_bank;
  logic [31:0] spad_read_address;
  logic spad_read_response_valid;
  logic spad_read_response_ready;
  logic [63:0] spad_read_response_data;
  logic conv_read_request_valid;
  logic conv_read_request_ready;
  logic conv_activation_read_bank;
  logic [31:0] conv_activation_read_address;
  logic conv_weight_read_bank;
  logic [31:0] conv_weight_read_address;
  logic conv_read_response_valid;
  logic conv_read_response_ready;
  logic [127:0] conv_activation_read_data;
  logic [511:0] conv_weight_read_data;
  logic conv_result_valid;
  logic conv_result_ready;
  logic [31:0] conv_result_address;
  logic [7:0] conv_result_lane_mask;
  logic [255:0] conv_result_data;
  logic conv_activation_request_valid;
  logic conv_activation_request_ready;
  logic conv_activation_response_valid;
  logic conv_activation_response_ready;
  logic conv_weight_request_valid;
  logic conv_weight_request_ready;
  logic conv_weight_response_valid;
  logic conv_weight_response_ready;
  logic conv_output_write_ready;
  logic [31:0] conv_output_write_strobe;
  logic conv_output_write_valid;
  logic [31:0] conv_output_write_address;
  logic [255:0] conv_output_write_data;
  logic [7:0] conv_output_write_lane_mask;
  logic conv_output_write_bank;
  logic [1:0][31:0] conv_output_address_fifo_q;
  logic [1:0][255:0] conv_output_data_fifo_q;
  logic [1:0][7:0] conv_output_mask_fifo_q;
  logic [1:0] conv_output_bank_fifo_q;
  logic conv_output_write_pointer_q;
  logic conv_output_read_pointer_q;
  logic [1:0] conv_output_count_q;
  logic conv_output_push;
  logic conv_output_pop;
  logic conv_controller_start_ready;
  logic conv_controller_busy;
  logic conv_controller_done;
  logic [7:0] conv_controller_event;
  logic conv_completion_pending_q;
  logic [7:0] conv_completion_event_q;
  logic conv_activation_sent_q;
  logic conv_weight_sent_q;
  logic conv_activation_fire;
  logic conv_weight_fire;
  logic conv_output_bank_q;

  assign busy_o = frontend_busy || engine_busy;
  assign internal_soft_reset = soft_reset_i && !busy_o;
  assign command_ready_o = frontend_command_ready && !soft_reset_i;
  assign error_pulse_o = engine_error || frontend_error;
  assign error_source_o = engine_error ? 2'd2
    : (frontend_error ? 2'd1 : 2'd0);
  assign error_reason_o = engine_error ? engine_error_reason
    : frontend_error_reason;
  assign error_detail_o = engine_error ? 4'd0 : frontend_error_detail;
  assign error_pc_o = engine_error ? engine_pc_q : frontend_pc_q;
  assign error_tag_o = engine_error ? engine_error_tag : frontend_error_tag;
  assign conv_start_ready_o = conv_controller_start_ready
    && !conv_completion_pending_q && conv_output_count_q == 0;
  assign conv_busy_o = conv_controller_busy || conv_completion_pending_q
    || conv_output_count_q != 0;

  assign conv_activation_request_valid = conv_read_request_valid
    && !conv_activation_sent_q;
  assign conv_weight_request_valid = conv_read_request_valid
    && !conv_weight_sent_q;
  assign conv_activation_fire = conv_activation_request_valid
    && conv_activation_request_ready;
  assign conv_weight_fire = conv_weight_request_valid
    && conv_weight_request_ready;
  assign conv_read_request_ready
    = (conv_activation_sent_q || conv_activation_fire)
      && (conv_weight_sent_q || conv_weight_fire);

  assign conv_read_response_valid = conv_activation_response_valid
    && conv_weight_response_valid;
  assign conv_activation_response_ready = conv_read_response_ready
    && conv_weight_response_valid;
  assign conv_weight_response_ready = conv_read_response_ready
    && conv_activation_response_valid;
  assign conv_result_ready = conv_output_count_q < 2;
  assign conv_output_push = conv_result_valid && conv_result_ready;
  assign conv_output_write_valid = conv_output_count_q != 0;
  assign conv_output_pop = conv_output_write_valid && conv_output_write_ready;
  assign conv_output_write_address
    = conv_output_address_fifo_q[conv_output_read_pointer_q];
  assign conv_output_write_data
    = conv_output_data_fifo_q[conv_output_read_pointer_q];
  assign conv_output_write_lane_mask
    = conv_output_mask_fifo_q[conv_output_read_pointer_q];
  assign conv_output_write_bank
    = conv_output_bank_fifo_q[conv_output_read_pointer_q];

  genvar output_lane;
  generate
    for (output_lane = 0; output_lane < 8;
         output_lane = output_lane + 1) begin : g_output_strobe
      assign conv_output_write_strobe[output_lane*4 +: 4]
        = {4{conv_output_write_lane_mask[output_lane]}};
    end
  endgenerate

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      conv_activation_sent_q <= 1'b0;
      conv_weight_sent_q <= 1'b0;
      conv_output_bank_q <= 1'b0;
      conv_output_write_pointer_q <= 1'b0;
      conv_output_read_pointer_q <= 1'b0;
      conv_output_count_q <= '0;
      conv_completion_pending_q <= 1'b0;
      conv_completion_event_q <= '0;
      conv_done_pulse_o <= 1'b0;
      conv_event_set_o <= '0;
    end else if (soft_reset_i) begin
      conv_activation_sent_q <= 1'b0;
      conv_weight_sent_q <= 1'b0;
      conv_output_write_pointer_q <= 1'b0;
      conv_output_read_pointer_q <= 1'b0;
      conv_output_count_q <= '0;
      conv_completion_pending_q <= 1'b0;
      conv_done_pulse_o <= 1'b0;
      conv_event_set_o <= '0;
    end else begin
      conv_done_pulse_o <= 1'b0;
      conv_event_set_o <= '0;
      if (conv_start_valid_i && conv_start_ready_o)
        conv_output_bank_q <= conv_output_bank_i;
      if (conv_read_request_valid && conv_read_request_ready) begin
        conv_activation_sent_q <= 1'b0;
        conv_weight_sent_q <= 1'b0;
      end else begin
        if (conv_activation_fire)
          conv_activation_sent_q <= 1'b1;
        if (conv_weight_fire)
          conv_weight_sent_q <= 1'b1;
      end
      case ({conv_output_push, conv_output_pop})
        2'b10: conv_output_count_q <= conv_output_count_q + 1'b1;
        2'b01: conv_output_count_q <= conv_output_count_q - 1'b1;
        default: conv_output_count_q <= conv_output_count_q;
      endcase
      if (conv_output_push) begin
        conv_output_address_fifo_q[conv_output_write_pointer_q]
          <= conv_result_address;
        conv_output_data_fifo_q[conv_output_write_pointer_q]
          <= conv_result_data;
        conv_output_mask_fifo_q[conv_output_write_pointer_q]
          <= conv_result_lane_mask;
        conv_output_bank_fifo_q[conv_output_write_pointer_q]
          <= conv_output_bank_q;
        conv_output_write_pointer_q <= conv_output_write_pointer_q + 1'b1;
      end
      if (conv_output_pop)
        conv_output_read_pointer_q <= conv_output_read_pointer_q + 1'b1;
      if (conv_controller_done) begin
        conv_completion_pending_q <= 1'b1;
        conv_completion_event_q <= conv_controller_event;
      end else if (conv_completion_pending_q
                   && conv_output_count_q == 0) begin
        conv_completion_pending_q <= 1'b0;
        conv_done_pulse_o <= 1'b1;
        conv_event_set_o <= conv_completion_event_q;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      frontend_pc_q <= '0;
      engine_pc_q <= '0;
    end else if (soft_reset_i && !busy_o) begin
      frontend_pc_q <= '0;
      engine_pc_q <= '0;
    end else begin
      if (command_valid_i && command_ready_o)
        frontend_pc_q <= command_pc_i;
      if (frontend_request_valid && frontend_request_ready)
        engine_pc_q <= frontend_pc_q;
    end
  end

  npu_dma_frontend frontend (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(internal_soft_reset),
    .command_valid_i(command_valid_i && !soft_reset_i),
    .command_ready_o(frontend_command_ready),
    .command_bits_i(command_bits_i),
    .tensor_desc_base_i(tensor_desc_base_i),
    .operator_desc_base_i(operator_desc_base_i),
    .quant_desc_base_i(quant_desc_base_i),
    .segment_desc_base_i(segment_desc_base_i),
    .activation_base_i(activation_base_i), .weight_base_i(weight_base_i),
    .bias_base_i(bias_base_i), .quant_param_base_i(quant_param_base_i),
    .descriptor_memory_request_valid_o(descriptor_memory_request_valid_o),
    .descriptor_memory_request_ready_i(descriptor_memory_request_ready_i),
    .descriptor_memory_request_address_o(descriptor_memory_request_address_o),
    .descriptor_memory_request_bytes_o(descriptor_memory_request_bytes_o),
    .descriptor_memory_response_valid_i(descriptor_memory_response_valid_i),
    .descriptor_memory_response_ready_o(descriptor_memory_response_ready_o),
    .descriptor_memory_response_data_i(descriptor_memory_response_data_i),
    .descriptor_memory_response_error_i(descriptor_memory_response_error_i),
    .request_valid_o(frontend_request_valid),
    .request_ready_i(frontend_request_ready),
    .request_bits_o(frontend_request_bits),
    .busy_o(frontend_busy), .error_pulse_o(frontend_error),
    .error_reason_o(frontend_error_reason),
    .error_detail_o(frontend_error_detail), .error_tag_o(frontend_error_tag));

  npu_dma_engine engine (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(internal_soft_reset),
    .request_valid_i(frontend_request_valid),
    .request_ready_o(frontend_request_ready),
    .request_bits_i(frontend_request_bits),
    .scratchpad_write_valid_o(spad_write_valid),
    .scratchpad_write_ready_i(spad_write_ready),
    .scratchpad_write_kind_o(spad_write_kind),
    .scratchpad_write_bank_o(spad_write_bank),
    .scratchpad_write_address_o(spad_write_address),
    .scratchpad_write_data_o(spad_write_data),
    .scratchpad_write_strobe_o(spad_write_strobe),
    .scratchpad_read_request_valid_o(spad_read_request_valid),
    .scratchpad_read_request_ready_i(spad_read_request_ready),
    .scratchpad_read_kind_o(spad_read_kind),
    .scratchpad_read_bank_o(spad_read_bank),
    .scratchpad_read_address_o(spad_read_address),
    .scratchpad_read_response_valid_i(spad_read_response_valid),
    .scratchpad_read_response_ready_o(spad_read_response_ready),
    .scratchpad_read_response_data_i(spad_read_response_data),
    .m_axi_araddr_o(m_axi_araddr_o), .m_axi_arlen_o(m_axi_arlen_o),
    .m_axi_arsize_o(m_axi_arsize_o), .m_axi_arburst_o(m_axi_arburst_o),
    .m_axi_arvalid_o(m_axi_arvalid_o), .m_axi_arready_i(m_axi_arready_i),
    .m_axi_rdata_i(m_axi_rdata_i), .m_axi_rresp_i(m_axi_rresp_i),
    .m_axi_rlast_i(m_axi_rlast_i), .m_axi_rvalid_i(m_axi_rvalid_i),
    .m_axi_rready_o(m_axi_rready_o),
    .m_axi_awaddr_o(m_axi_awaddr_o), .m_axi_awlen_o(m_axi_awlen_o),
    .m_axi_awsize_o(m_axi_awsize_o), .m_axi_awburst_o(m_axi_awburst_o),
    .m_axi_awvalid_o(m_axi_awvalid_o), .m_axi_awready_i(m_axi_awready_i),
    .m_axi_wdata_o(m_axi_wdata_o), .m_axi_wstrb_o(m_axi_wstrb_o),
    .m_axi_wlast_o(m_axi_wlast_o), .m_axi_wvalid_o(m_axi_wvalid_o),
    .m_axi_wready_i(m_axi_wready_i), .m_axi_bresp_i(m_axi_bresp_i),
    .m_axi_bvalid_i(m_axi_bvalid_i), .m_axi_bready_o(m_axi_bready_o),
    .busy_o(engine_busy), .done_pulse_o(), .event_set_o(event_set_o),
    .error_pulse_o(engine_error), .error_reason_o(engine_error_reason),
    .error_tag_o(engine_error_tag), .bytes_read_o(bytes_read_o),
    .bytes_written_o(bytes_written_o));

  npu_conv2d_controller conv2d_controller (
    .clk_i(clk_i), .rst_ni(rst_ni), .soft_reset_i(soft_reset_i),
    .start_valid_i(conv_start_valid_i && conv_start_ready_o),
    .start_ready_o(conv_controller_start_ready),
    .operator_desc_bits_i(conv_operator_desc_bits_i),
    .activation_bank_i(conv_activation_bank_i),
    .weight_bank_i(conv_weight_bank_i),
    .completion_event_i(conv_completion_event_i),
    .read_request_valid_o(conv_read_request_valid),
    .read_request_ready_i(conv_read_request_ready),
    .activation_read_bank_o(conv_activation_read_bank),
    .activation_read_address_o(conv_activation_read_address),
    .weight_read_bank_o(conv_weight_read_bank),
    .weight_read_address_o(conv_weight_read_address),
    .read_response_valid_i(conv_read_response_valid),
    .read_response_ready_o(conv_read_response_ready),
    .activation_read_data_i(conv_activation_read_data),
    .weight_read_data_i(conv_weight_read_data),
    .result_valid_o(conv_result_valid),
    .result_ready_i(conv_result_ready),
    .result_address_o(conv_result_address),
    .result_lane_mask_o(conv_result_lane_mask),
    .result_data_o(conv_result_data),
    .busy_o(conv_controller_busy), .done_pulse_o(conv_controller_done),
    .event_set_o(conv_controller_event), .error_pulse_o(conv_error_pulse_o),
    .error_reason_o(conv_error_reason_o));

  npu_scratchpad scratchpad (
    .clk_i(clk_i), .rst_ni(rst_ni),
    .dma_write_valid_i(spad_write_valid),
    .dma_write_ready_o(spad_write_ready), .dma_write_kind_i(spad_write_kind),
    .dma_write_bank_i(spad_write_bank), .dma_write_address_i(spad_write_address),
    .dma_write_data_i(spad_write_data), .dma_write_strobe_i(spad_write_strobe),
    .dma_read_request_valid_i(spad_read_request_valid),
    .dma_read_request_ready_o(spad_read_request_ready),
    .dma_read_kind_i(spad_read_kind), .dma_read_bank_i(spad_read_bank),
    .dma_read_address_i(spad_read_address),
    .dma_read_response_valid_o(spad_read_response_valid),
    .dma_read_response_ready_i(spad_read_response_ready),
    .dma_read_response_data_o(spad_read_response_data),
    .accelerator_write_valid_i(accelerator_write_valid_i),
    .accelerator_write_ready_o(accelerator_write_ready_o),
    .accelerator_write_kind_i(accelerator_write_kind_i),
    .accelerator_write_bank_i(accelerator_write_bank_i),
    .accelerator_write_address_i(accelerator_write_address_i),
    .accelerator_write_data_i(accelerator_write_data_i),
    .accelerator_write_strobe_i(accelerator_write_strobe_i),
    .accelerator_read_request_valid_i(accelerator_read_request_valid_i),
    .accelerator_read_request_ready_o(accelerator_read_request_ready_o),
    .accelerator_read_kind_i(accelerator_read_kind_i),
    .accelerator_read_bank_i(accelerator_read_bank_i),
    .accelerator_read_address_i(accelerator_read_address_i),
    .accelerator_read_response_valid_o(accelerator_read_response_valid_o),
    .accelerator_read_response_ready_i(accelerator_read_response_ready_i),
    .accelerator_read_response_data_o(accelerator_read_response_data_o),
    .compute_activation_read_request_valid_i(
      conv_activation_request_valid),
    .compute_activation_read_request_ready_o(
      conv_activation_request_ready),
    .compute_activation_read_bank_i(conv_activation_read_bank),
    .compute_activation_read_address_i(conv_activation_read_address),
    .compute_activation_read_response_valid_o(
      conv_activation_response_valid),
    .compute_activation_read_response_ready_i(
      conv_activation_response_ready),
    .compute_activation_read_response_data_o(conv_activation_read_data),
    .compute_weight_read_request_valid_i(conv_weight_request_valid),
    .compute_weight_read_request_ready_o(conv_weight_request_ready),
    .compute_weight_read_bank_i(conv_weight_read_bank),
    .compute_weight_read_address_i(conv_weight_read_address),
    .compute_weight_read_response_valid_o(conv_weight_response_valid),
    .compute_weight_read_response_ready_i(conv_weight_response_ready),
    .compute_weight_read_response_data_o(conv_weight_read_data),
    .compute_output_write_valid_i(conv_output_write_valid),
    .compute_output_write_ready_o(conv_output_write_ready),
    .compute_output_write_bank_i(conv_output_write_bank),
    .compute_output_write_address_i(conv_output_write_address),
    .compute_output_write_data_i(conv_output_write_data),
    .compute_output_write_strobe_i(conv_output_write_strobe),
    .collision_stall_o(scratchpad_collision_stall_o));

endmodule
