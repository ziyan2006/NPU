`timescale 1ns/1ps

module npu_dma_frontend (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         command_valid_i,
  output logic         command_ready_o,
  input  logic [127:0] command_bits_i,

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

  output logic         request_valid_o,
  input  logic         request_ready_i,
  output logic [npu_dma_pkg::NPU_DMA_REQUEST_BITS-1:0] request_bits_o,

  output logic         busy_o,
  output logic         error_pulse_o,
  output logic [3:0]   error_reason_o,
  output logic [3:0]   error_detail_o,
  output logic [15:0]  error_tag_o
);
  import npu_isa_pkg::*;
  import npu_dma_pkg::*;

  typedef enum logic [3:0] {
    DF_IDLE,
    DF_REQUEST_OPERATOR,
    DF_WAIT_OPERATOR,
    DF_REQUEST_SOURCE,
    DF_WAIT_SOURCE,
    DF_REQUEST_DESTINATION,
    DF_WAIT_DESTINATION,
    DF_REQUEST_QUANT,
    DF_WAIT_QUANT,
    DF_REQUEST_SEGMENT,
    DF_WAIT_SEGMENT,
    DF_ISSUE,
    DF_ERROR
  } dma_frontend_state_e;

  dma_frontend_state_e state_q;
  logic [127:0] command_q;
  logic [511:0] source_desc_q;
  logic [511:0] destination_desc_q;
  logic [511:0] operator_desc_q;
  logic [255:0] quant_desc_q;
  logic [63:0] segment_desc_q;
  logic [3:0] error_reason_q;
  logic [3:0] error_detail_q;

  npu_command_t command;
  npu_tensor_desc_t returned_tensor_desc;
  npu_tensor_desc_t destination_desc;
  logic segmented;
  logic [31:0] segment_record_index;
  logic segment_index_valid;

  logic cache_request_valid;
  logic cache_request_ready;
  logic [1:0] cache_request_kind;
  logic [15:0] cache_request_index;
  logic cache_response_valid;
  logic cache_response_ready;
  logic [511:0] cache_response_data;
  logic [6:0] cache_response_bytes;
  logic cache_response_error;

  logic agu_request_valid;
  logic [NPU_DMA_REQUEST_BITS-1:0] agu_request_bits;
  logic agu_error_valid;
  logic [3:0] agu_error_reason;

  assign command = npu_command_t'(command_q);
  assign returned_tensor_desc = npu_tensor_desc_t'(cache_response_data);
  assign destination_desc = npu_tensor_desc_t'(destination_desc_q);
  assign segmented = command.opcode == NPU_OP_DMA_LOAD
    && command.imm[NPU_IMM_SEGMENTED_BIT];
  assign segment_record_index = (destination_desc.segment_offset >> 3)
    + command.imm[NPU_IMM_SEGMENT_LSB +: NPU_IMM_SEGMENT_WIDTH];
  assign segment_index_valid = destination_desc.segment_offset[2:0] == 0
    && segment_record_index <= 32'h0000_ffff
    && command.imm[NPU_IMM_SEGMENT_LSB +: NPU_IMM_SEGMENT_WIDTH]
       < destination_desc.segment_count;

  assign command_ready_o = state_q == DF_IDLE && !soft_reset_i;
  assign busy_o = state_q != DF_IDLE;
  assign error_pulse_o = state_q == DF_ERROR;
  assign error_reason_o = error_reason_q;
  assign error_detail_o = error_detail_q;
  assign error_tag_o = command.tag;

  always @* begin
    cache_request_valid = 1'b0;
    cache_request_kind = NPU_DESC_TENSOR;
    cache_request_index = NPU_NONE_INDEX;
    case (state_q)
      DF_REQUEST_OPERATOR: begin
        cache_request_valid = 1'b1;
        cache_request_kind = NPU_DESC_OPERATOR;
        cache_request_index = command.op_desc;
      end
      DF_REQUEST_SOURCE: begin
        cache_request_valid = 1'b1;
        cache_request_kind = NPU_DESC_TENSOR;
        cache_request_index = command.src0_td;
      end
      DF_REQUEST_DESTINATION: begin
        cache_request_valid = 1'b1;
        cache_request_kind = NPU_DESC_TENSOR;
        cache_request_index = command.dst_td;
      end
      DF_REQUEST_QUANT: begin
        cache_request_valid = 1'b1;
        cache_request_kind = NPU_DESC_QUANT;
        cache_request_index = command.quant_desc;
      end
      DF_REQUEST_SEGMENT: begin
        cache_request_valid = segment_index_valid;
        cache_request_kind = NPU_DESC_SEGMENT;
        cache_request_index = segment_record_index[15:0];
      end
      default: begin end
    endcase
  end

  assign cache_response_ready = state_q == DF_WAIT_OPERATOR
    || state_q == DF_WAIT_SOURCE
    || state_q == DF_WAIT_DESTINATION
    || state_q == DF_WAIT_QUANT
    || state_q == DF_WAIT_SEGMENT;

  npu_descriptor_cache descriptor_cache (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .tensor_base_i(tensor_desc_base_i),
    .operator_base_i(operator_desc_base_i),
    .quant_base_i(quant_desc_base_i),
    .segment_base_i(segment_desc_base_i),
    .request_valid_i(cache_request_valid),
    .request_ready_o(cache_request_ready),
    .request_kind_i(cache_request_kind),
    .request_index_i(cache_request_index),
    .response_valid_o(cache_response_valid),
    .response_ready_i(cache_response_ready),
    .response_data_o(cache_response_data),
    .response_bytes_o(cache_response_bytes),
    .response_error_o(cache_response_error),
    .memory_request_valid_o(descriptor_memory_request_valid_o),
    .memory_request_ready_i(descriptor_memory_request_ready_i),
    .memory_request_address_o(descriptor_memory_request_address_o),
    .memory_request_bytes_o(descriptor_memory_request_bytes_o),
    .memory_response_valid_i(descriptor_memory_response_valid_i),
    .memory_response_ready_o(descriptor_memory_response_ready_o),
    .memory_response_data_i(descriptor_memory_response_data_i),
    .memory_response_error_i(descriptor_memory_response_error_i)
  );

  npu_dma_agu address_generator (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .soft_reset_i(soft_reset_i),
    .command_valid_i(state_q == DF_ISSUE),
    .command_ready_o(),
    .command_bits_i(command_q),
    .src_tensor_desc_bits_i(source_desc_q),
    .dst_tensor_desc_bits_i(destination_desc_q),
    .operator_desc_bits_i(operator_desc_q),
    .quant_desc_bits_i(quant_desc_q),
    .segment_desc_bits_i(segment_desc_q),
    .activation_base_i(activation_base_i),
    .weight_base_i(weight_base_i),
    .bias_base_i(bias_base_i),
    .quant_param_base_i(quant_param_base_i),
    .request_valid_o(agu_request_valid),
    .request_ready_i(request_ready_i),
    .request_bits_o(agu_request_bits),
    .error_valid_o(agu_error_valid),
    .error_reason_o(agu_error_reason)
  );

  assign request_valid_o = agu_request_valid;
  assign request_bits_o = agu_request_bits;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= DF_IDLE;
      command_q <= '0;
      source_desc_q <= '0;
      destination_desc_q <= '0;
      operator_desc_q <= '0;
      quant_desc_q <= '0;
      segment_desc_q <= '0;
      error_reason_q <= NPU_DMA_FE_OK;
      error_detail_q <= NPU_DMA_AGU_OK;
    end else if (soft_reset_i) begin
      state_q <= DF_IDLE;
      command_q <= '0;
      error_reason_q <= NPU_DMA_FE_OK;
      error_detail_q <= NPU_DMA_AGU_OK;
    end else begin
      case (state_q)
        DF_IDLE: begin
          if (command_valid_i) begin
            command_q <= command_bits_i;
            source_desc_q <= '0;
            destination_desc_q <= '0;
            operator_desc_q <= '0;
            quant_desc_q <= '0;
            segment_desc_q <= '0;
            error_reason_q <= NPU_DMA_FE_OK;
            error_detail_q <= NPU_DMA_AGU_OK;
            state_q <= DF_REQUEST_OPERATOR;
          end
        end

        DF_REQUEST_OPERATOR: begin
          if (cache_request_ready)
            state_q <= DF_WAIT_OPERATOR;
        end

        DF_WAIT_OPERATOR: begin
          if (cache_response_valid) begin
            if (cache_response_error) begin
              error_reason_q <= NPU_DMA_FE_DESCRIPTOR_READ;
              state_q <= DF_ERROR;
            end else begin
              operator_desc_q <= cache_response_data;
              if (command.src0_td != NPU_NONE_INDEX)
                state_q <= DF_REQUEST_SOURCE;
              else if (command.dst_td != NPU_NONE_INDEX)
                state_q <= DF_REQUEST_DESTINATION;
              else if (command.quant_desc != NPU_NONE_INDEX)
                state_q <= DF_REQUEST_QUANT;
              else
                state_q <= DF_ISSUE;
            end
          end
        end

        DF_REQUEST_SOURCE: begin
          if (cache_request_ready)
            state_q <= DF_WAIT_SOURCE;
        end

        DF_WAIT_SOURCE: begin
          if (cache_response_valid) begin
            if (cache_response_error) begin
              error_reason_q <= NPU_DMA_FE_DESCRIPTOR_READ;
              state_q <= DF_ERROR;
            end else begin
              source_desc_q <= cache_response_data;
              if (command.dst_td == command.src0_td) begin
                destination_desc_q <= cache_response_data;
                if (command.quant_desc != NPU_NONE_INDEX)
                  state_q <= DF_REQUEST_QUANT;
                else if (segmented) begin
                  if (returned_tensor_desc.segment_offset[2:0] != 0
                      || ((returned_tensor_desc.segment_offset >> 3)
                          + command.imm[NPU_IMM_SEGMENT_LSB
                                        +: NPU_IMM_SEGMENT_WIDTH])
                         > 32'h0000_ffff
                      || command.imm[NPU_IMM_SEGMENT_LSB
                                     +: NPU_IMM_SEGMENT_WIDTH]
                         >= returned_tensor_desc.segment_count) begin
                    error_reason_q <= NPU_DMA_FE_SEGMENT_INDEX;
                    state_q <= DF_ERROR;
                  end else begin
                    state_q <= DF_REQUEST_SEGMENT;
                  end
                end else begin
                  state_q <= DF_ISSUE;
                end
              end else if (command.dst_td != NPU_NONE_INDEX) begin
                state_q <= DF_REQUEST_DESTINATION;
              end else if (command.quant_desc != NPU_NONE_INDEX) begin
                state_q <= DF_REQUEST_QUANT;
              end else begin
                state_q <= DF_ISSUE;
              end
            end
          end
        end

        DF_REQUEST_DESTINATION: begin
          if (cache_request_ready)
            state_q <= DF_WAIT_DESTINATION;
        end

        DF_WAIT_DESTINATION: begin
          if (cache_response_valid) begin
            if (cache_response_error) begin
              error_reason_q <= NPU_DMA_FE_DESCRIPTOR_READ;
              state_q <= DF_ERROR;
            end else begin
              destination_desc_q <= cache_response_data;
              if (command.quant_desc != NPU_NONE_INDEX)
                state_q <= DF_REQUEST_QUANT;
              else if (segmented) begin
                if (returned_tensor_desc.segment_offset[2:0] != 0
                    || ((returned_tensor_desc.segment_offset >> 3)
                        + command.imm[NPU_IMM_SEGMENT_LSB
                                      +: NPU_IMM_SEGMENT_WIDTH])
                       > 32'h0000_ffff
                    || command.imm[NPU_IMM_SEGMENT_LSB
                                   +: NPU_IMM_SEGMENT_WIDTH]
                       >= returned_tensor_desc.segment_count) begin
                  error_reason_q <= NPU_DMA_FE_SEGMENT_INDEX;
                  state_q <= DF_ERROR;
                end else begin
                  state_q <= DF_REQUEST_SEGMENT;
                end
              end else begin
                state_q <= DF_ISSUE;
              end
            end
          end
        end

        DF_REQUEST_QUANT: begin
          if (cache_request_ready)
            state_q <= DF_WAIT_QUANT;
        end

        DF_WAIT_QUANT: begin
          if (cache_response_valid) begin
            if (cache_response_error) begin
              error_reason_q <= NPU_DMA_FE_DESCRIPTOR_READ;
              state_q <= DF_ERROR;
            end else begin
              quant_desc_q <= cache_response_data[255:0];
              if (segmented)
                state_q <= DF_REQUEST_SEGMENT;
              else
                state_q <= DF_ISSUE;
            end
          end
        end

        DF_REQUEST_SEGMENT: begin
          if (!segment_index_valid) begin
            error_reason_q <= NPU_DMA_FE_SEGMENT_INDEX;
            state_q <= DF_ERROR;
          end else if (cache_request_ready) begin
            state_q <= DF_WAIT_SEGMENT;
          end
        end

        DF_WAIT_SEGMENT: begin
          if (cache_response_valid) begin
            if (cache_response_error) begin
              error_reason_q <= NPU_DMA_FE_DESCRIPTOR_READ;
              state_q <= DF_ERROR;
            end else begin
              segment_desc_q <= cache_response_data[63:0];
              state_q <= DF_ISSUE;
            end
          end
        end

        DF_ISSUE: begin
          if (agu_error_valid) begin
            error_reason_q <= NPU_DMA_FE_ADDRESS_GENERATOR;
            error_detail_q <= agu_error_reason;
            state_q <= DF_ERROR;
          end else if (agu_request_valid && request_ready_i) begin
            state_q <= DF_IDLE;
          end
        end

        DF_ERROR: state_q <= DF_IDLE;
        default: begin
          error_reason_q <= NPU_DMA_FE_DESCRIPTOR_READ;
          state_q <= DF_ERROR;
        end
      endcase
    end
  end

endmodule
