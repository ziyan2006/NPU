`timescale 1ns/1ps

module npu_command_processor (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,
  input  logic         start_i,

  input  logic         command_valid_i,
  output logic         command_ready_o,
  input  logic [127:0] command_bits_i,

  output logic         dma_command_valid_o,
  input  logic         dma_command_ready_i,
  output logic [127:0] dma_command_bits_o,

  output logic         compute_command_valid_o,
  input  logic         compute_command_ready_i,
  output logic [127:0] compute_command_bits_o,

  output logic         vector_command_valid_o,
  input  logic         vector_command_ready_i,
  output logic [127:0] vector_command_bits_o,

  input  logic [7:0]   event_set_i,
  input  logic         vector_sync_done_i,
  input  logic         units_idle_i,
  input  logic [31:0]  watchdog_limit_i,

  input  logic         execution_error_i,
  input  logic [15:0]  execution_error_code_i,
  input  logic [31:0]  execution_error_pc_i,
  input  logic [15:0]  execution_error_tag_i,

  output logic         busy_o,
  output logic         done_pulse_o,
  output logic         irq_pulse_o,
  output logic         error_o,
  output logic [15:0]  error_code_o,
  output logic [31:0]  error_pc_o,
  output logic [15:0]  error_inst_tag_o,
  output logic [31:0]  command_pc_o,
  output logic [31:0]  commands_retired_o,
  output logic [63:0]  cycles_total_o,
  output logic [7:0]   event_state_o
);
  import npu_isa_pkg::*;

  typedef enum logic [3:0] {
    CP_IDLE,
    CP_RUN,
    CP_DISPATCH,
    CP_WAIT_EVENT,
    CP_WAIT_VECTOR,
    CP_WAIT_END,
    CP_ERROR
  } cp_state_e;

  cp_state_e state_q;
  npu_command_t command;
  logic [127:0] command_bits_q;
  logic [31:0] dispatch_pc_q;
  logic opcode_supported;
  logic flags_valid;
  logic fields_valid;
  logic command_validated;
  logic [15:0] decode_error_code;
  logic [7:0] wait_mask_q;
  logic pending_irq_q;
  logic [31:0] wait_pc_q;
  logic [15:0] wait_tag_q;
  logic [7:0] event_clear_mask;
  logic command_capture;
  logic command_dispatch;
  logic dispatch_ready;

  assign command = npu_command_t'(command_bits_q);
  assign dma_command_bits_o = command_bits_q;
  assign compute_command_bits_o = command_bits_q;
  assign vector_command_bits_o = command_bits_q;
  assign busy_o = (state_q != CP_IDLE) && (state_q != CP_ERROR);
  assign error_o = (state_q == CP_ERROR);
  assign command_capture = command_valid_i && command_ready_o;
  assign command_dispatch = state_q == CP_DISPATCH && dispatch_ready;

  always @* begin
    opcode_supported = 1'b1;
    flags_valid = 1'b1;
    fields_valid = 1'b1;

    case (command.opcode)
      NPU_OP_NOP: begin
        flags_valid = (command.flags == 8'h00);
        fields_valid = (command.dst_td == NPU_NONE_INDEX)
          && (command.src0_td == NPU_NONE_INDEX)
          && (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc == NPU_NONE_INDEX)
          && (command.quant_desc == NPU_NONE_INDEX)
          && (command.imm == 16'h0000);
      end

      NPU_OP_WAIT: begin
        flags_valid = (command.flags == 8'h00);
        fields_valid = (command.dst_td == NPU_NONE_INDEX)
          && (command.src0_td == NPU_NONE_INDEX)
          && (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc == NPU_NONE_INDEX)
          && (command.quant_desc == NPU_NONE_INDEX)
          && (command.imm[15:8] == 8'h00)
          && (command.imm[7:0] != 8'h00);
      end

      NPU_OP_END: begin
        flags_valid = ((command.flags & ~NPU_FLAG_IRQ) == 8'h00);
        fields_valid = (command.dst_td == NPU_NONE_INDEX)
          && (command.src0_td == NPU_NONE_INDEX)
          && (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc == NPU_NONE_INDEX)
          && (command.quant_desc == NPU_NONE_INDEX)
          && (command.imm == 16'h0000);
      end

      NPU_OP_DMA_LOAD: begin
        flags_valid = (command.flags == NPU_FLAG_ASYNC);
        fields_valid = (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc != NPU_NONE_INDEX)
          && (command.imm[15] == 1'b0)
          && (((command.src0_td != NPU_NONE_INDEX)
               && (command.dst_td != NPU_NONE_INDEX)
               && (command.quant_desc == NPU_NONE_INDEX))
              || ((command.src0_td == NPU_NONE_INDEX)
                  && (command.dst_td == NPU_NONE_INDEX)
                  && (command.quant_desc != NPU_NONE_INDEX)));
      end

      NPU_OP_DMA_STORE: begin
        flags_valid = (command.flags == NPU_FLAG_ASYNC);
        fields_valid = (command.dst_td != NPU_NONE_INDEX)
          && (command.src0_td != NPU_NONE_INDEX)
          && (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc != NPU_NONE_INDEX)
          && (command.quant_desc == NPU_NONE_INDEX)
          && ((command.imm & 16'hf700) == 16'h0000);
      end

      NPU_OP_CONV2D: begin
        flags_valid = (command.flags == (NPU_FLAG_ASYNC
                                        | NPU_FLAG_SATURATE
                                        | NPU_FLAG_FUSED_POST_OP));
        fields_valid = (command.dst_td != NPU_NONE_INDEX)
          && (command.src0_td != NPU_NONE_INDEX)
          && (command.src1_td != NPU_NONE_INDEX)
          && (command.op_desc != NPU_NONE_INDEX)
          && (command.quant_desc != NPU_NONE_INDEX)
          && ((command.imm & 16'hb000) == 16'h0000);
      end

      NPU_OP_VEC_ADD: begin
        flags_valid = (command.flags == NPU_FLAG_SATURATE);
        fields_valid = (command.dst_td != NPU_NONE_INDEX)
          && (command.src0_td != NPU_NONE_INDEX)
          && (command.src1_td != NPU_NONE_INDEX)
          && (command.op_desc != NPU_NONE_INDEX)
          && (command.quant_desc != NPU_NONE_INDEX)
          && ((command.imm & 16'hf6ff) == 16'h0000);
      end

      NPU_OP_ACT,
      NPU_OP_REQUANT: begin
        flags_valid = ((command.flags & ~(NPU_FLAG_SATURATE)) == 8'h00);
        fields_valid = (command.dst_td != NPU_NONE_INDEX)
          && (command.src0_td != NPU_NONE_INDEX)
          && (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc == NPU_NONE_INDEX)
          && (command.quant_desc != NPU_NONE_INDEX)
          && ((command.imm & 16'hf6ff) == 16'h0000);
      end

      NPU_OP_UPSAMPLE2X: begin
        flags_valid = (command.flags == 8'h00);
        fields_valid = (command.dst_td != NPU_NONE_INDEX)
          && (command.src0_td != NPU_NONE_INDEX)
          && (command.src1_td == NPU_NONE_INDEX)
          && (command.op_desc == NPU_NONE_INDEX)
          && (command.quant_desc == NPU_NONE_INDEX)
          && (command.imm == 16'h0000);
      end

      default: begin
        opcode_supported = 1'b0;
        flags_valid = 1'b0;
        fields_valid = 1'b0;
      end
    endcase
  end

  assign command_validated = opcode_supported && flags_valid && fields_valid;

  always @* begin
    if (!opcode_supported)
      decode_error_code = NPU_ERR_ILLEGAL_OPCODE;
    else if (!flags_valid)
      decode_error_code = NPU_ERR_ILLEGAL_FLAGS;
    else
      decode_error_code = NPU_ERR_ILLEGAL_FIELDS;
  end

  always @* begin
    command_ready_o = state_q == CP_RUN;
    dma_command_valid_o = 1'b0;
    compute_command_valid_o = 1'b0;
    vector_command_valid_o = 1'b0;
    dispatch_ready = 1'b0;

    if (state_q == CP_DISPATCH) begin
      if (!command_validated) begin
        dispatch_ready = 1'b1;
      end else begin
        case (command.opcode)
          NPU_OP_DMA_LOAD,
          NPU_OP_DMA_STORE: begin
            dma_command_valid_o = 1'b1;
            dispatch_ready = dma_command_ready_i;
          end

          NPU_OP_CONV2D: begin
            compute_command_valid_o = 1'b1;
            dispatch_ready = compute_command_ready_i;
          end

          NPU_OP_VEC_ADD,
          NPU_OP_ACT,
          NPU_OP_REQUANT,
          NPU_OP_UPSAMPLE2X: begin
            vector_command_valid_o = 1'b1;
            dispatch_ready = vector_command_ready_i;
          end

          default: dispatch_ready = 1'b1;
        endcase
      end
    end
  end

  always @* begin
    event_clear_mask = 8'h00;
    if (command_dispatch && command_validated
        && ((command.flags & NPU_FLAG_ASYNC) != 8'h00))
      event_clear_mask = npu_imm_event_mask(command.imm);
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= CP_IDLE;
      command_bits_q <= '0;
      dispatch_pc_q <= '0;
      wait_mask_q <= 8'h00;
      pending_irq_q <= 1'b0;
      wait_pc_q <= 32'h0000_0000;
      wait_tag_q <= 16'h0000;
      done_pulse_o <= 1'b0;
      irq_pulse_o <= 1'b0;
      error_code_o <= NPU_ERR_NONE;
      error_pc_o <= 32'h0000_0000;
      error_inst_tag_o <= 16'h0000;
      command_pc_o <= 32'h0000_0000;
      commands_retired_o <= 32'h0000_0000;
      cycles_total_o <= 64'h0000_0000_0000_0000;
      event_state_o <= 8'h00;
    end else if (soft_reset_i) begin
      state_q <= CP_IDLE;
      command_bits_q <= '0;
      dispatch_pc_q <= '0;
      wait_mask_q <= 8'h00;
      pending_irq_q <= 1'b0;
      wait_pc_q <= 32'h0000_0000;
      wait_tag_q <= 16'h0000;
      done_pulse_o <= 1'b0;
      irq_pulse_o <= 1'b0;
      error_code_o <= NPU_ERR_NONE;
      error_pc_o <= 32'h0000_0000;
      error_inst_tag_o <= 16'h0000;
      command_pc_o <= 32'h0000_0000;
      commands_retired_o <= 32'h0000_0000;
      cycles_total_o <= 64'h0000_0000_0000_0000;
      event_state_o <= 8'h00;
    end else begin
      done_pulse_o <= 1'b0;
      irq_pulse_o <= 1'b0;
      event_state_o <= (event_state_o | event_set_i) & ~event_clear_mask;
      if (busy_o)
        cycles_total_o <= cycles_total_o + 1'b1;

      if (execution_error_i && state_q != CP_IDLE && state_q != CP_ERROR) begin
        state_q <= CP_ERROR;
        error_code_o <= (execution_error_code_i == NPU_ERR_NONE)
          ? NPU_ERR_EXECUTION_ERROR : execution_error_code_i;
        error_pc_o <= execution_error_pc_i;
        error_inst_tag_o <= execution_error_tag_i;
      end else if (busy_o && watchdog_limit_i != 0
                   && cycles_total_o >= watchdog_limit_i) begin
        state_q <= CP_ERROR;
        cycles_total_o <= cycles_total_o;
        error_code_o <= NPU_ERR_WATCHDOG;
        if (state_q == CP_WAIT_EVENT || state_q == CP_WAIT_VECTOR
            || state_q == CP_WAIT_END) begin
          error_pc_o <= wait_pc_q;
          error_inst_tag_o <= wait_tag_q;
        end else if (state_q == CP_DISPATCH) begin
          error_pc_o <= dispatch_pc_q;
          error_inst_tag_o <= command.tag;
        end else begin
          error_pc_o <= command_pc_o;
          error_inst_tag_o <= 16'h0000;
        end
      end else begin
        case (state_q)
          CP_IDLE: begin
            if (start_i) begin
              state_q <= CP_RUN;
              wait_mask_q <= 8'h00;
              pending_irq_q <= 1'b0;
              wait_pc_q <= 32'h0000_0000;
              wait_tag_q <= 16'h0000;
              error_code_o <= NPU_ERR_NONE;
              error_pc_o <= 32'h0000_0000;
              error_inst_tag_o <= 16'h0000;
              command_pc_o <= 32'h0000_0000;
              commands_retired_o <= 32'h0000_0000;
              cycles_total_o <= 64'h0000_0000_0000_0000;
              event_state_o <= 8'h00;
            end
          end

          CP_RUN: begin
            if (command_capture) begin
              command_bits_q <= command_bits_i;
              dispatch_pc_q <= command_pc_o;
              command_pc_o <= command_pc_o + NPU_COMMAND_BYTES;
              state_q <= CP_DISPATCH;
            end
          end

          CP_DISPATCH: begin
            if (command_dispatch) begin
              if (!command_validated) begin
                state_q <= CP_ERROR;
                error_code_o <= decode_error_code;
                error_pc_o <= dispatch_pc_q;
                error_inst_tag_o <= command.tag;
              end else begin
                case (command.opcode)
                  NPU_OP_WAIT: begin
                    if (((event_state_o | event_set_i) & command.imm[7:0])
                        == command.imm[7:0]) begin
                      commands_retired_o <= commands_retired_o + 1'b1;
                      state_q <= CP_RUN;
                    end else begin
                      wait_mask_q <= command.imm[7:0];
                      wait_pc_q <= dispatch_pc_q;
                      wait_tag_q <= command.tag;
                      state_q <= CP_WAIT_EVENT;
                    end
                  end

                  NPU_OP_END: begin
                    pending_irq_q <= ((command.flags & NPU_FLAG_IRQ) != 8'h00);
                    if (units_idle_i) begin
                      commands_retired_o <= commands_retired_o + 1'b1;
                      done_pulse_o <= 1'b1;
                      irq_pulse_o <= ((command.flags & NPU_FLAG_IRQ) != 8'h00);
                      state_q <= CP_IDLE;
                    end else begin
                      wait_pc_q <= dispatch_pc_q;
                      wait_tag_q <= command.tag;
                      state_q <= CP_WAIT_END;
                    end
                  end

                  NPU_OP_DMA_LOAD,
                  NPU_OP_DMA_STORE: begin
                    commands_retired_o <= commands_retired_o + 1'b1;
                    state_q <= CP_RUN;
                  end

                  NPU_OP_CONV2D: begin
                    commands_retired_o <= commands_retired_o + 1'b1;
                    state_q <= CP_RUN;
                  end

                  NPU_OP_VEC_ADD,
                  NPU_OP_ACT,
                  NPU_OP_REQUANT,
                  NPU_OP_UPSAMPLE2X: begin
                    wait_pc_q <= dispatch_pc_q;
                    wait_tag_q <= command.tag;
                    state_q <= CP_WAIT_VECTOR;
                  end

                  default: begin
                    commands_retired_o <= commands_retired_o + 1'b1;
                    state_q <= CP_RUN;
                  end
                endcase
              end
            end
          end

          CP_WAIT_EVENT: begin
            if (((event_state_o | event_set_i) & wait_mask_q) == wait_mask_q) begin
              commands_retired_o <= commands_retired_o + 1'b1;
              state_q <= CP_RUN;
            end
          end

          CP_WAIT_VECTOR: begin
            if (vector_sync_done_i) begin
              commands_retired_o <= commands_retired_o + 1'b1;
              state_q <= CP_RUN;
            end
          end

          CP_WAIT_END: begin
            if (units_idle_i) begin
              commands_retired_o <= commands_retired_o + 1'b1;
              done_pulse_o <= 1'b1;
              irq_pulse_o <= pending_irq_q;
              state_q <= CP_IDLE;
            end
          end

          default: state_q <= CP_ERROR;
        endcase
      end
    end
  end

endmodule
