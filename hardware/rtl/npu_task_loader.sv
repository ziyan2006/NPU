`timescale 1ns/1ps

// Parses task-image metadata and preloads the 4096-entry tanh LUT.  Full CRC
// and SHA-256 verification is deliberately performed by trusted host software;
// this block enforces the execution-critical structural constraints.
module npu_task_loader (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         start_valid_i,
  output logic         start_ready_o,
  input  logic [63:0]  task_base_i,
  input  logic [31:0]  task_bytes_i,

  output logic         memory_request_valid_o,
  input  logic         memory_request_ready_i,
  output logic [63:0]  memory_request_address_o,
  output logic [6:0]   memory_request_bytes_o,
  input  logic         memory_response_valid_i,
  output logic         memory_response_ready_o,
  input  logic [511:0] memory_response_data_i,
  input  logic         memory_response_error_i,

  output logic         lut_write_valid_o,
  input  logic         lut_write_ready_i,
  output logic [11:0]  lut_write_address_o,
  output logic [15:0]  lut_write_data_o,

  output logic         busy_o,
  output logic         done_pulse_o,
  output logic         error_pulse_o,
  output logic [3:0]   error_reason_o,

  output logic [63:0]  command_base_o,
  output logic [31:0]  command_bytes_o,
  output logic [31:0]  command_count_o,
  output logic [63:0]  tensor_desc_base_o,
  output logic [63:0]  operator_desc_base_o,
  output logic [63:0]  quant_desc_base_o,
  output logic [63:0]  segment_desc_base_o,
  output logic [63:0]  activation_base_o,
  output logic [63:0]  weight_base_o,
  output logic [63:0]  bias_base_o,
  output logic [63:0]  quant_param_base_o,
  output logic [15:0]  input_tensor_o,
  output logic [15:0]  output_tensor_o
);
  localparam logic [63:0] TASK_MAGIC = 64'h0055_504e_4d45_5453;
  localparam logic [3:0] LOAD_ERROR_NONE = 4'h0;
  localparam logic [3:0] LOAD_ERROR_MEMORY = 4'h1;
  localparam logic [3:0] LOAD_ERROR_MAGIC = 4'h2;
  localparam logic [3:0] LOAD_ERROR_VERSION = 4'h3;
  localparam logic [3:0] LOAD_ERROR_SIZE = 4'h4;
  localparam logic [3:0] LOAD_ERROR_SECTION = 4'h5;
  localparam logic [3:0] LOAD_ERROR_LUT = 4'h6;

  typedef enum logic [3:0] {
    TL_IDLE,
    TL_HEADER0_REQUEST,
    TL_HEADER0_RESPONSE,
    TL_HEADER1_REQUEST,
    TL_HEADER1_RESPONSE,
    TL_VALIDATE,
    TL_LUT_REQUEST,
    TL_LUT_RESPONSE,
    TL_LUT_WRITE,
    TL_DONE,
    TL_ERROR
  } task_loader_state_e;

  task_loader_state_e state_q;
  logic [63:0] task_base_q;
  logic [31:0] task_bytes_q;
  logic [1023:0] header_q;
  logic [511:0] lut_block_q;
  logic [12:0] lut_cursor_q;
  logic [5:0] lut_word_q;
  logic [3:0] error_reason_q;

  wire [31:0] header_bytes = header_q[64 +: 32];
  wire [31:0] total_bytes = header_q[96 +: 32];
  wire [15:0] format_major = header_q[128 +: 16];
  wire [15:0] format_minor = header_q[144 +: 16];
  wire [15:0] isa_major = header_q[160 +: 16];
  wire [15:0] isa_minor = header_q[176 +: 16];
  wire [31:0] flags = header_q[192 +: 32];
  wire [31:0] header_command_count = header_q[224 +: 32];

  wire [31:0] command_offset = header_q[256 +: 32];
  wire [31:0] command_bytes = header_q[288 +: 32];
  wire [31:0] tensor_offset = header_q[320 +: 32];
  wire [31:0] tensor_bytes = header_q[352 +: 32];
  wire [31:0] operator_offset = header_q[384 +: 32];
  wire [31:0] operator_bytes = header_q[416 +: 32];
  wire [31:0] quant_offset = header_q[448 +: 32];
  wire [31:0] quant_bytes = header_q[480 +: 32];
  wire [31:0] segment_offset = header_q[512 +: 32];
  wire [31:0] segment_bytes = header_q[544 +: 32];
  wire [31:0] activation_offset = header_q[576 +: 32];
  wire [31:0] activation_bytes = header_q[608 +: 32];
  wire [31:0] weight_offset = header_q[640 +: 32];
  wire [31:0] weight_bytes = header_q[672 +: 32];
  wire [31:0] bias_offset = header_q[704 +: 32];
  wire [31:0] bias_bytes = header_q[736 +: 32];
  wire [31:0] quant_param_offset = header_q[768 +: 32];
  wire [31:0] quant_param_bytes = header_q[800 +: 32];
  wire [31:0] lut_offset = header_q[832 +: 32];
  wire [31:0] lut_bytes = header_q[864 +: 32];
  wire [15:0] header_input_tensor = header_q[896 +: 16];
  wire [15:0] header_output_tensor = header_q[912 +: 16];

  function automatic logic section_valid(
    input logic [31:0] offset,
    input logic [31:0] bytes,
    input logic [31:0] image_bytes
  );
    logic [32:0] section_end;
    begin
      section_end = {1'b0, offset} + {1'b0, bytes};
      section_valid = offset[5:0] == 6'd0
        && offset >= 32'd256
        && bytes != 0
        && !section_end[32]
        && section_end[31:0] <= image_bytes;
    end
  endfunction

  logic all_sections_valid;
  always_comb begin
    all_sections_valid = section_valid(command_offset, command_bytes, total_bytes)
      && section_valid(tensor_offset, tensor_bytes, total_bytes)
      && section_valid(operator_offset, operator_bytes, total_bytes)
      && section_valid(quant_offset, quant_bytes, total_bytes)
      && section_valid(segment_offset, segment_bytes, total_bytes)
      && section_valid(activation_offset, activation_bytes, total_bytes)
      && section_valid(weight_offset, weight_bytes, total_bytes)
      && section_valid(bias_offset, bias_bytes, total_bytes)
      && section_valid(quant_param_offset, quant_param_bytes, total_bytes)
      && section_valid(lut_offset, lut_bytes, total_bytes);
  end

  assign start_ready_o = state_q == TL_IDLE;
  assign busy_o = state_q != TL_IDLE && state_q != TL_DONE && state_q != TL_ERROR;
  assign done_pulse_o = state_q == TL_DONE;
  assign error_pulse_o = state_q == TL_ERROR;
  assign error_reason_o = error_reason_q;

  assign memory_request_valid_o = state_q == TL_HEADER0_REQUEST
    || state_q == TL_HEADER1_REQUEST || state_q == TL_LUT_REQUEST;
  assign memory_request_address_o = state_q == TL_HEADER0_REQUEST
      ? task_base_q
      : state_q == TL_HEADER1_REQUEST
        ? task_base_q + 64
        : task_base_q + {32'd0, lut_offset} + lut_cursor_q;
  assign memory_request_bytes_o = 7'd64;
  assign memory_response_ready_o = state_q == TL_HEADER0_RESPONSE
    || state_q == TL_HEADER1_RESPONSE || state_q == TL_LUT_RESPONSE;

  assign lut_write_valid_o = state_q == TL_LUT_WRITE;
  assign lut_write_address_o = lut_cursor_q[12:1] + lut_word_q[5:0];
  assign lut_write_data_o = lut_block_q[lut_word_q*16 +: 16];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q <= TL_IDLE;
      task_base_q <= '0;
      task_bytes_q <= '0;
      header_q <= '0;
      lut_block_q <= '0;
      lut_cursor_q <= '0;
      lut_word_q <= '0;
      error_reason_q <= LOAD_ERROR_NONE;
      command_base_o <= '0;
      command_bytes_o <= '0;
      command_count_o <= '0;
      tensor_desc_base_o <= '0;
      operator_desc_base_o <= '0;
      quant_desc_base_o <= '0;
      segment_desc_base_o <= '0;
      activation_base_o <= '0;
      weight_base_o <= '0;
      bias_base_o <= '0;
      quant_param_base_o <= '0;
      input_tensor_o <= '0;
      output_tensor_o <= '0;
    end else if (soft_reset_i) begin
      state_q <= TL_IDLE;
      error_reason_q <= LOAD_ERROR_NONE;
      lut_cursor_q <= '0;
      lut_word_q <= '0;
    end else begin
      case (state_q)
        TL_IDLE: begin
          if (start_valid_i) begin
            task_base_q <= task_base_i;
            task_bytes_q <= task_bytes_i;
            header_q <= '0;
            error_reason_q <= LOAD_ERROR_NONE;
            if (task_base_i[5:0] != 0 || task_bytes_i < 256) begin
              error_reason_q <= LOAD_ERROR_SIZE;
              state_q <= TL_ERROR;
            end else begin
              state_q <= TL_HEADER0_REQUEST;
            end
          end
        end

        TL_HEADER0_REQUEST: begin
          if (memory_request_ready_i)
            state_q <= TL_HEADER0_RESPONSE;
        end
        TL_HEADER0_RESPONSE: begin
          if (memory_response_valid_i) begin
            if (memory_response_error_i) begin
              error_reason_q <= LOAD_ERROR_MEMORY;
              state_q <= TL_ERROR;
            end else begin
              header_q[511:0] <= memory_response_data_i;
              state_q <= TL_HEADER1_REQUEST;
            end
          end
        end
        TL_HEADER1_REQUEST: begin
          if (memory_request_ready_i)
            state_q <= TL_HEADER1_RESPONSE;
        end
        TL_HEADER1_RESPONSE: begin
          if (memory_response_valid_i) begin
            if (memory_response_error_i) begin
              error_reason_q <= LOAD_ERROR_MEMORY;
              state_q <= TL_ERROR;
            end else begin
              header_q[1023:512] <= memory_response_data_i;
              state_q <= TL_VALIDATE;
            end
          end
        end

        TL_VALIDATE: begin
          if (header_q[63:0] != TASK_MAGIC) begin
            error_reason_q <= LOAD_ERROR_MAGIC;
            state_q <= TL_ERROR;
          end else if (format_major != 1 || format_minor != 0
                       || isa_major != 1 || isa_minor != 0
                       || flags[31:1] != 0) begin
            error_reason_q <= LOAD_ERROR_VERSION;
            state_q <= TL_ERROR;
          end else if (header_bytes != 256 || total_bytes != task_bytes_q
                       || total_bytes[5:0] != 0
                       || command_bytes[3:0] != 0
                       || command_bytes != (header_command_count << 4)) begin
            error_reason_q <= LOAD_ERROR_SIZE;
            state_q <= TL_ERROR;
          end else if (!all_sections_valid
                       || tensor_bytes[5:0] != 0
                       || operator_bytes[5:0] != 0
                       || quant_bytes[4:0] != 0
                       || segment_bytes[2:0] != 0
                       || quant_param_bytes[3:0] != 0
                       || lut_bytes != 8192) begin
            error_reason_q <= LOAD_ERROR_SECTION;
            state_q <= TL_ERROR;
          end else begin
            command_base_o <= task_base_q + command_offset;
            command_bytes_o <= command_bytes;
            command_count_o <= header_command_count;
            tensor_desc_base_o <= task_base_q + tensor_offset;
            operator_desc_base_o <= task_base_q + operator_offset;
            quant_desc_base_o <= task_base_q + quant_offset;
            segment_desc_base_o <= task_base_q + segment_offset;
            activation_base_o <= task_base_q + activation_offset;
            weight_base_o <= task_base_q + weight_offset;
            bias_base_o <= task_base_q + bias_offset;
            quant_param_base_o <= task_base_q + quant_param_offset;
            input_tensor_o <= header_input_tensor;
            output_tensor_o <= header_output_tensor;
            lut_cursor_q <= 0;
            state_q <= TL_LUT_REQUEST;
          end
        end

        TL_LUT_REQUEST: begin
          if (memory_request_ready_i)
            state_q <= TL_LUT_RESPONSE;
        end
        TL_LUT_RESPONSE: begin
          if (memory_response_valid_i) begin
            if (memory_response_error_i) begin
              error_reason_q <= LOAD_ERROR_LUT;
              state_q <= TL_ERROR;
            end else begin
              lut_block_q <= memory_response_data_i;
              lut_word_q <= 0;
              state_q <= TL_LUT_WRITE;
            end
          end
        end
        TL_LUT_WRITE: begin
          if (lut_write_ready_i) begin
            if (lut_word_q == 31) begin
              lut_word_q <= 0;
              if (lut_cursor_q == 13'd8128) begin
                state_q <= TL_DONE;
              end else begin
                lut_cursor_q <= lut_cursor_q + 64;
                state_q <= TL_LUT_REQUEST;
              end
            end else begin
              lut_word_q <= lut_word_q + 1'b1;
            end
          end
        end

        TL_DONE,
        TL_ERROR: state_q <= TL_IDLE;
        default: state_q <= TL_IDLE;
      endcase
    end
  end
endmodule
