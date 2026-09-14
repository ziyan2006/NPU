`timescale 1ns/1ps

package npu_dma_pkg;

  typedef enum logic [1:0] {
    NPU_DESC_TENSOR   = 2'd0,
    NPU_DESC_OPERATOR = 2'd1,
    NPU_DESC_QUANT    = 2'd2,
    NPU_DESC_SEGMENT  = 2'd3
  } npu_descriptor_kind_e;

  typedef enum logic [1:0] {
    NPU_DMA_SPACE_ACTIVATION = 2'd0,
    NPU_DMA_SPACE_WEIGHT     = 2'd1,
    NPU_DMA_SPACE_BIAS       = 2'd2,
    NPU_DMA_SPACE_QUANT      = 2'd3
  } npu_dma_space_e;

  typedef enum logic [1:0] {
    NPU_SPAD_A = 2'd0,
    NPU_SPAD_W = 2'd1,
    NPU_SPAD_O = 2'd2
  } npu_scratchpad_e;

  typedef enum logic [3:0] {
    NPU_DMA_AGU_OK              = 4'd0,
    NPU_DMA_AGU_BAD_COMMAND     = 4'd1,
    NPU_DMA_AGU_BAD_DESCRIPTOR  = 4'd2,
    NPU_DMA_AGU_EXTERNAL_BOUNDS = 4'd3,
    NPU_DMA_AGU_SPAD_BOUNDS     = 4'd4,
    NPU_DMA_AGU_ADDRESS_OVERFLOW= 4'd5
  } npu_dma_agu_error_e;

  typedef enum logic [3:0] {
    NPU_DMA_EXEC_OK                    = 4'd0,
    NPU_DMA_EXEC_BAD_REQUEST           = 4'd1,
    NPU_DMA_EXEC_UNSUPPORTED_ALIGNMENT = 4'd2,
    NPU_DMA_EXEC_AXI_READ              = 4'd3,
    NPU_DMA_EXEC_AXI_WRITE             = 4'd4,
    NPU_DMA_EXEC_AXI_PROTOCOL          = 4'd5
  } npu_dma_exec_error_e;

  typedef enum logic [3:0] {
    NPU_DMA_FE_OK                = 4'd0,
    NPU_DMA_FE_DESCRIPTOR_READ   = 4'd1,
    NPU_DMA_FE_SEGMENT_INDEX     = 4'd2,
    NPU_DMA_FE_ADDRESS_GENERATOR = 4'd3
  } npu_dma_frontend_error_e;

  // One request describes z_count planes, each containing y_count rows of
  // x_bytes contiguous data.  The execution engine advances the independent
  // external and scratchpad strides after each row and plane.
  typedef struct packed {
    logic [15:0] tag;
    logic [7:0]  event_mask;
    logic        store;
    logic [1:0]  memory_space;
    logic [1:0]  scratchpad;
    logic        bank;
    logic        clear_before;
    logic [63:0] external_address;
    logic [31:0] scratchpad_offset;
    logic [31:0] x_bytes;
    logic [15:0] y_count;
    logic [15:0] z_count;
    logic [31:0] external_y_stride;
    logic [31:0] external_z_stride;
    logic [31:0] scratchpad_y_stride;
    logic [31:0] scratchpad_z_stride;
    logic [31:0] clear_bytes;
  } npu_dma_request_t;

  localparam integer NPU_DMA_REQUEST_BITS = $bits(npu_dma_request_t);

  function automatic logic [31:0] npu_align8(input logic [31:0] value);
    npu_align8 = (value + 32'd7) & 32'hffff_fff8;
  endfunction

  function automatic logic [31:0] npu_align64(input logic [31:0] value);
    npu_align64 = (value + 32'd63) & 32'hffff_ffc0;
  endfunction

endpackage : npu_dma_pkg
