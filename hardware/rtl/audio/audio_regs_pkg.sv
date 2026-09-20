package audio_regs_pkg;
  localparam logic [31:0] AUDIO_IP_ID = 32'h3144_5541; // "AUD1"
  localparam logic [31:0] AUDIO_VERSION = 32'h0001_0000;
  localparam logic [31:0] AUDIO_FIFO_CAPACITY = 32'd16384;

  localparam logic [11:0] AUDIO_REG_IP_ID = 12'h000;
  localparam logic [11:0] AUDIO_REG_VERSION = 12'h004;
  localparam logic [11:0] AUDIO_REG_CONTROL = 12'h008;
  localparam logic [11:0] AUDIO_REG_STATUS = 12'h00c;
  localparam logic [11:0] AUDIO_REG_MIX_FRAME = 12'h010;
  localparam logic [11:0] AUDIO_REG_VOCAL_FRAME = 12'h014;
  localparam logic [11:0] AUDIO_REG_FIFO_LEVEL = 12'h018;
  localparam logic [11:0] AUDIO_REG_FIFO_CAPACITY = 12'h01c;
  localparam logic [11:0] AUDIO_REG_UNDERFLOW_COUNT = 12'h020;
  localparam logic [11:0] AUDIO_REG_OVERFLOW_COUNT = 12'h024;
  localparam logic [11:0] AUDIO_REG_PLAYED_FRAMES = 12'h028;
  localparam logic [11:0] AUDIO_REG_STEM_STATE = 12'h02c;
  localparam logic [11:0] AUDIO_REG_CODEC_STATUS = 12'h030;
  localparam logic [11:0] AUDIO_REG_TONE_CONTROL = 12'h034;

  localparam logic [31:0] AUDIO_CONTROL_ENABLE = 32'h0000_0001;
  localparam logic [31:0] AUDIO_CONTROL_SOFT_RESET = 32'h0000_0002;
  localparam logic [31:0] AUDIO_CONTROL_CODEC_REINIT = 32'h0000_0004;
  localparam logic [31:0] AUDIO_CONTROL_TONE_ENABLE = 32'h0000_0008;

  localparam logic [31:0] AUDIO_STATUS_READY = 32'h0000_0001;
  localparam logic [31:0] AUDIO_STATUS_MUTED = 32'h0000_0002;
  localparam logic [31:0] AUDIO_STATUS_CODEC_DONE = 32'h0000_0004;
  localparam logic [31:0] AUDIO_STATUS_CODEC_ERROR = 32'h0000_0008;
  localparam logic [31:0] AUDIO_STATUS_FIFO_FULL = 32'h0000_0010;
  localparam logic [31:0] AUDIO_STATUS_FIFO_EMPTY = 32'h0000_0020;
endpackage
