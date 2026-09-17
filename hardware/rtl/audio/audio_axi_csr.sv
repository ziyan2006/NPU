`timescale 1ns/1ps

module audio_axi_csr (
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

  input  logic         fifo_full_i,
  input  logic         fifo_empty_i,
  input  logic [13:0]  fifo_level_i,
  input  logic         underflow_pulse_i,
  input  logic         overflow_pulse_i,
  input  logic [31:0]  played_frames_i,
  input  logic         audio_ready_i,
  input  logic         codec_done_i,
  input  logic         codec_error_i,
  input  logic         codec_muted_i,
  input  logic [7:0]   codec_index_i,
  input  logic         codec_last_ack_i,
  input  logic         key_pressed_i,
  input  logic         stem_target_i,
  input  logic         stem_ramping_i,
  input  logic [15:0]  stem_gain_q15_i,

  output logic         enable_o,
  output logic         soft_reset_pulse_o,
  output logic         codec_reinit_pulse_o,
  output logic         tone_enable_o,
  output logic [31:0]  tone_control_o,
  output logic         frame_wr_valid_o,
  output logic [63:0]  frame_wr_data_o
);
  import audio_regs_pkg::*;

  logic aw_hold_q;
  logic [11:0] awaddr_q;
  logic w_hold_q;
  logic [31:0] wdata_q;
  logic [3:0] wstrb_q;
  logic bvalid_q;
  logic [1:0] bresp_q;
  logic rvalid_q;
  logic [31:0] rdata_q;
  logic [1:0] rresp_q;
  logic [31:0] read_data;
  logic [1:0] read_resp;

  logic [31:0] mix_stage_q;
  logic [31:0] vocal_stage_q;
  logic [31:0] underflow_count_q;
  logic [31:0] overflow_count_q;
  logic [31:0] merged_mix;
  logic [31:0] merged_vocal;
  logic [31:0] merged_control;
  logic [31:0] merged_tone;

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

  function automatic logic [31:0] increment_saturating(
    input logic [31:0] value
  );
    begin
      increment_saturating = (&value) ? value : value + 1'b1;
    end
  endfunction

  function automatic logic write_address_valid(input logic [11:0] address);
    begin
      case (address)
        AUDIO_REG_CONTROL, AUDIO_REG_MIX_FRAME, AUDIO_REG_VOCAL_FRAME,
        AUDIO_REG_UNDERFLOW_COUNT, AUDIO_REG_OVERFLOW_COUNT,
        AUDIO_REG_TONE_CONTROL: write_address_valid = 1'b1;
        default: write_address_valid = 1'b0;
      endcase
    end
  endfunction

  function automatic logic read_address_valid(input logic [11:0] address);
    begin
      read_address_valid = address <= AUDIO_REG_TONE_CONTROL
        && address[1:0] == 2'b00;
    end
  endfunction

  always_comb begin
    case (s_axi_araddr_i)
      AUDIO_REG_IP_ID: read_data = AUDIO_IP_ID;
      AUDIO_REG_VERSION: read_data = AUDIO_VERSION;
      AUDIO_REG_CONTROL: read_data = {28'd0, tone_enable_o, 2'd0, enable_o};
      AUDIO_REG_STATUS: read_data = {
        26'd0, fifo_empty_i, fifo_full_i, codec_error_i, codec_done_i,
        codec_muted_i, audio_ready_i
      };
      AUDIO_REG_MIX_FRAME: read_data = mix_stage_q;
      AUDIO_REG_VOCAL_FRAME: read_data = vocal_stage_q;
      AUDIO_REG_FIFO_LEVEL: read_data = {18'd0, fifo_level_i};
      AUDIO_REG_FIFO_CAPACITY: read_data = AUDIO_FIFO_CAPACITY;
      AUDIO_REG_UNDERFLOW_COUNT: read_data = underflow_count_q;
      AUDIO_REG_OVERFLOW_COUNT: read_data = overflow_count_q;
      AUDIO_REG_PLAYED_FRAMES: read_data = played_frames_i;
      AUDIO_REG_STEM_STATE: read_data = {
        stem_gain_q15_i, 13'd0, key_pressed_i, stem_ramping_i, stem_target_i
      };
      AUDIO_REG_CODEC_STATUS: read_data = {
        23'd0, codec_last_ack_i, codec_index_i
      };
      AUDIO_REG_TONE_CONTROL: read_data = tone_control_o;
      default: read_data = 32'd0;
    endcase
    read_resp = read_address_valid(s_axi_araddr_i) ? 2'b00 : 2'b10;

    merged_mix = merge_strobe(mix_stage_q, wdata_q, wstrb_q);
    merged_vocal = merge_strobe(vocal_stage_q, wdata_q, wstrb_q);
    merged_control = merge_strobe(
      {28'd0, tone_enable_o, 2'd0, enable_o}, wdata_q, wstrb_q
    );
    merged_tone = merge_strobe(tone_control_o, wdata_q, wstrb_q);
  end

  assign s_axi_awready_o = !aw_hold_q && !bvalid_q;
  assign s_axi_wready_o = !w_hold_q && !bvalid_q;
  assign s_axi_bvalid_o = bvalid_q;
  assign s_axi_bresp_o = bresp_q;
  assign s_axi_arready_o = !rvalid_q;
  assign s_axi_rvalid_o = rvalid_q;
  assign s_axi_rdata_o = rdata_q;
  assign s_axi_rresp_o = rresp_q;

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
      rdata_q <= '0;
      rresp_q <= 2'b00;
      mix_stage_q <= '0;
      vocal_stage_q <= '0;
      underflow_count_q <= '0;
      overflow_count_q <= '0;
      enable_o <= 1'b0;
      tone_enable_o <= 1'b0;
      tone_control_o <= '0;
      soft_reset_pulse_o <= 1'b0;
      codec_reinit_pulse_o <= 1'b0;
      frame_wr_valid_o <= 1'b0;
      frame_wr_data_o <= '0;
    end else begin
      soft_reset_pulse_o <= 1'b0;
      codec_reinit_pulse_o <= 1'b0;
      frame_wr_valid_o <= 1'b0;

      if (underflow_pulse_i)
        underflow_count_q <= increment_saturating(underflow_count_q);
      if (overflow_pulse_i)
        overflow_count_q <= increment_saturating(overflow_count_q);

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
        bresp_q <= (awaddr_q[1:0] == 2'b00
          && write_address_valid(awaddr_q)) ? 2'b00 : 2'b10;
        if (awaddr_q[1:0] == 2'b00 && write_address_valid(awaddr_q)) begin
          case (awaddr_q)
            AUDIO_REG_CONTROL: begin
              enable_o <= merged_control[0];
              soft_reset_pulse_o <= merged_control[1];
              codec_reinit_pulse_o <= merged_control[2];
              tone_enable_o <= merged_control[3];
            end
            AUDIO_REG_MIX_FRAME: mix_stage_q <= merged_mix;
            AUDIO_REG_VOCAL_FRAME: begin
              vocal_stage_q <= merged_vocal;
              if (!fifo_full_i) begin
                frame_wr_valid_o <= 1'b1;
                frame_wr_data_o <= {merged_vocal, mix_stage_q};
              end else begin
                overflow_count_q <= increment_saturating(overflow_count_q);
              end
            end
            AUDIO_REG_UNDERFLOW_COUNT: begin
              if (|merge_strobe(32'd0, wdata_q, wstrb_q))
                underflow_count_q <= 32'd0;
            end
            AUDIO_REG_OVERFLOW_COUNT: begin
              if (|merge_strobe(32'd0, wdata_q, wstrb_q))
                overflow_count_q <= 32'd0;
            end
            AUDIO_REG_TONE_CONTROL: tone_control_o <= merged_tone;
            default: begin end
          endcase
        end
      end

      if (s_axi_arvalid_i && s_axi_arready_o) begin
        rvalid_q <= 1'b1;
        rdata_q <= read_data;
        rresp_q <= read_resp;
      end else if (rvalid_q && s_axi_rready_i) begin
        rvalid_q <= 1'b0;
      end
    end
  end
endmodule
