`timescale 1ns/1ps

module tb_audio_out_axi;
  import audio_regs_pkg::*;

  logic aclk = 0;
  logic aresetn = 0;
  logic mclk = 0;
  logic audio_locked = 0;
  logic key_n = 1;
  logic [11:0] awaddr = 0;
  logic awvalid = 0;
  logic awready;
  logic [31:0] wdata = 0;
  logic [3:0] wstrb = 0;
  logic wvalid = 0;
  logic wready;
  logic [1:0] bresp;
  logic bvalid;
  logic bready = 1;
  logic [11:0] araddr = 0;
  logic arvalid = 0;
  logic arready;
  logic [31:0] rdata;
  logic [1:0] rresp;
  logic rvalid;
  logic rready = 0;
  tri1 aud_scl;
  tri1 aud_sda;
  logic codec_ack_low = 1;
  wire aud_mclk;
  wire aud_bclk;
  wire aud_lrclk;
  wire aud_dacdat;
  integer played_seen = 0;
  integer nonzero_bits = 0;
  integer timeout;
  logic [31:0] value;

  assign aud_sda = codec_ack_low ? 1'b0 : 1'bz;
  always #5 aclk = ~aclk;
  always #2 mclk = ~mclk;
  always @(posedge aud_bclk)
    if (aresetn && aud_dacdat)
      nonzero_bits = nonzero_bits + 1;

  audio_out_axi #(
    .AXI_CLOCK_HZ(1_000_000),
    .AUDIO_CLOCK_HZ(1_000),
    .KEY_DEBOUNCE_MS(2),
    .CODEC_I2C_HZ(250_000),
    .CODEC_FAST_START_CYCLES(5),
    .CODEC_OUTPUT_SETTLE_CYCLES(3)
  ) dut (
    .aclk(aclk), .aresetn(aresetn),
    .s_axi_audio_awaddr(awaddr), .s_axi_audio_awvalid(awvalid),
    .s_axi_audio_awready(awready), .s_axi_audio_wdata(wdata),
    .s_axi_audio_wstrb(wstrb), .s_axi_audio_wvalid(wvalid),
    .s_axi_audio_wready(wready), .s_axi_audio_bresp(bresp),
    .s_axi_audio_bvalid(bvalid), .s_axi_audio_bready(bready),
    .s_axi_audio_araddr(araddr), .s_axi_audio_arvalid(arvalid),
    .s_axi_audio_arready(arready), .s_axi_audio_rdata(rdata),
    .s_axi_audio_rresp(rresp), .s_axi_audio_rvalid(rvalid),
    .s_axi_audio_rready(rready), .audio_mclk_i(mclk),
    .audio_clock_locked_i(audio_locked), .key_n_i(key_n),
    .aud_mclk_o(aud_mclk), .aud_bclk_o(aud_bclk),
    .aud_dac_lrclk_o(aud_lrclk), .aud_dacdat_o(aud_dacdat),
    .aud_scl_io(aud_scl), .aud_sda_io(aud_sda)
  );

  task automatic axi_write(input logic [11:0] address,
                           input logic [31:0] data);
    begin
      @(negedge aclk); awaddr = address; awvalid = 1;
      wdata = data; wstrb = 4'hf; wvalid = 1;
      do @(posedge aclk); while (!(awready && wready));
      @(negedge aclk); awvalid = 0; wvalid = 0;
      wait (bvalid);
      if (bresp != 2'b00)
        $fatal(1, "AXI write failed addr=%h", address);
      @(posedge aclk);
    end
  endtask

  task automatic axi_read(input logic [11:0] address,
                          output logic [31:0] data);
    begin
      @(negedge aclk); araddr = address; arvalid = 1;
      do @(posedge aclk); while (!arready);
      @(negedge aclk); arvalid = 0;
      wait (rvalid);
      if (rresp != 2'b00)
        $fatal(1, "AXI read failed addr=%h", address);
      data = rdata;
      @(negedge aclk); rready = 1;
      @(posedge aclk);
      @(negedge aclk); rready = 0;
    end
  endtask

  initial begin
    repeat (6) @(posedge aclk);
    aresetn = 1;
    audio_locked = 1;
    axi_read(AUDIO_REG_IP_ID, value);
    if (value != AUDIO_IP_ID)
      $fatal(1, "audio top ID mismatch");

    timeout = 0;
    value = 0;
    while (!value[2] && timeout < 5000) begin
      axi_read(AUDIO_REG_STATUS, value);
      timeout = timeout + 1;
    end
    if (!value[2] || value[3])
      $fatal(1, "codec did not initialize status=%h", value);

    axi_write(AUDIO_REG_MIX_FRAME, 32'h2000_1000);
    axi_write(AUDIO_REG_VOCAL_FRAME, 32'h0000_0000);
    axi_write(AUDIO_REG_MIX_FRAME, 32'h4000_3000);
    axi_write(AUDIO_REG_VOCAL_FRAME, 32'h0000_0000);
    axi_write(AUDIO_REG_CONTROL, AUDIO_CONTROL_ENABLE);

    timeout = 0;
    while (!played_seen && timeout < 5000) begin
      axi_read(AUDIO_REG_PLAYED_FRAMES, value);
      played_seen = value >= 2;
      timeout = timeout + 1;
    end
    if (!played_seen || nonzero_bits == 0)
      $fatal(1, "FIFO frames did not reach I2S played=%0d bits=%0d",
             value, nonzero_bits);

    @(negedge mclk); key_n = 0;
    repeat (8) @(posedge mclk);
    @(negedge mclk); key_n = 1;
    repeat (8) @(posedge mclk);
    axi_read(AUDIO_REG_STEM_STATE, value);
    if (!value[0])
      $fatal(1, "KEY0 did not toggle STEM target state=%h", value);

    $display("audio_out_axi: PASS");
    $finish;
  end
endmodule
