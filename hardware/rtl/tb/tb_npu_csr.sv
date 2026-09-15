`timescale 1ns/1ps

module tb_npu_csr;
  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
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

  logic soft_reset_pulse;
  logic task_start_pulse;
  logic [63:0] task_base;
  logic [31:0] task_bytes;
  logic [31:0] task_tag;
  logic [31:0] watchdog_limit;
  logic irq;
  logic core_busy = 0;
  logic core_done = 0;
  logic core_error = 0;
  logic [15:0] core_error_code = 0;
  logic [31:0] core_error_pc = 0;
  logic [15:0] core_error_tag = 0;
  logic [31:0] completed_tag = 0;
  integer start_count_q = 0;
  integer reset_count_q = 0;

  always #5 clk_i = ~clk_i;

  npu_csr dut (
    .clk_i(clk_i), .rst_ni(rst_ni), .s_axi_awaddr_i(awaddr),
    .s_axi_awvalid_i(awvalid), .s_axi_awready_o(awready),
    .s_axi_wdata_i(wdata), .s_axi_wstrb_i(wstrb),
    .s_axi_wvalid_i(wvalid), .s_axi_wready_o(wready),
    .s_axi_bresp_o(bresp), .s_axi_bvalid_o(bvalid), .s_axi_bready_i(bready),
    .s_axi_araddr_i(araddr), .s_axi_arvalid_i(arvalid),
    .s_axi_arready_o(arready), .s_axi_rdata_o(rdata),
    .s_axi_rresp_o(rresp), .s_axi_rvalid_o(rvalid), .s_axi_rready_i(rready),
    .soft_reset_pulse_o(soft_reset_pulse),
    .task_start_pulse_o(task_start_pulse), .task_base_o(task_base),
    .task_bytes_o(task_bytes), .task_tag_o(task_tag),
    .watchdog_limit_o(watchdog_limit), .irq_o(irq),
    .core_busy_i(core_busy), .core_done_pulse_i(core_done),
    .core_error_pulse_i(core_error), .core_error_code_i(core_error_code),
    .core_error_pc_i(core_error_pc), .core_error_tag_i(core_error_tag),
    .completed_tag_i(completed_tag), .commands_retired_i(32'd1869),
    .cycles_total_i(64'h0123_4567_89ab_cdef),
    .compute_busy_cycles_i(64'h1111_2222_3333_4444),
    .read_wait_cycles_i(64'h5555_6666_7777_8888),
    .write_wait_cycles_i(64'h9999_aaaa_bbbb_cccc),
    .bank_stall_cycles_i(64'hdddd_eeee_ffff_0001),
    .bytes_read_i(64'h1000_2000_3000_4000),
    .bytes_written_i(64'h5000_6000_7000_8000),
    .read_highwater_i(64'h9000_a000_b000_c000),
    .write_highwater_i(64'hd000_e000_f000_0000));

  always_ff @(posedge clk_i) begin
    if (rst_ni) begin
      if (task_start_pulse)
        start_count_q <= start_count_q + 1;
      if (soft_reset_pulse)
        reset_count_q <= reset_count_q + 1;
    end
  end

  task automatic axi_write(
    input logic [11:0] address,
    input logic [31:0] data,
    input logic [3:0] strobe,
    input integer order,
    input logic [1:0] expected_response
  );
    begin
      if (order == 0) begin
        @(negedge clk_i); awaddr = address; awvalid = 1;
        do @(posedge clk_i); while (!awready);
        @(negedge clk_i); awvalid = 0;
        repeat (2) @(posedge clk_i);
        @(negedge clk_i); wdata = data; wstrb = strobe; wvalid = 1;
        do @(posedge clk_i); while (!wready);
        @(negedge clk_i); wvalid = 0;
      end else if (order == 1) begin
        @(negedge clk_i); wdata = data; wstrb = strobe; wvalid = 1;
        do @(posedge clk_i); while (!wready);
        @(negedge clk_i); wvalid = 0;
        repeat (2) @(posedge clk_i);
        @(negedge clk_i); awaddr = address; awvalid = 1;
        do @(posedge clk_i); while (!awready);
        @(negedge clk_i); awvalid = 0;
      end else begin
        @(negedge clk_i);
        awaddr = address; awvalid = 1;
        wdata = data; wstrb = strobe; wvalid = 1;
        do @(posedge clk_i); while (!(awready && wready));
        @(negedge clk_i); awvalid = 0; wvalid = 0;
      end
      wait(bvalid);
      if (bresp != expected_response)
        $fatal(1, "write response mismatch addr=%h got=%b", address, bresp);
      @(posedge clk_i);
    end
  endtask

  task automatic axi_read(
    input logic [11:0] address,
    input logic [31:0] expected_data,
    input logic [1:0] expected_response
  );
    begin
      @(negedge clk_i); araddr = address; arvalid = 1;
      do @(posedge clk_i); while (!arready);
      @(negedge clk_i); arvalid = 0; araddr = 12'h000;
      repeat (2) @(posedge clk_i);
      if (!rvalid)
        $fatal(1, "RVALID dropped under backpressure");
      if (rdata != expected_data || rresp != expected_response)
        $fatal(1, "read mismatch addr=%h data=%h/%h resp=%b/%b",
               address, rdata, expected_data, rresp, expected_response);
      @(negedge clk_i); rready = 1;
      @(posedge clk_i);
      @(negedge clk_i); rready = 0;
    end
  endtask

  initial begin
    repeat (5) @(posedge clk_i);
    rst_ni <= 1'b1;
    repeat (2) @(posedge clk_i);
    axi_read(12'h000, 32'h3155_504e, 2'b00);
    axi_read(12'h048, 32'd4_000_000, 2'b00);
    axi_read(12'h050, 32'h89ab_cdef, 2'b00);
    axi_read(12'h054, 32'h0123_4567, 2'b00);

    axi_write(12'h024, 32'h7654_3200, 4'hf, 0, 2'b00);
    axi_write(12'h028, 32'h0000_0001, 4'hf, 1, 2'b00);
    axi_write(12'h02c, 32'd1_851_712, 4'hf, 2, 2'b00);
    axi_write(12'h030, 32'h55aa_1234, 4'hf, 0, 2'b00);
    axi_write(12'h048, 32'd5_000_000, 4'hf, 1, 2'b00);
    axi_read(12'h024, 32'h7654_3200, 2'b00);
    axi_read(12'h028, 32'h0000_0001, 2'b00);
    axi_read(12'h02c, 32'd1_851_712, 2'b00);
    axi_read(12'h030, 32'h55aa_1234, 2'b00);

    axi_write(12'h020, 32'h0000_0007, 4'h1, 2, 2'b00);
    axi_write(12'h014, 32'h0000_0002, 4'h1, 2, 2'b00);
    axi_write(12'h034, 32'h0000_0001, 4'h1, 2, 2'b00);
    repeat (2) @(posedge clk_i);
    if (start_count_q != 1 || irq)
      $fatal(1, "doorbell/IRQ mismatch");

    core_busy = 1;
    axi_read(12'h018, 32'h0000_0002, 2'b00);
    @(negedge clk_i); core_busy = 0; completed_tag = task_tag;
    core_done = 1;
    @(posedge clk_i);
    @(negedge clk_i); core_done = 0;
    repeat (1) @(posedge clk_i);
    if (!irq)
      $fatal(1, "done IRQ missing");
    axi_read(12'h018, 32'h0000_0009, 2'b00);
    axi_read(12'h01c, 32'h0000_0001, 2'b00);
    axi_read(12'h038, 32'h55aa_1234, 2'b00);
    axi_write(12'h01c, 32'h0000_0001, 4'h1, 2, 2'b00);
    repeat (1) @(posedge clk_i);
    if (irq)
      $fatal(1, "W1C failed");

    core_error_code = 16'h1234;
    core_error_pc = 32'h0000_4560;
    core_error_tag = 16'h0777;
    core_error = 1;
    @(posedge clk_i);
    @(negedge clk_i); core_error = 0;
    repeat (1) @(posedge clk_i);
    axi_read(12'h03c, 32'h0000_1234, 2'b00);
    axi_read(12'h040, 32'h0000_4560, 2'b00);
    axi_read(12'h044, 32'h0000_0777, 2'b00);
    if (!irq)
      $fatal(1, "error IRQ missing");

    core_busy = 1;
    axi_write(12'h034, 32'h0000_0001, 4'h1, 1, 2'b00);
    repeat (1) @(posedge clk_i);
    axi_read(12'h03c, 32'h0000_f001, 2'b00);
    if (start_count_q != 1)
      $fatal(1, "busy doorbell incorrectly started task");
    core_busy = 0;

    axi_write(12'h014, 32'h0000_0003, 4'h1, 0, 2'b00);
    repeat (2) @(posedge clk_i);
    if (reset_count_q != 1)
      $fatal(1, "soft reset pulse missing");
    axi_read(12'h018, 32'h0000_0001, 2'b00);
    axi_write(12'h100, 32'hdead_beef, 4'hf, 2, 2'b10);
    axi_read(12'h0fc, 32'h0000_0000, 2'b10);
    $display("npu_csr: PASS");
    $finish;
  end
endmodule
