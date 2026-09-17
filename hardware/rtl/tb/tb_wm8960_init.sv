`timescale 1ns/1ps

module tb_wm8960_init;
  localparam integer WRITE_COUNT = 17;

  logic clk = 0;
  logic rst_n = 0;
  logic reinit = 0;
  logic mclk_stable = 1;
  logic scl_drive_low;
  logic sda_drive_low;
  logic slave_drive_low = 0;
  wire scl = scl_drive_low ? 1'b0 : 1'b1;
  wire sda = (sda_drive_low || slave_drive_low) ? 1'b0 : 1'b1;
  logic done;
  logic error;
  logic muted;
  logic [7:0] config_index;
  logic last_ack;

  logic [15:0] expected [0:WRITE_COUNT-1];
  logic [15:0] captured [0:WRITE_COUNT-1];
  logic [23:0] shift_word = 0;
  integer bit_in_transaction = 0;
  integer start_count = 0;
  integer completed_transactions = 0;
  integer nack_transaction = -1;
  integer index;
  integer timeout;

  always #5 clk = ~clk;

  wm8960_init #(
    .CLOCK_HZ(1_000_000),
    .I2C_HZ(250_000),
    .FAST_START_CYCLES(5),
    .OUTPUT_SETTLE_CYCLES(3)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n), .reinit_i(reinit),
    .mclk_stable_i(mclk_stable), .sda_i(sda),
    .scl_drive_low_o(scl_drive_low), .sda_drive_low_o(sda_drive_low),
    .codec_done_o(done), .codec_error_o(error), .codec_muted_o(muted),
    .config_index_o(config_index), .last_ack_o(last_ack)
  );

  initial begin
    expected[0] = {7'd15, 9'h000};
    expected[1] = {7'd25, 9'h1c0};
    expected[2] = {7'd25, 9'h0c0};
    expected[3] = {7'd5, 9'h008};
    expected[4] = {7'd6, 9'h008};
    expected[5] = {7'd4, 9'h000};
    expected[6] = {7'd7, 9'h00a};
    expected[7] = {7'd8, 9'h1c0};
    expected[8] = {7'd34, 9'h100};
    expected[9] = {7'd37, 9'h100};
    expected[10] = {7'd47, 9'h00c};
    expected[11] = {7'd40, 9'h079};
    expected[12] = {7'd41, 9'h179};
    expected[13] = {7'd51, 9'h080};
    expected[14] = {7'd26, 9'h198};
    expected[15] = {7'd49, 9'h0f7};
    expected[16] = {7'd5, 9'h000};
  end

  always @(negedge sda)
    if (scl) begin
      bit_in_transaction = 0;
      shift_word = 0;
      start_count = start_count + 1;
    end

  always @(posedge scl) begin
    if (rst_n && bit_in_transaction < 27) begin
      if ((bit_in_transaction % 9) < 8)
        shift_word = {shift_word[22:0], sda};
      bit_in_transaction = bit_in_transaction + 1;
    end
  end

  always @(negedge scl) begin
    if (rst_n && (bit_in_transaction == 8 || bit_in_transaction == 17
                  || bit_in_transaction == 26)) begin
      slave_drive_low = (completed_transactions != nack_transaction);
    end else begin
      slave_drive_low = 0;
    end
  end

  always @(posedge sda)
    if (scl && rst_n && bit_in_transaction > 0) begin
      if (completed_transactions < WRITE_COUNT && bit_in_transaction >= 27) begin
        if (shift_word[23:16] != 8'h34)
          $fatal(1, "I2C address mismatch got=%h", shift_word[23:16]);
        captured[completed_transactions] = shift_word[15:0];
      end
      completed_transactions = completed_transactions + 1;
      bit_in_transaction = 0;
      slave_drive_low = 0;
    end

  task automatic reset_case(input integer nack_at);
    begin
      rst_n = 0;
      slave_drive_low = 0;
      nack_transaction = nack_at;
      bit_in_transaction = 0;
      start_count = 0;
      completed_transactions = 0;
      repeat (4) @(posedge clk);
      rst_n = 1;
    end
  endtask

  task automatic wait_for_terminal;
    begin
      timeout = 0;
      while (!done && !error && timeout < 20_000) begin
        @(posedge clk);
        timeout = timeout + 1;
      end
      if (timeout == 20_000)
        $fatal(1, "WM8960 init timeout");
    end
  endtask

  initial begin
    reset_case(-1);
    wait_for_terminal();
    if (!done || error || muted || !last_ack)
      $fatal(1, "all-ACK sequence did not complete cleanly");
    if (completed_transactions != WRITE_COUNT || start_count != WRITE_COUNT)
      $fatal(1, "transaction count mismatch completed=%0d starts=%0d",
             completed_transactions, start_count);
    for (index = 0; index < WRITE_COUNT; index = index + 1)
      if (captured[index] !== expected[index])
        $fatal(1, "register sequence mismatch index=%0d got=%h expected=%h",
               index, captured[index], expected[index]);

    for (index = 0; index < WRITE_COUNT; index = index + 1) begin
      reset_case(index);
      wait_for_terminal();
      if (done || !error || !muted || last_ack || config_index != index)
        $fatal(1, "NACK state mismatch index=%0d cfg=%0d", index, config_index);
      if (start_count != index + 1)
        $fatal(1, "NACK did not stop sequence index=%0d starts=%0d",
               index, start_count);
      repeat (20) @(posedge clk);
      if (start_count != index + 1)
        $fatal(1, "write occurred after NACK index=%0d", index);
    end

    $display("wm8960_init: PASS");
    $finish;
  end
endmodule
