`timescale 1ns/1ps

module npu_tensor_mac_8x8 (
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         soft_reset_i,

  input  logic         input_valid_i,
  output logic         input_ready_o,
  input  logic         first_i,
  input  logic         last_i,
  input  logic [7:0]   input_lane_mask_i,
  input  logic [7:0]   output_lane_mask_i,
  input  logic [127:0] activation_i,
  input  logic [511:0] weight_i,

  output logic         result_valid_o,
  input  logic         result_ready_i,
  output logic [255:0] result_o
);
  (* use_dsp = "yes" *) logic signed [23:0] product_q [0:7][0:7];
  (* use_dsp = "no" *) logic signed [24:0] sum_l1_q [0:7][0:3];
  (* use_dsp = "no" *) logic signed [25:0] sum_l2_q [0:7][0:1];
  (* use_dsp = "no" *) logic signed [26:0] dot_q [0:7];
  (* use_dsp = "no" *) logic signed [31:0] accumulator_q [0:7];

  logic product_valid_q;
  logic sum_l1_valid_q;
  logic sum_l2_valid_q;
  logic dot_valid_q;
  logic product_first_q;
  logic sum_l1_first_q;
  logic sum_l2_first_q;
  logic dot_first_q;
  logic product_last_q;
  logic sum_l1_last_q;
  logic sum_l2_last_q;
  logic dot_last_q;
  logic [7:0] product_output_mask_q;
  logic [7:0] sum_l1_output_mask_q;
  logic [7:0] sum_l2_output_mask_q;
  logic [7:0] dot_output_mask_q;
  logic result_valid_q;
  logic [255:0] result_q;
  logic pipeline_advance;
  integer datapath_output_lane;
  integer datapath_input_lane;
  integer accumulator_lane;

  assign pipeline_advance = !result_valid_q || result_ready_i;
  assign input_ready_o = rst_ni && !soft_reset_i && pipeline_advance;
  assign result_valid_o = result_valid_q;
  assign result_o = result_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      product_valid_q <= 1'b0;
      sum_l1_valid_q <= 1'b0;
      sum_l2_valid_q <= 1'b0;
      dot_valid_q <= 1'b0;
      result_valid_q <= 1'b0;
    end else if (soft_reset_i) begin
      product_valid_q <= 1'b0;
      sum_l1_valid_q <= 1'b0;
      sum_l2_valid_q <= 1'b0;
      dot_valid_q <= 1'b0;
      result_valid_q <= 1'b0;
    end else if (pipeline_advance) begin
      dot_valid_q <= sum_l2_valid_q;
      sum_l2_valid_q <= sum_l1_valid_q;
      sum_l1_valid_q <= product_valid_q;
      product_valid_q <= input_valid_i && input_ready_o;
      result_valid_q <= dot_valid_q && dot_last_q;
    end
  end

  // Datapath and sideband registers do not require reset. Their valid bits are
  // reset separately, which keeps reset muxes out of the DSP/adder pipeline.
  always_ff @(posedge clk_i) begin
    if (pipeline_advance) begin
      dot_first_q <= sum_l2_first_q;
      dot_last_q <= sum_l2_last_q;
      dot_output_mask_q <= sum_l2_output_mask_q;
      for (datapath_output_lane = 0; datapath_output_lane < 8;
           datapath_output_lane = datapath_output_lane + 1)
        dot_q[datapath_output_lane]
          <= $signed({sum_l2_q[datapath_output_lane][0][25],
                      sum_l2_q[datapath_output_lane][0]})
           + $signed({sum_l2_q[datapath_output_lane][1][25],
                      sum_l2_q[datapath_output_lane][1]});

      sum_l2_first_q <= sum_l1_first_q;
      sum_l2_last_q <= sum_l1_last_q;
      sum_l2_output_mask_q <= sum_l1_output_mask_q;
      for (datapath_output_lane = 0; datapath_output_lane < 8;
           datapath_output_lane = datapath_output_lane + 1) begin
        sum_l2_q[datapath_output_lane][0]
          <= $signed({sum_l1_q[datapath_output_lane][0][24],
                      sum_l1_q[datapath_output_lane][0]})
           + $signed({sum_l1_q[datapath_output_lane][1][24],
                      sum_l1_q[datapath_output_lane][1]});
        sum_l2_q[datapath_output_lane][1]
          <= $signed({sum_l1_q[datapath_output_lane][2][24],
                      sum_l1_q[datapath_output_lane][2]})
           + $signed({sum_l1_q[datapath_output_lane][3][24],
                      sum_l1_q[datapath_output_lane][3]});
      end

      sum_l1_first_q <= product_first_q;
      sum_l1_last_q <= product_last_q;
      sum_l1_output_mask_q <= product_output_mask_q;
      for (datapath_output_lane = 0; datapath_output_lane < 8;
           datapath_output_lane = datapath_output_lane + 1)
        for (datapath_input_lane = 0; datapath_input_lane < 4;
             datapath_input_lane = datapath_input_lane + 1)
          sum_l1_q[datapath_output_lane][datapath_input_lane]
            <= $signed({product_q[datapath_output_lane]
                                  [datapath_input_lane*2][23],
                        product_q[datapath_output_lane]
                                  [datapath_input_lane*2]})
             + $signed({product_q[datapath_output_lane]
                                  [datapath_input_lane*2+1][23],
                        product_q[datapath_output_lane]
                                  [datapath_input_lane*2+1]});

      product_first_q <= first_i;
      product_last_q <= last_i;
      product_output_mask_q <= output_lane_mask_i;
      if (input_valid_i && input_ready_o)
        for (datapath_output_lane = 0; datapath_output_lane < 8;
             datapath_output_lane = datapath_output_lane + 1)
          for (datapath_input_lane = 0; datapath_input_lane < 8;
               datapath_input_lane = datapath_input_lane + 1)
            if (input_lane_mask_i[datapath_input_lane]
                && output_lane_mask_i[datapath_output_lane])
              product_q[datapath_output_lane][datapath_input_lane]
                <= $signed(activation_i[datapath_input_lane*16 +: 16])
                 * $signed(weight_i[
                     (datapath_output_lane*8+datapath_input_lane)*8 +: 8]);
            else
              product_q[datapath_output_lane][datapath_input_lane] <= '0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni || soft_reset_i) begin
      result_q <= '0;
      for (accumulator_lane = 0; accumulator_lane < 8;
           accumulator_lane = accumulator_lane + 1)
        accumulator_q[accumulator_lane] <= '0;
    end else if (pipeline_advance && dot_valid_q) begin
      for (accumulator_lane = 0; accumulator_lane < 8;
           accumulator_lane = accumulator_lane + 1) begin
        if (!dot_output_mask_q[accumulator_lane]) begin
          accumulator_q[accumulator_lane] <= '0;
          if (dot_last_q)
            result_q[accumulator_lane*32 +: 32] <= '0;
        end else if (dot_first_q) begin
          accumulator_q[accumulator_lane]
            <= {{5{dot_q[accumulator_lane][26]}},
                dot_q[accumulator_lane]};
          if (dot_last_q)
            result_q[accumulator_lane*32 +: 32]
              <= {{5{dot_q[accumulator_lane][26]}},
                  dot_q[accumulator_lane]};
        end else begin
          accumulator_q[accumulator_lane]
            <= accumulator_q[accumulator_lane]
             + {{5{dot_q[accumulator_lane][26]}},
                dot_q[accumulator_lane]};
          if (dot_last_q)
            result_q[accumulator_lane*32 +: 32]
              <= accumulator_q[accumulator_lane]
               + {{5{dot_q[accumulator_lane][26]}},
                  dot_q[accumulator_lane]};
        end
      end
    end
  end

endmodule
