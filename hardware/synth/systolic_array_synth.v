// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// =============================================================================
// Systolic Array - Verilog-2001 Compatible for Yosys Synthesis
// =============================================================================

`default_nettype none

module systolic_pe_synth #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     load_weight,
    input  wire                     clear_acc,
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    activation_in,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    output reg  [DATA_WIDTH-1:0]    activation_out,
    output reg  [ACC_WIDTH-1:0]     psum_out
);

    reg [DATA_WIDTH-1:0] weight_reg;
    wire signed [2*DATA_WIDTH-1:0] product;
    
    assign product = $signed(weight_reg) * $signed(activation_in);
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= 0;
            activation_out <= 0;
            psum_out       <= 0;
        end else begin
            if (load_weight) weight_reg <= weight_in;
            activation_out <= activation_in;
            if (clear_acc)
                psum_out <= 0;
            else
                psum_out <= psum_in + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
        end
    end
endmodule

// 8x8 Systolic Array for synthesis
module systolic_array_8x8 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     load_weights,
    input  wire                     clear_acc,
    input  wire                     compute_enable,
    input  wire [2:0]               weight_row_sel,
    input  wire [63:0]              weight_row_data, // 8 x 8 bits
    input  wire [63:0]              activation_in,   // 8 x 8 bits
    output wire [255:0]             result_out,      // 8 x 32 bits
    output wire                     result_valid
);

    // Extract individual bytes
    wire [7:0] w0 = weight_row_data[7:0];
    wire [7:0] w1 = weight_row_data[15:8];
    wire [7:0] w2 = weight_row_data[23:16];
    wire [7:0] w3 = weight_row_data[31:24];
    wire [7:0] w4 = weight_row_data[39:32];
    wire [7:0] w5 = weight_row_data[47:40];
    wire [7:0] w6 = weight_row_data[55:48];
    wire [7:0] w7 = weight_row_data[63:56];
    
    wire [7:0] a0 = activation_in[7:0];
    wire [7:0] a1 = activation_in[15:8];
    wire [7:0] a2 = activation_in[23:16];
    wire [7:0] a3 = activation_in[31:24];
    wire [7:0] a4 = activation_in[39:32];
    wire [7:0] a5 = activation_in[47:40];
    wire [7:0] a6 = activation_in[55:48];
    wire [7:0] a7 = activation_in[63:56];
    
    // Skew delay registers (simplified)
    reg [7:0] skew [0:7][0:7]; // [row][delay_stage]
    reg [4:0] cycle_count;
    
    // PE interconnect signals
    wire [7:0] act_h [0:8][0:8]; // Horizontal activation flow
    wire [31:0] psum_v [0:8][0:7]; // Vertical psum flow
    
    // Load enables per row
    wire load_row0 = load_weights && (weight_row_sel == 0);
    wire load_row1 = load_weights && (weight_row_sel == 1);
    wire load_row2 = load_weights && (weight_row_sel == 2);
    wire load_row3 = load_weights && (weight_row_sel == 3);
    wire load_row4 = load_weights && (weight_row_sel == 4);
    wire load_row5 = load_weights && (weight_row_sel == 5);
    wire load_row6 = load_weights && (weight_row_sel == 6);
    wire load_row7 = load_weights && (weight_row_sel == 7);
    
    // Top row psum inputs are zero
    assign psum_v[0][0] = 32'd0;
    assign psum_v[0][1] = 32'd0;
    assign psum_v[0][2] = 32'd0;
    assign psum_v[0][3] = 32'd0;
    assign psum_v[0][4] = 32'd0;
    assign psum_v[0][5] = 32'd0;
    assign psum_v[0][6] = 32'd0;
    assign psum_v[0][7] = 32'd0;
    
    // Skewing logic
    integer i, j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 0;
            for (i = 0; i < 8; i = i + 1)
                for (j = 0; j < 8; j = j + 1)
                    skew[i][j] <= 0;
        end else if (compute_enable) begin
            cycle_count <= cycle_count + 1;
            // Shift registers per row
            for (i = 0; i < 8; i = i + 1) begin
                skew[i][0] <= (i == 0) ? a0 : (i == 1) ? a1 : (i == 2) ? a2 : (i == 3) ? a3 :
                              (i == 4) ? a4 : (i == 5) ? a5 : (i == 6) ? a6 : a7;
                for (j = 1; j < 8; j = j + 1)
                    skew[i][j] <= skew[i][j-1];
            end
        end else begin
            cycle_count <= 0;
        end
    end
    
    // Left edge connections (skewed activations)
    assign act_h[0][0] = skew[0][0];
    assign act_h[1][0] = skew[1][0];
    assign act_h[2][0] = skew[2][1];
    assign act_h[3][0] = skew[3][2];
    assign act_h[4][0] = skew[4][3];
    assign act_h[5][0] = skew[5][4];
    assign act_h[6][0] = skew[6][5];
    assign act_h[7][0] = skew[7][6];
    
    // Generate 64 PEs (8x8 grid)
    // Row 0
    systolic_pe_synth pe_00 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w0), .activation_in(act_h[0][0]), .psum_in(psum_v[0][0]),
        .activation_out(act_h[0][1]), .psum_out(psum_v[1][0]));
    systolic_pe_synth pe_01 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w1), .activation_in(act_h[0][1]), .psum_in(psum_v[0][1]),
        .activation_out(act_h[0][2]), .psum_out(psum_v[1][1]));
    systolic_pe_synth pe_02 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w2), .activation_in(act_h[0][2]), .psum_in(psum_v[0][2]),
        .activation_out(act_h[0][3]), .psum_out(psum_v[1][2]));
    systolic_pe_synth pe_03 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w3), .activation_in(act_h[0][3]), .psum_in(psum_v[0][3]),
        .activation_out(act_h[0][4]), .psum_out(psum_v[1][3]));
    systolic_pe_synth pe_04 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w4), .activation_in(act_h[0][4]), .psum_in(psum_v[0][4]),
        .activation_out(act_h[0][5]), .psum_out(psum_v[1][4]));
    systolic_pe_synth pe_05 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w5), .activation_in(act_h[0][5]), .psum_in(psum_v[0][5]),
        .activation_out(act_h[0][6]), .psum_out(psum_v[1][5]));
    systolic_pe_synth pe_06 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w6), .activation_in(act_h[0][6]), .psum_in(psum_v[0][6]),
        .activation_out(act_h[0][7]), .psum_out(psum_v[1][6]));
    systolic_pe_synth pe_07 (.clk(clk), .rst_n(rst_n), .load_weight(load_row0), .clear_acc(clear_acc),
        .weight_in(w7), .activation_in(act_h[0][7]), .psum_in(psum_v[0][7]),
        .activation_out(act_h[0][8]), .psum_out(psum_v[1][7]));
    
    // Row 1
    systolic_pe_synth pe_10 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w0), .activation_in(act_h[1][0]), .psum_in(psum_v[1][0]),
        .activation_out(act_h[1][1]), .psum_out(psum_v[2][0]));
    systolic_pe_synth pe_11 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w1), .activation_in(act_h[1][1]), .psum_in(psum_v[1][1]),
        .activation_out(act_h[1][2]), .psum_out(psum_v[2][1]));
    systolic_pe_synth pe_12 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w2), .activation_in(act_h[1][2]), .psum_in(psum_v[1][2]),
        .activation_out(act_h[1][3]), .psum_out(psum_v[2][2]));
    systolic_pe_synth pe_13 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w3), .activation_in(act_h[1][3]), .psum_in(psum_v[1][3]),
        .activation_out(act_h[1][4]), .psum_out(psum_v[2][3]));
    systolic_pe_synth pe_14 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w4), .activation_in(act_h[1][4]), .psum_in(psum_v[1][4]),
        .activation_out(act_h[1][5]), .psum_out(psum_v[2][4]));
    systolic_pe_synth pe_15 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w5), .activation_in(act_h[1][5]), .psum_in(psum_v[1][5]),
        .activation_out(act_h[1][6]), .psum_out(psum_v[2][5]));
    systolic_pe_synth pe_16 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w6), .activation_in(act_h[1][6]), .psum_in(psum_v[1][6]),
        .activation_out(act_h[1][7]), .psum_out(psum_v[2][6]));
    systolic_pe_synth pe_17 (.clk(clk), .rst_n(rst_n), .load_weight(load_row1), .clear_acc(clear_acc),
        .weight_in(w7), .activation_in(act_h[1][7]), .psum_in(psum_v[1][7]),
        .activation_out(act_h[1][8]), .psum_out(psum_v[2][7]));
    
    // Rows 2-7 (abbreviated - connect remaining 48 PEs)
    // Row 2
    systolic_pe_synth pe_20 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w0), .activation_in(act_h[2][0]), .psum_in(psum_v[2][0]), .activation_out(act_h[2][1]), .psum_out(psum_v[3][0]));
    systolic_pe_synth pe_21 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w1), .activation_in(act_h[2][1]), .psum_in(psum_v[2][1]), .activation_out(act_h[2][2]), .psum_out(psum_v[3][1]));
    systolic_pe_synth pe_22 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w2), .activation_in(act_h[2][2]), .psum_in(psum_v[2][2]), .activation_out(act_h[2][3]), .psum_out(psum_v[3][2]));
    systolic_pe_synth pe_23 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w3), .activation_in(act_h[2][3]), .psum_in(psum_v[2][3]), .activation_out(act_h[2][4]), .psum_out(psum_v[3][3]));
    systolic_pe_synth pe_24 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w4), .activation_in(act_h[2][4]), .psum_in(psum_v[2][4]), .activation_out(act_h[2][5]), .psum_out(psum_v[3][4]));
    systolic_pe_synth pe_25 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w5), .activation_in(act_h[2][5]), .psum_in(psum_v[2][5]), .activation_out(act_h[2][6]), .psum_out(psum_v[3][5]));
    systolic_pe_synth pe_26 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w6), .activation_in(act_h[2][6]), .psum_in(psum_v[2][6]), .activation_out(act_h[2][7]), .psum_out(psum_v[3][6]));
    systolic_pe_synth pe_27 (.clk(clk), .rst_n(rst_n), .load_weight(load_row2), .clear_acc(clear_acc), .weight_in(w7), .activation_in(act_h[2][7]), .psum_in(psum_v[2][7]), .activation_out(act_h[2][8]), .psum_out(psum_v[3][7]));
    // Row 3
    systolic_pe_synth pe_30 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w0), .activation_in(act_h[3][0]), .psum_in(psum_v[3][0]), .activation_out(act_h[3][1]), .psum_out(psum_v[4][0]));
    systolic_pe_synth pe_31 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w1), .activation_in(act_h[3][1]), .psum_in(psum_v[3][1]), .activation_out(act_h[3][2]), .psum_out(psum_v[4][1]));
    systolic_pe_synth pe_32 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w2), .activation_in(act_h[3][2]), .psum_in(psum_v[3][2]), .activation_out(act_h[3][3]), .psum_out(psum_v[4][2]));
    systolic_pe_synth pe_33 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w3), .activation_in(act_h[3][3]), .psum_in(psum_v[3][3]), .activation_out(act_h[3][4]), .psum_out(psum_v[4][3]));
    systolic_pe_synth pe_34 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w4), .activation_in(act_h[3][4]), .psum_in(psum_v[3][4]), .activation_out(act_h[3][5]), .psum_out(psum_v[4][4]));
    systolic_pe_synth pe_35 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w5), .activation_in(act_h[3][5]), .psum_in(psum_v[3][5]), .activation_out(act_h[3][6]), .psum_out(psum_v[4][5]));
    systolic_pe_synth pe_36 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w6), .activation_in(act_h[3][6]), .psum_in(psum_v[3][6]), .activation_out(act_h[3][7]), .psum_out(psum_v[4][6]));
    systolic_pe_synth pe_37 (.clk(clk), .rst_n(rst_n), .load_weight(load_row3), .clear_acc(clear_acc), .weight_in(w7), .activation_in(act_h[3][7]), .psum_in(psum_v[3][7]), .activation_out(act_h[3][8]), .psum_out(psum_v[4][7]));
    // Row 4
    systolic_pe_synth pe_40 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w0), .activation_in(act_h[4][0]), .psum_in(psum_v[4][0]), .activation_out(act_h[4][1]), .psum_out(psum_v[5][0]));
    systolic_pe_synth pe_41 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w1), .activation_in(act_h[4][1]), .psum_in(psum_v[4][1]), .activation_out(act_h[4][2]), .psum_out(psum_v[5][1]));
    systolic_pe_synth pe_42 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w2), .activation_in(act_h[4][2]), .psum_in(psum_v[4][2]), .activation_out(act_h[4][3]), .psum_out(psum_v[5][2]));
    systolic_pe_synth pe_43 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w3), .activation_in(act_h[4][3]), .psum_in(psum_v[4][3]), .activation_out(act_h[4][4]), .psum_out(psum_v[5][3]));
    systolic_pe_synth pe_44 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w4), .activation_in(act_h[4][4]), .psum_in(psum_v[4][4]), .activation_out(act_h[4][5]), .psum_out(psum_v[5][4]));
    systolic_pe_synth pe_45 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w5), .activation_in(act_h[4][5]), .psum_in(psum_v[4][5]), .activation_out(act_h[4][6]), .psum_out(psum_v[5][5]));
    systolic_pe_synth pe_46 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w6), .activation_in(act_h[4][6]), .psum_in(psum_v[4][6]), .activation_out(act_h[4][7]), .psum_out(psum_v[5][6]));
    systolic_pe_synth pe_47 (.clk(clk), .rst_n(rst_n), .load_weight(load_row4), .clear_acc(clear_acc), .weight_in(w7), .activation_in(act_h[4][7]), .psum_in(psum_v[4][7]), .activation_out(act_h[4][8]), .psum_out(psum_v[5][7]));
    // Row 5
    systolic_pe_synth pe_50 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w0), .activation_in(act_h[5][0]), .psum_in(psum_v[5][0]), .activation_out(act_h[5][1]), .psum_out(psum_v[6][0]));
    systolic_pe_synth pe_51 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w1), .activation_in(act_h[5][1]), .psum_in(psum_v[5][1]), .activation_out(act_h[5][2]), .psum_out(psum_v[6][1]));
    systolic_pe_synth pe_52 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w2), .activation_in(act_h[5][2]), .psum_in(psum_v[5][2]), .activation_out(act_h[5][3]), .psum_out(psum_v[6][2]));
    systolic_pe_synth pe_53 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w3), .activation_in(act_h[5][3]), .psum_in(psum_v[5][3]), .activation_out(act_h[5][4]), .psum_out(psum_v[6][3]));
    systolic_pe_synth pe_54 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w4), .activation_in(act_h[5][4]), .psum_in(psum_v[5][4]), .activation_out(act_h[5][5]), .psum_out(psum_v[6][4]));
    systolic_pe_synth pe_55 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w5), .activation_in(act_h[5][5]), .psum_in(psum_v[5][5]), .activation_out(act_h[5][6]), .psum_out(psum_v[6][5]));
    systolic_pe_synth pe_56 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w6), .activation_in(act_h[5][6]), .psum_in(psum_v[5][6]), .activation_out(act_h[5][7]), .psum_out(psum_v[6][6]));
    systolic_pe_synth pe_57 (.clk(clk), .rst_n(rst_n), .load_weight(load_row5), .clear_acc(clear_acc), .weight_in(w7), .activation_in(act_h[5][7]), .psum_in(psum_v[5][7]), .activation_out(act_h[5][8]), .psum_out(psum_v[6][7]));
    // Row 6
    systolic_pe_synth pe_60 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w0), .activation_in(act_h[6][0]), .psum_in(psum_v[6][0]), .activation_out(act_h[6][1]), .psum_out(psum_v[7][0]));
    systolic_pe_synth pe_61 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w1), .activation_in(act_h[6][1]), .psum_in(psum_v[6][1]), .activation_out(act_h[6][2]), .psum_out(psum_v[7][1]));
    systolic_pe_synth pe_62 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w2), .activation_in(act_h[6][2]), .psum_in(psum_v[6][2]), .activation_out(act_h[6][3]), .psum_out(psum_v[7][2]));
    systolic_pe_synth pe_63 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w3), .activation_in(act_h[6][3]), .psum_in(psum_v[6][3]), .activation_out(act_h[6][4]), .psum_out(psum_v[7][3]));
    systolic_pe_synth pe_64 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w4), .activation_in(act_h[6][4]), .psum_in(psum_v[6][4]), .activation_out(act_h[6][5]), .psum_out(psum_v[7][4]));
    systolic_pe_synth pe_65 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w5), .activation_in(act_h[6][5]), .psum_in(psum_v[6][5]), .activation_out(act_h[6][6]), .psum_out(psum_v[7][5]));
    systolic_pe_synth pe_66 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w6), .activation_in(act_h[6][6]), .psum_in(psum_v[6][6]), .activation_out(act_h[6][7]), .psum_out(psum_v[7][6]));
    systolic_pe_synth pe_67 (.clk(clk), .rst_n(rst_n), .load_weight(load_row6), .clear_acc(clear_acc), .weight_in(w7), .activation_in(act_h[6][7]), .psum_in(psum_v[6][7]), .activation_out(act_h[6][8]), .psum_out(psum_v[7][7]));
    // Row 7
    systolic_pe_synth pe_70 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w0), .activation_in(act_h[7][0]), .psum_in(psum_v[7][0]), .activation_out(act_h[7][1]), .psum_out(psum_v[8][0]));
    systolic_pe_synth pe_71 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w1), .activation_in(act_h[7][1]), .psum_in(psum_v[7][1]), .activation_out(act_h[7][2]), .psum_out(psum_v[8][1]));
    systolic_pe_synth pe_72 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w2), .activation_in(act_h[7][2]), .psum_in(psum_v[7][2]), .activation_out(act_h[7][3]), .psum_out(psum_v[8][2]));
    systolic_pe_synth pe_73 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w3), .activation_in(act_h[7][3]), .psum_in(psum_v[7][3]), .activation_out(act_h[7][4]), .psum_out(psum_v[8][3]));
    systolic_pe_synth pe_74 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w4), .activation_in(act_h[7][4]), .psum_in(psum_v[7][4]), .activation_out(act_h[7][5]), .psum_out(psum_v[8][4]));
    systolic_pe_synth pe_75 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w5), .activation_in(act_h[7][5]), .psum_in(psum_v[7][5]), .activation_out(act_h[7][6]), .psum_out(psum_v[8][5]));
    systolic_pe_synth pe_76 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w6), .activation_in(act_h[7][6]), .psum_in(psum_v[7][6]), .activation_out(act_h[7][7]), .psum_out(psum_v[8][6]));
    systolic_pe_synth pe_77 (.clk(clk), .rst_n(rst_n), .load_weight(load_row7), .clear_acc(clear_acc), .weight_in(w7), .activation_in(act_h[7][7]), .psum_in(psum_v[7][7]), .activation_out(act_h[7][8]), .psum_out(psum_v[8][7]));
    
    // Output connections
    assign result_out[31:0]    = psum_v[8][0];
    assign result_out[63:32]   = psum_v[8][1];
    assign result_out[95:64]   = psum_v[8][2];
    assign result_out[127:96]  = psum_v[8][3];
    assign result_out[159:128] = psum_v[8][4];
    assign result_out[191:160] = psum_v[8][5];
    assign result_out[223:192] = psum_v[8][6];
    assign result_out[255:224] = psum_v[8][7];
    
    assign result_valid = (cycle_count >= 15);

endmodule

`default_nettype wire
