// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// Synthesizable RFT 8x8 Kernel Module for Cell Count Analysis
// This is the canonical RFT eigenbasis matrix multiply core
// Written in Verilog-2005 for Yosys compatibility

module rft_kernel_8x8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [127:0] sample_in,   // 8 x 16-bit Q1.15 samples (packed)
    output reg  [127:0] sample_out,  // 8 x 16-bit Q1.15 results (packed)
    output wire         done
);

    // State encoding
    localparam [1:0] IDLE    = 2'b00;
    localparam [1:0] COMPUTE = 2'b01;
    localparam [1:0] OUTPUT  = 2'b10;
    
    reg [1:0] state, next_state;
    reg [2:0] row_cnt;
    
    // Accumulators (8 x 35-bit)
    reg signed [34:0] accum0, accum1, accum2, accum3;
    reg signed [34:0] accum4, accum5, accum6, accum7;
    
    // Input registers (8 x 16-bit)
    reg signed [15:0] in0, in1, in2, in3, in4, in5, in6, in7;
    
    // Kernel ROM - row-major order, 64 coefficients
    // Canonical RFT eigenbasis (Q1.15)
    function signed [15:0] get_kernel;
        input [5:0] idx;
        case (idx)
            6'd0:  get_kernel = 16'sd11585;
            6'd1:  get_kernel = 16'sd11585;
            6'd2:  get_kernel = 16'sd11585;
            6'd3:  get_kernel = 16'sd11585;
            6'd4:  get_kernel = 16'sd11585;
            6'd5:  get_kernel = 16'sd11585;
            6'd6:  get_kernel = 16'sd11585;
            6'd7:  get_kernel = 16'sd11585;
            6'd8:  get_kernel = 16'sd16070;
            6'd9:  get_kernel = 16'sd13254;
            6'd10: get_kernel = 16'sd6393;
            6'd11: get_kernel = -16'sd2260;
            6'd12: get_kernel = -16'sd10394;
            6'd13: get_kernel = -16'sd15137;
            6'd14: get_kernel = -16'sd15678;
            6'd15: get_kernel = -16'sd11866;
            6'd16: get_kernel = 16'sd15137;
            6'd17: get_kernel = 16'sd6393;
            6'd18: get_kernel = -16'sd10394;
            6'd19: get_kernel = -16'sd15678;
            6'd20: get_kernel = -16'sd6393;
            6'd21: get_kernel = 16'sd10394;
            6'd22: get_kernel = 16'sd15678;
            6'd23: get_kernel = 16'sd6393;
            6'd24: get_kernel = 16'sd13254;
            6'd25: get_kernel = -16'sd2260;
            6'd26: get_kernel = -16'sd15137;
            6'd27: get_kernel = -16'sd11866;
            6'd28: get_kernel = 16'sd6393;
            6'd29: get_kernel = 16'sd15678;
            6'd30: get_kernel = 16'sd10394;
            6'd31: get_kernel = -16'sd6393;
            6'd32: get_kernel = 16'sd11585;
            6'd33: get_kernel = -16'sd11585;
            6'd34: get_kernel = -16'sd11585;
            6'd35: get_kernel = 16'sd11585;
            6'd36: get_kernel = 16'sd11585;
            6'd37: get_kernel = -16'sd11585;
            6'd38: get_kernel = -16'sd11585;
            6'd39: get_kernel = 16'sd11585;
            6'd40: get_kernel = 16'sd6393;
            6'd41: get_kernel = -16'sd15137;
            6'd42: get_kernel = 16'sd6393;
            6'd43: get_kernel = 16'sd15678;
            6'd44: get_kernel = -16'sd2260;
            6'd45: get_kernel = -16'sd10394;
            6'd46: get_kernel = 16'sd15678;
            6'd47: get_kernel = -16'sd13254;
            6'd48: get_kernel = 16'sd2260;
            6'd49: get_kernel = -16'sd10394;
            6'd50: get_kernel = 16'sd15678;
            6'd51: get_kernel = -16'sd15137;
            6'd52: get_kernel = 16'sd6393;
            6'd53: get_kernel = 16'sd6393;
            6'd54: get_kernel = -16'sd15137;
            6'd55: get_kernel = 16'sd15678;
            6'd56: get_kernel = -16'sd2260;
            6'd57: get_kernel = -16'sd6393;
            6'd58: get_kernel = 16'sd10394;
            6'd59: get_kernel = -16'sd13254;
            6'd60: get_kernel = 16'sd15137;
            6'd61: get_kernel = -16'sd15678;
            6'd62: get_kernel = 16'sd15137;
            6'd63: get_kernel = -16'sd10394;
            default: get_kernel = 16'sd0;
        endcase
    endfunction
    
    // Current column's input sample
    wire signed [15:0] cur_sample;
    assign cur_sample = (row_cnt == 0) ? in0 :
                        (row_cnt == 1) ? in1 :
                        (row_cnt == 2) ? in2 :
                        (row_cnt == 3) ? in3 :
                        (row_cnt == 4) ? in4 :
                        (row_cnt == 5) ? in5 :
                        (row_cnt == 6) ? in6 : in7;
    
    // Kernel values for current column
    wire signed [15:0] k0, k1, k2, k3, k4, k5, k6, k7;
    assign k0 = get_kernel({3'd0, row_cnt});
    assign k1 = get_kernel({3'd1, row_cnt});
    assign k2 = get_kernel({3'd2, row_cnt});
    assign k3 = get_kernel({3'd3, row_cnt});
    assign k4 = get_kernel({3'd4, row_cnt});
    assign k5 = get_kernel({3'd5, row_cnt});
    assign k6 = get_kernel({3'd6, row_cnt});
    assign k7 = get_kernel({3'd7, row_cnt});
    
    // State register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // Next state logic
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:    if (start) next_state = COMPUTE;
            COMPUTE: if (row_cnt == 3'd7) next_state = OUTPUT;
            OUTPUT:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    // Row counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            row_cnt <= 3'd0;
        else if (state == IDLE && start)
            row_cnt <= 3'd0;
        else if (state == COMPUTE)
            row_cnt <= row_cnt + 1;
    end
    
    // Capture inputs
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in0 <= 16'd0; in1 <= 16'd0; in2 <= 16'd0; in3 <= 16'd0;
            in4 <= 16'd0; in5 <= 16'd0; in6 <= 16'd0; in7 <= 16'd0;
        end else if (state == IDLE && start) begin
            in0 <= sample_in[15:0];
            in1 <= sample_in[31:16];
            in2 <= sample_in[47:32];
            in3 <= sample_in[63:48];
            in4 <= sample_in[79:64];
            in5 <= sample_in[95:80];
            in6 <= sample_in[111:96];
            in7 <= sample_in[127:112];
        end
    end
    
    // MAC operation - 8 parallel MACs per cycle
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum0 <= 35'd0; accum1 <= 35'd0; accum2 <= 35'd0; accum3 <= 35'd0;
            accum4 <= 35'd0; accum5 <= 35'd0; accum6 <= 35'd0; accum7 <= 35'd0;
        end else if (state == IDLE && start) begin
            accum0 <= 35'd0; accum1 <= 35'd0; accum2 <= 35'd0; accum3 <= 35'd0;
            accum4 <= 35'd0; accum5 <= 35'd0; accum6 <= 35'd0; accum7 <= 35'd0;
        end else if (state == COMPUTE) begin
            accum0 <= accum0 + (k0 * cur_sample);
            accum1 <= accum1 + (k1 * cur_sample);
            accum2 <= accum2 + (k2 * cur_sample);
            accum3 <= accum3 + (k3 * cur_sample);
            accum4 <= accum4 + (k4 * cur_sample);
            accum5 <= accum5 + (k5 * cur_sample);
            accum6 <= accum6 + (k6 * cur_sample);
            accum7 <= accum7 + (k7 * cur_sample);
        end
    end
    
    // Output - truncate from Q2.30 back to Q1.15
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_out <= 128'd0;
        end else if (state == OUTPUT) begin
            sample_out[15:0]    <= accum0[30:15];
            sample_out[31:16]   <= accum1[30:15];
            sample_out[47:32]   <= accum2[30:15];
            sample_out[63:48]   <= accum3[30:15];
            sample_out[79:64]   <= accum4[30:15];
            sample_out[95:80]   <= accum5[30:15];
            sample_out[111:96]  <= accum6[30:15];
            sample_out[127:112] <= accum7[30:15];
        end
    end
    
    assign done = (state == OUTPUT);

endmodule
