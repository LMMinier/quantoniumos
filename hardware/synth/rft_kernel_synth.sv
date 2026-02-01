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

module rft_kernel_8x8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [127:0] sample_in,   // 8 x 16-bit Q1.15 samples (packed)
    output reg  [127:0] sample_out,  // 8 x 16-bit Q1.15 results (packed)
    output wire         done
);

    // Canonical RFT Kernel Coefficients (Q1.15)
    // Eigenbasis of K = T(R_φ·d) - REAL ONLY
    localparam signed [15:0] K [0:63] = '{
        16'sd11585, 16'sd11585, 16'sd11585, 16'sd11585, 
        16'sd11585, 16'sd11585, 16'sd11585, 16'sd11585,
        16'sd16070, 16'sd13254, 16'sd6393,  -16'sd2260, 
        -16'sd10394, -16'sd15137, -16'sd15678, -16'sd11866,
        16'sd15137, 16'sd6393,  -16'sd10394, -16'sd15678, 
        -16'sd6393,  16'sd10394, 16'sd15678, 16'sd6393,
        16'sd13254, -16'sd2260, -16'sd15137, -16'sd11866, 
        16'sd6393,  16'sd15678, 16'sd10394, -16'sd6393,
        16'sd11585, -16'sd11585, -16'sd11585, 16'sd11585, 
        16'sd11585, -16'sd11585, -16'sd11585, 16'sd11585,
        16'sd6393,  -16'sd15137, 16'sd6393,  16'sd15678, 
        -16'sd2260, -16'sd10394, 16'sd15678, -16'sd13254,
        16'sd2260,  -16'sd10394, 16'sd15678, -16'sd15137, 
        16'sd6393,  16'sd6393,  -16'sd15137, 16'sd15678,
        -16'sd2260, -16'sd6393,  16'sd10394, -16'sd13254, 
        16'sd15137, -16'sd15678, 16'sd15137, -16'sd10394
    };

    // State machine
    typedef enum logic [1:0] {
        IDLE   = 2'b00,
        COMPUTE = 2'b01,
        OUTPUT = 2'b10
    } state_t;
    
    state_t state, next_state;
    
    // Row counter for sequential multiply
    logic [2:0] row_cnt;
    
    // Accumulator for dot product (needs extra bits for accumulation)
    logic signed [34:0] accum [7:0];
    
    // Input registers
    logic signed [15:0] in_reg [7:0];
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE:    if (start) next_state = COMPUTE;
            COMPUTE: if (row_cnt == 3'd7) next_state = OUTPUT;
            OUTPUT:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end
    
    // Row counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_cnt <= 3'd0;
        end else if (state == IDLE && start) begin
            row_cnt <= 3'd0;
        end else if (state == COMPUTE) begin
            row_cnt <= row_cnt + 1;
        end
    end
    
    // Capture inputs
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 8; i++) in_reg[i] <= 16'd0;
        end else if (state == IDLE && start) begin
            for (int i = 0; i < 8; i++) in_reg[i] <= $signed(sample_in[i]);
        end
    end
    
    // MAC operation - one row per cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 8; i++) accum[i] <= 35'd0;
        end else if (state == IDLE && start) begin
            for (int i = 0; i < 8; i++) accum[i] <= 35'd0;
        end else if (state == COMPUTE) begin
            for (int i = 0; i < 8; i++) begin
                // accum[i] += K[i*8 + row_cnt] * in_reg[row_cnt]
                accum[i] <= accum[i] + 
                    ($signed(K[{i[2:0], row_cnt}]) * $signed(in_reg[row_cnt]));
            end
        end
    end
    
    // Output - truncate back to Q1.15
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 8; i++) sample_out[i] <= 16'd0;
        end else if (state == OUTPUT) begin
            for (int i = 0; i < 8; i++) begin
                // Right shift by 15 to convert from Q2.30 back to Q1.15
                sample_out[i] <= accum[i][30:15];
            end
        end
    end
    
    assign done = (state == OUTPUT);

endmodule
