// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// Scale test for v2.3 streaming controller
`timescale 1ns/1ps
`include "rtl/systolic_array_v23_streaming.sv"

module v23_scale_test;
    parameter ARRAY_DIM = 16;  // 16x16 = 256 MACs
    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter NUM_VECTORS = 256;
    
    reg clk, rst_n;
    reg start;
    reg [15:0] num_vectors;
    reg weight_valid;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] weight_data;
    wire weight_ready;
    reg act_valid;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] act_data;
    wire act_ready;
    wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data;
    wire result_valid;
    wire busy, done;
    wire [31:0] perf_total, perf_weight, perf_compute, perf_idle;
    wire [7:0] utilization;
    
    rftpu_systolic_v23x #(.ARRAY_DIM(ARRAY_DIM)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .num_vectors(num_vectors),
        .weight_valid(weight_valid), .weight_data(weight_data), .weight_ready(weight_ready),
        .act_valid(act_valid), .act_data(act_data), .act_ready(act_ready),
        .result_data(result_data), .result_valid(result_valid),
        .busy(busy), .done(done),
        .perf_total_cycles(perf_total), .perf_weight_cycles(perf_weight),
        .perf_compute_cycles(perf_compute), .perf_idle_cycles(perf_idle),
        .utilization_pct(utilization)
    );
    
    initial clk = 0;
    always #5 clk = ~clk;
    
    integer i, results_cnt;
    real mac_per_cycle, gops_100mhz, gops_1ghz;
    
    initial begin
        $display("\n══════════════════════════════════════════════════════════════════");
        $display("  RFTPU v2.3x SCALE TEST: %0dx%0d = %0d MACs, %0d vectors", 
                 ARRAY_DIM, ARRAY_DIM, ARRAY_DIM*ARRAY_DIM, NUM_VECTORS);
        $display("══════════════════════════════════════════════════════════════════");
        
        rst_n = 0; start = 0; num_vectors = NUM_VECTORS;
        weight_valid = 0; weight_data = '0;
        act_valid = 0; act_data = '0;
        results_cnt = 0;
        
        #20 rst_n = 1; #10;
        
        @(posedge clk); start = 1; @(posedge clk); start = 0;
        
        // Load weights
        for (i = 0; i < ARRAY_DIM; i++) begin
            @(posedge clk);
            while (!weight_ready) @(posedge clk);
            weight_valid = 1;
            weight_data = {ARRAY_DIM{8'(i+1)}};
        end
        @(posedge clk); weight_valid = 0;
        
        // Stream activations
        i = 0;
        while (!done) begin
            @(posedge clk);
            if (act_ready && i < NUM_VECTORS) begin
                act_valid = 1;
                act_data = {ARRAY_DIM{8'(i & 8'hFF)}};
                i = i + 1;
            end else act_valid = 0;
            if (result_valid) results_cnt = results_cnt + 1;
            if (perf_total > 2000) begin $display("TIMEOUT"); $finish; end
        end
        
        // Results
        mac_per_cycle = (ARRAY_DIM * ARRAY_DIM * NUM_VECTORS) / (perf_total * 1.0);
        gops_100mhz = mac_per_cycle * 0.1;  // 100 MHz = 0.1 GHz
        gops_1ghz = mac_per_cycle;          // 1 GHz
        
        $display("\n══════════════════════════════════════════════════════════════════");
        $display("  RESULTS");
        $display("══════════════════════════════════════════════════════════════════");
        $display("  Array Size:       %0dx%0d (%0d MACs)", ARRAY_DIM, ARRAY_DIM, ARRAY_DIM*ARRAY_DIM);
        $display("  Vectors:          %0d", NUM_VECTORS);
        $display("  Total Cycles:     %0d", perf_total);
        $display("  Weight Cycles:    %0d", perf_weight);
        $display("  Compute Cycles:   %0d", perf_compute);
        $display("  UTILIZATION:      %0d%%", utilization);
        $display("──────────────────────────────────────────────────────────────────");
        $display("  Total MAC ops:    %0d", ARRAY_DIM*ARRAY_DIM*NUM_VECTORS);
        $display("  MAC/cycle:        %.2f", mac_per_cycle);
        $display("  GOPS @ 100 MHz:   %.2f", gops_100mhz);
        $display("  GOPS @ 1 GHz:     %.2f", gops_1ghz);
        $display("  TOPS @ 1 GHz:     %.3f", gops_1ghz/1000);
        $display("══════════════════════════════════════════════════════════════════\n");
        
        #50 $finish;
    end
endmodule
