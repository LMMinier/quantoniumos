// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// 16×16 Scale Benchmark
`timescale 1ns/1ps

module systolic_v21_16x16_tb;
    parameter int ARRAY_DIM   = 16;
    parameter int DATA_WIDTH  = 8;
    parameter int ACC_WIDTH   = 32;
    parameter int FIFO_DEPTH  = 16;
    parameter int UB_DEPTH    = 256;
    parameter int NUM_BATCHES = 50;
    parameter real CLK_PERIOD = 10.0;
    
    reg clk = 0;
    reg rst_n;
    reg start;
    reg [3:0] mode;
    reg [15:0] num_vectors;
    reg [7:0] k_tiles;
    reg weight_load_en;
    reg [$clog2(ARRAY_DIM)-1:0] weight_row_sel;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] weight_data;
    reg weight_fifo_wr_en;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] weight_fifo_wr_data;
    wire weight_fifo_full, weight_fifo_empty;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] activation_data;
    reg ub_wr_en;
    reg [$clog2(UB_DEPTH)-1:0] ub_addr;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] ub_wr_data;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] ub_rd_data;
    reg ub_bank_sel;
    reg sis_start;
    reg [ARRAY_DIM*DATA_WIDTH-1:0] sis_message;
    reg sis_load_matrix;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] sis_hash_out;
    wire sis_hash_valid;
    wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] quant_result_data;
    wire result_valid;
    wire busy, done, ready_for_weights, ready_for_activation;
    wire [31:0] perf_compute_cycles, perf_weight_cycles, perf_stall_cycles;
    wire [31:0] perf_mac_ops, perf_total_cycles;
    
    always #(CLK_PERIOD/2) clk = ~clk;
    
    rftpu_systolic_v21 #(
        .ARRAY_DIM(ARRAY_DIM), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .SATURATE(0), .ROUND_MODE(1), .FIFO_DEPTH(FIFO_DEPTH), .UB_DEPTH(UB_DEPTH)
    ) dut (.*);
    
    longint total_macs, total_cycles;
    real sustained_mac_per_cycle, utilization_pct, effective_gops;
    
    task automatic reset_dut();
        rst_n = 0; start = 0; mode = 4'hF; num_vectors = 16'd16; k_tiles = 8'd1;
        weight_load_en = 0; weight_row_sel = 0; weight_data = '0;
        weight_fifo_wr_en = 0; weight_fifo_wr_data = '0; activation_data = '0;
        ub_wr_en = 0; ub_addr = 0; ub_wr_data = '0; ub_bank_sel = 0;
        sis_start = 0; sis_message = '0; sis_load_matrix = 0;
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
    endtask
    
    task automatic load_matrix();
        for (int row = 0; row < ARRAY_DIM; row++) begin
            weight_row_sel = row[$clog2(ARRAY_DIM)-1:0];
            for (int col = 0; col < ARRAY_DIM; col++)
                weight_data[col*DATA_WIDTH +: DATA_WIDTH] = $urandom_range(0, 127);
            weight_load_en = 1;
            @(posedge clk);
        end
        weight_load_en = 0;
    endtask
    
    task automatic run_streaming(input int num_ops);
        longint start_time = $time;
        for (int op = 0; op < num_ops; op++) begin
            start = 1; @(posedge clk); start = 0;
            wait(ready_for_weights); load_matrix();
            wait(ready_for_activation);
            for (int i = 0; i < ARRAY_DIM; i++)
                activation_data[i*DATA_WIDTH +: DATA_WIDTH] = (op + i) & 8'hFF;
            wait(done); @(posedge clk);
        end
        total_cycles = ($time - start_time) / CLK_PERIOD;
    endtask
    
    initial begin
        $display("");
        $display("======================================================================");
        $display("     RFTPU v2.1 - 16×16 SCALE BENCHMARK");
        $display("======================================================================");
        $display("  Array Size: %0d × %0d (%0d MACs)", ARRAY_DIM, ARRAY_DIM, ARRAY_DIM*ARRAY_DIM);
        $display("  Batches:    %0d", NUM_BATCHES);
        $display("======================================================================");
        
        reset_dut();
        run_streaming(NUM_BATCHES);
        
        total_macs = longint'(NUM_BATCHES) * longint'(ARRAY_DIM) * longint'(ARRAY_DIM) * longint'(ARRAY_DIM);
        sustained_mac_per_cycle = real'(total_macs) / real'(total_cycles);
        utilization_pct = (sustained_mac_per_cycle / real'(ARRAY_DIM * ARRAY_DIM)) * 100.0;
        effective_gops = sustained_mac_per_cycle * (1000.0 / CLK_PERIOD) / 1000.0;
        
        $display("");
        $display("  RESULTS:");
        $display("    Total MACs:           %0d", total_macs);
        $display("    Total Cycles:         %0d", total_cycles);
        $display("    Sustained MAC/cycle:  %.2f / %0d peak", sustained_mac_per_cycle, ARRAY_DIM*ARRAY_DIM);
        $display("    Utilization:          %.1f%%", utilization_pct);
        $display("    @ 100 MHz:            %.2f GOPS", effective_gops);
        $display("    @ 500 MHz:            %.2f GOPS", effective_gops * 5.0);
        $display("    @ 1 GHz:              %.2f GOPS", effective_gops * 10.0);
        $display("");
        $display("======================================================================");
        $finish;
    end
    
    initial begin #(CLK_PERIOD * 5000000); $display("TIMEOUT!"); $finish; end
endmodule
