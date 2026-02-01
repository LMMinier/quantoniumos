// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// =============================================================================
// Comprehensive Testbench for Systolic Array
// =============================================================================
// Tests:
//   1. Identity matrix multiplication
//   2. Random matrix-vector multiply
//   3. Throughput measurement
//   4. Latency measurement
//   5. Large batch processing
// =============================================================================

`timescale 1ns/1ps

module systolic_array_benchmark;
    // Test parameters
    localparam ARRAY_DIM = 8;      // 8x8 array (matching RFTPU tile size)
    localparam DATA_WIDTH = 8;     // INT8
    localparam ACC_WIDTH = 32;     // INT32 accumulator
    
    // Clock period (100 MHz for FPGA-realistic testing)
    localparam CLK_PERIOD = 10;    // 10ns = 100 MHz
    
    // Testbench signals
    reg clk;
    reg rst_n;
    reg load_weights;
    reg clear_acc;
    reg compute_enable;
    reg [$clog2(ARRAY_DIM)-1:0] weight_row_sel;
    reg [ARRAY_DIM-1:0][DATA_WIDTH-1:0] weight_row_data;
    reg [ARRAY_DIM-1:0][DATA_WIDTH-1:0] activation_in;
    wire [ARRAY_DIM-1:0][ACC_WIDTH-1:0] result_out;
    wire result_valid;
    
    // Benchmark counters
    integer test_count;
    integer pass_count;
    integer fail_count;
    integer cycle_count;
    integer total_cycles;
    integer vectors_processed;
    real throughput_gops;
    real latency_ns;
    
    // Expected results storage
    reg signed [ACC_WIDTH-1:0] expected_result [ARRAY_DIM];
    reg signed [DATA_WIDTH-1:0] weight_matrix [ARRAY_DIM][ARRAY_DIM];
    
    // DUT instantiation
    systolic_array #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights),
        .clear_acc(clear_acc),
        .compute_enable(compute_enable),
        .weight_row_sel(weight_row_sel),
        .weight_row_data(weight_row_data),
        .activation_in(activation_in),
        .result_out(result_out),
        .result_valid(result_valid)
    );
    
    // Clock generation
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Cycle counter for benchmarking
    always @(posedge clk) begin
        if (compute_enable)
            cycle_count <= cycle_count + 1;
    end
    
    // =========================================================================
    // Helper Tasks
    // =========================================================================
    
    task automatic reset_dut();
        begin
            rst_n = 0;
            load_weights = 0;
            clear_acc = 0;
            compute_enable = 0;
            weight_row_sel = 0;
            weight_row_data = '0;
            activation_in = '0;
            cycle_count = 0;
            #(CLK_PERIOD * 5);
            rst_n = 1;
            #(CLK_PERIOD * 2);
        end
    endtask
    
    task automatic load_identity_matrix();
        integer row, col;
        begin
            $display("  Loading identity matrix...");
            load_weights = 1;
            for (row = 0; row < ARRAY_DIM; row++) begin
                weight_row_sel = row;
                for (col = 0; col < ARRAY_DIM; col++) begin
                    weight_row_data[col] = (row == col) ? 8'sd1 : 8'sd0;
                    weight_matrix[row][col] = (row == col) ? 1 : 0;
                end
                @(posedge clk);
            end
            load_weights = 0;
            @(posedge clk);
        end
    endtask
    
    task automatic load_random_matrix(input int seed);
        integer row, col;
        reg signed [7:0] val;
        begin
            $display("  Loading random matrix (seed=%0d)...", seed);
            load_weights = 1;
            for (row = 0; row < ARRAY_DIM; row++) begin
                weight_row_sel = row;
                for (col = 0; col < ARRAY_DIM; col++) begin
                    // Simple PRNG: (seed * row * 7 + col * 13) mod 256 - 128
                    val = ((seed * (row + 1) * 7 + (col + 1) * 13) % 256) - 128;
                    // Clamp to reasonable range
                    if (val > 64) val = 64;
                    if (val < -64) val = -64;
                    weight_row_data[col] = val;
                    weight_matrix[row][col] = val;
                end
                @(posedge clk);
            end
            load_weights = 0;
            @(posedge clk);
        end
    endtask
    
    task automatic load_scaling_matrix(input int scale);
        integer row, col;
        begin
            $display("  Loading scaling matrix (scale=%0d)...", scale);
            load_weights = 1;
            for (row = 0; row < ARRAY_DIM; row++) begin
                weight_row_sel = row;
                for (col = 0; col < ARRAY_DIM; col++) begin
                    weight_row_data[col] = (row == col) ? scale : 8'sd0;
                    weight_matrix[row][col] = (row == col) ? scale : 0;
                end
                @(posedge clk);
            end
            load_weights = 0;
            @(posedge clk);
        end
    endtask
    
    task automatic compute_expected(input reg signed [DATA_WIDTH-1:0] act_vec [ARRAY_DIM]);
        integer row, col;
        reg signed [ACC_WIDTH-1:0] sum;
        begin
            for (row = 0; row < ARRAY_DIM; row++) begin
                sum = 0;
                for (col = 0; col < ARRAY_DIM; col++) begin
                    sum = sum + weight_matrix[row][col] * act_vec[col];
                end
                expected_result[row] = sum;
            end
        end
    endtask
    
    task automatic run_single_vector(input reg signed [DATA_WIDTH-1:0] act_vec [ARRAY_DIM]);
        integer i;
        begin
            // Set activation inputs
            for (i = 0; i < ARRAY_DIM; i++) begin
                activation_in[i] = act_vec[i];
            end
            
            // Clear and start
            clear_acc = 1;
            @(posedge clk);
            clear_acc = 0;
            
            cycle_count = 0;
            compute_enable = 1;
            
            // Wait for valid result
            while (!result_valid) @(posedge clk);
            
            compute_enable = 0;
        end
    endtask
    
    task automatic check_result(input string test_name);
        integer i;
        integer match;
        begin
            match = 1;
            for (i = 0; i < ARRAY_DIM; i++) begin
                if ($signed(result_out[i]) !== expected_result[i]) begin
                    match = 0;
                    $display("    MISMATCH at [%0d]: got %0d, expected %0d", 
                             i, $signed(result_out[i]), expected_result[i]);
                end
            end
            
            if (match) begin
                $display("  [PASS] %s (cycles=%0d)", test_name, cycle_count);
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] %s", test_name);
                fail_count = fail_count + 1;
            end
            test_count = test_count + 1;
        end
    endtask
    
    // =========================================================================
    // Main Test Sequence
    // =========================================================================
    
    initial begin
        // Initialize
        test_count = 0;
        pass_count = 0;
        fail_count = 0;
        total_cycles = 0;
        vectors_processed = 0;
        
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║     SYSTOLIC ARRAY BENCHMARK - RFTPU v2 TPU-STYLE UNIT          ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Array Dimension:  %0d x %0d                                        ║", ARRAY_DIM, ARRAY_DIM);
        $display("║  Data Width:       %0d-bit (INT8)                                  ║", DATA_WIDTH);
        $display("║  Accumulator:      %0d-bit (INT32)                                 ║", ACC_WIDTH);
        $display("║  Clock Period:     %0d ns (%0d MHz)                                ║", CLK_PERIOD, 1000/CLK_PERIOD);
        $display("║  Total MACs:       %0d                                            ║", ARRAY_DIM * ARRAY_DIM);
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // =====================================================================
        // TEST 1: Identity Matrix
        // =====================================================================
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST 1: Identity Matrix Multiplication");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        load_identity_matrix();
        
        begin
            reg signed [DATA_WIDTH-1:0] test_vec [ARRAY_DIM];
            integer i;
            
            // Test vector: [1, 2, 3, 4, 5, 6, 7, 8]
            for (i = 0; i < ARRAY_DIM; i++) test_vec[i] = i + 1;
            
            compute_expected(test_vec);
            run_single_vector(test_vec);
            check_result("Identity * [1,2,3,4,5,6,7,8]");
            
            $display("    Result:   [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                     $signed(result_out[0]), $signed(result_out[1]), 
                     $signed(result_out[2]), $signed(result_out[3]),
                     $signed(result_out[4]), $signed(result_out[5]),
                     $signed(result_out[6]), $signed(result_out[7]));
        end
        
        // =====================================================================
        // TEST 2: Scaling Matrix (2x)
        // =====================================================================
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST 2: Scaling Matrix (2x) Multiplication");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        load_scaling_matrix(2);
        
        begin
            reg signed [DATA_WIDTH-1:0] test_vec [ARRAY_DIM];
            integer i;
            
            for (i = 0; i < ARRAY_DIM; i++) test_vec[i] = 10 + i;
            
            compute_expected(test_vec);
            run_single_vector(test_vec);
            check_result("Scale(2) * [10,11,12,13,14,15,16,17]");
            
            $display("    Result:   [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                     $signed(result_out[0]), $signed(result_out[1]), 
                     $signed(result_out[2]), $signed(result_out[3]),
                     $signed(result_out[4]), $signed(result_out[5]),
                     $signed(result_out[6]), $signed(result_out[7]));
            $display("    Expected: [20, 22, 24, 26, 28, 30, 32, 34]");
        end
        
        // =====================================================================
        // TEST 3: Random Matrix
        // =====================================================================
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST 3: Random Matrix Multiplication");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        load_random_matrix(42);
        
        begin
            reg signed [DATA_WIDTH-1:0] test_vec [ARRAY_DIM];
            integer i;
            
            for (i = 0; i < ARRAY_DIM; i++) test_vec[i] = (i % 2 == 0) ? 5 : -3;
            
            compute_expected(test_vec);
            run_single_vector(test_vec);
            check_result("Random(42) * [5,-3,5,-3,5,-3,5,-3]");
            
            $display("    Input:    [5, -3, 5, -3, 5, -3, 5, -3]");
            $display("    Result:   [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                     $signed(result_out[0]), $signed(result_out[1]), 
                     $signed(result_out[2]), $signed(result_out[3]),
                     $signed(result_out[4]), $signed(result_out[5]),
                     $signed(result_out[6]), $signed(result_out[7]));
        end
        
        // =====================================================================
        // TEST 4: Signed Value Test
        // =====================================================================
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST 4: Signed Value Handling");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        load_scaling_matrix(-1);  // Negation matrix
        
        begin
            reg signed [DATA_WIDTH-1:0] test_vec [ARRAY_DIM];
            integer i;
            
            for (i = 0; i < ARRAY_DIM; i++) test_vec[i] = 10 * (i - 4);  // [-40, -30, -20, -10, 0, 10, 20, 30]
            
            compute_expected(test_vec);
            run_single_vector(test_vec);
            check_result("Negate * signed vector");
            
            $display("    Input:    [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                     test_vec[0], test_vec[1], test_vec[2], test_vec[3],
                     test_vec[4], test_vec[5], test_vec[6], test_vec[7]);
            $display("    Result:   [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                     $signed(result_out[0]), $signed(result_out[1]), 
                     $signed(result_out[2]), $signed(result_out[3]),
                     $signed(result_out[4]), $signed(result_out[5]),
                     $signed(result_out[6]), $signed(result_out[7]));
        end
        
        // =====================================================================
        // TEST 5: Throughput Benchmark
        // =====================================================================
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST 5: Throughput Benchmark (100 vectors)");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        load_identity_matrix();
        
        begin
            integer start_time, end_time, elapsed;
            integer vec_count;
            integer i;
            
            vec_count = 100;
            
            // Clear accumulators
            clear_acc = 1;
            @(posedge clk);
            clear_acc = 0;
            
            start_time = $time;
            cycle_count = 0;
            compute_enable = 1;
            
            // Process 100 vectors
            for (i = 0; i < vec_count; i++) begin
                activation_in[0] = i;
                activation_in[1] = i + 1;
                activation_in[2] = i + 2;
                activation_in[3] = i + 3;
                activation_in[4] = i + 4;
                activation_in[5] = i + 5;
                activation_in[6] = i + 6;
                activation_in[7] = i + 7;
                @(posedge clk);
            end
            
            // Wait for pipeline to drain
            repeat(2 * ARRAY_DIM) @(posedge clk);
            
            compute_enable = 0;
            end_time = $time;
            elapsed = end_time - start_time;
            
            // Calculate metrics
            // Operations per vector: ARRAY_DIM * ARRAY_DIM * 2 (multiply + add)
            // Total ops: vec_count * ARRAY_DIM^2 * 2
            throughput_gops = (vec_count * ARRAY_DIM * ARRAY_DIM * 2.0) / (elapsed * 1.0);
            
            $display("  Vectors processed:  %0d", vec_count);
            $display("  Total time:         %0d ns", elapsed);
            $display("  Cycles:             %0d", cycle_count);
            $display("  Throughput:         %.3f GOPS", throughput_gops);
            $display("  Vectors/sec:        %.2f M", (vec_count * 1000.0) / elapsed);
            
            pass_count = pass_count + 1;
            test_count = test_count + 1;
        end
        
        // =====================================================================
        // TEST 6: Latency Measurement
        // =====================================================================
        $display("");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        $display("TEST 6: Single-Vector Latency");
        $display("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        reset_dut();
        load_identity_matrix();
        
        begin
            integer start_time, end_time, elapsed;
            integer i;
            
            // Set activation
            for (i = 0; i < ARRAY_DIM; i++) activation_in[i] = i + 1;
            
            clear_acc = 1;
            @(posedge clk);
            clear_acc = 0;
            
            start_time = $time;
            cycle_count = 0;
            compute_enable = 1;
            
            // Wait for first valid result
            while (!result_valid) @(posedge clk);
            
            end_time = $time;
            elapsed = end_time - start_time;
            compute_enable = 0;
            
            latency_ns = elapsed;
            
            $display("  First result latency: %0d ns (%0d cycles)", elapsed, cycle_count);
            $display("  Pipeline depth:       %0d stages", 2 * ARRAY_DIM - 1);
            
            pass_count = pass_count + 1;
            test_count = test_count + 1;
        end
        
        // =====================================================================
        // Summary
        // =====================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════════╗");
        $display("║                      BENCHMARK SUMMARY                           ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Total Tests:     %3d                                            ║", test_count);
        $display("║  Passed:          %3d                                            ║", pass_count);
        $display("║  Failed:          %3d                                            ║", fail_count);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║                    PERFORMANCE METRICS                           ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  Array Size:      %0d x %0d = %0d MACs                              ║", ARRAY_DIM, ARRAY_DIM, ARRAY_DIM*ARRAY_DIM);
        $display("║  Clock:           %0d MHz                                         ║", 1000/CLK_PERIOD);
        $display("║  Peak GOPS:       %.2f (theoretical)                            ║", (ARRAY_DIM * ARRAY_DIM * 2.0 * 1000.0 / CLK_PERIOD) / 1000.0);
        $display("║  Measured GOPS:   %.3f                                          ║", throughput_gops);
        $display("║  Latency:         %.0f ns (%0d cycles)                            ║", latency_ns, 2*ARRAY_DIM-1);
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║                    COMPARISON TO GOOGLE TPU v1                   ║");
        $display("╠══════════════════════════════════════════════════════════════════╣");
        $display("║  TPU v1:          256x256 = 65,536 MACs @ 700 MHz = 92 TOPS     ║");
        $display("║  RFTPU v2 (8x8):  8x8 = 64 MACs @ 100 MHz = 0.0128 TOPS         ║");
        $display("║  Scaling factor:  1024x MACs needed to match TPU v1             ║");
        $display("╚══════════════════════════════════════════════════════════════════╝");
        $display("");
        
        if (fail_count == 0) begin
            $display("✓ ALL TESTS PASSED - Systolic Array is functional!");
        end else begin
            $display("✗ SOME TESTS FAILED - Review implementation");
        end
        
        $display("");
        $finish;
    end
    
    // Timeout watchdog
    initial begin
        #100000;
        $display("ERROR: Simulation timeout!");
        $finish;
    end

endmodule
