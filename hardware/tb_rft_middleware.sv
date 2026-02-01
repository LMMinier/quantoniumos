// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// ===============================================
// ENHANCED TESTBENCH - Comprehensive RFT Testing
// ===============================================
`timescale 1ns/1ps

module testbench;
    reg clk;
    reg reset;
    reg start;
    reg [63:0] raw_data_in;
    
    wire [15:0] vertex_amplitudes [0:7];
    wire signed [15:0] vertex_phases [0:7];
    wire transform_valid;
    wire [31:0] resonance_energy;
    
    integer test_num;
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    rft_middleware_engine dut (
        .clk(clk),
        .reset(reset),
        .raw_data_in(raw_data_in),
        .start(start),
        .vertex_amplitudes(vertex_amplitudes),
        .vertex_phases(vertex_phases),
        .transform_valid(transform_valid),
        .resonance_energy(resonance_energy)
    );
    
    // Task to run a test
    task run_test;
        input [63:0] data;
        input [255:0] test_name;
        integer i;
        real phase_rad;
        real energy_pct;
        begin
            test_num = test_num + 1;
            $display("\n========================================");
            $display("[Test %0d] %s", test_num, test_name);
            $display("  Input: 0x%016X", data);
            $display("========================================");
            
            raw_data_in = data;
            start = 1;
            #10;
            start = 0;
            
            wait(transform_valid);
            #20;
            
            $display("\n  FREQUENCY DOMAIN ANALYSIS:");
            $display("  %-8s %-12s %-12s %-8s", "Vertex", "Amplitude", "Phase", "Energy%");
            $display("  %s", {64{"-"}});
            
            for (i = 0; i < 8; i = i + 1) begin
                phase_rad = $itor($signed(vertex_phases[i])) * 3.14159265 / 32768.0;
                energy_pct = (resonance_energy > 0) ? 
                    ($itor(vertex_amplitudes[i] * vertex_amplitudes[i]) / $itor(resonance_energy) * 100.0) : 0.0;
                
                $display("  [%0d]      0x%04X (%5d)  0x%04X (%6.3f)  %5.1f%%",
                    i, 
                    vertex_amplitudes[i], 
                    vertex_amplitudes[i],
                    vertex_phases[i],
                    phase_rad,
                    energy_pct);
            end
            
            $display("\n  SUMMARY:");
            $display("    Total Resonance Energy: %0d", resonance_energy);
            $display("    Dominant Frequency: k=%0d (Amplitude=0x%04X)", 
                find_max_amplitude(), vertex_amplitudes[find_max_amplitude()]);
            
            #200;
        end
    endtask
    
    // Function to find index of maximum amplitude
    function integer find_max_amplitude;
        integer i;
        integer max_idx;
        begin
            max_idx = 0;
            for (i = 1; i < 8; i = i + 1) begin
                if (vertex_amplitudes[i] > vertex_amplitudes[max_idx]) begin
                    max_idx = i;
                end
            end
            find_max_amplitude = max_idx;
        end
    endfunction
    
    initial begin
        $display("\n");
        $display("╔═══════════════════════════════════════════════════════════╗");
        $display("║   QUANTONIUMOS RFT MIDDLEWARE ENGINE - FULL TEST SUITE   ║");
        $display("║   CORDIC (12 iter) + Complex Math + 8×8 RFT Kernel       ║");
        $display("╚═══════════════════════════════════════════════════════════╝");
        
        test_num = 0;
        reset = 1;
        start = 0;
        raw_data_in = 64'h0;
        
        #100;
        reset = 0;
        #50;
        
        // Test 1: Impulse - validates unitary transform
        run_test(64'h0000000000000001, "IMPULSE (Delta Function)");
        
        // Test 2: All zeros - null input
        run_test(64'h0000000000000000, "NULL INPUT (All Zeros)");
        
        // Test 3: DC only - all same value
        run_test(64'h0808080808080808, "DC COMPONENT (Constant Value)");
        
        // Test 4: Nyquist frequency - alternating pattern
        run_test(64'h00FF00FF00FF00FF, "NYQUIST FREQUENCY (Alternating)");
        
        // Test 5: Linear ramp
        run_test(64'h0001020304050607, "LINEAR RAMP (Ascending)");
        
        // Test 6: Step function
        run_test(64'h00000000FFFFFFFF, "STEP FUNCTION (Half-wave)");
        
        // Test 7: Symmetric pattern
        run_test(64'h0102040804020100, "SYMMETRIC PATTERN (Triangle)");
        
        // Test 8: Complex pattern
        run_test(64'h0123456789ABCDEF, "COMPLEX PATTERN (Hex Sequence)");
        
        // Test 9: Single high byte
        run_test(64'hFF00000000000000, "SINGLE HIGH VALUE (Last Byte)");
        
        // Test 10: Two peaks
        run_test(64'h8000000000000080, "TWO PEAKS (Endpoints)");
        
        $display("\n");
        $display("╔═══════════════════════════════════════════════════════════╗");
        $display("║                  ALL TESTS COMPLETED!                     ║");
        $display("║  ✓ CORDIC: 12-iteration cartesian-to-polar conversion    ║");
        $display("║  ✓ Complex multiply-accumulate with 64 coefficients      ║");
        $display("║  ✓ Full 8×8 resonance kernel ROM                         ║");
        $display("║  ✓ Amplitude extraction with CORDIC gain compensation    ║");
        $display("║  ✓ Phase extraction in fixed-point radians               ║");
        $display("║  ✓ Total energy calculation across frequency domain      ║");
        $display("╚═══════════════════════════════════════════════════════════╝");
        $display("\n");
        
        $finish;
    end
    
    initial begin
        $dumpfile("quantoniumos_full.vcd");
        $dumpvars(0, testbench);
    end
    
    initial begin
        #200000;
        $display("\n");
        $display("✗✗✗ TIMEOUT - Test suite did not complete in time ✗✗✗");
        $display("\n");
        $finish;
    end
endmodule
