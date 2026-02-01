// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
`timescale 1ns / 1ps

module fpga_top_tb;

    // Inputs
    reg WF_CLK;
    reg WF_BUTTON;

    // Outputs
    wire [7:0] WF_LED;

    // Instantiate the Unit Under Test (UUT)
    fpga_top uut (
        .WF_CLK(WF_CLK),
        .WF_BUTTON(WF_BUTTON),
        .WF_LED(WF_LED)
    );

    // Clock generation (12 MHz -> ~83.33 ns period)
    initial begin
        WF_CLK = 0;
        forever #41.67 WF_CLK = ~WF_CLK;
    end

    // Test Stimulus
    initial begin
        // Generate VCD waveform file
        $dumpfile("quantoniumos_sim.vcd");
        $dumpvars(0, fpga_top_tb);
        
        // Initialize Inputs (Button is active low, so 1 is unpressed)
        WF_BUTTON = 1; 

        $display("------------------------------------------------");
        $display(" QUANTONIUMOS FPGA SIMULATION START");
        $display("------------------------------------------------");
        
        // 1. Wait for internal reset (10 cycles)
        #1000;
        $display("[Time %0t] Internal Reset Complete.", $time);

        // 2. Observe Default Mode (Mode 14: Quantum Sim)
        #10000; 
        $display("[Time %0t] Mode 14 (Quantum Sim) Result LEDs: %b", $time, WF_LED);

        // 3. Switch to Mode 15 (Roundtrip)
        $display("[Time %0t] Pressing Button (Next Mode)...", $time);
        WF_BUTTON = 0; 
        #2500; 
        WF_BUTTON = 1; 
        $display("[Time %0t] Button Released.", $time);

        #10000;
        $display("[Time %0t] Mode 15 (Roundtrip) Result LEDs: %b", $time, WF_LED);

        // 4. Switch to Mode 0 (Golden Ratio)
        $display("[Time %0t] Pressing Button (Next Mode)...", $time);
        WF_BUTTON = 0;
        #2500;
        WF_BUTTON = 1;
        $display("[Time %0t] Button Released.", $time);

        #10000;
        $display("[Time %0t] Mode 0 (Golden) Result LEDs: %b", $time, WF_LED);

        $display("------------------------------------------------");
        $display(" SIMULATION COMPLETE");
        $display("------------------------------------------------");
        $finish;
    end

endmodule
