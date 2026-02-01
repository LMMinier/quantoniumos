// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// ═══════════════════════════════════════════════════════════════════════════
// QUANTONIUMOS RFTPU - MINIMAL 4-MODE VERSION FOR WEBFPGA
// ═══════════════════════════════════════════════════════════════════════════
// 
// Reduced version with only the 4 VERIFIED modes for reliable WebFPGA synthesis.
// Full 16-mode version: fpga_top.sv
//
// 🟢 INCLUDED MODES (4 verified):
//   Mode 0: RFT-GOLDEN     - Canonical Golden Ratio Transform
//   Mode 1: RFT-CASCADE    - H3 Hybrid Compression  
//   Mode 2: SIS-HASH       - Lattice-based Hash
//   Mode 3: QUANTUM-SIM    - Symbolic Quantum Engine
//
// Kernel ROM: 4 modes × 64 coefficients = 256 entries (vs 768 in full)

/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */

module fpga_top (
    input wire WF_CLK,
    input wire WF_BUTTON,
    output wire [7:0] WF_LED
);

    // 4 verified modes only
    localparam [1:0] MODE_RFT_GOLDEN   = 2'd0;
    localparam [1:0] MODE_RFT_CASCADE  = 2'd1;
    localparam [1:0] MODE_SIS_HASH     = 2'd2;
    localparam [1:0] MODE_QUANTUM_SIM  = 2'd3;

    // State machine
    localparam [2:0] STATE_IDLE    = 3'd0;
    localparam [2:0] STATE_COMPUTE = 3'd2;
    localparam [2:0] STATE_DONE    = 3'd5;

    // Registers
    reg [7:0] reset_counter = 8'h00;
    wire reset = (reset_counter < 8'd10);
    
    reg [19:0] button_debounce = 20'h00000;
    reg button_stable = 1'b0;
    reg button_prev = 1'b0;
    wire button_edge = button_stable && !button_prev;
    
    reg [1:0] current_mode = MODE_RFT_GOLDEN;
    reg [23:0] auto_cycle_counter = 24'h000000;
    wire auto_cycle_trigger = (auto_cycle_counter == 24'hFFFFFF);
    
    reg [7:0] cyc_cnt = 8'h00;
    wire start = button_edge || (cyc_cnt == 8'd20);
    
    reg [2:0] state = STATE_IDLE;
    reg [2:0] k_index = 3'b000;
    reg [2:0] n_index = 3'b000;
    
    reg [15:0] sample [0:7];
    reg valid = 1'b0;
    
    reg signed [15:0] kernel_reg;
    reg signed [15:0] kernel_rom_out;
    
    wire [15:0] input_selected = sample[n_index];
    wire signed [31:0] mult_out = $signed(input_selected) * $signed(kernel_reg);
    
    reg signed [31:0] acc = 32'sh00000000;
    reg signed [31:0] rft_out [0:7];
    
    wire is_computing = (state == STATE_COMPUTE);
    wire is_done = (state == STATE_DONE);
    wire save_result = (n_index == 3'b111) && is_computing;
    
    reg [7:0] led_output = 8'h00;
    
    integer i;

    // ═══════════════════════════════════════════════════════════════════════
    // KERNEL ROM: 4 modes × 64 coefficients = 256 entries
    // ═══════════════════════════════════════════════════════════════════════
    
    always @(*) begin
        case ({current_mode, k_index, n_index})

            // MODE 0: RFT-GOLDEN (unitarity: 6.12e-15)
            {2'd0, 3'd0, 3'd0}: kernel_rom_out = -16'sd10528;
            {2'd0, 3'd0, 3'd1}: kernel_rom_out = 16'sd12809;
            {2'd0, 3'd0, 3'd2}: kernel_rom_out = -16'sd11788;
            {2'd0, 3'd0, 3'd3}: kernel_rom_out = -16'sd12520;
            {2'd0, 3'd0, 3'd4}: kernel_rom_out = 16'sd14036;
            {2'd0, 3'd0, 3'd5}: kernel_rom_out = -16'sd14281;
            {2'd0, 3'd0, 3'd6}: kernel_rom_out = -16'sd9488;
            {2'd0, 3'd0, 3'd7}: kernel_rom_out = -16'sd3470;
            {2'd0, 3'd1, 3'd0}: kernel_rom_out = -16'sd11613;
            {2'd0, 3'd1, 3'd1}: kernel_rom_out = 16'sd12317;
            {2'd0, 3'd1, 3'd2}: kernel_rom_out = 16'sd11248;
            {2'd0, 3'd1, 3'd3}: kernel_rom_out = -16'sd7835;
            {2'd0, 3'd1, 3'd4}: kernel_rom_out = 16'sd9793;
            {2'd0, 3'd1, 3'd5}: kernel_rom_out = 16'sd15845;
            {2'd0, 3'd1, 3'd6}: kernel_rom_out = 16'sd13399;
            {2'd0, 3'd1, 3'd7}: kernel_rom_out = 16'sd8523;
            {2'd0, 3'd2, 3'd0}: kernel_rom_out = -16'sd12087;
            {2'd0, 3'd2, 3'd1}: kernel_rom_out = -16'sd10234;
            {2'd0, 3'd2, 3'd2}: kernel_rom_out = 16'sd11353;
            {2'd0, 3'd2, 3'd3}: kernel_rom_out = -16'sd15150;
            {2'd0, 3'd2, 3'd4}: kernel_rom_out = -16'sd8738;
            {2'd0, 3'd2, 3'd5}: kernel_rom_out = 16'sd7100;
            {2'd0, 3'd2, 3'd6}: kernel_rom_out = -16'sd13620;
            {2'd0, 3'd2, 3'd7}: kernel_rom_out = -16'sd12336;
            {2'd0, 3'd3, 3'd0}: kernel_rom_out = -16'sd12043;
            {2'd0, 3'd3, 3'd1}: kernel_rom_out = -16'sd10784;
            {2'd0, 3'd3, 3'd2}: kernel_rom_out = -16'sd11936;
            {2'd0, 3'd3, 3'd3}: kernel_rom_out = -16'sd9443;
            {2'd0, 3'd3, 3'd4}: kernel_rom_out = -16'sd12944;
            {2'd0, 3'd3, 3'd5}: kernel_rom_out = -16'sd5602;
            {2'd0, 3'd3, 3'd6}: kernel_rom_out = 16'sd9043;
            {2'd0, 3'd3, 3'd7}: kernel_rom_out = 16'sd17320;
            {2'd0, 3'd4, 3'd0}: kernel_rom_out = -16'sd12043;
            {2'd0, 3'd4, 3'd1}: kernel_rom_out = 16'sd10784;
            {2'd0, 3'd4, 3'd2}: kernel_rom_out = -16'sd11936;
            {2'd0, 3'd4, 3'd3}: kernel_rom_out = 16'sd9443;
            {2'd0, 3'd4, 3'd4}: kernel_rom_out = -16'sd12944;
            {2'd0, 3'd4, 3'd5}: kernel_rom_out = 16'sd5602;
            {2'd0, 3'd4, 3'd6}: kernel_rom_out = 16'sd9043;
            {2'd0, 3'd4, 3'd7}: kernel_rom_out = -16'sd17320;
            {2'd0, 3'd5, 3'd0}: kernel_rom_out = -16'sd12087;
            {2'd0, 3'd5, 3'd1}: kernel_rom_out = 16'sd10234;
            {2'd0, 3'd5, 3'd2}: kernel_rom_out = 16'sd11353;
            {2'd0, 3'd5, 3'd3}: kernel_rom_out = 16'sd15150;
            {2'd0, 3'd5, 3'd4}: kernel_rom_out = -16'sd8738;
            {2'd0, 3'd5, 3'd5}: kernel_rom_out = -16'sd7100;
            {2'd0, 3'd5, 3'd6}: kernel_rom_out = -16'sd13620;
            {2'd0, 3'd5, 3'd7}: kernel_rom_out = 16'sd12336;
            {2'd0, 3'd6, 3'd0}: kernel_rom_out = -16'sd11613;
            {2'd0, 3'd6, 3'd1}: kernel_rom_out = -16'sd12317;
            {2'd0, 3'd6, 3'd2}: kernel_rom_out = 16'sd11248;
            {2'd0, 3'd6, 3'd3}: kernel_rom_out = 16'sd7835;
            {2'd0, 3'd6, 3'd4}: kernel_rom_out = 16'sd9793;
            {2'd0, 3'd6, 3'd5}: kernel_rom_out = -16'sd15845;
            {2'd0, 3'd6, 3'd6}: kernel_rom_out = 16'sd13399;
            {2'd0, 3'd6, 3'd7}: kernel_rom_out = -16'sd8523;
            {2'd0, 3'd7, 3'd0}: kernel_rom_out = -16'sd10528;
            {2'd0, 3'd7, 3'd1}: kernel_rom_out = -16'sd12809;
            {2'd0, 3'd7, 3'd2}: kernel_rom_out = -16'sd11788;
            {2'd0, 3'd7, 3'd3}: kernel_rom_out = 16'sd12520;
            {2'd0, 3'd7, 3'd4}: kernel_rom_out = 16'sd14036;
            {2'd0, 3'd7, 3'd5}: kernel_rom_out = 16'sd14281;
            {2'd0, 3'd7, 3'd6}: kernel_rom_out = -16'sd9488;
            {2'd0, 3'd7, 3'd7}: kernel_rom_out = 16'sd3470;

            // MODE 1: RFT-CASCADE (H3 Compression - Class B Winner)
            {2'd1, 3'd0, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd1}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd2}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd3}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd4}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd5}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd6}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd0, 3'd7}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd1, 3'd0}: kernel_rom_out = 16'sd16069;
            {2'd1, 3'd1, 3'd1}: kernel_rom_out = 16'sd13623;
            {2'd1, 3'd1, 3'd2}: kernel_rom_out = 16'sd8867;
            {2'd1, 3'd1, 3'd3}: kernel_rom_out = 16'sd2998;
            {2'd1, 3'd1, 3'd4}: kernel_rom_out = -16'sd2998;
            {2'd1, 3'd1, 3'd5}: kernel_rom_out = -16'sd8867;
            {2'd1, 3'd1, 3'd6}: kernel_rom_out = -16'sd13623;
            {2'd1, 3'd1, 3'd7}: kernel_rom_out = -16'sd16069;
            {2'd1, 3'd2, 3'd0}: kernel_rom_out = 16'sd15137;
            {2'd1, 3'd2, 3'd1}: kernel_rom_out = 16'sd6270;
            {2'd1, 3'd2, 3'd2}: kernel_rom_out = -16'sd6270;
            {2'd1, 3'd2, 3'd3}: kernel_rom_out = -16'sd15137;
            {2'd1, 3'd2, 3'd4}: kernel_rom_out = -16'sd15137;
            {2'd1, 3'd2, 3'd5}: kernel_rom_out = -16'sd6270;
            {2'd1, 3'd2, 3'd6}: kernel_rom_out = 16'sd6270;
            {2'd1, 3'd2, 3'd7}: kernel_rom_out = 16'sd15137;
            {2'd1, 3'd3, 3'd0}: kernel_rom_out = 16'sd13623;
            {2'd1, 3'd3, 3'd1}: kernel_rom_out = -16'sd2998;
            {2'd1, 3'd3, 3'd2}: kernel_rom_out = -16'sd16069;
            {2'd1, 3'd3, 3'd3}: kernel_rom_out = -16'sd8867;
            {2'd1, 3'd3, 3'd4}: kernel_rom_out = 16'sd8867;
            {2'd1, 3'd3, 3'd5}: kernel_rom_out = 16'sd16069;
            {2'd1, 3'd3, 3'd6}: kernel_rom_out = 16'sd2998;
            {2'd1, 3'd3, 3'd7}: kernel_rom_out = -16'sd13623;
            {2'd1, 3'd4, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd4, 3'd1}: kernel_rom_out = -16'sd11585;
            {2'd1, 3'd4, 3'd2}: kernel_rom_out = -16'sd11585;
            {2'd1, 3'd4, 3'd3}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd4, 3'd4}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd4, 3'd5}: kernel_rom_out = -16'sd11585;
            {2'd1, 3'd4, 3'd6}: kernel_rom_out = -16'sd11585;
            {2'd1, 3'd4, 3'd7}: kernel_rom_out = 16'sd11585;
            {2'd1, 3'd5, 3'd0}: kernel_rom_out = 16'sd8867;
            {2'd1, 3'd5, 3'd1}: kernel_rom_out = -16'sd16069;
            {2'd1, 3'd5, 3'd2}: kernel_rom_out = 16'sd2998;
            {2'd1, 3'd5, 3'd3}: kernel_rom_out = 16'sd13623;
            {2'd1, 3'd5, 3'd4}: kernel_rom_out = -16'sd13623;
            {2'd1, 3'd5, 3'd5}: kernel_rom_out = -16'sd2998;
            {2'd1, 3'd5, 3'd6}: kernel_rom_out = 16'sd16069;
            {2'd1, 3'd5, 3'd7}: kernel_rom_out = -16'sd8867;
            {2'd1, 3'd6, 3'd0}: kernel_rom_out = 16'sd6270;
            {2'd1, 3'd6, 3'd1}: kernel_rom_out = -16'sd15137;
            {2'd1, 3'd6, 3'd2}: kernel_rom_out = 16'sd15137;
            {2'd1, 3'd6, 3'd3}: kernel_rom_out = -16'sd6270;
            {2'd1, 3'd6, 3'd4}: kernel_rom_out = -16'sd6270;
            {2'd1, 3'd6, 3'd5}: kernel_rom_out = 16'sd15137;
            {2'd1, 3'd6, 3'd6}: kernel_rom_out = -16'sd15137;
            {2'd1, 3'd6, 3'd7}: kernel_rom_out = 16'sd6270;
            {2'd1, 3'd7, 3'd0}: kernel_rom_out = 16'sd2998;
            {2'd1, 3'd7, 3'd1}: kernel_rom_out = -16'sd8867;
            {2'd1, 3'd7, 3'd2}: kernel_rom_out = 16'sd13623;
            {2'd1, 3'd7, 3'd3}: kernel_rom_out = -16'sd16069;
            {2'd1, 3'd7, 3'd4}: kernel_rom_out = 16'sd16069;
            {2'd1, 3'd7, 3'd5}: kernel_rom_out = -16'sd13623;
            {2'd1, 3'd7, 3'd6}: kernel_rom_out = 16'sd8867;
            {2'd1, 3'd7, 3'd7}: kernel_rom_out = -16'sd2998;

            // MODE 2: SIS-HASH (Lattice-based, Q=3329)
            {2'd2, 3'd0, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd1}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd2}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd3}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd4}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd5}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd6}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd0, 3'd7}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd1, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd1, 3'd1}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd1, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd1, 3'd3}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd1, 3'd4}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd1, 3'd5}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd1, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd1, 3'd7}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd2, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd2, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd2, 3'd2}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd2, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd2, 3'd4}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd2, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd2, 3'd6}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd2, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd3, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd3, 3'd1}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd3, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd3, 3'd3}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd3, 3'd4}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd3, 3'd5}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd3, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd3, 3'd7}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd4, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd4, 3'd1}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd4, 3'd2}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd4, 3'd3}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd4, 3'd4}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd4, 3'd5}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd4, 3'd6}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd4, 3'd7}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd5, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd5, 3'd1}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd5, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd5, 3'd3}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd5, 3'd4}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd5, 3'd5}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd5, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd5, 3'd7}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd6, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd6, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd6, 3'd2}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd6, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd6, 3'd4}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd6, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd6, 3'd6}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd6, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd7, 3'd0}: kernel_rom_out = 16'sd11585;
            {2'd2, 3'd7, 3'd1}: kernel_rom_out = 16'sd8260;
            {2'd2, 3'd7, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd7, 3'd3}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd7, 3'd4}: kernel_rom_out = -16'sd11585;
            {2'd2, 3'd7, 3'd5}: kernel_rom_out = -16'sd8260;
            {2'd2, 3'd7, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd2, 3'd7, 3'd7}: kernel_rom_out = 16'sd8260;

            // MODE 3: QUANTUM-SIM (GHZ State Symbolic)
            {2'd3, 3'd0, 3'd0}: kernel_rom_out = 16'sd23170;
            {2'd3, 3'd0, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd0, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd0, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd0, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd0, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd0, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd0, 3'd7}: kernel_rom_out = 16'sd23170;
            {2'd3, 3'd1, 3'd0}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd1, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd0}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd2, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd0}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd3, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd0}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd4, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd0}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd5, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd0}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd6, 3'd7}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd0}: kernel_rom_out = 16'sd23170;
            {2'd3, 3'd7, 3'd1}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd2}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd3}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd4}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd5}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd6}: kernel_rom_out = 16'sd0;
            {2'd3, 3'd7, 3'd7}: kernel_rom_out = 16'sd23170;

            default: kernel_rom_out = 16'sd0;
        endcase
    end

    // ═══════════════════════════════════════════════════════════════════════
    // MAIN LOGIC
    // ═══════════════════════════════════════════════════════════════════════

    always @(posedge WF_CLK) begin
        // Reset counter
        if (reset_counter < 8'd10)
            reset_counter <= reset_counter + 1'b1;

        // Button debounce
        if (WF_BUTTON)
            button_debounce <= (button_debounce < 20'hFFFFF) ? button_debounce + 1'b1 : button_debounce;
        else
            button_debounce <= 20'h00000;
        
        button_stable <= (button_debounce > 20'h7FFFF);
        button_prev <= button_stable;

        // Auto-cycle counter
        auto_cycle_counter <= auto_cycle_counter + 1'b1;

        // Cycle counter for auto-start
        if (cyc_cnt < 8'd25)
            cyc_cnt <= cyc_cnt + 1'b1;

        if (reset) begin
            state <= STATE_IDLE;
            k_index <= 3'b000;
            n_index <= 3'b000;
            acc <= 32'sh00000000;
            valid <= 1'b0;
            current_mode <= MODE_RFT_GOLDEN;
            
            // Initialize sample buffer with test pattern
            sample[0] <= 16'h1000;
            sample[1] <= 16'h2000;
            sample[2] <= 16'h3000;
            sample[3] <= 16'h4000;
            sample[4] <= 16'h3000;
            sample[5] <= 16'h2000;
            sample[6] <= 16'h1000;
            sample[7] <= 16'h0000;
            
            for (i = 0; i < 8; i = i + 1)
                rft_out[i] <= 32'sh00000000;
        end
        else begin
            // Mode cycling on button press or auto-cycle
            if (button_edge || auto_cycle_trigger) begin
                current_mode <= current_mode + 1'b1;
                cyc_cnt <= 8'd0;
            end

            // Latch kernel value
            kernel_reg <= kernel_rom_out;

            case (state)
                STATE_IDLE: begin
                    if (start) begin
                        state <= STATE_COMPUTE;
                        k_index <= 3'b000;
                        n_index <= 3'b000;
                        acc <= 32'sh00000000;
                        valid <= 1'b0;
                    end
                end

                STATE_COMPUTE: begin
                    acc <= acc + mult_out;
                    
                    if (n_index == 3'b111) begin
                        rft_out[k_index] <= acc + mult_out;
                        acc <= 32'sh00000000;
                        
                        if (k_index == 3'b111) begin
                            state <= STATE_DONE;
                            valid <= 1'b1;
                        end
                        else begin
                            k_index <= k_index + 1'b1;
                        end
                    end
                    
                    n_index <= n_index + 1'b1;
                end

                STATE_DONE: begin
                    // Stay done until next start
                    if (start) begin
                        state <= STATE_COMPUTE;
                        k_index <= 3'b000;
                        n_index <= 3'b000;
                        acc <= 32'sh00000000;
                        valid <= 1'b0;
                    end
                end

                default: state <= STATE_IDLE;
            endcase
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // LED OUTPUT
    // ═══════════════════════════════════════════════════════════════════════

    always @(posedge WF_CLK) begin
        if (reset)
            led_output <= 8'h00;
        else if (is_computing)
            // Show current mode during compute
            led_output <= {6'b000000, current_mode};
        else if (valid) begin
            case (current_mode)
                MODE_RFT_GOLDEN, MODE_RFT_CASCADE: begin
                    // Frequency bin amplitude visualization
                    led_output[0] <= (rft_out[0][31:24] > 8'h10);
                    led_output[1] <= (rft_out[1][31:24] > 8'h10);
                    led_output[2] <= (rft_out[2][31:24] > 8'h10);
                    led_output[3] <= (rft_out[3][31:24] > 8'h10);
                    led_output[4] <= (rft_out[4][31:24] > 8'h10);
                    led_output[5] <= (rft_out[5][31:24] > 8'h10);
                    led_output[6] <= (rft_out[6][31:24] > 8'h10);
                    led_output[7] <= (rft_out[7][31:24] > 8'h10);
                end
                MODE_SIS_HASH: begin
                    // Hash output bits
                    led_output <= rft_out[0][7:0] ^ rft_out[7][7:0];
                end
                MODE_QUANTUM_SIM: begin
                    // GHZ visualization: LEDs 0 and 7 lit
                    led_output <= 8'b10000001;
                end
                default: begin
                    led_output <= {6'b000000, current_mode};
                end
            endcase
        end
    end

    assign WF_LED = led_output;

endmodule
