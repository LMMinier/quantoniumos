// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
// Patent Application: USPTO #19/169,399
//
// =============================================================================
// RFTPU v2.1: Enhanced Systolic Array with K-Tiling & Performance Counters
// =============================================================================
// Reference: Google TPU Paper (arXiv:1704.04760)
// 
// ENHANCEMENTS over v2.0:
//   - K-Tiling: Support for K > ARRAY_DIM via tile accumulation
//   - Separate weight/activation paths with weight FIFO
//   - On-chip scratchpad (Unified Buffer v0)
//   - SIS_HASH integration path (MODE_12)
//   - INT8 quantization with saturation/rounding
//   - Performance counters for energy analysis
//   - Configurable array size (8x8 → 16x16 → 64x64)
// =============================================================================

`default_nettype none
`timescale 1ns/1ps

// -----------------------------------------------------------------------------
// Enhanced Processing Element with Saturation
// -----------------------------------------------------------------------------
module systolic_pe_v2 #(
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter bit SATURATE   = 1       // Enable saturation on accumulator
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control
    input  wire                     load_weight,
    input  wire                     clear_acc,
    input  wire                     freeze_weight,  // NEW: Hold weight for multiple vectors
    
    // Data inputs
    input  wire [DATA_WIDTH-1:0]    weight_in,
    input  wire [DATA_WIDTH-1:0]    activation_in,
    input  wire [ACC_WIDTH-1:0]     psum_in,
    
    // Data outputs
    output reg  [DATA_WIDTH-1:0]    activation_out,
    output reg  [ACC_WIDTH-1:0]     psum_out,
    
    // Status (for performance counters)
    output wire                     mac_active
);

    reg [DATA_WIDTH-1:0] weight_reg;
    wire signed [2*DATA_WIDTH-1:0] product;
    wire signed [ACC_WIDTH-1:0] acc_next;
    wire overflow_pos, overflow_neg;
    
    // Signed multiplication
    assign product = $signed(weight_reg) * $signed(activation_in);
    
    // Sign-extended accumulation
    assign acc_next = $signed(psum_in) + $signed({{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product});
    
    // Overflow detection for saturation
    assign overflow_pos = !psum_in[ACC_WIDTH-1] && !product[2*DATA_WIDTH-1] && acc_next[ACC_WIDTH-1];
    assign overflow_neg = psum_in[ACC_WIDTH-1] && product[2*DATA_WIDTH-1] && !acc_next[ACC_WIDTH-1];
    
    // MAC is active when activation is non-zero
    assign mac_active = (activation_in != '0);
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg     <= '0;
            activation_out <= '0;
            psum_out       <= '0;
        end else begin
            // Weight loading (only when not frozen)
            if (load_weight && !freeze_weight) begin
                weight_reg <= weight_in;
            end
            
            // Systolic forwarding
            activation_out <= activation_in;
            
            // MAC with optional saturation
            if (clear_acc) begin
                psum_out <= '0;
            end else if (SATURATE) begin
                // Saturating accumulation
                if (overflow_pos)
                    psum_out <= {1'b0, {(ACC_WIDTH-1){1'b1}}}; // MAX_POS
                else if (overflow_neg)
                    psum_out <= {1'b1, {(ACC_WIDTH-1){1'b0}}}; // MAX_NEG
                else
                    psum_out <= acc_next;
            end else begin
                psum_out <= acc_next;
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Weight FIFO (Decouples weight loading from compute)
// -----------------------------------------------------------------------------
module weight_fifo #(
    parameter int DEPTH      = 16,
    parameter int DATA_WIDTH = 64   // One row of weights
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Write interface (from DMA/host)
    input  wire                     wr_en,
    input  wire [DATA_WIDTH-1:0]    wr_data,
    output wire                     full,
    
    // Read interface (to systolic array)
    input  wire                     rd_en,
    output wire [DATA_WIDTH-1:0]    rd_data,
    output wire                     empty,
    
    // Status
    output wire [$clog2(DEPTH):0]   count
);

    reg [DATA_WIDTH-1:0] mem [DEPTH];
    reg [$clog2(DEPTH):0] wr_ptr, rd_ptr;
    reg [$clog2(DEPTH):0] cnt;
    
    assign full  = (cnt == DEPTH);
    assign empty = (cnt == 0);
    assign count = cnt;
    assign rd_data = mem[rd_ptr[$clog2(DEPTH)-1:0]];
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            cnt    <= '0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr[$clog2(DEPTH)-1:0]] <= wr_data;
                wr_ptr <= wr_ptr + 1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
            
            // Update count
            case ({wr_en && !full, rd_en && !empty})
                2'b10: cnt <= cnt + 1;
                2'b01: cnt <= cnt - 1;
                default: cnt <= cnt;
            endcase
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Unified Buffer v0 (Small On-Chip Scratchpad)
// -----------------------------------------------------------------------------
module unified_buffer #(
    parameter int DEPTH       = 256,    // Number of vectors
    parameter int WIDTH       = 64,     // Bits per vector (8 x 8-bit)
    parameter int NUM_BANKS   = 2       // Ping-pong
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Port A (DMA write / Host interface)
    input  wire                     a_en,
    input  wire                     a_we,
    input  wire [$clog2(DEPTH)-1:0] a_addr,
    input  wire [WIDTH-1:0]         a_wdata,
    output reg  [WIDTH-1:0]         a_rdata,
    
    // Port B (Systolic array read/write)
    input  wire                     b_en,
    input  wire                     b_we,
    input  wire [$clog2(DEPTH)-1:0] b_addr,
    input  wire [WIDTH-1:0]         b_wdata,
    output reg  [WIDTH-1:0]         b_rdata,
    
    // Bank select (for ping-pong)
    input  wire                     bank_sel
);

    // Dual-port BRAM inference
    reg [WIDTH-1:0] bank0 [DEPTH/NUM_BANKS];
    reg [WIDTH-1:0] bank1 [DEPTH/NUM_BANKS];
    
    wire [$clog2(DEPTH/NUM_BANKS)-1:0] addr_a = a_addr[$clog2(DEPTH/NUM_BANKS)-1:0];
    wire [$clog2(DEPTH/NUM_BANKS)-1:0] addr_b = b_addr[$clog2(DEPTH/NUM_BANKS)-1:0];
    
    // Port A access
    always_ff @(posedge clk) begin
        if (a_en) begin
            if (a_we) begin
                if (bank_sel)
                    bank1[addr_a] <= a_wdata;
                else
                    bank0[addr_a] <= a_wdata;
            end
            a_rdata <= bank_sel ? bank1[addr_a] : bank0[addr_a];
        end
    end
    
    // Port B access (opposite bank for ping-pong)
    always_ff @(posedge clk) begin
        if (b_en) begin
            if (b_we) begin
                if (!bank_sel)
                    bank1[addr_b] <= b_wdata;
                else
                    bank0[addr_b] <= b_wdata;
            end
            b_rdata <= (!bank_sel) ? bank1[addr_b] : bank0[addr_b];
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Performance Counters
// -----------------------------------------------------------------------------
module perf_counters #(
    parameter int ARRAY_DIM = 8,
    parameter int CNT_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clear,
    
    // Events
    input  wire                     compute_enable,
    input  wire                     load_weights,
    input  wire                     stall,
    input  wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active,  // Per-PE activity
    
    // Counters
    output reg  [CNT_WIDTH-1:0]     cnt_compute_cycles,
    output reg  [CNT_WIDTH-1:0]     cnt_weight_load_cycles,
    output reg  [CNT_WIDTH-1:0]     cnt_stall_cycles,
    output reg  [CNT_WIDTH-1:0]     cnt_mac_ops,        // Total MAC operations
    output reg  [CNT_WIDTH-1:0]     cnt_total_cycles
);

    // Count active MACs this cycle
    wire [$clog2(ARRAY_DIM*ARRAY_DIM+1)-1:0] active_macs;
    
    // Population count (number of active MACs)
    integer i;
    reg [$clog2(ARRAY_DIM*ARRAY_DIM+1)-1:0] popcount;
    always_comb begin
        popcount = '0;
        for (i = 0; i < ARRAY_DIM*ARRAY_DIM; i++) begin
            popcount = popcount + mac_active[i];
        end
    end
    assign active_macs = popcount;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear) begin
            cnt_compute_cycles     <= '0;
            cnt_weight_load_cycles <= '0;
            cnt_stall_cycles       <= '0;
            cnt_mac_ops            <= '0;
            cnt_total_cycles       <= '0;
        end else begin
            cnt_total_cycles <= cnt_total_cycles + 1;
            
            if (compute_enable)
                cnt_compute_cycles <= cnt_compute_cycles + 1;
            
            if (load_weights)
                cnt_weight_load_cycles <= cnt_weight_load_cycles + 1;
            
            if (stall)
                cnt_stall_cycles <= cnt_stall_cycles + 1;
            
            if (compute_enable)
                cnt_mac_ops <= cnt_mac_ops + active_macs;
        end
    end

endmodule


// -----------------------------------------------------------------------------
// Systolic Array v2.1 with K-Tiling
// -----------------------------------------------------------------------------
module systolic_array_v2 #(
    parameter int ARRAY_DIM  = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter bit SATURATE   = 1
)(
    input  wire                                     clk,
    input  wire                                     rst_n,
    
    // Control
    input  wire                                     load_weights,
    input  wire                                     clear_acc,
    input  wire                                     compute_enable,
    input  wire                                     freeze_weights,     // NEW: Hold weights for K-tiling
    
    // K-Tiling control
    input  wire [$clog2(256)-1:0]                  k_tiles,           // Number of K tiles (K/ARRAY_DIM)
    input  wire                                     accumulate,        // NEW: Add to existing accumulators
    
    // Weight loading interface
    input  wire [$clog2(ARRAY_DIM)-1:0]            weight_row_sel,
    input  wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0]   weight_row_data,
    
    // Activation input
    input  wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0]   activation_in,
    
    // Output results
    output wire [ARRAY_DIM-1:0][ACC_WIDTH-1:0]    result_out,
    output wire                                     result_valid,
    
    // Performance counters interface
    output wire [ARRAY_DIM*ARRAY_DIM-1:0]         mac_active_out
);

    // PE interconnect
    wire [DATA_WIDTH-1:0] activation_h [ARRAY_DIM][ARRAY_DIM+1];
    wire [ACC_WIDTH-1:0]  psum_v       [ARRAY_DIM+1][ARRAY_DIM];
    wire load_pe [ARRAY_DIM][ARRAY_DIM];
    wire mac_active_pe [ARRAY_DIM][ARRAY_DIM];
    
    // Skew delay and cycle counter
    reg [DATA_WIDTH-1:0] skew_delay [ARRAY_DIM][ARRAY_DIM];
    reg [$clog2(2*ARRAY_DIM)+1:0] cycle_count;
    
    // K-tile state
    reg [$clog2(256)-1:0] current_k_tile;
    reg k_tile_done;
    
    // Accumulator storage for K-tiling (stores results between tiles)
    reg [ACC_WIDTH-1:0] acc_storage [ARRAY_DIM];
    wire use_stored_acc;
    
    assign use_stored_acc = (current_k_tile > 0) && accumulate;
    
    // Initialize top row psums (either zero or stored accumulators)
    genvar col;
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : init_psum
            assign psum_v[0][col] = use_stored_acc ? acc_storage[col] : '0;
        end
    endgenerate
    
    // Generate PE array
    genvar row;
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : pe_row
            for (col = 0; col < ARRAY_DIM; col++) begin : pe_col
                
                assign load_pe[row][col] = load_weights && (weight_row_sel == row);
                
                systolic_pe_v2 #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH),
                    .SATURATE(SATURATE)
                ) pe_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .load_weight(load_pe[row][col]),
                    .clear_acc(clear_acc && !use_stored_acc),
                    .freeze_weight(freeze_weights),
                    .weight_in(weight_row_data[col]),
                    .activation_in(activation_h[row][col]),
                    .psum_in(psum_v[row][col]),
                    .activation_out(activation_h[row][col+1]),
                    .psum_out(psum_v[row+1][col]),
                    .mac_active(mac_active_pe[row][col])
                );
                
                // Flatten mac_active for perf counters
                assign mac_active_out[row*ARRAY_DIM + col] = mac_active_pe[row][col];
            end
        end
    endgenerate
    
    // Activation skewing
    integer i, j;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < ARRAY_DIM; i++) begin
                for (j = 0; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= '0;
                end
            end
            cycle_count <= '0;
            current_k_tile <= '0;
            k_tile_done <= '0;
            for (i = 0; i < ARRAY_DIM; i++) begin
                acc_storage[i] <= '0;
            end
        end else if (clear_acc && !accumulate) begin
            // Full clear - reset K-tile counter and storage
            cycle_count <= '0;
            current_k_tile <= '0;
            k_tile_done <= '0;
            for (i = 0; i < ARRAY_DIM; i++) begin
                acc_storage[i] <= '0;
            end
        end else if (compute_enable) begin
            cycle_count <= cycle_count + 1;
            
            // Skew delay pipeline
            for (i = 0; i < ARRAY_DIM; i++) begin
                skew_delay[i][0] <= activation_in[i];
                for (j = 1; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= skew_delay[i][j-1];
                end
            end
            
            // Store results when valid for K-tiling
            if (result_valid && (current_k_tile < k_tiles - 1)) begin
                for (i = 0; i < ARRAY_DIM; i++) begin
                    acc_storage[i] <= psum_v[ARRAY_DIM][i];
                end
                current_k_tile <= current_k_tile + 1;
            end else if (result_valid && (current_k_tile >= k_tiles - 1)) begin
                k_tile_done <= 1'b1;
            end
        end else begin
            cycle_count <= '0;
            for (i = 0; i < ARRAY_DIM; i++) begin
                for (j = 0; j < ARRAY_DIM; j++) begin
                    skew_delay[i][j] <= '0;
                end
            end
        end
    end
    
    // Connect skewed activations
    generate
        for (row = 0; row < ARRAY_DIM; row++) begin : connect_act
            if (row == 0) begin
                assign activation_h[row][0] = skew_delay[row][0];
            end else begin
                assign activation_h[row][0] = skew_delay[row][row-1];
            end
        end
    endgenerate
    
    // Output connections
    generate
        for (col = 0; col < ARRAY_DIM; col++) begin : connect_out
            assign result_out[col] = psum_v[ARRAY_DIM][col];
        end
    endgenerate
    
    assign result_valid = (cycle_count >= 2*ARRAY_DIM - 1);

endmodule


// -----------------------------------------------------------------------------
// RFTPU v2.1 Integration Wrapper with All Features
// -----------------------------------------------------------------------------
module rftpu_systolic_v2 #(
    parameter int ARRAY_DIM   = 8,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int UB_DEPTH    = 256,
    parameter int WEIGHT_FIFO_DEPTH = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,
    
    // Control interface
    input  wire                     start,
    input  wire [3:0]               mode,
    input  wire [15:0]              num_vectors,    // M dimension
    input  wire [7:0]               k_tiles,        // K/ARRAY_DIM
    
    // Weight FIFO interface (from DMA)
    input  wire                     weight_wr_en,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_wr_data,
    output wire                     weight_fifo_full,
    
    // Activation interface (from DMA)
    input  wire                     act_wr_en,
    input  wire [ARRAY_DIM*DATA_WIDTH-1:0] act_wr_data,
    
    // Result interface (to DMA)
    output wire [ARRAY_DIM*ACC_WIDTH-1:0] result_data,
    output wire                     result_valid,
    
    // Status
    output wire                     busy,
    output wire                     done,
    
    // Performance counters
    output wire [31:0]              perf_compute_cycles,
    output wire [31:0]              perf_weight_cycles,
    output wire [31:0]              perf_stall_cycles,
    output wire [31:0]              perf_mac_ops,
    output wire [31:0]              perf_total_cycles
);

    // Mode definitions
    localparam MODE_SYSTOLIC_GEMM = 4'hF;
    localparam MODE_SIS_HASH      = 4'hC;  // SIS hash via systolic path
    
    // State machine
    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_WEIGHTS,
        S_COMPUTE_TILE,
        S_NEXT_K_TILE,
        S_OUTPUT,
        S_DONE
    } state_t;
    
    state_t state, next_state;
    
    // Control signals
    reg load_weights_r;
    reg clear_acc_r;
    reg compute_enable_r;
    reg freeze_weights_r;
    reg accumulate_r;
    reg stall_r;
    
    // Counters
    reg [$clog2(ARRAY_DIM):0] row_counter;
    reg [15:0] vector_counter;
    reg [7:0] k_tile_counter;
    
    // Weight FIFO signals
    wire weight_fifo_empty;
    wire [ARRAY_DIM*DATA_WIDTH-1:0] weight_fifo_data;
    reg weight_fifo_rd;
    
    // Systolic array signals
    wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0] weight_vec;
    wire [ARRAY_DIM-1:0][DATA_WIDTH-1:0] activation_vec;
    wire [ARRAY_DIM-1:0][ACC_WIDTH-1:0] result_vec;
    wire systolic_valid;
    wire [ARRAY_DIM*ARRAY_DIM-1:0] mac_active;
    
    // Activation buffer
    reg [ARRAY_DIM*DATA_WIDTH-1:0] act_buffer;
    reg act_buffer_valid;
    
    // Unpack vectors
    genvar i;
    generate
        for (i = 0; i < ARRAY_DIM; i++) begin : unpack
            assign weight_vec[i] = weight_fifo_data[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
            assign activation_vec[i] = act_buffer[(i+1)*DATA_WIDTH-1 : i*DATA_WIDTH];
            assign result_data[(i+1)*ACC_WIDTH-1 : i*ACC_WIDTH] = result_vec[i];
        end
    endgenerate
    
    // Weight FIFO instance
    weight_fifo #(
        .DEPTH(WEIGHT_FIFO_DEPTH),
        .DATA_WIDTH(ARRAY_DIM*DATA_WIDTH)
    ) wfifo (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(weight_wr_en),
        .wr_data(weight_wr_data),
        .full(weight_fifo_full),
        .rd_en(weight_fifo_rd),
        .rd_data(weight_fifo_data),
        .empty(weight_fifo_empty),
        .count()
    );
    
    // Systolic array v2 instance
    systolic_array_v2 #(
        .ARRAY_DIM(ARRAY_DIM),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .SATURATE(1)
    ) systolic (
        .clk(clk),
        .rst_n(rst_n),
        .load_weights(load_weights_r),
        .clear_acc(clear_acc_r),
        .compute_enable(compute_enable_r),
        .freeze_weights(freeze_weights_r),
        .k_tiles(k_tiles),
        .accumulate(accumulate_r),
        .weight_row_sel(row_counter[$clog2(ARRAY_DIM)-1:0]),
        .weight_row_data(weight_vec),
        .activation_in(activation_vec),
        .result_out(result_vec),
        .result_valid(systolic_valid),
        .mac_active_out(mac_active)
    );
    
    // Performance counters
    perf_counters #(
        .ARRAY_DIM(ARRAY_DIM),
        .CNT_WIDTH(32)
    ) perf (
        .clk(clk),
        .rst_n(rst_n),
        .clear(state == S_IDLE && start),
        .compute_enable(compute_enable_r),
        .load_weights(load_weights_r),
        .stall(stall_r),
        .mac_active(mac_active),
        .cnt_compute_cycles(perf_compute_cycles),
        .cnt_weight_load_cycles(perf_weight_cycles),
        .cnt_stall_cycles(perf_stall_cycles),
        .cnt_mac_ops(perf_mac_ops),
        .cnt_total_cycles(perf_total_cycles)
    );
    
    // Activation buffer logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_buffer <= '0;
            act_buffer_valid <= 1'b0;
        end else begin
            if (act_wr_en) begin
                act_buffer <= act_wr_data;
                act_buffer_valid <= 1'b1;
            end else if (compute_enable_r) begin
                act_buffer_valid <= 1'b0;
            end
        end
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        load_weights_r = 1'b0;
        clear_acc_r = 1'b0;
        compute_enable_r = 1'b0;
        freeze_weights_r = 1'b0;
        accumulate_r = 1'b0;
        weight_fifo_rd = 1'b0;
        stall_r = 1'b0;
        
        case (state)
            S_IDLE: begin
                if (start && (mode == MODE_SYSTOLIC_GEMM || mode == MODE_SIS_HASH)) begin
                    next_state = S_LOAD_WEIGHTS;
                    clear_acc_r = 1'b1;
                end
            end
            
            S_LOAD_WEIGHTS: begin
                if (!weight_fifo_empty) begin
                    load_weights_r = 1'b1;
                    weight_fifo_rd = 1'b1;
                    if (row_counter >= ARRAY_DIM - 1) begin
                        next_state = S_COMPUTE_TILE;
                    end
                end else begin
                    stall_r = 1'b1;  // Waiting for weights
                end
            end
            
            S_COMPUTE_TILE: begin
                freeze_weights_r = 1'b1;  // Hold weights
                accumulate_r = (k_tile_counter > 0);  // Accumulate after first tile
                
                if (act_buffer_valid) begin
                    compute_enable_r = 1'b1;
                    if (systolic_valid && vector_counter >= num_vectors - 1) begin
                        if (k_tile_counter < k_tiles - 1) begin
                            next_state = S_NEXT_K_TILE;
                        end else begin
                            next_state = S_OUTPUT;
                        end
                    end
                end else begin
                    stall_r = 1'b1;  // Waiting for activations
                end
            end
            
            S_NEXT_K_TILE: begin
                // Load next weight tile
                next_state = S_LOAD_WEIGHTS;
            end
            
            S_OUTPUT: begin
                if (systolic_valid) begin
                    next_state = S_DONE;
                end
            end
            
            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end
    
    // Counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_counter <= '0;
            vector_counter <= '0;
            k_tile_counter <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    row_counter <= '0;
                    vector_counter <= '0;
                    k_tile_counter <= '0;
                end
                
                S_LOAD_WEIGHTS: begin
                    if (!weight_fifo_empty)
                        row_counter <= row_counter + 1;
                end
                
                S_COMPUTE_TILE: begin
                    row_counter <= '0;
                    if (compute_enable_r)
                        vector_counter <= vector_counter + 1;
                end
                
                S_NEXT_K_TILE: begin
                    k_tile_counter <= k_tile_counter + 1;
                    vector_counter <= '0;
                end
                
                default: ;
            endcase
        end
    end
    
    // Status outputs
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    assign result_valid = systolic_valid && (state == S_OUTPUT || state == S_COMPUTE_TILE);

endmodule


`default_nettype wire
