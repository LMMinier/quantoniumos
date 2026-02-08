// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
//
// Theorem 8 Option A: Energy-dump testbench
// =========================================
//
// Contract:
//   - "Hardware" computes canonical RFT outputs y = U_phi^H x (here: behavioral LUT model).
//   - Testbench dumps coefficient energies |y[k]|^2 to CSV.
//   - Host computes K99, mean gap, CI, pass/fail.
//
// This avoids implementing sort/top-K logic in RTL.
//
// Usage (from repo root):
//   make -C hardware/tb theorem8-run
//
// Outputs:
//   hardware/tb/theorem8_hw_energies.csv

`timescale 1ns/1ps

module tb_theorem8_energy_dump;

  `ifndef THEOREM8_N
    `define THEOREM8_N 64
  `endif
  `ifndef THEOREM8_CASES
    `define THEOREM8_CASES 200
  `endif

  localparam int N = `THEOREM8_N;
  localparam int SAMPLE_WIDTH = 16; // Q1.15
  localparam int NUM_CASES_DEFAULT = `THEOREM8_CASES;
  localparam int WORDS_PER_CASE = 2 * N;
  localparam int MAX_CASES = 2000;
  localparam int MAX_WORDS = MAX_CASES * WORDS_PER_CASE;
  localparam int CORE_LATENCY = 1;

  // Clock + reset
  logic clk;
  logic rst_n;


  // Start/valid handshake
  logic start;
  logic output_valid;

  // Inputs
  logic [N*SAMPLE_WIDTH-1:0] samples_real;
  logic [N*SAMPLE_WIDTH-1:0] samples_imag;

  // Vector memory (max sized)
  logic [SAMPLE_WIDTH-1:0] vec_mem [0:MAX_WORDS-1];

  // Kernel memory: conj(U)[n,k] stored as kernel[k*N + n]
  logic signed [15:0] kernel_real [0:N*N-1];
  logic signed [15:0] kernel_imag [0:N*N-1];

  // Clock generation
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  // Valid pipeline (CORE_LATENCY=1)
  logic valid_d;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      valid_d <= 1'b0;
      output_valid <= 1'b0;
    end else begin
      valid_d <= start;
      output_valid <= valid_d;
    end
  end

  // Compute y = U^H x with complex kernel (Q1.15 × Q1.15 → Q2.30; sum over N)
  logic signed [15:0] input_real_arr [0:N-1];
  logic signed [15:0] input_imag_arr [0:N-1];
  logic signed [63:0] acc_real [0:N-1];
  logic signed [63:0] acc_imag [0:N-1];
  logic signed [31:0] y_real_q15 [0:N-1];
  logic signed [31:0] y_imag_q15 [0:N-1];
  logic signed [63:0] y_real_shift;
  logic signed [63:0] y_imag_shift;

  always_comb begin
    for (int i = 0; i < N; i++) begin
      input_real_arr[i] = samples_real[i*SAMPLE_WIDTH +: SAMPLE_WIDTH];
      input_imag_arr[i] = samples_imag[i*SAMPLE_WIDTH +: SAMPLE_WIDTH];
    end

    for (int k = 0; k < N; k++) begin
      acc_real[k] = 0;
      acc_imag[k] = 0;
      for (int n = 0; n < N; n++) begin
        logic signed [15:0] a;
        logic signed [15:0] b;
        logic signed [15:0] c;
        logic signed [15:0] d;
        logic signed [31:0] ac;
        logic signed [31:0] bd;
        logic signed [31:0] ad;
        logic signed [31:0] bc;

        a = kernel_real[k*N + n];
        b = kernel_imag[k*N + n];
        c = input_real_arr[n];
        d = input_imag_arr[n];

        ac = a * c;
        bd = b * d;
        ad = a * d;
        bc = b * c;

        // (a+jb)*(c+jd) = (ac - bd) + j(ad + bc)
        acc_real[k] = acc_real[k] + (ac - bd);
        acc_imag[k] = acc_imag[k] + (ad + bc);
      end
      // Convert accumulated Q2.30 to Q1.15-ish by shifting 15.
      // Keep 32-bit for stable squaring without overflow.
      y_real_shift = $signed(acc_real[k] >>> 15);
      y_imag_shift = $signed(acc_imag[k] >>> 15);
      y_real_q15[k] = y_real_shift[31:0];
      y_imag_q15[k] = y_imag_shift[31:0];
    end
  end

  task automatic load_case(input int tid);
    int base;
    begin
      base = tid * WORDS_PER_CASE;
      for (int i = 0; i < N; i++) begin
        samples_real[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = vec_mem[base + i];
        samples_imag[i*SAMPLE_WIDTH +: SAMPLE_WIDTH] = vec_mem[base + N + i];
      end
    end
  endtask

  integer output_file;
  string vec_file;
  string kreal_file;
  string kimag_file;
  string out_csv;
  int num_cases;

  initial begin
    vec_file = "theorem8_vectors_N64_q15.memh";
    kreal_file = "theorem8_kernel_uconj_real_N64_q15.memh";
    kimag_file = "theorem8_kernel_uconj_imag_N64_q15.memh";
    out_csv = "theorem8_hw_energies.csv";
    num_cases = NUM_CASES_DEFAULT;

    void'($value$plusargs("VEC=%s", vec_file));
    void'($value$plusargs("KREAL=%s", kreal_file));
    void'($value$plusargs("KIMAG=%s", kimag_file));
    void'($value$plusargs("OUT=%s", out_csv));
    void'($value$plusargs("CASES=%d", num_cases));

    if (num_cases > MAX_CASES) begin
      $display("ERROR: CASES=%0d exceeds MAX_CASES=%0d", num_cases, MAX_CASES);
      $finish;
    end

    $display("Loading vectors: %s", vec_file);
    $readmemh(vec_file, vec_mem);
    $display("Loading kernel real: %s", kreal_file);
    $readmemh(kreal_file, kernel_real);
    $display("Loading kernel imag: %s", kimag_file);
    $readmemh(kimag_file, kernel_imag);

    output_file = $fopen(out_csv, "w");
    if (output_file == 0) begin
      $display("ERROR: Cannot open %s", out_csv);
      $finish;
    end
    $fwrite(output_file, "case_id,k,out_real,out_imag,energy\n");

    // Reset
    rst_n = 1'b0;
    start = 1'b0;
    samples_real = '0;
    samples_imag = '0;
    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    $display("=== Theorem 8 Option A: dumping coefficient energies (N=%0d, cases=%0d) ===", N, num_cases);
    for (int case_id = 0; case_id < num_cases; case_id++) begin
      load_case(case_id);

      @(posedge clk);
      start = 1'b1;
      @(posedge clk);
      start = 1'b0;

      wait(output_valid);
      @(posedge clk);

      for (int k = 0; k < N; k++) begin
        logic signed [31:0] yr;
        logic signed [31:0] yi;
        logic signed [63:0] e;
        yr = y_real_q15[k];
        yi = y_imag_q15[k];
        e = (yr * yr) + (yi * yi); // scale cancels for K99
        $fwrite(output_file, "%0d,%0d,%0d,%0d,%0d\n", case_id, k, yr, yi, e);
      end

      if ((case_id % 25) == 0) $display("  case %0d/%0d", case_id, num_cases);
      repeat (3) @(posedge clk);
    end

    $fclose(output_file);
    $display("Wrote: %s", out_csv);
    $finish;
  end

  // Timeout watchdog
  initial begin
    #500000;
    $display("ERROR: Simulation timeout");
    $finish;
  end

endmodule
