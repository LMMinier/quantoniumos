#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Theorem 8: Golden Spectral Concentration Inequality - Hardware Test Suite

Purpose
-------
Generate deterministic, hardware-ready test vectors + statistically certified benchmarks
for verifying Theorem 8 on FPGA/ASIC implementations of the canonical RFT basis.

What this script *claims* (honest)
----------------------------------
Finite-N empirical inequality under a fully specified ensemble and metric:

    For x ~ E_phi (defined below),
        E[ K0.99(U_phi^H x) ] < E[ K0.99(F^H x) ].

This script reports:
- paired improvement Δ := K_FFT(x) - K_RFT(x)
- bootstrap 95% CI for mean(Δ)
- win rate = P[Δ >= 0] with Wilson 95% CI

What this script does NOT claim
-------------------------------
- Any asymptotic limsup/liminf theorem without a separate mathematical proof.
- Universal superiority over FFT on arbitrary signals.

Ensemble (canonical, fully specified)
-------------------------------------
E_phi: x[n] = exp(i 2π (f0*n + a*frac(n*phi))),
    f0 ~ Uniform[0,1),
    a  ~ Uniform[a_min, a_max]  (default [-1,1])
    phi = (1+sqrt(5))/2
    frac(t) = t mod 1

Metric
------
K_threshold(c) = smallest K such that top-K coefficient energies capture >= threshold of energy.

Contract discipline:
- Theorem 8 is an *expectation* claim over the ensemble. Individual samples may go either way.
- Hardware validation should therefore gate on aggregated quantities like mean(Δ) > 0, not per-case wins.

Hardware I/O format
-------------------
Test vectors are written to a single sequential .memh file per N:

WORDS_PER_CASE = 2*N + 1  (all 32-bit hex words)

Case layout (word indices relative to case base):
  [0 .. N-1]           : signal_real[i]  in Q16.16 (signed 32-bit packed as unsigned hex)
  [N .. 2N-1]          : signal_imag[i]  in Q16.16
    [2N]                 : packed reference K values (software reference only):
                                                     bits [15:0]  = K_RFT
                                                     bits [31:16] = K_FFT

This makes $readmemh straightforward with a single memory and deterministic indexing.
"""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from statistics import NormalDist
from typing import List, Dict, Tuple, Optional

# Add project root to path (repo root assumed parent of "hardware/")
sys.path.insert(0, str(Path(__file__).parent.parent))

PHI = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

DEFAULT_THRESHOLD = 0.99
DEFAULT_A_MIN = -1.0
DEFAULT_A_MAX = 1.0

Q_FRAC_BITS = 16
Q_SCALE = 1 << Q_FRAC_BITS


# =============================================================================
# Statistical utilities (paired certification)
# =============================================================================

@dataclass(frozen=True)
class BootstrapCI:
    alpha: float
    lower: float
    upper: float
    n_resamples: int


@dataclass(frozen=True)
class WilsonCI:
    alpha: float
    lower: float
    upper: float


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.05,
    n_resamples: int = 5000,
) -> BootstrapCI:
    """
    Nonparametric bootstrap CI for the mean of 'values' (paired deltas recommended).
    Returns percentile CI [alpha/2, 1-alpha/2].
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return BootstrapCI(alpha=alpha, lower=float("nan"), upper=float("nan"), n_resamples=n_resamples)

    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return BootstrapCI(alpha=alpha, lower=lo, upper=hi, n_resamples=n_resamples)


def wilson_score_interval(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> WilsonCI:
    """
    Wilson score interval for a Bernoulli proportion.
    Uses normal approximation for z; adequate for reporting and CI gating.
    """
    if trials <= 0:
        return WilsonCI(alpha=alpha, lower=float("nan"), upper=float("nan"))

    # z for two-sided alpha, computed without SciPy.
    z = float(NormalDist().inv_cdf(1.0 - alpha / 2.0))
    n = float(trials)
    phat = float(successes) / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return WilsonCI(alpha=alpha, lower=float(lo), upper=float(hi))


def paired_effect_size_d(delta: np.ndarray) -> float:
    """
    Paired Cohen's d for deltas: mean(delta) / std(delta).
    If std is ~0, returns 0 (conservative).
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size < 2:
        return 0.0
    sd = float(np.std(delta, ddof=1))
    if sd <= 1e-15:
        return 0.0
    return float(np.mean(delta) / sd)


# =============================================================================
# Core transform implementations (self-contained, stable)
# =============================================================================

def canonical_rft_basis(N: int) -> np.ndarray:
    """
    Canonical RFT basis U = Φ (ΦᴴΦ)^(-1/2) using a Hermitian eigendecomposition.

    This is numerically safer than sqrtm(inv(G)) and preserves Hermitian structure.
    """
    import scipy.linalg  # local import

    k = np.arange(N, dtype=np.float64)
    f_k = np.mod((k + 1.0) * PHI, 1.0)  # frac((k+1)phi)

    n = np.arange(N, dtype=np.float64).reshape(-1, 1)
    Phi = np.exp(2j * np.pi * (n * f_k.reshape(1, -1))) / np.sqrt(N)

    # Gram matrix (Hermitian PD)
    G = Phi.conj().T @ Phi
    # Hermitian eigendecomposition
    w, V = scipy.linalg.eigh(G)

    w_min = float(np.min(w))
    if w_min <= 0.0:
        raise ValueError(f"Gram matrix is not positive definite (min eigenvalue {w_min:.3e}) at N={N}")

    inv_sqrt_w = 1.0 / np.sqrt(w)
    G_inv_sqrt = (V * inv_sqrt_w.reshape(1, -1)) @ V.conj().T
    U = Phi @ G_inv_sqrt

    return U.astype(np.complex128, copy=False)


def fft_unitary_basis(N: int) -> np.ndarray:
    """Unitary DFT basis F[n,k] = exp(-i 2π nk/N) / √N."""
    n = np.arange(N, dtype=np.float64).reshape(-1, 1)
    k = np.arange(N, dtype=np.float64).reshape(1, -1)
    F = np.exp(-2j * np.pi * (n * k) / float(N)) / np.sqrt(N)
    return F.astype(np.complex128, copy=False)


def golden_drift_signal(N: int, f0: float, a: float) -> np.ndarray:
    """x[n] = exp(i 2π (f0*n + a*frac(n*phi)))."""
    n = np.arange(N, dtype=np.float64)
    frac_n_phi = np.mod(n * PHI, 1.0)
    x = np.exp(2j * np.pi * (f0 * n + a * frac_n_phi))
    return x.astype(np.complex128, copy=False)


def harmonic_signal(N: int, k: int, phase: float) -> np.ndarray:
    """FFT-native negative control: pure harmonic at integer bin k with random phase."""
    n = np.arange(N, dtype=np.float64)
    x = np.exp(1j * (2.0 * np.pi * float(k) * n / float(N) + phase))
    return x.astype(np.complex128, copy=False)


def k99(coeffs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> int:
    """Smallest K such that top-K coefficient energies capture ≥threshold of energy."""
    energy = np.abs(coeffs) ** 2
    total = float(energy.sum())
    if total <= 1e-15:
        return 1
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy) / total
    return int(np.searchsorted(cumsum, threshold) + 1)


# =============================================================================
# Hardware test vector generation
# =============================================================================

@dataclass
class ConcentrationTestCase:
    test_id: int
    N: int
    f0: float
    a: float
    signal_real: np.ndarray
    signal_imag: np.ndarray
    k99_rft: int
    k99_fft: int

    @property
    def delta(self) -> int:
        """Paired improvement: Δ = K_FFT - K_RFT (positive => RFT better)."""
        return int(self.k99_fft - self.k99_rft)

    @property
    def rft_wins(self) -> bool:
        """RFT win condition consistent with inequality statement."""
        return bool(self.k99_rft <= self.k99_fft)


def float_to_q16_16(value: float) -> int:
    """Convert float to Q16.16 fixed-point (32-bit signed), returned as 32-bit unsigned int."""
    fixed = int(np.round(float(value) * Q_SCALE))
    fixed = max(-(1 << 31), min((1 << 31) - 1, fixed))
    if fixed < 0:
        fixed = (1 << 32) + fixed
    return fixed & 0xFFFFFFFF


def generate_theorem8_test_vectors(
    N: int,
    num_cases: int,
    seed: int,
    threshold: float = DEFAULT_THRESHOLD,
    a_min: float = DEFAULT_A_MIN,
    a_max: float = DEFAULT_A_MAX,
    output_dir: Optional[Path] = None,
    bootstrap_resamples: int = 5000,
) -> List[ConcentrationTestCase]:
    """
    Generate software-ground-truth vectors for hardware verification.

    Writes:
            - theorem8_vectors_N{N}.hex   (single-memory, sequential layout; $readmemh compatible)
      - theorem8_summary_N{N}.json  (stats + per-case metadata)
            - theorem8_tb_N{N}.sv         (testbench skeleton consistent with .hex layout)
    """
    if output_dir is None:
        output_dir = Path("hardware_test_vectors/theorem8")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Bases
    U_phi = canonical_rft_basis(N)
    F = fft_unitary_basis(N)

    test_cases: List[ConcentrationTestCase] = []

    print(f"\n{'='*70}")
    print("THEOREM 8: Golden Spectral Concentration Inequality")
    print("Hardware Test Vector Generation")
    print(f"{'='*70}")
    print(f"N = {N}")
    print(f"cases = {num_cases}")
    print(f"seed = {seed}")
    print(f"K-threshold = {threshold}")
    print(f"a ~ Uniform[{a_min}, {a_max}]")
    print(f"phi = {PHI:.15f}")
    print(f"{'='*70}\n")

    for test_id in range(num_cases):
        f0 = float(rng.uniform(0.0, 1.0))
        a = float(rng.uniform(a_min, a_max))
        x = golden_drift_signal(N, f0, a)

        rft_coeffs = U_phi.conj().T @ x
        fft_coeffs = F.conj().T @ x

        k_rft = k99(rft_coeffs, threshold=threshold)
        k_fft = k99(fft_coeffs, threshold=threshold)

        test_cases.append(
            ConcentrationTestCase(
                test_id=test_id,
                N=N,
                f0=f0,
                a=a,
                signal_real=x.real.astype(np.float64, copy=False),
                signal_imag=x.imag.astype(np.float64, copy=False),
                k99_rft=int(k_rft),
                k99_fft=int(k_fft),
            )
        )

    # Paired statistics
    deltas = np.array([tc.delta for tc in test_cases], dtype=np.float64)
    wins = int(np.sum(deltas >= 0.0))
    mean_rft = float(np.mean([tc.k99_rft for tc in test_cases]))
    mean_fft = float(np.mean([tc.k99_fft for tc in test_cases]))
    mean_delta = float(np.mean(deltas))
    d_paired = float(paired_effect_size_d(deltas))

    # Certified intervals
    ci_rng = np.random.default_rng(seed + 1000003 + N)
    delta_ci = bootstrap_mean_ci(deltas, rng=ci_rng, alpha=0.05, n_resamples=bootstrap_resamples)
    win_ci = wilson_score_interval(wins, num_cases, alpha=0.05)

    gap_pct = (mean_delta / mean_fft * 100.0) if mean_fft > 1e-15 else 0.0

    print("Results Summary (paired certification):")
    print(f"  E[K99(RFT)] = {mean_rft:.3f}")
    print(f"  E[K99(FFT)] = {mean_fft:.3f}")
    print(f"  Δ = E[FFT - RFT] = {mean_delta:.3f}  ({gap_pct:.2f}%)")
    print(
        f"  95% bootstrap CI for mean(Δ): [{delta_ci.lower:.3f}, {delta_ci.upper:.3f}]  (B={delta_ci.n_resamples})"
    )
    print(f"  Win rate P[Δ>=0] = {wins}/{num_cases} = {wins/num_cases:.1%}")
    print(f"  95% Wilson CI for win rate: [{win_ci.lower:.1%}, {win_ci.upper:.1%}]")
    print(f"  Paired effect size d = {d_paired:.3f}")
    print(f"  Inequality check (means): {mean_rft < mean_fft}")
    if delta_ci.lower > 0.0:
        print("  ✓ Certified: mean(Δ) > 0 at 95% bootstrap CI")
    else:
        print("  ⚠ Not CI-certified at 95%: CI includes 0 (increase cases/resamples or adjust ensemble strength)")

    # Write artifacts
    mem_path = output_dir / f"theorem8_vectors_N{N}.hex"
    json_path = output_dir / f"theorem8_summary_N{N}.json"
    sv_path = output_dir / f"theorem8_tb_N{N}.sv"

    _write_memh_vectors(test_cases, mem_path)
    _write_json_summary(
        test_cases=test_cases,
        filepath=json_path,
        threshold=threshold,
        a_min=a_min,
        a_max=a_max,
        bootstrap_resamples=bootstrap_resamples,
        delta_ci=delta_ci,
        win_ci=win_ci,
        paired_d=d_paired,
    )
    _write_verilog_testbench(test_cases, sv_path, mem_filename=mem_path.name)

    return test_cases


def _write_memh_vectors(test_cases: List[ConcentrationTestCase], filepath: Path) -> None:
    """Write a single sequential memory file (32-bit words) for Verilog $readmemh."""
    if not test_cases:
        raise ValueError("No test cases provided")

    N = int(test_cases[0].N)
    words_per_case = 2 * N

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("// Theorem 8: Golden Spectral Concentration Inequality Test Vectors\n")
        f.write("// Generated by hardware/theorem8_concentration_test.py\n")
        f.write(f"// N = {N}, cases = {len(test_cases)}\n")
        f.write(f"// WORDS_PER_CASE = {words_per_case} (real[N], imag[N])\n")
        f.write("// Layout per case:\n")
        f.write("//   [0..N-1]     signal_real Q16.16\n")
        f.write("//   [N..2N-1]    signal_imag Q16.16\n")
        f.write("//\n")
        f.write("// NOTE (contract discipline):\n")
        f.write("// - Theorem 8 is an expectation statement; per-sample wins/losses are normal.\n")
        f.write("// - This .hex intentionally carries *no* per-case expected K values.\n")
        f.write("//   Hardware verification should gate on an ensemble statistic (e.g., mean(Δ)>0)\n")
        f.write("//   computed from DUT outputs, not on float64 software expectations.\n")
        f.write("\n")

        for tc in test_cases:
            f.write(f"// Case {tc.test_id}: f0={tc.f0:.9f}, a={tc.a:.9f}\n")

            # Write real words
            for i in range(N):
                w = float_to_q16_16(float(tc.signal_real[i]))
                f.write(f"{w:08X}\n")

            # Write imag words
            for i in range(N):
                w = float_to_q16_16(float(tc.signal_imag[i]))
                f.write(f"{w:08X}\n")

    print(f"  ✓ Vectors written: {filepath}")


def _write_json_summary(
    test_cases: List[ConcentrationTestCase],
    filepath: Path,
    threshold: float,
    a_min: float,
    a_max: float,
    bootstrap_resamples: int,
    delta_ci: BootstrapCI,
    win_ci: WilsonCI,
    paired_d: float,
) -> None:
    """Write JSON summary including certified paired statistics and per-case metadata."""
    N = int(test_cases[0].N)
    num_cases = len(test_cases)

    k_rft = np.array([tc.k99_rft for tc in test_cases], dtype=np.float64)
    k_fft = np.array([tc.k99_fft for tc in test_cases], dtype=np.float64)
    deltas = k_fft - k_rft
    wins = int(np.sum(deltas >= 0.0))

    words_per_case = 2 * N

    summary = {
        "theorem": "Theorem 8: Golden Spectral Concentration Inequality (hardware vectors + certified stats)",
        "date_context": "2026-02-06",
        "ensemble_spec": {
            "signal": "x[n] = exp(i 2π (f0*n + a*frac(n*phi)))",
            "phi": float(PHI),
            "f0_distribution": "Uniform[0,1)",
            "a_distribution": f"Uniform[{a_min},{a_max}]",
            "frac_definition": "frac(t) = t mod 1",
        },
        "metric_spec": {
            "name": "K0.99",
            "threshold": float(threshold),
            "definition": "smallest K such that top-K energy mass >= threshold",
        },
        "hardware_format": {
            "file": str(filepath.name),
            "word_bits": 32,
            "q_format": "Q16.16 (signed stored as 32-bit hex words)",
            "words_per_case": int(words_per_case),
            "layout": {
                "real_words": f"[0..{N-1}]",
                "imag_words": f"[{N}..{2*N-1}]",
                "note": "No per-case expected K values stored in vector file; see JSON for software reference only.",
            },
        },
        "N": int(N),
        "num_cases": int(num_cases),
        "statistics": {
            "mean_k99_rft": float(np.mean(k_rft)),
            "mean_k99_fft": float(np.mean(k_fft)),
            "std_k99_rft": float(np.std(k_rft)),
            "std_k99_fft": float(np.std(k_fft)),
            "mean_delta_fft_minus_rft": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "paired_effect_size_d": float(paired_d),
            "win_rate": float(wins / num_cases),
            "wins": int(wins),
            "bootstrap_mean_delta_ci_95": {
                "lower": float(delta_ci.lower),
                "upper": float(delta_ci.upper),
                "resamples": int(delta_ci.n_resamples),
                "alpha": float(delta_ci.alpha),
            },
            "wilson_win_rate_ci_95": {
                "lower": float(win_ci.lower),
                "upper": float(win_ci.upper),
                "alpha": float(win_ci.alpha),
            },
        },
        "test_cases": [
            {
                "id": int(tc.test_id),
                "f0": float(tc.f0),
                "a": float(tc.a),
                "k99_rft": int(tc.k99_rft),
                "k99_fft": int(tc.k99_fft),
                "delta": int(tc.delta),
                "rft_wins": bool(tc.rft_wins),
            }
            for tc in test_cases
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ JSON summary written: {filepath}")


def _write_verilog_testbench(
    test_cases: List[ConcentrationTestCase],
    filepath: Path,
    mem_filename: str,
) -> None:
    """
    Generate a SystemVerilog testbench skeleton consistent with the single-memory .memh layout.

        NOTE:
        - DUT instantiation is left as a placeholder because module names/ports vary by your RFTPU design.
        - This testbench intentionally does NOT compare per-case K values to software float expectations.
            It demonstrates expectation/ensemble gating by computing mean(Δ) from DUT outputs.
    """
    N = int(test_cases[0].N)
    num_cases = len(test_cases)
    words_per_case = 2 * N
    total_words = num_cases * words_per_case

    sv = f"""// Theorem 8: Golden Spectral Concentration Inequality - Hardware Testbench
// Auto-generated by hardware/theorem8_concentration_test.py
// N = {N}, cases = {num_cases}
//
// Memory file: {mem_filename}
// WORDS_PER_CASE = {words_per_case}
// Layout per case:
//   base+0 .. base+N-1      : real[i] Q16.16
//   base+N .. base+2N-1     : imag[i] Q16.16
//
// Contract discipline:
// - Theorem 8 is an expectation statement; do NOT require per-sample wins.
// - Gate on an ensemble statistic from DUT outputs, e.g. mean(Δ)>0 where Δ=K_fft-K_rft.

`timescale 1ns/1ps

module theorem8_tb;

  localparam int N = {N};
  localparam int NUM_TESTS = {num_cases};
  localparam int WORDS_PER_CASE = {words_per_case};
  localparam int TOTAL_WORDS = {total_words};
  localparam int DATA_WIDTH = 32;

  reg clk = 0;
  reg rst_n = 0;
  reg start = 0;

  // Single memory holding all vectors sequentially
  reg [DATA_WIDTH-1:0] mem [0:TOTAL_WORDS-1];

  // Current test vectors
  reg [DATA_WIDTH-1:0] signal_real [0:N-1];
  reg [DATA_WIDTH-1:0] signal_imag [0:N-1];

  // DUT outputs (wire these to your actual module)
  wire done;
  wire [15:0] k99_rft_hw;
  wire [15:0] k99_fft_hw;

  integer test_id;
  integer i;
    integer sum_delta = 0;
    integer win_count = 0;

  // Clock generation
  always #10 clk = ~clk;

  initial begin
    $display("Loading vectors: {mem_filename}");
    $readmemh("{mem_filename}", mem);
  end

  // Example DUT placeholder (replace with your RFTPU module)
  // rftpu_concentration_engine #(.N(N)) dut (
  //   .clk(clk),
  //   .rst_n(rst_n),
  //   .start(start),
  //   .signal_real(signal_real),
  //   .signal_imag(signal_imag),
  //   .done(done),
  //   .k99_rft(k99_rft_hw),
  //   .k99_fft(k99_fft_hw)
  // );

  task automatic load_case(input int tid);
    int base;
    begin
      base = tid * WORDS_PER_CASE;
      for (i = 0; i < N; i = i + 1) begin
        signal_real[i] = mem[base + i];
        signal_imag[i] = mem[base + N + i];
      end
    end
  endtask

  initial begin
    $display("=========================================================");
    $display("THEOREM 8: Golden Spectral Concentration Inequality");
    $display("Hardware Verification Testbench");
    $display("N = %0d, cases = %0d", N, NUM_TESTS);
    $display("=========================================================");

    // Reset
    rst_n = 0;
    repeat (10) @(posedge clk);
    rst_n = 1;
    repeat (5) @(posedge clk);

    // Run all tests
    for (test_id = 0; test_id < NUM_TESTS; test_id = test_id + 1) begin
      load_case(test_id);

      start = 1;
      @(posedge clk);
      start = 0;

      // Wait for done (replace with your actual done handshake)
      // Here we just delay as a placeholder.
      repeat (200) @(posedge clk);

            // Once DUT is wired, compute ensemble statistic from DUT outputs.
            // Suggested handshake:
            //   wait(done);
            //   @(posedge clk);
            begin
                integer delta;
                delta = $signed({{16{1'b0}}, k99_fft_hw}) - $signed({{16{1'b0}}, k99_rft_hw});
                sum_delta = sum_delta + delta;
                if (delta >= 0) win_count = win_count + 1;
            end
    end

    $display("=========================================================");
        $display("VERIFICATION SUMMARY (ensemble / expectation gating)");
        $display("  sum_delta = %0d", sum_delta);
        $display("  mean_delta = %0f", sum_delta * 1.0 / NUM_TESTS);
        $display("  win_rate (Δ>=0) = %0f", win_count * 1.0 / NUM_TESTS);
        if (sum_delta > 0) begin
            $display("  PASS: mean(Δ) > 0 for this deterministic ensemble");
        end else begin
            $fatal(1, "FAIL: mean(Δ) <= 0 for this deterministic ensemble");
        end
    $display("=========================================================");
    $finish;
  end

endmodule
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sv)

    print(f"  ✓ Verilog testbench written: {filepath}")


# =============================================================================
# Benchmarks for paper (certified + LaTeX)
# =============================================================================

def run_theorem8_hardware_benchmark(
    sizes: List[int],
    samples_per_size: int,
    seed: int,
    threshold: float = DEFAULT_THRESHOLD,
    a_min: float = DEFAULT_A_MIN,
    a_max: float = DEFAULT_A_MAX,
    bootstrap_resamples: int = 8000,
    include_negative_control: bool = True,
) -> Dict:
    """
    Run certified finite-N benchmark for paper documentation.

    Returns a dict with per-N results including:
    - mean K99 for RFT and FFT
    - paired mean Δ and bootstrap CI
    - win rate and Wilson CI
    """
    print("\n" + "=" * 70)
    print("THEOREM 8: Certified Hardware Benchmark (finite-N)")
    print("=" * 70)
    print(f"sizes = {sizes}")
    print(f"samples_per_size = {samples_per_size}")
    print(f"seed = {seed}")
    print(f"threshold = {threshold}")
    print(f"a ~ Uniform[{a_min},{a_max}]")
    print(f"bootstrap_resamples = {bootstrap_resamples}")
    print("=" * 70)

    results: Dict = {
        "theorem": "Theorem 8: Golden Spectral Concentration Inequality (finite-N certified benchmark)",
        "date_context": "2026-02-06",
        "ensemble_spec": {
            "signal": "x[n] = exp(i 2π (f0*n + a*frac(n*phi)))",
            "phi": float(PHI),
            "f0_distribution": "Uniform[0,1)",
            "a_distribution": f"Uniform[{a_min},{a_max}]",
        },
        "metric_spec": {
            "name": "K0.99",
            "threshold": float(threshold),
        },
        "sizes": [],
        "negative_control": None,
    }

    for N in sizes:
        # Independent RNG stream per N (deterministic but avoids coupling across sizes)
        rng = np.random.default_rng(seed + 10007 * N)

        U_phi = canonical_rft_basis(N)
        F = fft_unitary_basis(N)

        k_rft = np.empty(samples_per_size, dtype=np.float64)
        k_fft = np.empty(samples_per_size, dtype=np.float64)

        for i in range(samples_per_size):
            f0 = float(rng.uniform(0.0, 1.0))
            a = float(rng.uniform(a_min, a_max))
            x = golden_drift_signal(N, f0, a)

            rft_coeffs = U_phi.conj().T @ x
            fft_coeffs = F.conj().T @ x

            k_rft[i] = float(k99(rft_coeffs, threshold=threshold))
            k_fft[i] = float(k99(fft_coeffs, threshold=threshold))

        delta = k_fft - k_rft
        wins = int(np.sum(delta >= 0.0))

        ci_rng = np.random.default_rng(seed + 777777 + N)
        delta_ci = bootstrap_mean_ci(delta, rng=ci_rng, alpha=0.05, n_resamples=bootstrap_resamples)
        win_ci = wilson_score_interval(wins, samples_per_size, alpha=0.05)

        mean_rft = float(np.mean(k_rft))
        mean_fft = float(np.mean(k_fft))
        mean_delta = float(np.mean(delta))
        std_rft = float(np.std(k_rft))
        std_fft = float(np.std(k_fft))
        std_delta = float(np.std(delta))
        d_paired = float(paired_effect_size_d(delta))
        gap_pct = (mean_delta / mean_fft * 100.0) if mean_fft > 1e-15 else 0.0

        size_result = {
            "N": int(N),
            "samples": int(samples_per_size),
            "E_K99_RFT": round(mean_rft, 4),
            "E_K99_FFT": round(mean_fft, 4),
            "std_RFT": round(std_rft, 4),
            "std_FFT": round(std_fft, 4),
            "mean_delta_fft_minus_rft": round(mean_delta, 4),
            "std_delta": round(std_delta, 4),
            "gap_percent": round(gap_pct, 3),
            "wins": int(wins),
            "win_rate": round(float(wins / samples_per_size), 4),
            "win_rate_wilson_ci_95": {
                "lower": round(float(win_ci.lower), 4),
                "upper": round(float(win_ci.upper), 4),
            },
            "mean_delta_bootstrap_ci_95": {
                "lower": round(float(delta_ci.lower), 4),
                "upper": round(float(delta_ci.upper), 4),
                "resamples": int(delta_ci.n_resamples),
            },
            "paired_effect_size_d": round(float(d_paired), 4),
            "inequality_holds_by_means": bool(mean_rft < mean_fft),
            "ci_certified_mean_delta_positive": bool(delta_ci.lower > 0.0),
        }

        results["sizes"].append(size_result)

        print(f"\nN = {N}:")
        print(f"  E[K99(RFT)] = {mean_rft:.3f} ± {std_rft:.3f}")
        print(f"  E[K99(FFT)] = {mean_fft:.3f} ± {std_fft:.3f}")
        print(f"  mean Δ = {mean_delta:.3f} ({gap_pct:.2f}%)")
        print(f"  95% bootstrap CI mean(Δ): [{delta_ci.lower:.3f}, {delta_ci.upper:.3f}]")
        print(f"  win rate = {wins/samples_per_size:.1%}  (Wilson 95%: [{win_ci.lower:.1%}, {win_ci.upper:.1%}])")
        print(f"  paired d = {d_paired:.3f}")
        print(f"  CI-certified mean(Δ) > 0: {delta_ci.lower > 0.0}")

    if include_negative_control:
        # Negative control: harmonics (FFT native)
        nc_sizes = sizes
        nc_rows = []
        print("\n" + "=" * 70)
        print("Negative Control: Harmonic Ensemble (FFT-native)")
        print("=" * 70)

        for N in nc_sizes:
            rng = np.random.default_rng(seed + 99991 * N)
            U_phi = canonical_rft_basis(N)
            F = fft_unitary_basis(N)

            k_rft = np.empty(samples_per_size, dtype=np.float64)
            k_fft = np.empty(samples_per_size, dtype=np.float64)

            for i in range(samples_per_size):
                kbin = int(rng.integers(0, N))
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                x = harmonic_signal(N, kbin, phase)

                k_rft[i] = float(k99(U_phi.conj().T @ x, threshold=threshold))
                k_fft[i] = float(k99(F.conj().T @ x, threshold=threshold))

            row = {
                "N": int(N),
                "samples": int(samples_per_size),
                "E_K99_RFT": round(float(np.mean(k_rft)), 4),
                "E_K99_FFT": round(float(np.mean(k_fft)), 4),
                "std_RFT": round(float(np.std(k_rft)), 4),
                "std_FFT": round(float(np.std(k_fft)), 4),
            }
            nc_rows.append(row)

            print(f"\nN = {N}:")
            print(f"  E[K99(FFT)] = {row['E_K99_FFT']:.3f} (should be near 1)")
            print(f"  E[K99(RFT)] = {row['E_K99_RFT']:.3f} (should be >> FFT)")

        results["negative_control"] = {
            "ensemble": "harmonic",
            "definition": "x[n] = exp(i (2π k n/N + phase)), k integer, phase uniform",
            "rows": nc_rows,
        }

    # LaTeX table (certified)
    results["latex_table_certified"] = _generate_latex_table_certified(results)
    results["latex_table_negative_control"] = (
        _generate_latex_table_negative_control(results) if include_negative_control else ""
    )

    return results


def _generate_latex_table_certified(results: Dict) -> str:
    """Generate a reviewer-safe LaTeX table with paired Δ + CI + win rate."""
    rows = results["sizes"]

    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Theorem 8 (finite-$N$): paired improvement $\\Delta = K_{0.99}(\\mathrm{FFT}) - K_{0.99}(\\mathrm{RFT})$ with bootstrap CI and win-rate CI under the golden drift ensemble.}\n"
        "\\label{tab:theorem8_certified}\n"
        "\\begin{tabular}{c|cc|c|c|c}\n"
        "\\toprule\n"
        "$N$ & $\\mathbb{E}[K_{0.99}(\\mathrm{RFT})]$ & $\\mathbb{E}[K_{0.99}(\\mathrm{FFT})]$ & $\\mathbb{E}[\\Delta]$ & 95\\% CI (mean $\\Delta$) & Win-rate (95\\% CI) \\\\" "\n"
        "\\midrule\n"
    )
    for s in rows:
        ci = s["mean_delta_bootstrap_ci_95"]
        wci = s["win_rate_wilson_ci_95"]
        latex += (
            f"{s['N']} & "
            f"{s['E_K99_RFT']:.2f} & {s['E_K99_FFT']:.2f} & "
            f"{s['mean_delta_fft_minus_rft']:.2f} & "
            f"[{ci['lower']:.2f}, {ci['upper']:.2f}] & "
            f"{100.0*s['win_rate']:.1f}\\% "
            f"([{100.0*wci['lower']:.1f}\\%, {100.0*wci['upper']:.1f}\\%]) \\\\\n"
        )

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex.strip() + "\n"


def _generate_latex_table_negative_control(results: Dict) -> str:
    nc = results.get("negative_control")
    if not nc:
        return ""

    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Negative control (harmonics): FFT is sparse on its native ensemble ($K_{0.99}\\approx 1$), while RFT is not.}\n"
        "\\label{tab:theorem8_negative_control}\n"
        "\\begin{tabular}{c|cc|cc}\n"
        "\\toprule\n"
        "$N$ & $\\mathbb{E}[K_{0.99}(\\mathrm{RFT})]$ & $\\sigma$ & $\\mathbb{E}[K_{0.99}(\\mathrm{FFT})]$ & $\\sigma$ \\\\" "\n"
        "\\midrule\n"
    )
    for r in nc["rows"]:
        latex += (
            f"{r['N']} & {r['E_K99_RFT']:.2f} & {r['std_RFT']:.2f} & {r['E_K99_FFT']:.2f} & {r['std_FFT']:.2f} \\\\\n"
        )

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return latex.strip() + "\n"


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Theorem 8 hardware vectors + certified benchmark tables."
    )
    parser.add_argument("--output-dir", type=str, default="hardware_test_vectors/theorem8")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--vector-sizes",
        type=str,
        default="8,16,32,64",
        help="Comma-separated N sizes for vector generation.",
    )
    parser.add_argument("--num-cases", type=int, default=100)

    parser.add_argument(
        "--benchmark-sizes",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated N sizes for benchmark table.",
    )
    parser.add_argument("--samples", type=int, default=500)

    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--a-min", type=float, default=DEFAULT_A_MIN)
    parser.add_argument("--a-max", type=float, default=DEFAULT_A_MAX)

    parser.add_argument(
        "--bootstrap-resamples", type=int, default=8000, help="Bootstrap resamples for benchmark CI."
    )
    parser.add_argument(
        "--vector-bootstrap-resamples",
        type=int,
        default=5000,
        help="Bootstrap resamples for per-N vector summary CI.",
    )

    parser.add_argument("--no-vectors", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-negative-control", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_sizes = [int(x.strip()) for x in args.vector_sizes.split(",") if x.strip()]
    benchmark_sizes = [int(x.strip()) for x in args.benchmark_sizes.split(",") if x.strip()]

    print("=" * 70)
    print("THEOREM 8: Golden Spectral Concentration Inequality")
    print("Hardware Test Suite Generator (certified finite-N)")
    print("=" * 70)

    if not args.no_vectors:
        for N in vector_sizes:
            print(f"\n{'='*70}")
            print(f"Generating hardware vectors for N = {N}")
            generate_theorem8_test_vectors(
                N=N,
                num_cases=args.num_cases,
                seed=args.seed,
                threshold=args.threshold,
                a_min=args.a_min,
                a_max=args.a_max,
                output_dir=out_dir,
                bootstrap_resamples=args.vector_bootstrap_resamples,
            )

    benchmark_results = None
    if not args.no_benchmark:
        benchmark_results = run_theorem8_hardware_benchmark(
            sizes=benchmark_sizes,
            samples_per_size=args.samples,
            seed=args.seed,
            threshold=args.threshold,
            a_min=args.a_min,
            a_max=args.a_max,
            bootstrap_resamples=args.bootstrap_resamples,
            include_negative_control=(not args.no_negative_control),
        )

        # Save benchmark JSON and LaTeX
        bench_json = out_dir / "theorem8_benchmark_results.json"
        with open(bench_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"\n✓ Saved benchmark JSON: {bench_json}")

        latex_cert = out_dir / "theorem8_latex_table_certified.tex"
        with open(latex_cert, "w", encoding="utf-8") as f:
            f.write(benchmark_results["latex_table_certified"])
        print(f"✓ Saved LaTeX (certified): {latex_cert}")

        if not args.no_negative_control:
            latex_nc = out_dir / "theorem8_latex_table_negative_control.tex"
            with open(latex_nc, "w", encoding="utf-8") as f:
                f.write(benchmark_results["latex_table_negative_control"])
            print(f"✓ Saved LaTeX (negative control): {latex_nc}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print("\nGenerated files:")
    for f in sorted(out_dir.glob("*")):
        print(f"  - {f.name}")

    if benchmark_results is not None:
        print("\n" + benchmark_results["latex_table_certified"])
        if not args.no_negative_control:
            print("\n" + benchmark_results["latex_table_negative_control"])

    print("\nNext steps for hardware verification (real DUT):")
    print("1) Wire the DUT into theorem8_tb_N*.sv (done/handshake + k99 outputs)")
    print("2) Implement K-threshold logic (or stream energies + compute K on host)")
    print("3) Gate 'PASS' on an ensemble statistic from DUT outputs (e.g., mean(Δ)>0), not per-case wins")
    print("4) Use the certified LaTeX table in the paper (paired Δ + CI + win-rate CI)")


if __name__ == "__main__":
    main()
