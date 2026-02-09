# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
HONEST SPARSITY & COMPRESSION BENCHMARK
=======================================
Resonant Fourier Transform (RFT) vs FFT vs DCT

NO BS RULES:
1. Same signal family, multiple instances (not just one cherry-picked signal)
2. Sparsity = fraction of coefficients needed to capture X% of energy
3. Compression = same number of coefficients kept, compare reconstruction MSE
4. Test on in-family AND out-of-family signals
"""

import numpy as np
from scipy.fft import fft, ifft, dct, idct
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithms.rft.kernels.operator_eigenbasis_rft import (
    build_rft_kernel, rft_forward, rft_inverse,
    generate_golden_quasiperiodic, generate_fibonacci_modulated, generate_phyllotaxis_signal,
    PHI
)


def energy_sparsity(coeffs: np.ndarray, energy_threshold: float = 0.99) -> int:
    """
    Count how many coefficients (sorted by magnitude) are needed to 
    capture `energy_threshold` fraction of total energy.
    
    Returns: number of coefficients needed
    """
    energy = np.abs(coeffs) ** 2
    total_energy = np.sum(energy)
    
    sorted_indices = np.argsort(energy)[::-1]
    cumulative = 0.0
    for i, idx in enumerate(sorted_indices):
        cumulative += energy[idx]
        if cumulative / total_energy >= energy_threshold:
            return i + 1
    return len(coeffs)


def compression_mse(signal: np.ndarray, coeffs: np.ndarray, 
                    inverse_fn, keep_fraction: float = 0.1) -> float:
    """
    Keep top `keep_fraction` of coefficients (by magnitude), zero rest,
    reconstruct, and return MSE.
    """
    N = len(coeffs)
    K = max(1, int(keep_fraction * N))
    
    # Zero out smallest coefficients
    threshold_idx = np.argsort(np.abs(coeffs))[-K:]
    sparse_coeffs = np.zeros_like(coeffs)
    sparse_coeffs[threshold_idx] = coeffs[threshold_idx]
    
    # Reconstruct
    rec = inverse_fn(sparse_coeffs)
    if np.iscomplexobj(rec):
        rec = np.real(rec)
    
    mse = np.mean((signal - rec) ** 2)
    return mse


def psnr_from_mse(mse: float, max_val: float) -> float:
    """Compute PSNR from MSE."""
    if mse < 1e-15:
        return 100.0
    return 10 * np.log10(max_val**2 / mse)


def run_single_test(signal: np.ndarray, Phi: np.ndarray, label: str, keep_frac: float = 0.1):
    """Run sparsity and compression tests for one signal."""
    N = len(signal)
    
    # Transforms
    coeffs_fft = fft(signal)
    coeffs_dct = dct(signal, norm='ortho')
    coeffs_rft = rft_forward(signal, Phi)
    
    # Sparsity: coefficients for 99% energy
    sparse_fft = energy_sparsity(coeffs_fft, 0.99)
    sparse_dct = energy_sparsity(coeffs_dct, 0.99)
    sparse_rft = energy_sparsity(coeffs_rft, 0.99)
    
    # Compression: MSE at fixed keep ratio
    mse_fft = compression_mse(signal, coeffs_fft, lambda c: np.real(ifft(c)), keep_frac)
    mse_dct = compression_mse(signal, coeffs_dct, lambda c: idct(c, norm='ortho'), keep_frac)
    mse_rft = compression_mse(signal, coeffs_rft, lambda c: rft_inverse(c, Phi), keep_frac)
    
    max_val = np.max(np.abs(signal))
    psnr_fft = psnr_from_mse(mse_fft, max_val)
    psnr_dct = psnr_from_mse(mse_dct, max_val)
    psnr_rft = psnr_from_mse(mse_rft, max_val)
    
    return {
        'label': label,
        'sparse_fft': sparse_fft,
        'sparse_dct': sparse_dct,
        'sparse_rft': sparse_rft,
        'psnr_fft': psnr_fft,
        'psnr_dct': psnr_dct,
        'psnr_rft': psnr_rft,
    }


def main():
    print("="*80)
    print("HONEST BENCHMARK: Resonant Fourier Transform (RFT) vs FFT vs DCT")
    print("="*80)
    
    N = 256
    Phi = build_rft_kernel(N)
    keep_frac = 0.10  # Keep 10% of coefficients
    
    results = []
    
    # =========================================================================
    # IN-FAMILY SIGNALS (RFT should win)
    # =========================================================================
    print("\n" + "-"*80)
    print("IN-FAMILY SIGNALS (Golden Quasi-Periodic)")
    print("-"*80)
    
    # Multiple instances with varying parameters
    for i, (A, B, phase) in enumerate([
        (1.0, 1.0, 0.0),
        (1.0, 0.5, np.pi/4),
        (0.8, 1.2, np.pi/2),
        (1.0, 1.0, np.pi),
        (0.5, 0.5, 0.0),
    ]):
        sig = generate_golden_quasiperiodic(N, f0=10.0, A=A, B=B, phase=phase)
        res = run_single_test(sig, Phi, f"GoldenQP_{i+1}", keep_frac)
        results.append(res)
    
    # Fibonacci modulated
    for depth in [3, 5, 7]:
        sig = generate_fibonacci_modulated(N, f0=10.0, depth=depth)
        res = run_single_test(sig, Phi, f"Fibonacci_d{depth}", keep_frac)
        results.append(res)
    
    # Phyllotaxis
    for spirals in [5, 8, 13]:
        sig = generate_phyllotaxis_signal(N, spirals=spirals)
        res = run_single_test(sig, Phi, f"Phyllotaxis_s{spirals}", keep_frac)
        results.append(res)
    
    # =========================================================================
    # OUT-OF-FAMILY SIGNALS (RFT should NOT win - sanity check)
    # =========================================================================
    print("\n" + "-"*80)
    print("OUT-OF-FAMILY SIGNALS (DCT/FFT home turf)")
    print("-"*80)
    
    # Pure sinusoid (DCT/FFT optimal)
    sig = np.sin(2 * np.pi * 7 * np.linspace(0, 1, N))
    results.append(run_single_test(sig, Phi, "PureSine_7Hz", keep_frac))
    
    # White noise (nothing should win)
    np.random.seed(42)
    sig = np.random.randn(N)
    results.append(run_single_test(sig, Phi, "WhiteNoise", keep_frac))
    
    # Chirp (DCT-friendly)
    t = np.linspace(0, 1, N)
    sig = np.sin(2 * np.pi * (5 + 20*t) * t)
    results.append(run_single_test(sig, Phi, "LinearChirp", keep_frac))
    
    # Square wave (FFT optimal, DCT also good)
    sig = np.sign(np.sin(2 * np.pi * 4 * t))
    results.append(run_single_test(sig, Phi, "SquareWave", keep_frac))
    
    # =========================================================================
    # RESULTS TABLE
    # =========================================================================
    print("\n" + "="*80)
    print(f"RESULTS: 99% Energy Sparsity (fewer = better) | PSNR at {int(keep_frac*100)}% coeffs (higher = better)")
    print("="*80)
    print(f"{'Signal':<20} | {'Sparse FFT':>10} | {'Sparse DCT':>10} | {'Sparse RFT':>10} || {'PSNR FFT':>10} | {'PSNR DCT':>10} | {'PSNR RFT':>10} | {'Winner':>8}")
    print("-"*120)
    
    in_family_rft_wins = 0
    in_family_total = 0
    out_family_rft_wins = 0
    out_family_total = 0
    
    for r in results:
        psnrs = [r['psnr_fft'], r['psnr_dct'], r['psnr_rft']]
        best_idx = np.argmax(psnrs)
        winner = ['FFT', 'DCT', 'RFT'][best_idx]
        
        is_in_family = r['label'].startswith(('Golden', 'Fib', 'Phyl'))
        if is_in_family:
            in_family_total += 1
            if winner == 'RFT':
                in_family_rft_wins += 1
        else:
            out_family_total += 1
            if winner == 'RFT':
                out_family_rft_wins += 1
        
        print(f"{r['label']:<20} | {r['sparse_fft']:>10} | {r['sparse_dct']:>10} | {r['sparse_rft']:>10} || {r['psnr_fft']:>10.2f} | {r['psnr_dct']:>10.2f} | {r['psnr_rft']:>10.2f} | {winner:>8}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"In-Family:     RFT wins {in_family_rft_wins}/{in_family_total} ({100*in_family_rft_wins/in_family_total:.1f}%)")
    print(f"Out-of-Family: RFT wins {out_family_rft_wins}/{out_family_total} ({100*out_family_rft_wins/out_family_total:.1f}%)")
    
    if in_family_rft_wins > in_family_total // 2 and out_family_rft_wins < out_family_total // 2:
        print("\n✅ VALID RESULT: RFT is specialized for its target family and does NOT overfit to all signals.")
    elif in_family_rft_wins <= in_family_total // 2:
        print("\n❌ FAILED: RFT does not beat FFT/DCT even on its home turf.")
    else:
        print("\n⚠️ WARNING: RFT wins on out-of-family signals too. May be overfitting or test bug.")


if __name__ == "__main__":
    main()
