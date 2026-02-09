# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RFT Multi-Scale Benchmark
=========================
Test RFT performance across multiple transform sizes (N=64 to 1024)
with randomized golden-modulated datasets.

This validates that RFT's advantage is consistent across scales,
not an artifact of a particular N.
"""

import numpy as np
from scipy.fft import fft, ifft, dct, idct
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from algorithms.rft.kernels.operator_eigenbasis_rft import (
    build_rft_kernel, rft_forward, rft_inverse,
    generate_golden_quasiperiodic, generate_fibonacci_modulated,
    PHI
)


def energy_sparsity(coeffs: np.ndarray, threshold: float = 0.99) -> int:
    """Count coefficients needed to capture threshold fraction of energy."""
    energy = np.abs(coeffs) ** 2
    total = np.sum(energy)
    sorted_energy = np.sort(energy)[::-1]
    cumsum = np.cumsum(sorted_energy)
    idx = np.searchsorted(cumsum, threshold * total)
    return min(idx + 1, len(coeffs))


def compression_psnr(signal: np.ndarray, coeffs: np.ndarray, 
                     inverse_fn, keep_frac: float) -> float:
    """PSNR when keeping top keep_frac coefficients."""
    N = len(coeffs)
    K = max(1, int(keep_frac * N))
    
    top_idx = np.argsort(np.abs(coeffs))[-K:]
    sparse = np.zeros_like(coeffs)
    sparse[top_idx] = coeffs[top_idx]
    
    rec = inverse_fn(sparse)
    if np.iscomplexobj(rec):
        rec = np.real(rec)
    
    mse = np.mean((signal - rec) ** 2)
    if mse < 1e-15:
        return 100.0
    return 10 * np.log10(np.max(signal)**2 / mse)


def generate_randomized_golden_family(N: int, num_samples: int = 20, seed: int = None) -> list:
    """
    Generate randomized samples from the golden quasi-periodic family.
    Varies: amplitude ratio, phase, base frequency, noise level.
    """
    if seed is not None:
        np.random.seed(seed)
    
    signals = []
    for _ in range(num_samples):
        A = np.random.uniform(0.3, 1.0)
        B = np.random.uniform(0.3, 1.0)
        phase = np.random.uniform(0, 2*np.pi)
        f0 = np.random.uniform(5, 20)
        noise = np.random.uniform(0, 0.05)
        
        sig = generate_golden_quasiperiodic(N, f0=f0, A=A, B=B, phase=phase, noise_std=noise)
        signals.append(sig)
    
    # Add some Fibonacci-modulated signals
    for depth in [3, 4, 5, 6]:
        f0 = np.random.uniform(5, 15)
        sig = generate_fibonacci_modulated(N, f0=f0, depth=depth)
        signals.append(sig)
    
    return signals


def benchmark_single_N(N: int, keep_frac: float = 0.1, num_samples: int = 20):
    """Run benchmark for a single transform size."""
    
    # Build RFT kernel (cached)
    t0 = time.perf_counter()
    Phi = build_rft_kernel(N)
    kernel_time = time.perf_counter() - t0
    
    # Generate test signals
    signals = generate_randomized_golden_family(N, num_samples=num_samples, seed=42)
    
    results = {
        'N': N,
        'kernel_time': kernel_time,
        'sparse_fft': [],
        'sparse_dct': [],
        'sparse_rft': [],
        'psnr_fft': [],
        'psnr_dct': [],
        'psnr_rft': [],
    }
    
    for sig in signals:
        # Transforms
        c_fft = fft(sig)
        c_dct = dct(sig, norm='ortho')
        c_rft = rft_forward(sig, Phi)
        
        # Sparsity
        results['sparse_fft'].append(energy_sparsity(c_fft))
        results['sparse_dct'].append(energy_sparsity(c_dct))
        results['sparse_rft'].append(energy_sparsity(c_rft))
        
        # PSNR at fixed compression
        results['psnr_fft'].append(compression_psnr(sig, c_fft, lambda c: np.real(ifft(c)), keep_frac))
        results['psnr_dct'].append(compression_psnr(sig, c_dct, lambda c: idct(c, norm='ortho'), keep_frac))
        results['psnr_rft'].append(compression_psnr(sig, c_rft, lambda c: rft_inverse(c, Phi), keep_frac))
    
    return results


def main():
    print("="*80)
    print("RFT MULTI-SCALE BENCHMARK")
    print("="*80)
    print("Testing RFT vs FFT vs DCT across multiple transform sizes")
    print("Signal family: Randomized Golden Quasi-Periodic + Fibonacci-modulated")
    print("Metric: PSNR at 10% coefficient retention")
    print()
    
    Ns = [64, 128, 256, 512, 1024]
    keep_frac = 0.10
    num_samples = 24  # 20 golden + 4 fibonacci
    
    all_results = []
    
    for N in Ns:
        print(f"Testing N={N}...", end=" ", flush=True)
        results = benchmark_single_N(N, keep_frac, num_samples=20)
        all_results.append(results)
        print(f"done (kernel build: {results['kernel_time']*1000:.1f}ms)")
    
    # Summary table
    print("\n" + "="*80)
    print(f"RESULTS: Mean PSNR at {int(keep_frac*100)}% coefficient retention (higher = better)")
    print("="*80)
    print(f"{'N':>6} | {'FFT PSNR':>10} | {'DCT PSNR':>10} | {'RFT PSNR':>10} | {'RFT Gain':>10} | {'Winner':>8}")
    print("-"*70)
    
    rft_wins = 0
    total = len(Ns)
    
    for r in all_results:
        mean_fft = np.mean(r['psnr_fft'])
        mean_dct = np.mean(r['psnr_dct'])
        mean_rft = np.mean(r['psnr_rft'])
        
        best = max(mean_fft, mean_dct, mean_rft)
        winner = 'RFT' if mean_rft == best else ('DCT' if mean_dct == best else 'FFT')
        gain = mean_rft - max(mean_fft, mean_dct)
        
        if winner == 'RFT':
            rft_wins += 1
        
        print(f"{r['N']:>6} | {mean_fft:>10.2f} | {mean_dct:>10.2f} | {mean_rft:>10.2f} | {gain:>+10.2f} | {winner:>8}")
    
    # Sparsity comparison
    print("\n" + "="*80)
    print("SPARSITY: Mean coefficients for 99% energy (fewer = better)")
    print("="*80)
    print(f"{'N':>6} | {'FFT Sparse':>10} | {'DCT Sparse':>10} | {'RFT Sparse':>10} | {'RFT Reduction':>12}")
    print("-"*60)
    
    for r in all_results:
        mean_fft = np.mean(r['sparse_fft'])
        mean_dct = np.mean(r['sparse_dct'])
        mean_rft = np.mean(r['sparse_rft'])
        
        baseline = min(mean_fft, mean_dct)
        reduction = (1 - mean_rft/baseline) * 100
        
        print(f"{r['N']:>6} | {mean_fft:>10.1f} | {mean_dct:>10.1f} | {mean_rft:>10.1f} | {reduction:>+11.1f}%")
    
    # Overall summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"RFT wins PSNR comparison: {rft_wins}/{total} ({100*rft_wins/total:.0f}%)")
    
    avg_gain = np.mean([np.mean(r['psnr_rft']) - max(np.mean(r['psnr_fft']), np.mean(r['psnr_dct'])) 
                        for r in all_results])
    print(f"Average RFT PSNR gain: {avg_gain:+.2f} dB")
    
    avg_sparse_reduction = np.mean([
        (1 - np.mean(r['sparse_rft'])/min(np.mean(r['sparse_fft']), np.mean(r['sparse_dct']))) * 100
        for r in all_results
    ])
    print(f"Average sparsity reduction: {avg_sparse_reduction:+.1f}%")
    
    # Complexity note
    print("\n" + "-"*80)
    print("COMPLEXITY NOTE")
    print("-"*80)
    print("Current RFT: O(N²) transform, O(N³) kernel construction (one-time)")
    print("FFT/DCT:     O(N log N) transform")
    print("\nFor N=1024, RFT is ~100x slower per transform.")
    print("Future work: Fast RFT via structured matrix approximations.")


if __name__ == "__main__":
    main()
