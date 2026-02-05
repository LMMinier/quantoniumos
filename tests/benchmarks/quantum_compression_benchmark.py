#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Quantum-Inspired State Compression Benchmark
=============================================

Test hypothesis: RFT better compresses quantum-inspired states with
golden-ratio / quasi-periodic structure than FFT/DCT.

**Note:** "Quantum-inspired states" here refers to complex wavefunctions $\psi(x)$
simulated on classical hardware, widely used in quantum simulation methods (e.g. split-step).
This is NOT a benchmark of quantum bits (qubits) or quantum error correction.

State classes tested:
1. Fibonacci-based quasi-periodic waveforms
2. Golden-ratio phase-modulated coherent states (Glauber states with $\phi$ phase)
3. Graph states on Penrose tilings (φ symmetry)
4. Quasi-crystal-like superposition states

Metrics:
- Compression ratio at fixed fidelity (F > 0.99)
- Fidelity at fixed compression ratio
- Kolmogorov complexity proxy (gzip size of coefficient representation)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from scipy.fftpack import dct, idct
from algorithms.rft.core.phi_phase_fft_optimized import rft_forward, rft_inverse
import json
from typing import Dict, Tuple
import gzip
import math

PHI = (1 + np.sqrt(5)) / 2


def generate_quantum_states(n: int = 512) -> Dict[str, np.ndarray]:
    """
    Generate quantum-inspired state vectors with golden-ratio structure
    
    Note: These are normalized complex vectors |ψ⟩, not density matrices
    """
    states = {}
    
    # 1. Fibonacci quasi-periodic state
    # |ψ⟩ = Σ fib[k] |k⟩ where fib follows golden ratio
    fib = np.zeros(n)
    fib[0], fib[1] = 1, 1
    for i in range(2, n):
        fib[i] = fib[i-1] + fib[i-2]
    # Convert to quantum amplitudes (complex, normalized)
    amplitudes = fib / np.linalg.norm(fib)
    phases = np.exp(1j * 2 * np.pi * np.arange(n) / PHI)
    states['fibonacci_quasiperiodic'] = amplitudes * phases
    
    # 2. Golden-ratio coherent state
    # |α⟩ = exp(-|α|²/2) Σ (α^n / √n!) |n⟩ with α = φ exp(iθ)
    alpha = PHI * np.exp(1j * np.pi / 4)
    factorials = np.array([float(math.factorial(k)) if k < 170 else 1e308 for k in range(n)])
    coherent = np.exp(-np.abs(alpha)**2 / 2) * (alpha**np.arange(n)) / np.sqrt(factorials)
    states['golden_coherent'] = coherent / np.linalg.norm(coherent)
    
    # 3. Penrose tiling-inspired state
    # Superposition with φ-ratio weights between sites
    k = np.arange(n)
    weights = np.cos(2*np.pi * k / PHI) + np.cos(2*np.pi * k / PHI**2)
    phases_penrose = np.exp(1j * 2*np.pi * k * PHI)
    states['penrose_tiling'] = (weights * phases_penrose) / np.linalg.norm(weights * phases_penrose)
    
    # 4. Quasi-crystal superposition
    # Multi-scale interference pattern at φ-related frequencies
    t = np.linspace(0, 2*np.pi, n)
    quasicrystal = (np.exp(1j * t) + 
                   np.exp(1j * PHI * t) + 
                   np.exp(1j * PHI**2 * t))
    states['quasicrystal'] = quasicrystal / np.linalg.norm(quasicrystal)
    
    # 5. Harper model eigenstate (φ appears in phase)
    # Tight-binding model with irrational flux φ/2π
    k_vals = np.arange(n)
    harper = np.exp(1j * 2*np.pi * PHI * k_vals) / (1 + np.cos(2*np.pi * k_vals / n))
    states['harper_eigenstate'] = harper / np.linalg.norm(harper)
    
    # 6. Control: random state (no structure)
    random_state = np.random.randn(n) + 1j * np.random.randn(n)
    states['random_control'] = random_state / np.linalg.norm(random_state)
    
    return states


def compute_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Quantum fidelity: F = |⟨ψ₁|ψ₂⟩|²
    
    Measures overlap between quantum states (0 = orthogonal, 1 = identical)
    """
    overlap = np.abs(np.vdot(state1, state2))**2
    return float(overlap)


def compress_rft(state: np.ndarray, keep_fraction: float) -> Tuple[np.ndarray, int]:
    """
    Compress quantum state using RFT + coefficient thresholding
    
    Returns: (reconstructed_state, num_bits)
    """
    n = len(state)
    
    # RFT transform
    coeffs = rft_forward(state)
    
    # Keep top-k coefficients
    k = max(1, int(keep_fraction * n))
    idx = np.argsort(np.abs(coeffs))[-k:]
    
    compressed = np.zeros_like(coeffs)
    compressed[idx] = coeffs[idx]
    
    # Reconstruct
    recon = rft_inverse(compressed)
    recon = recon / np.linalg.norm(recon)  # Re-normalize
    
    # Estimate bits: k indices + k complex coefficients
    bits_per_index = int(np.ceil(np.log2(n)))
    bits_per_coeff = 64  # Complex128 = 2 * 32-bit floats
    num_bits = k * (bits_per_index + bits_per_coeff)
    
    return recon, num_bits


def compress_dct(state: np.ndarray, keep_fraction: float) -> Tuple[np.ndarray, int]:
    """
    Compress using DCT on real and imaginary parts separately
    
    Returns: (reconstructed_state, num_bits)
    """
    n = len(state)
    
    # DCT on real and imaginary parts
    coeffs_real = dct(state.real, norm='ortho')
    coeffs_imag = dct(state.imag, norm='ortho')
    
    # Keep top-k
    k = max(1, int(keep_fraction * n))
    
    idx_real = np.argsort(np.abs(coeffs_real))[-k:]
    idx_imag = np.argsort(np.abs(coeffs_imag))[-k:]
    
    compressed_real = np.zeros_like(coeffs_real)
    compressed_imag = np.zeros_like(coeffs_imag)
    compressed_real[idx_real] = coeffs_real[idx_real]
    compressed_imag[idx_imag] = coeffs_imag[idx_imag]
    
    # Reconstruct
    recon_real = idct(compressed_real, norm='ortho')
    recon_imag = idct(compressed_imag, norm='ortho')
    recon = recon_real + 1j * recon_imag
    recon = recon / np.linalg.norm(recon)
    
    # Estimate bits
    bits_per_index = int(np.ceil(np.log2(n)))
    bits_per_coeff = 32  # Float32
    num_bits = 2 * k * (bits_per_index + bits_per_coeff)  # Real + imag
    
    return recon, num_bits


def compress_fft(state: np.ndarray, keep_fraction: float) -> Tuple[np.ndarray, int]:
    """
    Compress using standard FFT + thresholding
    
    Returns: (reconstructed_state, num_bits)
    """
    n = len(state)
    
    # FFT
    coeffs = np.fft.fft(state) / np.sqrt(n)
    
    # Keep top-k
    k = max(1, int(keep_fraction * n))
    idx = np.argsort(np.abs(coeffs))[-k:]
    
    compressed = np.zeros_like(coeffs)
    compressed[idx] = coeffs[idx]
    
    # Reconstruct
    recon = np.fft.ifft(compressed * np.sqrt(n))
    recon = recon / np.linalg.norm(recon)
    
    # Estimate bits
    bits_per_index = int(np.ceil(np.log2(n)))
    bits_per_coeff = 64  # Complex128
    num_bits = k * (bits_per_index + bits_per_coeff)
    
    return recon, num_bits


def kolmogorov_complexity_proxy(coeffs: np.ndarray) -> int:
    """
    Estimate Kolmogorov complexity using gzip compression
    
    Lower = more structure / compressibility
    """
    # Convert to bytes
    coeffs_bytes = coeffs.tobytes()
    
    # Compress with gzip
    compressed = gzip.compress(coeffs_bytes, compresslevel=9)
    
    return len(compressed)


def benchmark_quantum_compression(state: np.ndarray, 
                                 keep_fractions: list = [0.05, 0.10, 0.20]) -> Dict:
    """
    Compare compression methods on quantum state
    
    Returns rate-distortion results for all methods
    """
    results = {
        'rft': [],
        'dct': [],
        'fft': []
    }
    
    for keep_frac in keep_fractions:
        # RFT
        recon_rft, bits_rft = compress_rft(state, keep_frac)
        fidelity_rft = compute_fidelity(state, recon_rft)
        coeffs_rft = rft_forward(state)
        kc_rft = kolmogorov_complexity_proxy(coeffs_rft)
        
        results['rft'].append({
            'keep_fraction': keep_frac,
            'fidelity': float(fidelity_rft),
            'bits': bits_rft,
            'bits_per_value': bits_rft / len(state),
            'kolmogorov_proxy': kc_rft
        })
        
        # DCT
        recon_dct, bits_dct = compress_dct(state, keep_frac)
        fidelity_dct = compute_fidelity(state, recon_dct)
        coeffs_dct_real = dct(state.real, norm='ortho')
        coeffs_dct_imag = dct(state.imag, norm='ortho')
        kc_dct = kolmogorov_complexity_proxy(coeffs_dct_real) + kolmogorov_complexity_proxy(coeffs_dct_imag)
        
        results['dct'].append({
            'keep_fraction': keep_frac,
            'fidelity': float(fidelity_dct),
            'bits': bits_dct,
            'bits_per_value': bits_dct / len(state),
            'kolmogorov_proxy': kc_dct
        })
        
        # FFT
        recon_fft, bits_fft = compress_fft(state, keep_frac)
        fidelity_fft = compute_fidelity(state, recon_fft)
        coeffs_fft = np.fft.fft(state) / np.sqrt(len(state))
        kc_fft = kolmogorov_complexity_proxy(coeffs_fft)
        
        results['fft'].append({
            'keep_fraction': keep_frac,
            'fidelity': float(fidelity_fft),
            'bits': bits_fft,
            'bits_per_value': bits_fft / len(state),
            'kolmogorov_proxy': kc_fft
        })
    
    return results


def run_quantum_compression_benchmark(n: int = 512):
    """Run comprehensive quantum state compression benchmark"""
    print("=" * 80)
    print("QUANTUM-STATE COMPRESSION BENCHMARK")
    print("=" * 80)
    print(f"State dimension: {n}")
    print("Testing: φ-structured quantum states")
    print()
    
    states = generate_quantum_states(n)
    keep_fractions = [0.05, 0.10, 0.20, 0.30]
    
    all_results = {}
    
    for state_name, state in states.items():
        print(f"\n{'='*80}")
        print(f"State: {state_name.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        results = benchmark_quantum_compression(state, keep_fractions)
        all_results[state_name] = results
        
        # Display results
        print(f"\n  {'Method':<6} {'Keep%':<8} {'Fidelity':<12} {'Bits/val':<10} {'KC Proxy':<12}")
        print(f"  {'-'*70}")
        
        for method in ['rft', 'dct', 'fft']:
            for point in results[method]:
                print(f"  {method.upper():<6} "
                      f"{point['keep_fraction']*100:<8.0f} "
                      f"{point['fidelity']:<12.6f} "
                      f"{point['bits_per_value']:<10.2f} "
                      f"{point['kolmogorov_proxy']:<12}")
    
    return all_results


def analyze_quantum_advantage(results: Dict):
    """Analyze where RFT provides advantage on quantum states"""
    print("\n" + "="*80)
    print("QUANTUM-STATE ANALYSIS")
    print("="*80)
    
    for state_name, state_results in results.items():
        print(f"\n{state_name.upper().replace('_', ' ')}:")
        
        # For each compression level, find best fidelity
        num_levels = len(state_results['rft'])
        
        for i in range(num_levels):
            fid_rft = state_results['rft'][i]['fidelity']
            fid_dct = state_results['dct'][i]['fidelity']
            fid_fft = state_results['fft'][i]['fidelity']
            keep = state_results['rft'][i]['keep_fraction']
            
            best_method = max([('RFT', fid_rft), ('DCT', fid_dct), ('FFT', fid_fft)], 
                            key=lambda x: x[1])
            
            print(f"  Keep {keep*100:.0f}%: {best_method[0]} (F={best_method[1]:.6f})")


def main():
    """Run full benchmark suite"""
    results = run_quantum_compression_benchmark(n=512)
    analyze_quantum_advantage(results)
    
    # Save results
    output_path = 'quantum_compression_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Summary verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    rft_wins = 0
    total_tests = 0
    
    for state_name, state_results in results.items():
        num_levels = len(state_results['rft'])
        
        for i in range(num_levels):
            total_tests += 1
            
            fids = {
                'rft': state_results['rft'][i]['fidelity'],
                'dct': state_results['dct'][i]['fidelity'],
                'fft': state_results['fft'][i]['fidelity']
            }
            
            winner = max(fids.items(), key=lambda x: x[1])[0]
            if winner == 'rft':
                rft_wins += 1
    
    win_rate = rft_wins / total_tests * 100
    
    print(f"\nRFT achieves best fidelity: {rft_wins}/{total_tests} tests ({win_rate:.1f}%)")
    
    if win_rate >= 60:
        print("✅ RFT superior for φ-structured quantum state compression")
        print("   → Use case: Quasi-periodic quantum systems, Harper model, Fibonacci chains")
    else:
        print("❌ DCT/FFT remain competitive even on golden-ratio states")
        print("   → RFT advantage not significant")


if __name__ == "__main__":
    main()
