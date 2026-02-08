#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Twisted-Convolution Filtering Benchmark
========================================

Test hypothesis: RFT diagonalizes specific convolution operators through
golden-ratio phase modulation, enabling efficient filtering in transform domain.

Key insight: If a filter's impulse response has φ-modulated structure,
RFT should diagonalize the convolution operator, making multiplication
in RFT-domain equivalent to expensive time-domain convolution.

Comparison:
- Time-domain convolution: O(n²) naive, O(n log n) FFT-based
- RFT-domain multiplication: O(n) if pre-diagonalized
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import time
from scipy import signal as sp_signal
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse
import json
from typing import Dict, Tuple
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2


def design_phi_modulated_filter(n: int, filter_type: str = 'lowpass') -> np.ndarray:
    """
    Design filters with golden-ratio phase structure that RFT should diagonalize
    
    Types:
    - 'phi_decay': Exponential decay with φ modulation
    - 'phi_oscillatory': Oscillatory with φ-spaced frequencies
    - 'phi_chirp': Chirp with φ instantaneous frequency
    - 'gaussian': Control (no φ structure)
    """
    t = np.arange(n)
    
    if filter_type == 'phi_decay':
        # h[n] = exp(-n/φ) * cos(2π n / φ²)
        h = np.exp(-t / PHI**2) * np.cos(2 * np.pi * t / PHI**2)
    
    elif filter_type == 'phi_oscillatory':
        # Multi-scale oscillations at φ-spaced frequencies
        h = (np.sin(2*np.pi * t / PHI) + 
             np.sin(2*np.pi * t / PHI**2) + 
             np.sin(2*np.pi * t / PHI**3)) / 3
        h *= np.exp(-t / (n/4))  # Taper
    
    elif filter_type == 'phi_chirp':
        # Chirp with φ-modulated instantaneous frequency
        phase = 2 * np.pi * np.cumsum(1 / (PHI + 0.1*t))
        h = np.sin(phase) * np.exp(-t / (n/3))
    
    elif filter_type == 'phi_spiral':
        # Spiral decay mimicking φ-based phyllotaxis
        r = np.exp(-t / (n/5))
        theta = 2 * np.pi * PHI * t
        h = r * np.cos(theta)
    
    elif filter_type == 'gaussian':
        # Control: standard Gaussian (no φ structure)
        h = np.exp(-((t - n/2)**2) / (2 * (n/10)**2))
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Normalize
    h = h / np.linalg.norm(h)
    return h


def generate_test_signals(n: int = 1024) -> Dict[str, np.ndarray]:
    """Generate signals to filter"""
    t = np.linspace(0, 1, n)
    signals = {}
    
    # Random noise
    signals['white_noise'] = np.random.randn(n)
    
    # Phi-modulated signal (should couple well with phi filter)
    signals['phi_signal'] = np.sin(2*np.pi * 50 * t * PHI**t)
    
    # Multi-frequency signal
    signals['multi_tone'] = (np.sin(2*np.pi * 10 * t) + 
                             np.sin(2*np.pi * 25 * t) + 
                             np.sin(2*np.pi * 50 * t)) / 3
    
    # Impulse train
    signals['impulse_train'] = np.zeros(n)
    signals['impulse_train'][::int(n/20)] = 1.0
    
    return signals


def time_domain_convolution(signal: np.ndarray, filt: np.ndarray) -> Tuple[np.ndarray, float]:
    """Time-domain convolution (using scipy's optimized implementation)"""
    start = time.perf_counter()
    result = sp_signal.convolve(signal, filt, mode='same')
    elapsed = time.perf_counter() - start
    return result, elapsed


def fft_domain_filtering(signal: np.ndarray, filt: np.ndarray) -> Tuple[np.ndarray, float]:
    """FFT-based filtering (overlap-add not needed for same-size)"""
    n = len(signal)
    start = time.perf_counter()
    
    # Transform
    signal_fft = np.fft.fft(signal)
    filt_fft = np.fft.fft(filt, n=n)
    
    # Multiply
    result_fft = signal_fft * filt_fft
    
    # Inverse
    result = np.fft.ifft(result_fft).real
    
    elapsed = time.perf_counter() - start
    return result, elapsed


def rft_domain_filtering(signal: np.ndarray, filt: np.ndarray) -> Tuple[np.ndarray, float]:
    """RFT-based filtering"""
    n = len(signal)
    signal_complex = signal.astype(np.complex128)
    filt_complex = filt.astype(np.complex128)
    
    # Zero-pad filter if needed
    if len(filt) < n:
        filt_padded = np.zeros(n, dtype=np.complex128)
        filt_padded[:len(filt)] = filt_complex
        filt_complex = filt_padded
    
    start = time.perf_counter()
    
    # Transform
    signal_rft = rft_forward(signal_complex)
    filt_rft = rft_forward(filt_complex)
    
    # Multiply
    result_rft = signal_rft * filt_rft
    
    # Inverse
    result = rft_inverse(result_rft).real
    
    elapsed = time.perf_counter() - start
    return result, elapsed


def measure_diagonalization_quality(filt: np.ndarray, method: str = 'rft') -> float:
    """
    Measure how well transform diagonalizes the convolution operator
    
    A perfect diagonalization means the convolution matrix becomes diagonal
    in transform domain. We measure off-diagonal energy.
    """
    n = len(filt)
    
    # Build circulant convolution matrix in time domain
    from scipy.linalg import circulant
    C = circulant(filt)
    
    # Transform to frequency domain
    if method == 'rft':
        filt_complex = filt.astype(np.complex128)
        # Get RFT basis (this is expensive but done once for analysis)
        # We'll approximate by transforming standard basis vectors
        T = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            e_i = np.zeros(n, dtype=np.complex128)
            e_i[i] = 1.0
            T[:, i] = rft_forward(e_i)
        
        # Transform convolution matrix: T^H @ C @ T
        C_transformed = np.conj(T.T) @ C @ T
        
    elif method == 'fft':
        # FFT perfectly diagonalizes circulant matrices
        filt_fft = np.fft.fft(filt)
        C_transformed = np.diag(filt_fft)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Measure off-diagonal energy
    diag_energy = np.sum(np.abs(np.diag(C_transformed))**2)
    total_energy = np.sum(np.abs(C_transformed)**2)
    off_diag_energy = total_energy - diag_energy
    
    # Diagonalization quality: 1.0 = perfect, 0.0 = no diagonalization
    quality = 1.0 - (off_diag_energy / max(total_energy, 1e-16))
    
    return float(quality)


def benchmark_filtering(signal: np.ndarray, filt: np.ndarray, 
                       trials: int = 10) -> Dict:
    """Compare all three filtering methods"""
    results = {}
    
    # Time-domain (reference result)
    result_time, time_time = time_domain_convolution(signal, filt)
    
    # FFT-domain
    result_fft, time_fft_single = fft_domain_filtering(signal, filt)
    # Average over multiple trials
    times_fft = []
    for _ in range(trials):
        _, t = fft_domain_filtering(signal, filt)
        times_fft.append(t)
    time_fft = np.mean(times_fft)
    
    # RFT-domain
    result_rft, time_rft_single = rft_domain_filtering(signal, filt)
    times_rft = []
    for _ in range(trials):
        _, t = rft_domain_filtering(signal, filt)
        times_rft.append(t)
    time_rft = np.mean(times_rft)
    
    # Errors relative to time-domain
    error_fft = np.linalg.norm(result_fft - result_time) / np.linalg.norm(result_time)
    error_rft = np.linalg.norm(result_rft - result_time) / np.linalg.norm(result_time)
    
    # Diagonalization quality
    diag_fft = measure_diagonalization_quality(filt, method='fft')
    diag_rft = measure_diagonalization_quality(filt, method='rft')
    
    results = {
        'time_domain_ms': time_time * 1000,
        'fft': {
            'time_ms': time_fft * 1000,
            'relative_error': float(error_fft),
            'speedup_vs_time': time_time / time_fft,
            'diagonalization_quality': diag_fft
        },
        'rft': {
            'time_ms': time_rft * 1000,
            'relative_error': float(error_rft),
            'speedup_vs_time': time_time / time_rft,
            'speedup_vs_fft': time_fft / time_rft,
            'diagonalization_quality': diag_rft
        }
    }
    
    return results


def run_twisted_convolution_benchmark(n: int = 1024, trials: int = 10):
    """Run comprehensive twisted convolution benchmark"""
    print("=" * 80)
    print("TWISTED-CONVOLUTION FILTERING BENCHMARK")
    print("=" * 80)
    print(f"Signal length: {n}")
    print(f"Trials per test: {trials}")
    print()
    
    filter_types = ['phi_decay', 'phi_oscillatory', 'phi_chirp', 'phi_spiral', 'gaussian']
    signals = generate_test_signals(n)
    
    all_results = {}
    
    for filter_type in filter_types:
        print(f"\n{'='*80}")
        print(f"Filter: {filter_type.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        filt = design_phi_modulated_filter(n, filter_type=filter_type)
        filter_results = {}
        
        for signal_name, sig in signals.items():
            print(f"\n  Signal: {signal_name}")
            
            result = benchmark_filtering(sig, filt, trials=trials)
            filter_results[signal_name] = result
            
            print(f"    Time-domain: {result['time_domain_ms']:.3f} ms")
            print(f"    FFT: {result['fft']['time_ms']:.3f} ms "
                  f"(speedup: {result['fft']['speedup_vs_time']:.1f}×, "
                  f"error: {result['fft']['relative_error']:.2e}, "
                  f"diag: {result['fft']['diagonalization_quality']:.4f})")
            print(f"    RFT: {result['rft']['time_ms']:.3f} ms "
                  f"(speedup: {result['rft']['speedup_vs_time']:.1f}×, "
                  f"error: {result['rft']['relative_error']:.2e}, "
                  f"diag: {result['rft']['diagonalization_quality']:.4f})")
            
            if result['rft']['speedup_vs_fft'] > 1.0:
                print(f"    ✅ RFT {result['rft']['speedup_vs_fft']:.2f}× faster than FFT")
            else:
                print(f"    ❌ RFT {1/result['rft']['speedup_vs_fft']:.2f}× slower than FFT")
        
        all_results[filter_type] = filter_results
    
    return all_results


def analyze_diagonalization_advantage(results: Dict):
    """Analyze where RFT's diagonalization helps"""
    print("\n" + "="*80)
    print("DIAGONALIZATION ANALYSIS")
    print("="*80)
    
    print(f"\n{'Filter Type':<20} {'RFT Diag':<12} {'FFT Diag':<12} {'RFT Advantage':<15}")
    print("-" * 80)
    
    for filter_type, filter_results in results.items():
        # Average diagonalization across signals
        rft_diags = []
        fft_diags = []
        
        for signal_name, result in filter_results.items():
            rft_diags.append(result['rft']['diagonalization_quality'])
            fft_diags.append(result['fft']['diagonalization_quality'])
        
        avg_rft_diag = np.mean(rft_diags)
        avg_fft_diag = np.mean(fft_diags)
        
        if avg_rft_diag > avg_fft_diag:
            advantage = f"✅ +{(avg_rft_diag - avg_fft_diag)*100:.1f}%"
        else:
            advantage = f"❌ {(avg_rft_diag - avg_fft_diag)*100:.1f}%"
        
        print(f"{filter_type:<20} {avg_rft_diag:<12.4f} {avg_fft_diag:<12.4f} {advantage:<15}")


def main():
    """Run full benchmark suite"""
    results = run_twisted_convolution_benchmark(n=1024, trials=10)
    analyze_diagonalization_advantage(results)
    
    # Save results
    output_path = 'twisted_convolution_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Summary verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    rft_wins = 0
    total_tests = 0
    
    for filter_type, filter_results in results.items():
        for signal_name, result in filter_results.items():
            total_tests += 1
            # RFT wins if it has better diagonalization
            if result['rft']['diagonalization_quality'] > result['fft']['diagonalization_quality']:
                rft_wins += 1
    
    win_rate = rft_wins / total_tests * 100
    
    print(f"\nRFT achieves better diagonalization: {rft_wins}/{total_tests} tests ({win_rate:.1f}%)")
    
    if win_rate >= 60:
        print("✅ RFT shows superior diagonalization on φ-structured filters")
        print("   → Use case: Filtering with golden-ratio impulse responses")
    else:
        print("❌ FFT's perfect circulant diagonalization dominates")
        print("   → RFT does not provide filtering advantage")


if __name__ == "__main__":
    main()
