#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
verify_ascii_bottleneck.py
--------------------------
Tests the "ASCII Bottleneck": The difficulty of compressing discrete symbol sequences (text/code)
using continuous spectral transforms.

We compare Φ-RFT variants against DFT and DCT on:
1. Natural Text
2. Source Code
3. Random ASCII
4. Mixed Signals (Wave + Text) - Validating Theorem 10

New Transforms Documented:
- Log-Periodic Φ-RFT: Uses log-warped phase to capture low-frequency symbol statistics.
- Convex Mixed Φ-RFT: A convex combination of Standard and Log-Periodic phases.
"""

import numpy as np
import sys
import os
from scipy.fft import dct
from pathlib import Path

# Ensure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import irrevocable_truths as it
from algorithms.rft.hybrids.legacy_mca import adaptive_hybrid_compress, braided_hybrid_mca

def gini_coefficient(x):
    """Calculate the Gini coefficient of a numpy array."""
    # Based on bottom-up integration
    x = np.abs(x)
    if np.sum(x) == 0: return 0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * x)) / (n * np.sum(x))) - ((n + 1) / n)

def energy_concentration(x, percentile=0.99):
    """Return fraction of coefficients needed to capture 'percentile' of total energy."""
    energy = np.abs(x)**2
    total_energy = np.sum(energy)
    if total_energy == 0: return 1.0
    
    sorted_energy = np.sort(energy)[::-1]
    cumulative = np.cumsum(sorted_energy)
    
    # Find index where cumulative energy crosses threshold
    threshold = total_energy * percentile
    idx = np.searchsorted(cumulative, threshold)
    
    return (idx + 1) / len(x)

def text_to_signal(text, N):
    """Convert text string to normalized signal vector."""
    # Convert to bytes
    b = text.encode('utf-8') if isinstance(text, str) else text
    
    # Repeat or truncate to N
    if len(b) < N:
        reps = (N // len(b)) + 1
        b = (b * reps)[:N]
    else:
        b = b[:N]
        
    arr = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    # Normalize to [-1, 1]
    return (arr / 128.0) - 1.0

def run_ascii_bottleneck_test():
    N = 256
    print(f"\n{'='*60}")
    print(f" TEST: THE ASCII BOTTLENECK (N={N})")
    print(f"{'='*60}")
    print("Comparing spectral sparsity on discrete symbol data.")
    print("Lower 'Coeffs for 99% Energy' is BETTER (more compression).")
    print("Higher 'Gini' is BETTER (more sparsity).\n")

    # 1. Define Datasets & Signals
    raw_datasets = {
        "Natural Text": "The quick brown fox jumps over the lazy dog. " * 10,
        "Python Code": "def rft(x): return np.fft.fft(x) * np.exp(1j*phi) " * 5,
        "Random ASCII": "".join([chr(x) for x in np.random.randint(32, 127, N)])
    }
    
    test_signals = {k: text_to_signal(v, N) for k, v in raw_datasets.items()}
    
    # Add Mixed Signal (Theorem 10 Validation)
    # A Phi-resonant wave + text structure
    t = np.arange(N)
    phi = (1 + np.sqrt(5)) / 2
    wave = 0.6 * np.sin(2 * np.pi * (1/phi**2) * t)
    text_part = text_to_signal("Theorem 10 separates structure from texture.", N)
    test_signals["Mixed (Wave + Text)"] = wave + text_part

    # 2. Define Transforms
    # Pre-compute matrices for RFT variants
    print("Generating transform matrices...")
    U_orig = it.generate_original_phi_rft(N)
    U_harm = it.generate_harmonic_phase(N)
    U_fib  = it.generate_fibonacci_tilt(N)
    U_chaos = it.generate_chaotic_mix(N)
    
    def hybrid_transform(x):
        """Adaptive Hybrid (Theorem 10) - auto-routes to DCT/RFT"""
        x_struct, x_texture, _, _ = adaptive_hybrid_compress(x, verbose=False)
        return np.concatenate([x_struct, x_texture])

    def hybrid_transform_logphi(x):
        """Log-Periodic RFT (Corollary 10.1)"""
        x_struct, x_texture, _, _ = adaptive_hybrid_compress(x, verbose=False, rft_kind="logphi")
        return np.concatenate([x_struct, x_texture])

    def hybrid_transform_mixed(x):
        """Convex Mixed RFT (Corollary 10.2)"""
        x_struct, x_texture, _, _ = adaptive_hybrid_compress(x, verbose=False, rft_kind="mixed", rft_mix=0.5)
        return np.concatenate([x_struct, x_texture])

    def hybrid_transform_braided(x):
        """Braided MCA (Parallel Competition)"""
        res = braided_hybrid_mca(x, verbose=False, threshold=0.05)
        return np.concatenate([res.structural, res.texture])
    
    transforms = {
        "DFT (Complex)": lambda x: np.fft.fft(x, norm="ortho"),
        "DCT (Real)":    lambda x: dct(x, norm="ortho"),
        "Original Φ-RFT": lambda x: U_orig @ x,
        "Harmonic-Phase": lambda x: U_harm @ x,
        "Fibonacci Tilt": lambda x: U_fib @ x,
        "Chaotic Mix":    lambda x: U_chaos @ x,
        "RFT Hybrid Basis (T10)": hybrid_transform,
        "Log-Periodic RFT (New)": hybrid_transform_logphi,
        "Convex Mixed RFT (New)": hybrid_transform_mixed,
        "Braided MCA (Parallel)": hybrid_transform_braided
    }

    # 3. Run Tests
    for data_name, signal in test_signals.items():
        print(f"\n--- Dataset: {data_name} ---")
        # signal is already prepared
        
        print(f"{'Transform':<20} | {'Gini Coeff':<10} | {'% Coeffs (99% E)':<18} | {'Verdict'}")
        print("-" * 60)
        
        results = []
        
        for trans_name, func in transforms.items():
            coeffs = func(signal)
            gini = gini_coefficient(coeffs)
            sparsity_99 = energy_concentration(coeffs, 0.99) * 100
            
            results.append((trans_name, gini, sparsity_99))
        
        # Sort by Sparsity (Lower % is better)
        results.sort(key=lambda x: x[2])
        
        for name, g, s in results:
            # Simple verdict relative to DCT (Standard)
            dct_score = [r[2] for r in results if "DCT" in r[0]][0]
            if s < dct_score * 0.95:
                verdict = "✅ Beats DCT"
            elif s > dct_score * 1.05:
                verdict = "❌ Worse"
            else:
                verdict = "⚖️ Parity"
                
            print(f"{name:<20} | {g:.4f}     | {s:.2f}%             | {verdict}")

if __name__ == "__main__":
    run_ascii_bottleneck_test()
