# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Optimization Script for Operator-Based ARFT
Tests if the eigenbasis of a golden-ratio autocorrelation operator beats FFT.
"""

import numpy as np
import sys
import os

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from algorithms.rft.kernels.operator_arft_kernel import build_operator_kernel, arft_forward

def optimize_operator_arft():
    print("="*60)
    print("OPERATOR-BASED ARFT OPTIMIZATION")
    print("="*60)
    
    N = 256
    t = np.linspace(0, 1, N)
    phi = (1 + np.sqrt(5)) / 2
    
    # Target Signal: Golden Quasi-Periodic
    x_qp = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 10 * phi * t)
    
    # Compute Autocorrelation of the Target Signal
    # This defines the "ideal" operator for this signal class
    autocorr = np.correlate(x_qp, x_qp, mode='full')
    autocorr = autocorr[autocorr.size // 2:] # Keep positive lags
    autocorr = autocorr[:N] # Truncate to size N
    
    # Build Kernel from Autocorrelation
    print("Building operator kernel from signal autocorrelation...")
    kernel = build_operator_kernel(N, autocorr)
    
    # Baseline FFT Sparsity
    y_fft = np.fft.fft(x_qp, norm='ortho')
    
    def gini(array):
        array = np.abs(array) + 1e-9
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)
        
    base_gini = gini(y_fft)
    print(f"Baseline FFT Gini: {base_gini:.4f}")
    
    # Test Operator ARFT
    y_arft = arft_forward(x_qp, kernel)
    arft_gini = gini(y_arft)
    
    print(f"Operator ARFT Gini: {arft_gini:.4f}")
    
    gain = (arft_gini / base_gini - 1) * 100
    print(f"Improvement:        {gain:.2f}%")
    
    if arft_gini > base_gini:
        print("✅ Operator ARFT successfully beats FFT sparsity!")
        print("   (This confirms KLT optimality for this signal)")
    else:
        print("❌ Operator ARFT failed to beat FFT sparsity.")

if __name__ == "__main__":
    optimize_operator_arft()
