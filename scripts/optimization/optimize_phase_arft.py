# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Optimization Script for Phase-Coupled ARFT
"""

import numpy as np
import sys
import os

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from algorithms.rft.kernels.phase_arft_kernel import build_phase_coupled_kernel, arft_forward

def optimize_phase_arft():
    print("="*60)
    print("PHASE-COUPLED ARFT OPTIMIZATION")
    print("="*60)
    
    N = 256
    t = np.linspace(0, 1, N)
    phi = (1 + np.sqrt(5)) / 2
    
    # Target Signal: Golden Quasi-Periodic
    x_qp = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 10 * phi * t)
    
    # Resonance Map: Matched to the signal structure
    resonance_map = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 10 * phi * t)
    
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
    
    # Sweep Coupling Parameters (Smaller range for phase warping)
    couplings = np.linspace(-0.1, 0.1, 21)
    best_gini = 0
    best_coupling = 0
    
    print("\nSweeping coupling parameters...")
    print(f"{'Coupling':<10} | {'Gini':<10} | {'Gain':<10}")
    print("-" * 36)
    
    for c in couplings:
        if abs(c) < 1e-4: continue
        
        kernel = build_phase_coupled_kernel(N, resonance_map=resonance_map, coupling=c)
        y_arft = arft_forward(x_qp, kernel)
        g = gini(y_arft)
        
        gain = (g / base_gini - 1) * 100
        print(f"{c:<10.4f} | {g:<10.4f} | {gain:<10.2f}%")
        
        if g > best_gini:
            best_gini = g
            best_coupling = c
            
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Coupling: {best_coupling:.4f}")
    print(f"Best Gini:     {best_gini:.4f}")
    print(f"Improvement:   {(best_gini/base_gini - 1)*100:.2f}%")
    
    if best_gini > base_gini:
        print("✅ ARFT successfully beats FFT sparsity!")
    else:
        print("❌ ARFT failed to beat FFT sparsity.")

if __name__ == "__main__":
    optimize_phase_arft()
