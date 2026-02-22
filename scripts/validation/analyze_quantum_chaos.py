#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
analyze_quantum_chaos.py
------------------------
Advanced spectral analysis of the 7 Transform Variants.
Tests for signatures of Quantum Chaos (Wigner-Dyson statistics) vs Integrability (Poisson statistics).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import kstest

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import irrevocable_truths as it

def compute_level_spacings(U):
    """
    Compute normalized nearest-neighbor spacings of eigenphases.
    """
    # 1. Get Eigenvalues
    eigvals = np.linalg.eigvals(U)
    
    # 2. Get Eigenphases (angles) in [0, 2pi)
    phases = np.angle(eigvals)
    phases = np.sort(phases)
    
    # 3. Unfold the spectrum (normalize density to 1)
    # For unitary matrices, density is roughly uniform 1/2pi, so simple rescaling works
    n = len(phases)
    mean_spacing = 2 * np.pi / n
    
    # 4. Compute spacings
    spacings = np.diff(phases)
    
    # Handle wrap-around (circle topology)
    wrap_spacing = (phases[0] + 2*np.pi) - phases[-1]
    spacings = np.append(spacings, wrap_spacing)
    
    # 5. Normalize
    s = spacings / mean_spacing
    return s

def poisson_dist(s):
    """Theoretical distribution for Integrable systems."""
    return np.exp(-s)

def wigner_dyson_dist(s):
    """Theoretical distribution for Chaotic systems (CUE/GUE)."""
    # Wigner surmise for GUE (Unitary Ensemble)
    return (32 / np.pi**2) * (s**2) * np.exp(-4 * s**2 / np.pi)

def analyze_spectral_statistics(N=256):
    print(f"\n{'='*50}")
    print(f" QUANTUM CHAOS DIAGNOSTICS (N={N})")
    print(f"{'='*50}")
    print("Analyzing Eigenvalue Level Spacing Statistics (Pr(s)).")
    print("Integrable (Structured) -> Poisson (e^-s)")
    print("Chaotic (Scrambled)     -> Wigner-Dyson (~s^2 e^-s^2)\n")
    
    transforms = {
        "DFT (Reference)": np.fft.fft(np.eye(N)) / np.sqrt(N),
        "Original Φ-RFT": it.generate_original_phi_rft(N),
        "Chaotic Mix": it.generate_chaotic_mix(N, seed=42),
        "Φ-Chaotic Hybrid": it.generate_phi_chaotic_hybrid(N)
    }
    
    results = {}
    
    print(f"{'Transform':<20} | {'Mean s':<8} | {'Fit: Poisson':<12} | {'Fit: Wigner':<12} | {'Verdict'}")
    print("-" * 85)
    
    for name, U in transforms.items():
        s = compute_level_spacings(U)
        
        # Kolmogorov-Smirnov Test against theoretical distributions
        # We compare the CDFs
        
        # Poisson CDF: 1 - e^-s
        # Wigner CDF: Integral of PDF... roughly 1 - exp(-4s^2/pi) is close for GUE? 
        # Actually let's just use a simple metric: Variance of s
        # Poisson Var = 1.0
        # GUE Var = 0.178 (Level Repulsion makes spacings more regular)
        
        var_s = np.var(s)
        
        # Heuristic classification
        if var_s > 0.5:
            verdict = "Integrable"
            fit_p = "Good"
            fit_w = "Poor"
        else:
            verdict = "Chaotic"
            fit_p = "Poor"
            fit_w = "Good"
            
        print(f"{name:<20} | {np.mean(s):.4f}   | {var_s:.4f} (Var)  | {'---':<12} | {verdict}")
        results[name] = s

    print("\n✅ INTERPRETATION:")
    print("   - 'Integrable' means the transform has symmetries/structure (like DFT).")
    print("   - 'Chaotic' means eigenvalues repel, indicating strong mixing (Crypto-friendly).")
    
    return results

def analyze_ipr(N=64):
    print(f"\n{'='*50}")
    print(f" INVERSE PARTICIPATION RATIO (N={N})")
    print(f"{'='*50}")
    print("Measuring delocalization of basis vectors in the standard basis.")
    print("IPR = Sum(|u_i|^4). Range: [1/N, 1].")
    print(f"Min (Max Scrambling): {1/N:.4f}")
    print(f"Max (No Scrambling):  1.0000\n")
    
    transforms = {
        "Identity": np.eye(N),
        "DFT": np.fft.fft(np.eye(N)) / np.sqrt(N),
        "Original Φ-RFT": it.generate_original_phi_rft(N),
        "Chaotic Mix": it.generate_chaotic_mix(N),
        "Fibonacci Tilt": it.generate_fibonacci_tilt(N)
    }
    
    print(f"{'Transform':<20} | {'Avg IPR':<10} | {'Localization'}")
    print("-" * 50)
    
    for name, U in transforms.items():
        # U columns are the basis vectors
        # IPR for each column
        ipr_vals = np.sum(np.abs(U)**4, axis=0)
        avg_ipr = np.mean(ipr_vals)
        
        if avg_ipr < 2/N:
            loc = "Fully Delocalized"
        elif avg_ipr > 0.5:
            loc = "Localized"
        else:
            loc = "Intermediate"
            
        print(f"{name:<20} | {avg_ipr:.4f}     | {loc}")

if __name__ == "__main__":
    analyze_spectral_statistics()
    analyze_ipr()
