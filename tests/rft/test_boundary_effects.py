#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Proof of Boundary Effect Mitigation (Theorem 10).

This test demonstrates the "Boundary Problem" where DFT/RFT fail to efficiently
represent non-periodic signals due to implicit periodic extension discontinuities,
and proves that the Hybrid Basis (via DCT) solves this.
"""
import numpy as np
import pytest
from scipy.fftpack import dct, idct
from numpy.fft import fft, ifft
from algorithms.rft.hybrids.legacy_mca import adaptive_hybrid_compress, rft_forward, rft_inverse

def test_boundary_discontinuity_reconstruction():
    """
    Test reconstruction accuracy at signal boundaries for a linear ramp.
    
    A linear ramp f(t) = t has the worst-case boundary discontinuity for DFT
    because f(0) != f(N-1).
    """
    N = 256
    t = np.linspace(0, 1, N)
    # Linear ramp: Strong discontinuity if treated as periodic
    signal = t 
    
    # 1. Pure RFT (DFT-like) Reconstruction with 90% compression
    # We keep top 10% coefficients
    rft_coeffs = rft_forward(signal)
    threshold_rft = np.sort(np.abs(rft_coeffs))[-int(N * 0.1)]
    rft_sparse = rft_coeffs * (np.abs(rft_coeffs) >= threshold_rft)
    rft_recon = rft_inverse(rft_sparse).real
    
    # 2. Hybrid Reconstruction
    hybrid_struct, hybrid_text, _, _ = adaptive_hybrid_compress(signal, max_iter=5)
    hybrid_recon = hybrid_struct + hybrid_text
    
    # Measure error at the boundaries (first and last 5 samples)
    boundary_indices = np.concatenate([np.arange(5), np.arange(N-5, N)])
    
    rft_boundary_mse = np.mean((signal[boundary_indices] - rft_recon[boundary_indices])**2)
    hybrid_boundary_mse = float(np.real(np.mean((signal[boundary_indices] - hybrid_recon[boundary_indices])**2)))
    
    print(f"\nBoundary MSE (RFT): {rft_boundary_mse:.6f}")
    print(f"Boundary MSE (Hybrid): {hybrid_boundary_mse:.6f}")
    
    # Hybrid (using DCT) should be orders of magnitude better at boundaries
    # because DCT implies symmetric extension (no discontinuity)
    # We accept a 20x improvement (0.05 ratio) as proof of mitigation
    assert hybrid_boundary_mse < rft_boundary_mse * 0.05, f"Hybrid basis failed to mitigate boundary effects (Ratio: {hybrid_boundary_mse/rft_boundary_mse:.3f})"
    assert hybrid_boundary_mse < 5e-3, "Hybrid reconstruction at boundary is too noisy"

if __name__ == "__main__":
    test_boundary_discontinuity_reconstruction()
