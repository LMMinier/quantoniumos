# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
LCT/FrFT Non-Equivalence Test
Tests that Φ-RFT's golden-ratio phase is NOT purely quadratic
"""
import numpy as np
from algorithms.rft.core.resonant_fourier_transform import rft_phase_vectors_canonical as rft_phase_vectors, PHI

def _unwrap_phase(z: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(z))

def _fit_quadratic_phase(theta: np.ndarray) -> tuple[float, float, float, float]:
    """
    Fit theta[k] ≈ a k^2 + b k + c (mod 2π). We unwrap first, then least-squares fit.
    Returns: a,b,c, rms_residual
    """
    n = theta.size
    k = np.arange(n, dtype=np.float64)
    M = np.column_stack([k*k, k, np.ones_like(k)])
    coeffs, *_ = np.linalg.lstsq(M, theta, rcond=None)
    fit = M @ coeffs
    rms = float(np.sqrt(np.mean((theta - fit)**2)))
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), rms

def test_non_quadratic_phase_residual():
    """
    Proves Φ-RFT is NOT LCT/FrFT-equivalent by showing golden-ratio phase
    cannot be approximated by pure quadratic function
    """
    n = 256
    beta, sigma = 0.83, 1.25
    D_phi, _ = rft_phase_vectors(n, beta=beta, sigma=sigma, phi=PHI)
    theta = _unwrap_phase(D_phi)
    a, b, c, rms = _fit_quadratic_phase(theta)
    # If D_phi were LCT/FrFT-like (pure quadratic), rms → ~0. It shouldn't.
    assert rms > 1e-2, f"Quadratic residual too small (rms={rms}); D_phi looks quadratic."
    print(f"✓ Non-quadratic phase confirmed: RMS residual = {rms:.6f} (>> 0)")
