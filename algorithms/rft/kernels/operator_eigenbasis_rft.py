# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Resonant Fourier Transform (RFT) - OPERATOR VARIANT (NOT CANONICAL)
====================================================================

**NOTE:** This is the OPERATOR-BASED eigenbasis variant, NOT the canonical RFT.
For the canonical (patent-compliant) RFT, use:
    algorithms/rft/core/resonant_fourier_transform.py

This file implements the eigenbasis of a modeled autocorrelation operator,
which is a DIFFERENT mathematical object than the canonical golden-ratio basis.

A FIXED, structured unitary transform designed for quasi-periodic signal families.

Key Difference from Per-Signal KLT:
- KLT: Computes eigenbasis of EACH signal's autocorrelation (O(N³) per signal, basis changes)
- RFT: Computes eigenbasis of the EXPECTED autocorrelation for a signal FAMILY (fixed matrix, reusable)

The "Golden" Structure:
The autocorrelation of quasi-periodic signals with frequency ratios φ = (1+√5)/2 
has a specific Toeplitz structure. We model this analytically:

    r[k] = cos(2πf₀k) + cos(2πf₀φk) * decay(k)

This gives a FIXED operator that is optimal (in KLT sense) for the entire family
of golden-ratio quasi-periodic signals.
"""

import numpy as np
from scipy.linalg import toeplitz, eigh
from functools import lru_cache
from typing import Optional, Tuple

# Optional operator-based variant support
try:
    from algorithms.rft.variants.operator_variants import get_operator_variant
except Exception:  # pragma: no cover - fallback when variants package unavailable
    get_operator_variant = None

# The Golden Ratio
PHI = (1 + np.sqrt(5)) / 2

DEFAULT_VARIANT: Optional[str] = None


@lru_cache(maxsize=16)
def build_rft_kernel(N: int, f0: float = 10.0, decay_rate: float = 0.01) -> np.ndarray:
    """
    Build the FIXED Resonant Fourier Transform kernel for size N.
    
    This is NOT per-signal adaptive. It models the expected autocorrelation
    structure of the golden quasi-periodic signal family:
        x(t) = A*sin(2πf₀t) + B*sin(2πf₀φt + θ)
    
    The autocorrelation of this family is:
        r[k] ≈ (A²/2)cos(2πf₀k) + (B²/2)cos(2πf₀φk)
    
    With a decay envelope to ensure the matrix is well-conditioned.
    
    Args:
        N: Signal length
        f0: Base frequency (normalized, default 10 cycles per unit time)
        decay_rate: Exponential decay rate for conditioning
        
    Returns:
        Φ: N×N unitary matrix (RFT basis - eigenvectors of the modeled autocorrelation)
    """
    k = np.arange(N)
    
    # Model the expected autocorrelation for golden quasi-periodic signals
    # Normalized time: we assume signal is sampled at N points over [0, 1]
    t_k = k / N
    
    # Autocorrelation components
    r_fundamental = np.cos(2 * np.pi * f0 * t_k)
    r_golden = np.cos(2 * np.pi * f0 * PHI * t_k)
    
    # Decay envelope for numerical stability
    decay = np.exp(-decay_rate * k)
    
    # Combined autocorrelation model
    # Weight: assume equal power in both components
    r = (r_fundamental + r_golden) * decay
    r[0] = 1.0  # Normalize
    
    # Build Toeplitz autocorrelation matrix
    R = toeplitz(r)
    
    # Diagonalize to get the transform basis
    eigenvalues, eigenvectors = eigh(R)

    # Sort by eigenvalue (descending) for energy compaction
    idx = np.argsort(eigenvalues)[::-1]
    Phi = eigenvectors[:, idx]

    return Phi.astype(np.float64)


def _select_basis(n: int, variant: Optional[str] = None, Phi: Optional[np.ndarray] = None) -> np.ndarray:
    """Resolve which basis to use for forward/inverse transforms."""
    if Phi is not None:
        return Phi
    chosen_variant = variant or DEFAULT_VARIANT
    if chosen_variant and get_operator_variant is not None:
        return get_operator_variant(chosen_variant, n)
    return build_rft_kernel(n)


def rft_forward(x: np.ndarray, Phi: np.ndarray = None, variant: Optional[str] = None) -> np.ndarray:
    """Apply Resonant Fourier Transform (optionally selecting a variant)."""
    basis = _select_basis(len(x), variant=variant, Phi=Phi)
    return basis.T @ x


def rft_inverse(X: np.ndarray, Phi: np.ndarray = None, variant: Optional[str] = None) -> np.ndarray:
    """Apply Inverse Resonant Fourier Transform (optionally selecting a variant)."""
    basis = _select_basis(len(X), variant=variant, Phi=Phi)
    return basis @ X


def verify_unitarity(Phi: np.ndarray) -> Tuple[bool, float]:
    """Verify that Φ is unitary (Φ^T Φ = I)."""
    I = np.eye(Phi.shape[0])
    error = np.linalg.norm(Phi.T @ Phi - I, 'fro')
    return error < 1e-10, error


# ============================================================================
# SIGNAL FAMILY GENERATORS
# ============================================================================

def generate_golden_quasiperiodic(N: int, f0: float = 10.0, 
                                   A: float = 1.0, B: float = 1.0,
                                   phase: float = 0.0, 
                                   noise_std: float = 0.0) -> np.ndarray:
    """
    Generate a sample from the Golden Quasi-Periodic signal family.
    
    x(t) = A*sin(2πf₀t) + B*sin(2πf₀φt + θ) + noise
    """
    t = np.linspace(0, 1, N)
    x = A * np.sin(2 * np.pi * f0 * t) + B * np.sin(2 * np.pi * f0 * PHI * t + phase)
    if noise_std > 0:
        x += np.random.randn(N) * noise_std
    return x


def generate_fibonacci_modulated(N: int, f0: float = 10.0, 
                                  depth: int = 5) -> np.ndarray:
    """
    Generate Fibonacci-modulated signal (more complex member of the family).
    Uses Fibonacci sequence as amplitude modulation.
    """
    t = np.linspace(0, 1, N)
    fib = [1, 1]
    for _ in range(depth):
        fib.append(fib[-1] + fib[-2])
    
    x = np.zeros(N)
    for i, f in enumerate(fib):
        freq = f0 * (PHI ** i)
        x += (1.0 / f) * np.sin(2 * np.pi * freq * t)
    
    return x / np.max(np.abs(x))  # Normalize


def generate_phyllotaxis_signal(N: int, spirals: int = 8) -> np.ndarray:
    """
    Generate signal based on a fixed golden-angle phase sequence.
    Uses divergence angle 2π/φ² ≈ 137.5° (complement 2π/φ ≈ 222.5°).
    This is an irrational-rotation sequence only (no packing/growth model).
    """
    golden_angle = 2 * np.pi / (PHI ** 2)  # ~137.5 degrees
    t = np.linspace(0, 1, N)
    
    x = np.zeros(N)
    for s in range(1, spirals + 1):
        x += np.sin(s * golden_angle * np.arange(N) + 2 * np.pi * s * t)
    
    return x / np.max(np.abs(x))


# ============================================================================
# TEST SUITE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RESONANT FOURIER TRANSFORM - KERNEL VERIFICATION")
    print("="*70)
    
    N = 256
    Phi = build_rft_kernel(N)
    
    is_unitary, error = verify_unitarity(Phi)
    print(f"\nKernel Size: {N}x{N}")
    print(f"Unitarity Check: {'PASS' if is_unitary else 'FAIL'} (error: {error:.2e})")
    
    # Test perfect reconstruction
    x = generate_golden_quasiperiodic(N)
    X = rft_forward(x, Phi)
    x_rec = rft_inverse(X, Phi)
    rec_error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
    print(f"Reconstruction Error: {rec_error:.2e}")
