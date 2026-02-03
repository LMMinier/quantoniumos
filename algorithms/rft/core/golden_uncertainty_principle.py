# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
⚠️ DEPRECATED: Heisenberg-Style Uncertainty (Use Maassen-Uffink Instead)
=========================================================================

This module uses Heisenberg-style spread products which are INCORRECT for
finite-dimensional discrete transforms. The continuous bound Δx·Δp ≥ ℏ/2
does NOT apply to finite N.

For the CORRECT uncertainty principle, use:
    from algorithms.rft.core.maassen_uffink_uncertainty import (
        measure_entropic_uncertainty,
        verify_theorem_9,
    )

See: docs/theory/DEFINITION_FIREWALL.md for details on why this is deprecated.

---

LEGACY: Golden-RFT Uncertainty Principle
========================================

This module derives and implements a general uncertainty principle for the
canonical Resonant Fourier Transform (RFT), connecting:

1. Theorem 2: RFT unitarity (U^H U = I)
2. Theorem 4: Twisted convolution diagonalization
3. Theorem 6: RFT ≠ DFT (non-equivalence)
4. Theorem 8: Golden Spectral Concentration Inequality

THE GOLDEN-RFT UNCERTAINTY PRINCIPLE
====================================

For any signal x ∈ ℂ^N and the canonical RFT basis U_φ:

    Δ_t(x) · Δ_φ(x) ≥ (1/4π) · (1 - μ(U_φ)²)

where:
    Δ_t(x) = time-domain spread (position uncertainty)
    Δ_φ(x) = golden-frequency spread (RFT uncertainty)
    μ(U_φ) = mutual coherence between U_φ and the identity basis

COMPARISON WITH CLASSICAL UNCERTAINTY:

| Principle | Lower Bound | Equality Condition |
|-----------|-------------|-------------------|
| Heisenberg-Fourier | 1/(4π) | Gaussian |
| Gabor (DFT) | 1/(4π) | Chirped Gaussian |
| Golden-RFT (NEW) | (1/4π)(1-μ²) | Golden quasi-periodic |

KEY INSIGHT: The golden ratio's irrationality creates a *different* uncertainty
trade-off compared to the DFT. For golden quasi-periodic signals, the RFT
achieves better localization than predicted by the Fourier uncertainty principle.

COROLLARY (Concentration-Uncertainty Duality):
    Low K₀.₉₉(U_φ,x) ⟺ High Δ_t(x) for golden quasi-periodic x
    
This explains Theorem 8: RFT concentrates golden signals better because
it trades time-spread for frequency-concentration in the φ-grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.linalg

from .resonant_fourier_transform import PHI, rft_basis_matrix


# =============================================================================
# Spread Functionals
# =============================================================================

def time_spread(x: np.ndarray) -> float:
    """Time-domain spread Δ_t(x) = √(⟨n²⟩ - ⟨n⟩²) normalized by N.
    
    This measures how "spread out" the signal is in the time domain.
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = len(x)
    
    prob = np.abs(x) ** 2
    prob = prob / prob.sum()
    
    n = np.arange(N, dtype=np.float64)
    mean_n = np.sum(n * prob)
    mean_n2 = np.sum(n**2 * prob)
    
    variance = mean_n2 - mean_n**2
    return float(np.sqrt(max(0, variance)) / N)


def frequency_spread(X: np.ndarray, freqs: np.ndarray = None) -> float:
    """Frequency-domain spread Δ_f(X) = √(⟨f²⟩ - ⟨f⟩²).
    
    If freqs is None, uses standard DFT frequencies k/N.
    For RFT, pass the φ-grid frequencies.
    """
    X = np.asarray(X, dtype=np.complex128).ravel()
    N = len(X)
    
    if freqs is None:
        freqs = np.arange(N, dtype=np.float64) / N
    freqs = np.asarray(freqs, dtype=np.float64)
    
    prob = np.abs(X) ** 2
    prob = prob / prob.sum()
    
    # Handle circular statistics for frequencies in [0,1)
    # Use complex representation: z = exp(i 2π f)
    z = np.exp(2j * np.pi * freqs)
    mean_z = np.sum(z * prob)
    
    # Circular variance = 1 - |mean_z|
    circ_var = 1 - np.abs(mean_z)
    
    return float(np.sqrt(max(0, circ_var)))


def phi_frequencies(N: int) -> np.ndarray:
    """Golden ratio frequency grid: f_k = frac((k+1)φ)."""
    k = np.arange(N, dtype=np.float64)
    return np.mod((k + 1) * PHI, 1.0)


def golden_frequency_spread(X_rft: np.ndarray) -> float:
    """RFT frequency spread using the φ-grid."""
    N = len(X_rft)
    return frequency_spread(X_rft, phi_frequencies(N))


# =============================================================================
# Mutual Coherence
# =============================================================================

def mutual_coherence(U: np.ndarray, V: np.ndarray = None) -> float:
    """Mutual coherence μ(U,V) = max_{j,k} |⟨u_j, v_k⟩|.
    
    If V is None, uses the identity basis (standard basis vectors).
    
    The mutual coherence measures how "similar" two bases are:
    - μ = 1/√N for maximally incoherent bases (e.g., DFT)
    - μ = 1 for identical bases
    """
    U = np.asarray(U, dtype=np.complex128)
    N = U.shape[0]
    
    if V is None:
        # V = identity = standard basis
        V = np.eye(N, dtype=np.complex128)
    
    # G[j,k] = |⟨u_j, v_k⟩| = |U^H V|_{jk}
    G = np.abs(U.conj().T @ V)
    
    return float(np.max(G))


def rft_dft_coherence(N: int) -> float:
    """Mutual coherence between RFT and DFT bases."""
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    F_dft = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    return mutual_coherence(U_rft, F_dft)


# =============================================================================
# Golden-RFT Uncertainty Principle
# =============================================================================

@dataclass(frozen=True)
class UncertaintyMeasurement:
    """Complete uncertainty measurement for a signal."""
    N: int
    time_spread: float
    dft_spread: float
    rft_spread: float
    product_dft: float
    product_rft: float
    heisenberg_bound: float
    golden_bound: float
    rft_beats_dft: bool
    uncertainty_ratio: float  # product_rft / product_dft


def measure_uncertainty(x: np.ndarray) -> UncertaintyMeasurement:
    """Compute all uncertainty measures for a signal.
    
    Returns complete comparison of DFT vs RFT uncertainty products.
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = len(x)
    
    # Time spread (same for both)
    dt = time_spread(x)
    
    # DFT coefficients and spread
    X_dft = np.fft.fft(x, norm='ortho')
    df_dft = frequency_spread(X_dft)
    
    # RFT coefficients and spread
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_rft = U_rft.conj().T @ x
    df_rft = golden_frequency_spread(X_rft)
    
    # Uncertainty products
    prod_dft = dt * df_dft
    prod_rft = dt * df_rft
    
    # Bounds
    heisenberg = 1 / (4 * np.pi)
    mu = mutual_coherence(U_rft)
    golden = heisenberg * (1 - mu**2)
    
    return UncertaintyMeasurement(
        N=N,
        time_spread=dt,
        dft_spread=df_dft,
        rft_spread=df_rft,
        product_dft=prod_dft,
        product_rft=prod_rft,
        heisenberg_bound=heisenberg,
        golden_bound=golden,
        rft_beats_dft=(prod_rft < prod_dft),
        uncertainty_ratio=prod_rft / prod_dft if prod_dft > 1e-15 else float('inf'),
    )


def golden_uncertainty_bound(N: int) -> Tuple[float, float]:
    """Compute the Golden-RFT uncertainty bound.
    
    Returns: (heisenberg_bound, golden_bound)
    
    The golden bound is: (1/4π)(1 - μ²) where μ is the RFT-identity coherence.
    """
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    mu = mutual_coherence(U_rft)
    
    heisenberg = 1 / (4 * np.pi)
    golden = heisenberg * (1 - mu**2)
    
    return (heisenberg, golden)


# =============================================================================
# Test Signals for Uncertainty Principle
# =============================================================================

def gaussian_signal(N: int, center: float = 0.5, width: float = 0.1) -> np.ndarray:
    """Gaussian pulse (achieves equality in Heisenberg-Fourier)."""
    n = np.arange(N, dtype=np.float64) / N
    x = np.exp(-((n - center) / width)**2)
    return x / np.linalg.norm(x)


def golden_quasiperiodic_signal(N: int, f0: float = 0.3, a: float = 0.5) -> np.ndarray:
    """Golden quasi-periodic signal (native to RFT).
    
    x[n] = exp(i 2π (f₀ n + a · frac(n φ)))
    """
    n = np.arange(N, dtype=np.float64)
    frac = np.mod(n * PHI, 1.0)
    x = np.exp(2j * np.pi * (f0 * n + a * frac))
    return x / np.linalg.norm(x)


def harmonic_signal(N: int, k: int = 3) -> np.ndarray:
    """Pure harmonic (native to DFT)."""
    n = np.arange(N, dtype=np.float64)
    x = np.exp(2j * np.pi * k * n / N)
    return x / np.linalg.norm(x)


def chirp_signal(N: int, rate: float = 0.1) -> np.ndarray:
    """Linear chirp signal."""
    n = np.arange(N, dtype=np.float64)
    x = np.exp(1j * np.pi * rate * n**2 / N)
    return x / np.linalg.norm(x)


# =============================================================================
# Concentration-Uncertainty Duality (connects to Theorem 8)
# =============================================================================

def k99(X: np.ndarray, threshold: float = 0.99) -> int:
    """Smallest K such that top-K coefficients capture ≥ threshold energy."""
    p = np.abs(X) ** 2
    p = p / p.sum()
    idx = np.argsort(p)[::-1]
    c = np.cumsum(p[idx])
    return int(np.searchsorted(c, threshold) + 1)


@dataclass(frozen=True)
class ConcentrationDuality:
    """Measures showing concentration-uncertainty duality."""
    k99_rft: int
    k99_dft: int
    spread_rft: float
    spread_dft: float
    efficiency_rft: float  # k99 / spread (lower = better localization)
    efficiency_dft: float


def concentration_uncertainty_duality(x: np.ndarray) -> ConcentrationDuality:
    """Measure the concentration-uncertainty duality for a signal.
    
    This connects Theorem 8 (concentration inequality) with the
    uncertainty principle.
    
    Key relationship:
        Low K₀.₉₉ (few coefficients needed) ⟺ High efficiency
        
    For golden quasi-periodic signals, RFT achieves better efficiency
    because it trades time-spread for frequency-concentration.
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = len(x)
    
    # DFT
    X_dft = np.fft.fft(x, norm='ortho')
    k99_dft = k99(X_dft)
    spread_dft = frequency_spread(X_dft)
    
    # RFT
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_rft = U_rft.conj().T @ x
    k99_rft = k99(X_rft)
    spread_rft = golden_frequency_spread(X_rft)
    
    # Efficiency = K99 / (N * spread) - how many coefficients per unit spread
    # Lower is better (concentrated in fewer coefficients with less spread)
    eff_dft = k99_dft / (spread_dft + 1e-15) / N
    eff_rft = k99_rft / (spread_rft + 1e-15) / N
    
    return ConcentrationDuality(
        k99_rft=k99_rft,
        k99_dft=k99_dft,
        spread_rft=spread_rft,
        spread_dft=spread_dft,
        efficiency_rft=eff_rft,
        efficiency_dft=eff_dft,
    )


# =============================================================================
# Main: Demonstrate the Golden-RFT Uncertainty Principle
# =============================================================================

def print_uncertainty_comparison(N: int = 64, num_samples: int = 100):
    """Print a comprehensive uncertainty comparison."""
    
    print("=" * 70)
    print("GOLDEN-RFT UNCERTAINTY PRINCIPLE")
    print("=" * 70)
    print()
    
    # Compute bounds
    heisenberg, golden = golden_uncertainty_bound(N)
    print(f"N = {N}")
    print(f"Heisenberg bound: 1/(4π) = {heisenberg:.6f}")
    print(f"Golden-RFT bound: (1/4π)(1-μ²) = {golden:.6f}")
    print()
    
    # Test signals
    signals = [
        ("Gaussian", gaussian_signal(N)),
        ("Golden QP", golden_quasiperiodic_signal(N)),
        ("Harmonic", harmonic_signal(N, k=3)),
        ("Chirp", chirp_signal(N)),
    ]
    
    print(f"{'Signal':<15} {'Δt·Δf(DFT)':<12} {'Δt·Δf(RFT)':<12} {'Ratio':<8} {'RFT wins'}")
    print("-" * 70)
    
    for name, x in signals:
        m = measure_uncertainty(x)
        print(f"{name:<15} {m.product_dft:<12.4f} {m.product_rft:<12.4f} "
              f"{m.uncertainty_ratio:<8.4f} {'✓' if m.rft_beats_dft else ''}")
    
    print()
    print("=" * 70)
    print("CONCENTRATION-UNCERTAINTY DUALITY (Theorem 8 connection)")
    print("=" * 70)
    print()
    
    print(f"{'Signal':<15} {'K99(DFT)':<10} {'K99(RFT)':<10} {'Eff(DFT)':<10} {'Eff(RFT)':<10}")
    print("-" * 70)
    
    for name, x in signals:
        d = concentration_uncertainty_duality(x)
        print(f"{name:<15} {d.k99_dft:<10} {d.k99_rft:<10} "
              f"{d.efficiency_dft:<10.4f} {d.efficiency_rft:<10.4f}")
    
    print()
    print("=" * 70)
    print("STATISTICAL TEST (Golden quasi-periodic ensemble)")
    print("=" * 70)
    print()
    
    rng = np.random.default_rng(42)
    rft_wins = 0
    
    for _ in range(num_samples):
        f0 = rng.uniform(0, 1)
        a = rng.uniform(-1, 1)
        x = golden_quasiperiodic_signal(N, f0, a)
        m = measure_uncertainty(x)
        if m.rft_beats_dft:
            rft_wins += 1
    
    print(f"Over {num_samples} golden QP signals:")
    print(f"  RFT achieves lower uncertainty: {rft_wins}/{num_samples} ({100*rft_wins/num_samples:.1f}%)")
    print()
    print("INTERPRETATION:")
    print("  The Golden-RFT basis achieves a different uncertainty trade-off")
    print("  compared to the DFT. For signals with golden quasi-periodic structure,")
    print("  RFT trades time-spread for better frequency concentration.")


if __name__ == "__main__":
    print_uncertainty_comparison(N=64, num_samples=200)
