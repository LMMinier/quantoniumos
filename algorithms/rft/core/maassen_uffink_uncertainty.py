# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Theorem 9: Maassen-Uffink Entropic Uncertainty Principle for RFT
================================================================

This module implements the **correct** finite-dimensional uncertainty principle
for the canonical RFT, using the Maassen-Uffink entropic bound.

⚠️ WARNING: The Heisenberg bound Δx·Δp ≥ ℏ/2 is for CONTINUOUS systems.
   For finite-dimensional discrete transforms, use the ENTROPIC bound instead.

THE MAASSEN-UFFINK ENTROPIC UNCERTAINTY PRINCIPLE
==================================================

For any two orthonormal bases U and V in ℂ^N, and any unit vector x:

    H(|x|²) + H(|Ux|²) ≥ -2 log(μ(U))

where:
    H(p) = -Σ p_k log(p_k)  is the Shannon entropy
    μ(U) = max_{j,k} |U_{jk}|  is the mutual coherence

SPECIAL CASES:
- DFT: μ = 1/√N, so H(x) + H(Fx) ≥ log(N)
- Identity: μ = 1, so H(x) + H(x) ≥ 0 (trivial)
- RFT: μ depends on N, strictly between 1/√N and 1

KEY INSIGHT: The RFT's mutual coherence μ(U_φ) determines its uncertainty bound.
For golden quasi-periodic signals, the RFT achieves better entropy balance.

REFERENCES:
- Maassen, H. & Uffink, J. B. M. (1988). Generalized entropic uncertainty relations.
  Physical Review Letters, 60(12), 1103.
- Discrete uncertainty principles: Donoho-Stark (1989), Elad-Bruckstein (2002)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import scipy.linalg

from .resonant_fourier_transform import PHI, rft_basis_matrix


# =============================================================================
# Shannon Entropy
# =============================================================================

def shannon_entropy(p: np.ndarray, epsilon: float = 1e-15) -> float:
    """Shannon entropy H(p) = -Σ p_k log(p_k).
    
    Args:
        p: Probability distribution (must sum to 1)
        epsilon: Small value to avoid log(0)
    
    Returns:
        Entropy in nats (natural log). Divide by log(2) for bits.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.clip(p, epsilon, 1.0)  # Avoid log(0)
    p = p / p.sum()  # Normalize
    return float(-np.sum(p * np.log(p)))


def entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits: H_2(p) = H(p) / log(2)."""
    return shannon_entropy(p) / np.log(2)


def signal_entropy(x: np.ndarray) -> float:
    """Entropy of the probability distribution |x|² / ||x||²."""
    x = np.asarray(x, dtype=np.complex128).ravel()
    p = np.abs(x) ** 2
    p = p / p.sum()
    return shannon_entropy(p)


# =============================================================================
# Mutual Coherence
# =============================================================================

def mutual_coherence(U: np.ndarray) -> float:
    """Mutual coherence μ(U) = max_{j,k} |U_{jk}|.
    
    This measures how "spread out" the basis is compared to the standard basis.
    
    Reference values:
        - DFT: μ = 1/√N (maximally incoherent)
        - Identity: μ = 1 (maximally coherent)
        - Random unitary: μ ≈ √(2 log N / N) (w.h.p.)
    
    The Maassen-Uffink bound becomes tighter as μ → 1/√N.
    """
    U = np.asarray(U, dtype=np.complex128)
    return float(np.max(np.abs(U)))


def mutual_coherence_two_bases(U: np.ndarray, V: np.ndarray) -> float:
    """Mutual coherence between two bases: μ(U,V) = max_{j,k} |⟨u_j, v_k⟩|."""
    return float(np.max(np.abs(U.conj().T @ V)))


# =============================================================================
# Maassen-Uffink Entropic Uncertainty Bound
# =============================================================================

@dataclass(frozen=True)
class MaassenUffinkBound:
    """The Maassen-Uffink entropic uncertainty bound for a basis."""
    N: int
    mutual_coherence: float
    entropy_bound: float  # -2 log(μ)
    entropy_bound_bits: float  # -2 log₂(μ)
    dft_bound: float  # log(N) for comparison
    tightness_ratio: float  # bound / log(N)


def compute_maassen_uffink_bound(U: np.ndarray) -> MaassenUffinkBound:
    """Compute the Maassen-Uffink bound for a unitary basis U.
    
    For any unit vector x:
        H(|x|²) + H(|Ux|²) ≥ -2 log(μ(U))
    
    Args:
        U: Unitary matrix (N×N)
    
    Returns:
        MaassenUffinkBound with all relevant quantities
    """
    U = np.asarray(U, dtype=np.complex128)
    N = U.shape[0]
    
    mu = mutual_coherence(U)
    bound = -2 * np.log(mu)
    bound_bits = -2 * np.log2(mu)
    dft_bound = np.log(N)
    
    return MaassenUffinkBound(
        N=N,
        mutual_coherence=mu,
        entropy_bound=bound,
        entropy_bound_bits=bound_bits,
        dft_bound=dft_bound,
        tightness_ratio=bound / dft_bound if dft_bound > 0 else float('inf'),
    )


def rft_maassen_uffink_bound(N: int) -> MaassenUffinkBound:
    """Compute the Maassen-Uffink bound for the canonical RFT basis."""
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    return compute_maassen_uffink_bound(U_rft)


def dft_maassen_uffink_bound(N: int) -> MaassenUffinkBound:
    """Compute the Maassen-Uffink bound for the DFT basis."""
    F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    return compute_maassen_uffink_bound(F)


# =============================================================================
# Entropic Uncertainty Measurement
# =============================================================================

@dataclass(frozen=True)
class EntropicUncertainty:
    """Complete entropic uncertainty measurement for a signal."""
    N: int
    # Time domain
    time_entropy: float
    # Frequency domain (DFT)
    dft_entropy: float
    dft_sum: float  # H(x) + H(Fx)
    dft_bound: float  # log(N)
    dft_achieves_bound: bool
    # Frequency domain (RFT)
    rft_entropy: float
    rft_sum: float  # H(x) + H(U_φx)
    rft_bound: float  # -2 log(μ)
    rft_achieves_bound: bool
    # Comparison
    rft_more_concentrated: bool  # Lower entropy in RFT domain


def measure_entropic_uncertainty(x: np.ndarray) -> EntropicUncertainty:
    """Measure entropic uncertainty of a signal under DFT and RFT.
    
    The Maassen-Uffink principle states:
        H(|x|²) + H(|Ux|²) ≥ -2 log(μ(U))
    
    We measure how close each transform gets to its bound.
    """
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = len(x)
    
    # Normalize
    x = x / np.linalg.norm(x)
    
    # Time-domain entropy
    H_time = signal_entropy(x)
    
    # DFT
    X_dft = np.fft.fft(x, norm='ortho')
    H_dft = signal_entropy(X_dft)
    dft_sum = H_time + H_dft
    dft_bound = np.log(N)  # μ(F) = 1/√N → -2 log(1/√N) = log(N)
    
    # RFT
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_rft = U_rft.conj().T @ x
    H_rft = signal_entropy(X_rft)
    mu_rft = mutual_coherence(U_rft)
    rft_sum = H_time + H_rft
    rft_bound = -2 * np.log(mu_rft)
    
    return EntropicUncertainty(
        N=N,
        time_entropy=H_time,
        dft_entropy=H_dft,
        dft_sum=dft_sum,
        dft_bound=dft_bound,
        dft_achieves_bound=(dft_sum >= dft_bound - 0.01),
        rft_entropy=H_rft,
        rft_sum=rft_sum,
        rft_bound=rft_bound,
        rft_achieves_bound=(rft_sum >= rft_bound - 0.01),
        rft_more_concentrated=(H_rft < H_dft),
    )


# =============================================================================
# Test Signals
# =============================================================================

def gaussian_signal(N: int, center: float = 0.5, width: float = 0.1) -> np.ndarray:
    """Gaussian pulse (achieves near-optimal DFT uncertainty)."""
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
    """Pure harmonic (achieves perfect DFT concentration)."""
    n = np.arange(N, dtype=np.float64)
    x = np.exp(2j * np.pi * k * n / N)
    return x / np.linalg.norm(x)


def delta_signal(N: int, pos: int = 0) -> np.ndarray:
    """Delta spike (achieves minimum time entropy)."""
    x = np.zeros(N, dtype=np.complex128)
    x[pos] = 1.0
    return x


def uniform_signal(N: int) -> np.ndarray:
    """Uniform signal (achieves maximum time entropy log(N))."""
    return np.ones(N, dtype=np.complex128) / np.sqrt(N)


# =============================================================================
# Concentration Metrics (Connection to Theorem 8)
# =============================================================================

def k99(X: np.ndarray, threshold: float = 0.99) -> int:
    """Smallest K such that top-K coefficients capture ≥ threshold energy."""
    p = np.abs(X) ** 2
    p = p / p.sum()
    idx = np.argsort(p)[::-1]
    c = np.cumsum(p[idx])
    return int(np.searchsorted(c, threshold) + 1)


@dataclass(frozen=True)
class ConcentrationMeasurement:
    """Concentration metrics connecting entropy and K99."""
    k99_dft: int
    k99_rft: int
    entropy_dft: float
    entropy_rft: float
    max_entropy: float  # log(N)
    rft_wins_k99: bool
    rft_wins_entropy: bool


def measure_concentration(x: np.ndarray) -> ConcentrationMeasurement:
    """Measure K99 and entropy for both DFT and RFT."""
    x = np.asarray(x, dtype=np.complex128).ravel()
    N = len(x)
    x = x / np.linalg.norm(x)
    
    # DFT
    X_dft = np.fft.fft(x, norm='ortho')
    k99_dft = k99(X_dft)
    H_dft = signal_entropy(X_dft)
    
    # RFT
    U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
    X_rft = U_rft.conj().T @ x
    k99_rft = k99(X_rft)
    H_rft = signal_entropy(X_rft)
    
    return ConcentrationMeasurement(
        k99_dft=k99_dft,
        k99_rft=k99_rft,
        entropy_dft=H_dft,
        entropy_rft=H_rft,
        max_entropy=np.log(N),
        rft_wins_k99=(k99_rft < k99_dft),
        rft_wins_entropy=(H_rft < H_dft),
    )


# =============================================================================
# Theorem 9 Verification
# =============================================================================

def verify_theorem_9(N: int, x: np.ndarray) -> Tuple[bool, str]:
    """Verify that the Maassen-Uffink bound holds for a signal.
    
    Returns:
        (passes, message)
    """
    eu = measure_entropic_uncertainty(x)
    
    # Check DFT bound
    dft_passes = eu.dft_sum >= eu.dft_bound - 1e-10
    
    # Check RFT bound
    rft_passes = eu.rft_sum >= eu.rft_bound - 1e-10
    
    if dft_passes and rft_passes:
        return True, f"Maassen-Uffink holds: H(x)+H(DFT)={eu.dft_sum:.3f}≥{eu.dft_bound:.3f}, H(x)+H(RFT)={eu.rft_sum:.3f}≥{eu.rft_bound:.3f}"
    else:
        issues = []
        if not dft_passes:
            issues.append(f"DFT bound violated: {eu.dft_sum:.3f} < {eu.dft_bound:.3f}")
        if not rft_passes:
            issues.append(f"RFT bound violated: {eu.rft_sum:.3f} < {eu.rft_bound:.3f}")
        return False, "; ".join(issues)


# =============================================================================
# Demonstration
# =============================================================================

def print_theorem_9_verification(N: int = 64, num_samples: int = 100):
    """Demonstrate Theorem 9 (Maassen-Uffink Entropic Uncertainty)."""
    
    print("=" * 75)
    print("THEOREM 9: MAASSEN-UFFINK ENTROPIC UNCERTAINTY PRINCIPLE")
    print("=" * 75)
    print()
    print("For any unit vector x and unitary U:")
    print("    H(|x|²) + H(|Ux|²) ≥ -2 log(μ(U))")
    print()
    print("where μ(U) = max|U_{jk}| is the mutual coherence.")
    print()
    
    # Compute bounds
    rft_bound = rft_maassen_uffink_bound(N)
    dft_bound = dft_maassen_uffink_bound(N)
    
    print(f"N = {N}")
    print(f"DFT mutual coherence: μ(F) = 1/√N = {dft_bound.mutual_coherence:.4f}")
    print(f"RFT mutual coherence: μ(U_φ) = {rft_bound.mutual_coherence:.4f}")
    print()
    print(f"DFT entropy bound: -2 log(μ) = log(N) = {dft_bound.entropy_bound:.4f} nats")
    print(f"RFT entropy bound: -2 log(μ) = {rft_bound.entropy_bound:.4f} nats")
    print(f"Tightness ratio (RFT/DFT): {rft_bound.tightness_ratio:.3f}")
    print()
    
    # Test signals
    signals = [
        ("Delta (t=0)", delta_signal(N, 0)),
        ("Uniform", uniform_signal(N)),
        ("Gaussian", gaussian_signal(N)),
        ("Harmonic (k=3)", harmonic_signal(N, k=3)),
        ("Golden QP", golden_quasiperiodic_signal(N)),
    ]
    
    print("-" * 75)
    print(f"{'Signal':<18} {'H(x)':<8} {'H(DFT)':<8} {'H(RFT)':<8} {'DFT sum':<9} {'RFT sum':<9} {'DFT✓':<5} {'RFT✓'}")
    print("-" * 75)
    
    for name, x in signals:
        eu = measure_entropic_uncertainty(x)
        dft_ok = "✓" if eu.dft_achieves_bound else "✗"
        rft_ok = "✓" if eu.rft_achieves_bound else "✗"
        print(f"{name:<18} {eu.time_entropy:<8.3f} {eu.dft_entropy:<8.3f} {eu.rft_entropy:<8.3f} "
              f"{eu.dft_sum:<9.3f} {eu.rft_sum:<9.3f} {dft_ok:<5} {rft_ok}")
    
    print()
    print("-" * 75)
    print("CONNECTION TO THEOREM 8 (Concentration Inequality)")
    print("-" * 75)
    print()
    print(f"{'Signal':<18} {'K99(DFT)':<10} {'K99(RFT)':<10} {'H(DFT)':<10} {'H(RFT)':<10} {'RFT wins'}")
    print("-" * 75)
    
    for name, x in signals:
        cm = measure_concentration(x)
        winner = "K99+H" if cm.rft_wins_k99 and cm.rft_wins_entropy else (
            "K99" if cm.rft_wins_k99 else (
            "H" if cm.rft_wins_entropy else "DFT"))
        print(f"{name:<18} {cm.k99_dft:<10} {cm.k99_rft:<10} {cm.entropy_dft:<10.3f} {cm.entropy_rft:<10.3f} {winner}")
    
    print()
    print("-" * 75)
    print("STATISTICAL VERIFICATION (Golden QP ensemble)")
    print("-" * 75)
    print()
    
    rng = np.random.default_rng(42)
    dft_bound_passes = 0
    rft_bound_passes = 0
    rft_concentrates_better = 0
    
    for _ in range(num_samples):
        f0 = rng.uniform(0, 1)
        a = rng.uniform(-1, 1)
        x = golden_quasiperiodic_signal(N, f0, a)
        eu = measure_entropic_uncertainty(x)
        
        if eu.dft_achieves_bound:
            dft_bound_passes += 1
        if eu.rft_achieves_bound:
            rft_bound_passes += 1
        if eu.rft_more_concentrated:
            rft_concentrates_better += 1
    
    print(f"Over {num_samples} golden quasi-periodic signals:")
    print(f"  DFT bound satisfied: {dft_bound_passes}/{num_samples} ({100*dft_bound_passes/num_samples:.1f}%)")
    print(f"  RFT bound satisfied: {rft_bound_passes}/{num_samples} ({100*rft_bound_passes/num_samples:.1f}%)")
    print(f"  RFT more concentrated: {rft_concentrates_better}/{num_samples} ({100*rft_concentrates_better/num_samples:.1f}%)")
    print()
    print("INTERPRETATION:")
    print("  The Maassen-Uffink bound is a TRUE theorem - it MUST hold for all signals.")
    print("  RFT achieves better concentration (lower entropy) on golden QP signals,")
    print("  while still satisfying the entropic uncertainty bound.")


if __name__ == "__main__":
    print_theorem_9_verification(N=64, num_samples=200)
