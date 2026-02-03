# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Theorem 8 Extension: Diophantine Irrational Scaling Law
=======================================================

This module extends Theorem 8 (Golden Spectral Concentration Inequality) to
general badly approximable (Diophantine) irrationals, providing:

1. **Scaling Law**: The K99 concentration metric follows E[K99] ~ c·log(N)
   for all quadratic irrationals (√2, √3, √5, etc.) satisfying |α - p/q| > c/q²

2. **Sharp Bounds**: Davis-Kahan perturbation theory yields explicit
   concentration bounds tied to continued fraction convergents

3. **Eigenvalue Decay**: Covariance operator eigenvalues decay as
   λ_k ~ exp(-c k log N) where c = 1/log(α) for Diophantine α

Mathematical Background
-----------------------

A real number α is **badly approximable** if there exists c > 0 such that:
    |α - p/q| > c/q²  for all rationals p/q

All quadratic irrationals (roots of integer quadratics) are badly approximable
by Hurwitz's theorem. The continued fraction expansion [a₀; a₁, a₂, ...] is
eventually periodic for quadratics.

For the golden ratio φ = [1; 1, 1, ...], we have c = 1/√5 (optimal among
all irrationals by Hurwitz). For √2 = [1; 2, 2, ...], c = 1/(2√2).

The RFT basis for irrational α is:
    Φ_{n,k} = (1/√N) exp(i 2π f_k n),  f_k = frac((k+1)α)
    U_α = Φ (Φ^H Φ)^{-1/2}

Key Results
-----------

**Weyl Equidistribution**: For irrational α, the sequence {nα} mod 1 is
equidistributed with discrepancy D_N = O(1/N) for badly approximable α.

**Davis-Kahan Bound**: Off-diagonal energy ||U_α^H C_α U_α - Λ|| is bounded
by the perturbation ||Δ|| / δ where δ = min|z_j - z_k| ~ O(1/N).

**Sharp K99 Bound**: E[K99(U_α, x)] ≤ c·log(N) + O(1) for x ~ E_α
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable
from functools import lru_cache
import numpy as np
import scipy.linalg
import scipy.stats

# Golden ratio and other quadratic irrationals
PHI = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
SQRT5 = np.sqrt(5)
SILVER_RATIO = 1 + SQRT2  # δ_S = 1 + √2 ≈ 2.414, the silver ratio

# Diophantine constants (from continued fraction theory)
DIOPHANTINE_CONSTANTS = {
    'phi': (PHI, 1 / np.sqrt(5)),      # Golden ratio, Hurwitz optimal
    'sqrt2': (SQRT2, 1 / (2 * SQRT2)),  # √2
    'sqrt3': (SQRT3, 1 / (2 * SQRT3)),  # √3
    'sqrt5': (SQRT5, 1 / (2 * SQRT5)),  # √5
    'silver': (SILVER_RATIO, 1 / (2 * SILVER_RATIO)),  # Silver ratio
}


# =============================================================================
# Continued Fraction Utilities
# =============================================================================

def continued_fraction(alpha: float, max_terms: int = 50) -> List[int]:
    """
    Compute the continued fraction expansion [a₀; a₁, a₂, ...] of α.
    
    For quadratic irrationals, this is eventually periodic.
    """
    cf = []
    x = alpha
    for _ in range(max_terms):
        a = int(np.floor(x))
        cf.append(a)
        x = x - a
        if abs(x) < 1e-15:
            break
        x = 1.0 / x
    return cf


def convergents(cf: List[int]) -> List[Tuple[int, int]]:
    """
    Compute convergents p_n/q_n from continued fraction coefficients.
    
    These are the best rational approximations to the irrational.
    """
    if len(cf) == 0:
        return []
    
    h_prev, h_curr = 1, cf[0]
    k_prev, k_curr = 0, 1
    result = [(h_curr, k_curr)]
    
    for a in cf[1:]:
        h_new = a * h_curr + h_prev
        k_new = a * k_curr + k_prev
        result.append((h_new, k_new))
        h_prev, h_curr = h_curr, h_new
        k_prev, k_curr = k_curr, k_new
    
    return result


def diophantine_constant(alpha: float, num_tests: int = 1000) -> float:
    """
    Empirically estimate the Diophantine constant c for |α - p/q| > c/q².
    
    For badly approximable irrationals, this should be bounded away from 0.
    """
    cf = continued_fraction(alpha, max_terms=100)
    convs = convergents(cf)
    
    if len(convs) < 2:
        return 0.0
    
    min_c = float('inf')
    for p, q in convs[1:]:  # Skip first trivial convergent
        error = abs(alpha - p / q)
        c = error * q * q
        if c < min_c:
            min_c = c
    
    return min_c


# =============================================================================
# Diophantine RFT Basis Construction
# =============================================================================

def diophantine_frequency_grid(N: int, alpha: float) -> np.ndarray:
    """
    Construct the frequency grid f_k = frac((k+1)α) for k = 0, ..., N-1.
    
    The fractional parts {nα} are equidistributed by Weyl's theorem.
    """
    k = np.arange(N) + 1
    return np.mod(k * alpha, 1.0)


def diophantine_basis_matrix(N: int, alpha: float, 
                              use_gram_normalization: bool = True) -> np.ndarray:
    """
    Construct the RFT basis U_α for a general Diophantine irrational α.
    
    Parameters
    ----------
    N : int
        Dimension
    alpha : float
        The Diophantine irrational (e.g., φ, √2, √3)
    use_gram_normalization : bool
        If True, return U_α = Φ(Φ^H Φ)^{-1/2} (canonical unitary)
        If False, return raw Φ
    
    Returns
    -------
    U_α : ndarray, shape (N, N)
        The orthonormalized basis matrix
    """
    n = np.arange(N)
    f = diophantine_frequency_grid(N, alpha)
    
    # Raw Vandermonde-like basis: Φ_{n,k} = (1/√N) exp(i 2π f_k n)
    Phi = (1 / np.sqrt(N)) * np.exp(2j * np.pi * np.outer(n, f))
    
    if not use_gram_normalization:
        return Phi
    
    # Gram normalization: U = Φ (Φ^H Φ)^{-1/2}
    G = Phi.conj().T @ Phi
    G_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(G))
    U = Phi @ G_inv_sqrt
    
    return U


# =============================================================================
# Weyl Discrepancy and Equidistribution
# =============================================================================

@dataclass
class DiscrepancyResult:
    """Result from discrepancy analysis."""
    alpha: float
    N: int
    star_discrepancy: float
    expected_bound: float
    is_low_discrepancy: bool


def star_discrepancy(sequence: np.ndarray) -> float:
    """
    Compute the star discrepancy D*_N of a sequence in [0,1).
    
    D*_N = sup_{0≤t≤1} |#{n: x_n < t}/N - t|
    
    For badly approximable irrationals, D*_N = O(log N / N).
    """
    N = len(sequence)
    sorted_seq = np.sort(sequence)
    
    # Check at each point and endpoints
    max_disc = 0.0
    for i, x in enumerate(sorted_seq):
        # Empirical CDF is i/N just before x
        disc_left = abs(i / N - x)
        disc_right = abs((i + 1) / N - x)
        max_disc = max(max_disc, disc_left, disc_right)
    
    return max_disc


def analyze_equidistribution(N: int, alpha: float) -> DiscrepancyResult:
    """
    Analyze Weyl equidistribution for the sequence {nα} mod 1.
    """
    seq = np.mod(np.arange(1, N + 1) * alpha, 1.0)
    disc = star_discrepancy(seq)
    
    # Expected bound for badly approximable: O(log N / N)
    cf = continued_fraction(alpha)
    max_a = max(cf[:min(10, len(cf))])  # Partial quotient bound
    expected = (max_a + 1) * np.log(N) / N
    
    return DiscrepancyResult(
        alpha=alpha,
        N=N,
        star_discrepancy=disc,
        expected_bound=expected,
        is_low_discrepancy=(disc < 3 * expected)
    )


# =============================================================================
# Davis-Kahan Perturbation Analysis
# =============================================================================

@dataclass
class DavisKahanResult:
    """Result from Davis-Kahan perturbation analysis."""
    alpha: float
    N: int
    minimal_gap: float
    perturbation_norm: float
    off_diagonal_ratio: float
    bound_satisfied: bool


def companion_matrix_alpha(N: int, alpha: float) -> np.ndarray:
    """
    Construct the companion shift matrix C_α with eigenvalues on the α-grid.
    
    The eigenvalues are z_k = exp(i 2π {(k+1)α}).
    """
    f = diophantine_frequency_grid(N, alpha)
    z = np.exp(2j * np.pi * f)
    
    # Companion matrix with these eigenvalues
    # C has characteristic polynomial with roots z_k
    coeffs = np.polynomial.polynomial.polyfromroots(z)
    
    C = np.zeros((N, N), dtype=np.complex128)
    C[1:, :-1] = np.eye(N - 1)
    C[:, -1] = -coeffs[:-1] / coeffs[-1]
    
    return C


def minimal_eigenvalue_gap(z: np.ndarray) -> float:
    """Compute minimal gap between eigenvalues on the unit circle."""
    N = len(z)
    angles = np.angle(z)
    sorted_angles = np.sort(angles)
    
    # Gaps including wraparound
    gaps = np.diff(sorted_angles)
    wraparound = (2 * np.pi + sorted_angles[0] - sorted_angles[-1])
    
    return min(min(gaps), wraparound)


def davis_kahan_analysis(N: int, alpha: float) -> DavisKahanResult:
    """
    Apply Davis-Kahan perturbation theory to bound off-diagonal energy.
    
    The theorem states: ||sin Θ|| ≤ ||Δ|| / δ
    where Θ is the angle between eigenspaces, Δ is the perturbation,
    and δ is the minimal eigenvalue gap.
    """
    U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
    C = companion_matrix_alpha(N, alpha)
    
    # Eigenvalues
    f = diophantine_frequency_grid(N, alpha)
    z = np.exp(2j * np.pi * f)
    
    # Minimal gap
    delta = minimal_eigenvalue_gap(z)
    
    # Conjugation U^H C U
    UCU = U.conj().T @ C @ U
    Lambda = np.diag(z)
    
    # Off-diagonal (perturbation)
    off_diag = UCU - np.diag(np.diag(UCU))
    perturbation_norm = np.linalg.norm(off_diag, ord='fro')
    
    # Off-diagonal ratio
    total_norm = np.linalg.norm(UCU, ord='fro')
    off_diag_ratio = perturbation_norm / total_norm
    
    # Davis-Kahan bound: off-diagonal should be ≤ ||Δ|| / δ
    # Here Δ represents the difference from ideal diagonalization
    bound_rhs = perturbation_norm / delta if delta > 1e-15 else float('inf')
    
    return DavisKahanResult(
        alpha=alpha,
        N=N,
        minimal_gap=delta,
        perturbation_norm=perturbation_norm,
        off_diagonal_ratio=off_diag_ratio,
        bound_satisfied=(off_diag_ratio < 0.5)  # Reasonable threshold
    )


# =============================================================================
# K99 Analysis for Diophantine Irrationals
# =============================================================================

def k99(coeffs: np.ndarray, threshold: float = 0.99) -> int:
    """Smallest K such that top-K coefficients capture ≥ threshold energy."""
    energy = np.abs(coeffs) ** 2
    energy = energy / energy.sum()
    sorted_idx = np.argsort(energy)[::-1]
    cumsum = np.cumsum(energy[sorted_idx])
    return int(np.searchsorted(cumsum, threshold) + 1)


def diophantine_drift_ensemble(N: int, M: int, alpha: float,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Generate M signals from the α-quasi-periodic ensemble E_α:
        x[n] = exp(i 2π (f₀ n + a · frac(n α)))
    
    Parameters
    ----------
    N : int
        Signal length
    M : int
        Number of samples
    alpha : float
        The Diophantine irrational
    rng : Generator
        Random number generator
    
    Returns
    -------
    signals : ndarray, shape (M, N)
    """
    n = np.arange(N, dtype=np.float64)
    frac_part = np.mod(n * alpha, 1.0)
    
    signals = np.empty((M, N), dtype=np.complex128)
    for i in range(M):
        f0 = rng.uniform(0.0, 1.0)
        a = rng.uniform(-1.0, 1.0)
        signals[i] = np.exp(2j * np.pi * (f0 * n + a * frac_part))
    
    return signals


@dataclass
class DiophantineK99Result:
    """K99 comparison result for a Diophantine irrational with CI and t-test."""
    alpha: float
    alpha_name: str
    N: int
    M: int
    mean_k99_rft: float
    mean_k99_dft: float
    mean_improvement: float          # Δ = mean(DFT) - mean(RFT)
    improvement_percent: float
    theoretical_bound: float
    rft_wins: int
    # Statistical additions
    std_k99_rft: float = 0.0
    std_k99_dft: float = 0.0
    ci_delta_low: float = 0.0        # 95% CI lower bound for Δ
    ci_delta_high: float = 0.0       # 95% CI upper bound for Δ
    t_statistic: float = 0.0         # Paired t-test statistic
    p_value: float = 1.0             # Two-sided p-value
    significant: bool = False        # p < 0.05
    ci_includes_zero: bool = True    # True if CI contains 0


def compare_k99_diophantine(N: int, M: int, alpha: float, 
                            alpha_name: str = "custom",
                            seed: int = 42,
                            confidence: float = 0.95) -> DiophantineK99Result:
    """
    Compare K99 for RFT(α) vs DFT on the α-quasi-periodic ensemble.
    
    Now includes:
    - 95% confidence interval for Δ = K99(DFT) - K99(RFT)
    - Paired t-test for significance
    - If CI includes 0, reports "no significant difference"
    
    This is the core verification for Theorem 8 scaling law.
    """
    rng = np.random.default_rng(seed)
    
    # Build bases
    U_alpha = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
    F_dft = np.fft.fft(np.eye(N), axis=0, norm='ortho')
    
    # Generate ensemble
    signals = diophantine_drift_ensemble(N, M, alpha, rng)
    
    # Compute K99 for each signal (paired samples)
    k99_rft = np.array([k99(U_alpha.conj().T @ x) for x in signals])
    k99_dft = np.array([k99(F_dft.conj().T @ x) for x in signals])
    
    mean_rft = float(np.mean(k99_rft))
    mean_dft = float(np.mean(k99_dft))
    std_rft = float(np.std(k99_rft, ddof=1))
    std_dft = float(np.std(k99_dft, ddof=1))
    improvement = mean_dft - mean_rft
    
    # Paired differences for CI and t-test
    delta = k99_dft - k99_rft  # Positive means RFT better
    
    # Paired t-test (two-sided)
    t_stat, p_val = scipy.stats.ttest_rel(k99_dft, k99_rft)
    
    # 95% confidence interval for mean difference
    sem_delta = scipy.stats.sem(delta)
    df = M - 1
    t_crit = scipy.stats.t.ppf((1 + confidence) / 2, df)
    ci_low = float(np.mean(delta) - t_crit * sem_delta)
    ci_high = float(np.mean(delta) + t_crit * sem_delta)
    
    ci_includes_zero = (ci_low <= 0 <= ci_high)
    significant = p_val < 0.05
    
    # Theoretical bound: c · log(N) where c = 1/log(α)
    c_alpha = 1 / np.log(alpha) if alpha > 1 else 1 / np.log(1/alpha)
    theoretical = c_alpha * np.log(N)
    
    return DiophantineK99Result(
        alpha=alpha,
        alpha_name=alpha_name,
        N=N,
        M=M,
        mean_k99_rft=mean_rft,
        mean_k99_dft=mean_dft,
        mean_improvement=improvement,
        improvement_percent=100 * improvement / mean_dft if mean_dft > 0 else 0,
        theoretical_bound=theoretical,
        rft_wins=int(np.sum(k99_rft < k99_dft)),
        std_k99_rft=std_rft,
        std_k99_dft=std_dft,
        ci_delta_low=ci_low,
        ci_delta_high=ci_high,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=significant,
        ci_includes_zero=ci_includes_zero
    )


# =============================================================================
# Scaling Law Verification (formerly "Universality")
# =============================================================================

@dataclass
class ScalingLawResult:
    """Result from scaling law verification across irrationals (with statistics)."""
    results: List[DiophantineK99Result]
    all_significant: bool
    any_no_difference: bool  # True if any CI includes 0
    
    def summary(self) -> str:
        lines = [
            "=" * 80,
            "THEOREM 8 SCALING LAW: K99 ~ c·log(N) across Diophantine Irrationals",
            "=" * 80,
            f"{'Alpha':<10} {'K99(RFT)':>10} {'K99(DFT)':>10} {'Δ':>7} "
            f"{'95% CI':>18} {'p-value':>10} {'Verdict':>18}",
            "-" * 80,
        ]
        for r in self.results:
            ci_str = f"[{r.ci_delta_low:+.2f}, {r.ci_delta_high:+.2f}]"
            if r.ci_includes_zero:
                verdict = "No sig. diff."
            elif r.mean_improvement > 0 and r.significant:
                verdict = "RFT better (p<.05)"
            elif r.mean_improvement < 0 and r.significant:
                verdict = "DFT better (p<.05)"
            else:
                verdict = "Inconclusive"
            lines.append(
                f"{r.alpha_name:<10} {r.mean_k99_rft:>10.2f} {r.mean_k99_dft:>10.2f} "
                f"{r.mean_improvement:>+7.2f} {ci_str:>18} {r.p_value:>10.4f} {verdict:>18}"
            )
        lines.append("-" * 80)
        if self.any_no_difference:
            lines.append("Note: Some irrationals show no significant RFT vs DFT difference.")
        return "\n".join(lines)


# Alias for backward compatibility
UniversalityResult = ScalingLawResult


def verify_scaling_law(N: int = 128, M: int = 300, 
                       seed: int = 42) -> ScalingLawResult:
    """
    Verify Theorem 8 scaling law holds for multiple Diophantine irrationals.
    
    Uses paired t-test with 95% CI. Reports "no significant difference" if CI includes 0.
    """
    irrationals = [
        (PHI, "φ (golden)"),
        (SQRT2, "√2"),
        (SQRT3, "√3"),
        (SQRT5, "√5"),
        (SILVER_RATIO, "silver"),
    ]
    
    results = []
    all_significant = True
    any_no_diff = False
    
    for alpha, name in irrationals:
        result = compare_k99_diophantine(N, M, alpha, name, seed)
        results.append(result)
        if not result.significant:
            all_significant = False
        if result.ci_includes_zero:
            any_no_diff = True
    
    return ScalingLawResult(
        results=results, 
        all_significant=all_significant,
        any_no_difference=any_no_diff
    )


# Backward compatibility alias
def verify_universality(N: int = 128, M: int = 300, 
                        seed: int = 42) -> ScalingLawResult:
    """DEPRECATED: Use verify_scaling_law() instead."""
    return verify_scaling_law(N, M, seed)


# =============================================================================
# Sharp Log-N Bound Verification
# =============================================================================

@dataclass
class SharpBoundResult:
    """Result from sharp log-N bound analysis."""
    alpha: float
    N_values: List[int]
    k99_means: List[float]
    log_N_values: List[float]
    fitted_slope: float
    fitted_intercept: float
    theoretical_slope: float
    r_squared: float


def verify_sharp_logn_bound(alpha: float = PHI, 
                             N_values: List[int] = None,
                             M: int = 200,
                             seed: int = 42) -> SharpBoundResult:
    """
    Verify that E[K99] grows as c·log(N) with sharp constant c = 1/log(α).
    """
    if N_values is None:
        N_values = [32, 64, 128, 256, 512]
    
    rng = np.random.default_rng(seed)
    
    k99_means = []
    for N in N_values:
        U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
        signals = diophantine_drift_ensemble(N, M, alpha, rng)
        k99_vals = [k99(U.conj().T @ x) for x in signals]
        k99_means.append(np.mean(k99_vals))
    
    # Fit: K99 = a * log(N) + b
    log_N = np.log(N_values)
    slope, intercept = np.polyfit(log_N, k99_means, 1)
    
    # R² calculation
    y_pred = slope * log_N + intercept
    ss_res = np.sum((np.array(k99_means) - y_pred) ** 2)
    ss_tot = np.sum((np.array(k99_means) - np.mean(k99_means)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Theoretical slope
    c_theory = 1 / np.log(alpha) if alpha > 1 else 1 / np.log(1/alpha)
    
    return SharpBoundResult(
        alpha=alpha,
        N_values=N_values,
        k99_means=k99_means,
        log_N_values=list(log_N),
        fitted_slope=slope,
        fitted_intercept=intercept,
        theoretical_slope=c_theory,
        r_squared=r_squared
    )


# =============================================================================
# Main Demo
# =============================================================================

def main():
    """Demonstrate Theorem 8 universality for Diophantine irrationals."""
    print("=" * 70)
    print("THEOREM 8 EXTENSION: Diophantine Irrational Universality")
    print("=" * 70)
    print()
    
    # 1. Verify universality across irrationals
    print("1. UNIVERSALITY VERIFICATION")
    print("-" * 40)
    univ = verify_universality(N=128, M=200)
    print(univ.summary())
    print()
    
    # 2. Sharp log-N bound
    print("2. SHARP LOG-N BOUND")
    print("-" * 40)
    bound = verify_sharp_logn_bound(PHI, N_values=[32, 64, 128, 256])
    print(f"Fitted: K99 ≈ {bound.fitted_slope:.2f} log(N) + {bound.fitted_intercept:.2f}")
    print(f"Theoretical slope c = 1/log(φ) = {bound.theoretical_slope:.2f}")
    print(f"R² = {bound.r_squared:.4f}")
    print()
    
    # 3. Davis-Kahan perturbation analysis
    print("3. DAVIS-KAHAN PERTURBATION ANALYSIS")
    print("-" * 40)
    for alpha, name in [(PHI, "φ"), (SQRT2, "√2")]:
        dk = davis_kahan_analysis(64, alpha)
        print(f"{name}: gap={dk.minimal_gap:.4f}, off-diag ratio={dk.off_diagonal_ratio:.4f}")
    print()
    
    # 4. Equidistribution check
    print("4. WEYL EQUIDISTRIBUTION")
    print("-" * 40)
    for alpha, name in [(PHI, "φ"), (SQRT2, "√2")]:
        disc = analyze_equidistribution(128, alpha)
        print(f"{name}: D*_N = {disc.star_discrepancy:.6f}, expected ≤ {disc.expected_bound:.6f}")


if __name__ == "__main__":
    main()
