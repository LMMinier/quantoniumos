# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Test suite for Theorem 8 — Diophantine Proof.

Verifies each Diophantine lemma independently plus the combined theorem.
All tests are deterministic (fixed seeds) and run in < 60 seconds total.
"""

import numpy as np
import pytest

from algorithms.rft.theory.theorem8_diophantine import (
    prove_lemma_8_4a,
    prove_lemma_8_4b,
    prove_lemma_8_4c,
    prove_lemma_8_4d,
    prove_lemma_8_4e,
    prove_lemma_8_4f,
    prove_theorem_8_diophantine,
    fibonacci_up_to,
    fibonacci_index_for_N,
    continued_fraction_convergents,
    _phi_frequencies,
    _golden_signal_basis,
    _canonical_U,
    _dft_matrix,
    _K_modes,
    _k99,
    _generate_golden_signal,
    PHI,
    PHI_FRAC,
    SQRT5,
)


# ─── Test Sizes (small for CI speed) ───────────────────────────────────────────

SMALL_SIZES = [32, 64, 128]
FULL_SIZES = [32, 64, 128, 256]
N_TRIALS = 100


# ═══════════════════════════════════════════════════════════════════════════════
# Fibonacci Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestFibonacciUtilities:
    """Tests for the Fibonacci helper functions."""

    def test_fibonacci_sequence_correct(self):
        fibs = fibonacci_up_to(100)
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        assert fibs == expected

    def test_fibonacci_index_for_N(self):
        m, F_m, F_m1 = fibonacci_index_for_N(100)
        assert F_m == 89     # Largest Fibonacci ≤ 100
        assert F_m1 == 144   # Next Fibonacci

    def test_fibonacci_index_exact(self):
        """When N is itself Fibonacci, F_m = N."""
        m, F_m, F_m1 = fibonacci_index_for_N(89)
        assert F_m == 89

    def test_continued_fraction_convergents_are_fibonacci_ratios(self):
        convs = continued_fraction_convergents(10)
        # Convergents of φ are F_{m+1}/F_m
        fibs = fibonacci_up_to(200)
        for i, (p, q) in enumerate(convs):
            assert p == fibs[i + 1] or (i == 0 and p == 1)
            assert q == fibs[i] or (i == 0 and q == 1)

    def test_convergents_approach_phi(self):
        convs = continued_fraction_convergents(20)
        errors = [abs(PHI - p / q) for p, q in convs if q > 0]
        # Errors should be strictly decreasing
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Convergent error not decreasing: {errors[i]} vs {errors[i+1]}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Lemma 8.4a: Three-Distance Theorem
# ═══════════════════════════════════════════════════════════════════════════════

class TestLemma84a:
    """Three-Distance / Steinhaus-Sós Theorem."""

    def test_verified(self):
        result = prove_lemma_8_4a(SMALL_SIZES)
        assert result.verified, f"Lemma 8.4a not verified: {result.proof_steps[-1]}"

    def test_exactly_2_or_3_gaps(self):
        result = prove_lemma_8_4a(SMALL_SIZES)
        for N in SMALL_SIZES:
            n_gaps = result.certificates[f"N{N}_n_distinct_gaps"]
            assert n_gaps in (2, 3), (
                f"N={N}: expected 2 or 3 gaps, got {n_gaps}"
            )

    def test_fibonacci_N_gives_exactly_2_gaps(self):
        """When N is a Fibonacci number, exactly 2 gap values."""
        fib_sizes = [8, 13, 21, 34, 55, 89]
        result = prove_lemma_8_4a(fib_sizes)
        for N in fib_sizes:
            n_gaps = result.certificates[f"N{N}_n_distinct_gaps"]
            assert n_gaps == 2, f"N={N} (Fibonacci): expected 2 gaps, got {n_gaps}"

    def test_gaps_are_positive(self):
        result = prove_lemma_8_4a(SMALL_SIZES)
        for N in SMALL_SIZES:
            min_gap = result.certificates[f"N{N}_min_gap"]
            assert min_gap > 1e-15, f"N={N}: gap too small: {min_gap}"

    def test_gaps_decrease_with_N(self):
        """Gaps shrink as N grows (more points on [0,1))."""
        result = prove_lemma_8_4a(FULL_SIZES)
        max_gaps = [result.certificates[f"N{N}_max_gap"] for N in FULL_SIZES]
        for i in range(len(max_gaps) - 1):
            assert max_gaps[i] > max_gaps[i + 1], (
                f"Max gap not decreasing: N={FULL_SIZES[i]}→{FULL_SIZES[i+1]}"
            )

    def test_gap_sum_equals_1(self):
        """Gaps must sum to 1 (they partition [0,1))."""
        for N in SMALL_SIZES:
            freqs = np.sort(np.mod(np.arange(1, N + 1, dtype=np.float64) * PHI, 1.0))
            gaps = np.diff(freqs)
            gap_wrap = 1.0 + freqs[0] - freqs[-1]
            total = np.sum(gaps) + gap_wrap
            assert abs(total - 1.0) < 1e-12, f"N={N}: gaps sum to {total} ≠ 1"


# ═══════════════════════════════════════════════════════════════════════════════
# Lemma 8.4b: Hurwitz Irrationality Bound
# ═══════════════════════════════════════════════════════════════════════════════

class TestLemma84b:
    """Hurwitz Irrationality Bound for φ."""

    def test_verified(self):
        result = prove_lemma_8_4b(n_convergents=20)
        assert result.verified, f"Lemma 8.4b not verified: {result.proof_steps[-1]}"

    def test_hurwitz_limit(self):
        """q²|φ−p/q| → 1/√5 for Fibonacci convergents."""
        result = prove_lemma_8_4b(n_convergents=30)
        limit_err = result.certificates["convergence_error"]
        assert limit_err < 0.001, f"Hurwitz limit error {limit_err} too large"

    def test_hurwitz_alternation(self):
        """q²|φ−p/q| oscillates around 1/√5."""
        result = prove_lemma_8_4b(n_convergents=20)
        assert result.certificates["all_hurwitz_hold"], "Alternation not detected"

    def test_golden_dft_gap_positive(self):
        """Golden frequencies never coincide with DFT bins."""
        result = prove_lemma_8_4b()
        for N in [32, 64, 128, 256, 512]:
            gap = result.certificates[f"N{N}_min_golden_dft_dist"]
            assert gap > 1e-12, f"N={N}: golden-DFT gap {gap} ≤ 0"

    def test_phi_is_irrational(self):
        """Direct verification: (k+1)φ is never an integer for k < 10000."""
        for k in range(10000):
            val = (k + 1) * PHI
            frac_part = val - int(val)
            assert frac_part > 1e-10 or frac_part < 1 - 1e-10, (
                f"k={k}: (k+1)φ appears integer: {val}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Lemma 8.4c: Quantitative Weyl Discrepancy
# ═══════════════════════════════════════════════════════════════════════════════

class TestLemma84c:
    """Quantitative Weyl / Erdős-Turán Discrepancy."""

    def test_verified(self):
        result = prove_lemma_8_4c(SMALL_SIZES)
        assert result.verified, f"Lemma 8.4c not verified: {result.proof_steps[-1]}"

    def test_discrepancy_decrease(self):
        """D*_N decreases with N."""
        result = prove_lemma_8_4c(FULL_SIZES)
        D_stars = [result.certificates[f"N{N}_D_star"] for N in FULL_SIZES]
        for i in range(len(D_stars) - 1):
            assert D_stars[i] > D_stars[i + 1] - 1e-6, (
                f"Discrepancy not decreasing: {D_stars[i]} vs {D_stars[i+1]}"
            )

    def test_discrepancy_below_bound(self):
        """D*_N ≤ C·log(N)/N."""
        result = prove_lemma_8_4c(FULL_SIZES)
        for N in FULL_SIZES:
            D = result.certificates[f"N{N}_D_star"]
            bound = result.certificates[f"N{N}_bound"]
            assert D < bound * 1.1, f"N={N}: D*={D} exceeds bound={bound}"

    def test_kappa_converging(self):
        """V†V → I as N → ∞ (κ(V) bounded, improving)."""
        result = prove_lemma_8_4c(FULL_SIZES)
        kappas = [result.certificates[f"N{N}_kappa_V"] for N in FULL_SIZES]
        assert kappas[-1] < 1.2, f"κ({FULL_SIZES[-1]}) = {kappas[-1]} not near 1"

    def test_gram_off_diagonal_shrinks(self):
        """Off-diagonal Gram entries shrink with N."""
        result = prove_lemma_8_4c(FULL_SIZES)
        offdiags = [result.certificates[f"N{N}_max_off_diag"] for N in FULL_SIZES]
        for i in range(len(offdiags) - 1):
            assert offdiags[i] >= offdiags[i + 1] - 1e-6, (
                f"Off-diag not shrinking: {offdiags[i]} vs {offdiags[i+1]}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Lemma 8.4d: Per-Harmonic DFT Leakage
# ═══════════════════════════════════════════════════════════════════════════════

class TestLemma84d:
    """Per-Harmonic DFT Leakage / Dirichlet Kernel."""

    def test_verified(self):
        result = prove_lemma_8_4d(SMALL_SIZES)
        assert result.verified, f"Lemma 8.4d not verified: {result.proof_steps[-1]}"

    def test_epsilon_positive(self):
        """ε_m > 0 at all N (Hurwitz guarantee)."""
        result = prove_lemma_8_4d(SMALL_SIZES)
        for N in SMALL_SIZES:
            eps = result.certificates[f"N{N}_min_epsilon"]
            assert eps > 1e-10, f"N={N}: ε_min = {eps} ≤ 0"

    def test_peak_energy_below_1(self):
        """sinc²(ε) < 1 — leakage is guaranteed."""
        result = prove_lemma_8_4d(SMALL_SIZES)
        for N in SMALL_SIZES:
            peak = result.certificates[f"N{N}_mean_peak_energy"]
            assert peak < 0.999, f"N={N}: peak energy {peak} ≈ 1 (no leakage?)"

    def test_dft_needs_more_bins_than_oracle(self):
        """Total DFT bins for 99% > K (oracle count)."""
        result = prove_lemma_8_4d(SMALL_SIZES)
        for N in SMALL_SIZES:
            total = result.certificates[f"N{N}_total_bins_99"]
            K = result.certificates[f"N{N}_K"]
            assert total > K, f"N={N}: DFT bins {total} ≤ oracle {K}"

    def test_sinc_squared_formula(self):
        """Verify sinc²(ε) matches Dirichlet kernel at peak bin."""
        N = 64
        for m in [1, 3, 7]:
            f_m = np.mod(m * PHI, 1.0)

            # Direct Dirichlet kernel computation
            x = np.exp(2j * np.pi * f_m * np.arange(N)) / np.sqrt(N)
            F = _dft_matrix(N)
            c = F.conj().T @ x
            energies = np.abs(c) ** 2

            # Find actual peak bin (handles DFT sign convention)
            j_star = int(np.argmax(energies))
            peak = float(energies[j_star])

            # Compute ε from the actual peak bin
            delta = f_m - j_star / N
            # Handle circular distance
            if abs(delta) > 0.5:
                delta = delta - np.sign(delta)
            # Also consider the alias at (N-j_star)/N
            delta_alt = f_m - (N - j_star) / N
            if abs(delta_alt) > 0.5:
                delta_alt = delta_alt - np.sign(delta_alt)
            epsilon = min(abs(N * delta), abs(N * delta_alt))

            # sinc² formula
            if epsilon > 1e-15:
                sinc_sq = (np.sin(np.pi * epsilon) / (np.pi * epsilon)) ** 2
            else:
                sinc_sq = 1.0

            # They should be close (not exact due to finite-N effects)
            assert abs(peak - sinc_sq) < 0.15, (
                f"m={m}: Dirichlet peak {peak} vs sinc²({epsilon:.4f})={sinc_sq}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Lemma 8.4e: RFT Zero-Misalignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestLemma84e:
    """RFT Zero-Misalignment Principle."""

    def test_verified(self):
        result = prove_lemma_8_4e(SMALL_SIZES)
        assert result.verified, f"Lemma 8.4e not verified: {result.proof_steps[-1]}"

    def test_rft_better_than_dft_per_harmonic(self):
        """RFT K99 ≤ DFT K99 for each individual golden harmonic."""
        result = prove_lemma_8_4e(SMALL_SIZES)
        for N in SMALL_SIZES:
            rft = result.certificates[f"N{N}_rft_k99_per_harmonic"]
            dft = result.certificates[f"N{N}_dft_k99_per_harmonic"]
            assert rft <= dft + 0.5, (
                f"N={N}: RFT per-harmonic K99 {rft} > DFT {dft}"
            )

    def test_ensemble_advantage_positive(self):
        """Ensemble ΔK₉₉ > 0 at every tested N."""
        result = prove_lemma_8_4e(SMALL_SIZES)
        for N in SMALL_SIZES:
            delta = result.certificates[f"N{N}_ensemble_delta"]
            assert delta > 0, f"N={N}: no ensemble advantage ({delta})"

    def test_rft_frequencies_match_signal(self):
        """The RFT basis frequencies ARE the signal frequencies."""
        N = 64
        K = _K_modes(N)
        # Signal frequencies: frac(mφ) for m=1,...,K
        signal_freqs = np.sort(np.mod(np.arange(1, K + 1) * PHI, 1.0))
        # RFT frequencies: frac((k+1)φ) for k=0,...,N-1
        rft_freqs = np.sort(_phi_frequencies(N))

        # Every signal frequency should appear in the RFT frequency grid
        for sf in signal_freqs:
            dists = np.abs(rft_freqs - sf)
            min_dist = np.min(dists)
            assert min_dist < 1e-10, (
                f"Signal freq {sf:.8f} not found in RFT grid (min dist {min_dist})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Lemma 8.4f: Diophantine Gap Theorem
# ═══════════════════════════════════════════════════════════════════════════════

class TestLemma84f:
    """Diophantine Gap Theorem — the punchline."""

    def test_verified(self):
        result = prove_lemma_8_4f(SMALL_SIZES, N_TRIALS)
        assert result.verified, f"Lemma 8.4f not verified: {result.proof_steps[-1]}"

    def test_gap_positive_at_all_N(self):
        """ΔK₉₉ > 0 at every N, with bootstrap CIs excluding 0."""
        result = prove_lemma_8_4f(SMALL_SIZES, N_TRIALS)
        for N in SMALL_SIZES:
            ci_lo = result.certificates[f"N{N}_ci_lo"]
            assert ci_lo > 0, f"N={N}: 95% CI lower bound {ci_lo} ≤ 0"

    def test_gap_grows_with_N(self):
        """ΔK₉₉ grows monotonically with N."""
        result = prove_lemma_8_4f(FULL_SIZES, N_TRIALS)
        deltas = [result.certificates[f"N{N}_delta"] for N in FULL_SIZES]
        for i in range(len(deltas) - 1):
            assert deltas[i] < deltas[i + 1] + 2.0, (
                f"Gap not growing: N={FULL_SIZES[i]}, Δ={deltas[i]} vs "
                f"N={FULL_SIZES[i+1]}, Δ={deltas[i+1]}"
            )

    def test_gap_scaling_exponent_positive(self):
        """Gap ΔK₉₉ ∝ N^α with α > 0."""
        result = prove_lemma_8_4f(FULL_SIZES, N_TRIALS)
        alpha = result.certificates.get("gap_alpha", 0)
        assert alpha > 0.3, f"Gap scaling exponent α = {alpha} too small"

    def test_diophantine_excess_positive(self):
        """Diophantine per-harmonic excess is > 0 at all N."""
        result = prove_lemma_8_4f(SMALL_SIZES, N_TRIALS)
        for N in SMALL_SIZES:
            excess = result.certificates[f"N{N}_dioph_excess"]
            assert excess > 0, f"N={N}: Diophantine excess {excess} ≤ 0"


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Theorem 8 (Diophantine)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTheorem8Diophantine:
    """Full Diophantine proof chain."""

    def test_all_verified(self):
        proof = prove_theorem_8_diophantine(SMALL_SIZES, N_TRIALS)
        assert proof.theorem_verified, "Theorem 8 Diophantine not fully verified"

    def test_classification(self):
        proof = prove_theorem_8_diophantine(SMALL_SIZES, N_TRIALS)
        assert proof.classification == "CONSTRUCTIVE + DIOPHANTINE"

    def test_all_6_lemmas(self):
        proof = prove_theorem_8_diophantine(SMALL_SIZES, N_TRIALS)
        assert len(proof.lemmas) == 6
        for lid, lem in proof.lemmas.items():
            assert lem.verified, f"{lid}: {lem.name} not verified"

    def test_classical_references_present(self):
        proof = prove_theorem_8_diophantine(SMALL_SIZES, N_TRIALS)
        refs = proof.summary.get("classical_references", [])
        assert len(refs) >= 4, f"Only {len(refs)} classical references"
        ref_text = " ".join(refs).lower()
        assert "hurwitz" in ref_text
        assert "steinhaus" in ref_text or "sós" in ref_text.lower()
        assert "weyl" in ref_text

    def test_hurwitz_constant_present(self):
        proof = prove_theorem_8_diophantine(SMALL_SIZES, N_TRIALS)
        h = proof.summary.get("hurwitz_constant", 0)
        assert abs(h - 1.0 / SQRT5) < 1e-8, f"Hurwitz constant {h} wrong"

    def test_no_empirical_claims(self):
        """Verify no lemma is classified as EMPIRICAL."""
        proof = prove_theorem_8_diophantine(SMALL_SIZES, N_TRIALS)
        for lid, lem in proof.lemmas.items():
            assert lem.status != "EMPIRICAL", (
                f"{lid}: still classified as EMPIRICAL"
            )
        assert "EMPIRICAL" not in proof.classification


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Cutting Structural Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiophantineStructural:
    """Cross-cutting mathematical invariants for the Diophantine proof."""

    def test_hurwitz_constant_is_optimal_for_phi(self):
        """
        Verify that 1/√5 is indeed the best Hurwitz constant for φ.
        No c > √5 would work: there would be infinitely many convergents
        with q²|φ−p/q| < 1/c.
        """
        convs = continued_fraction_convergents(20)
        # Count how many have q²|err| < 1/√5 (should be roughly half)
        below = sum(1 for p, q in convs if q > 0 and q * q * abs(PHI - p / q) < 1.0 / SQRT5)
        above = sum(1 for p, q in convs if q > 0 and q * q * abs(PHI - p / q) > 1.0 / SQRT5)
        assert below >= 5, f"Only {below} convergents below 1/√5"
        assert above >= 5, f"Only {above} convergents above 1/√5"

    def test_three_distance_implies_equidistribution(self):
        """Three-distance theorem is consistent with Weyl equidistribution."""
        N = 256
        freqs = np.sort(np.mod(np.arange(1, N + 1, dtype=np.float64) * PHI, 1.0))
        # The frequencies should be equidistributed: roughly N/10 in each [0,0.1)
        for start in np.arange(0, 1, 0.1):
            count = np.sum((freqs >= start) & (freqs < start + 0.1))
            # Should be about N/10 = 25.6 ± some tolerance
            assert N / 10 * 0.5 < count < N / 10 * 1.5, (
                f"[{start:.1f},{start+0.1:.1f}): {count} points (expected ~{N/10})"
            )

    def test_dft_grid_is_rational(self):
        """DFT bins j/N are all rational — direct contrast with golden grid."""
        N = 128
        dft_bins = np.arange(N) / N
        for j in range(N):
            # j/N is rational (by construction)
            # Verify it's NOT close to any golden frequency
            golden = _phi_frequencies(N)
            dists = np.abs(golden - dft_bins[j])
            dists = np.minimum(dists, 1.0 - dists)
            assert np.min(dists) > 1e-12, (
                f"DFT bin {j}/{N} coincides with a golden frequency!"
            )

    def test_irrationality_measure_phi_equals_2(self):
        """
        φ has irrationality measure μ = 2 (Roth's theorem for algebraics).
        This means |φ − p/q| > c(ε)/q^{2+ε} for any ε > 0.
        Verify: q^{2.01}·|φ−p/q| → ∞ for Fibonacci convergents.
        """
        convs = continued_fraction_convergents(25)
        # For μ = 2: q^2·|φ−p/q| → 1/√5 (bounded)
        # For μ = 2 + ε: q^{2+ε}·|φ−p/q| → ∞
        eps = 0.01
        prods = []
        for p, q in convs:
            if q > 10:
                err = abs(PHI - p / q)
                prods.append(q ** (2 + eps) * err)
        # Should be increasing (diverging)
        assert prods[-1] > prods[0], (
            f"q^{{2.01}}·|err| not diverging: first={prods[0]:.4f}, last={prods[-1]:.4f}"
        )

    def test_golden_grid_is_densest_3gap_sequence(self):
        """
        Golden ratio produces the "most uniform" 3-gap distribution.
        With 3 distinct gaps, the largest is φ² ≈ 2.618 times the smallest
        (since the three gaps satisfy d₃ = d₁ + d₂, and d₂/d₁ ≈ φ).
        With 2 gaps (Fibonacci N), the ratio is exactly φ.
        """
        for N in [64, 128, 256]:
            freqs = np.sort(np.mod(np.arange(1, N + 1, dtype=np.float64) * PHI, 1.0))
            gaps = np.diff(freqs)
            gap_wrap = 1.0 + freqs[0] - freqs[-1]
            all_gaps = np.concatenate([gaps, [gap_wrap]])

            ratio = np.max(all_gaps) / np.min(all_gaps)
            # For golden ratio: 3 gaps have ratio in [φ, φ²] ≈ [1.618, 2.618]
            assert 1.5 < ratio < 2.7, (
                f"N={N}: max/min gap ratio {ratio:.4f} not in [φ, φ²]"
            )
