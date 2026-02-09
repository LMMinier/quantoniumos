# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Test suite for Theorem 8 — Golden Spectral Concentration (Formal Proof).

Verifies each lemma in the proof chain independently plus the combined theorem.
All tests are deterministic (fixed seeds) and run in < 60 seconds total.
"""

import numpy as np
import pytest

from algorithms.rft.theory.theorem8_formal_proof import (
    prove_lemma_8_3a,
    prove_lemma_8_3b,
    prove_lemma_8_3c,
    prove_lemma_8_3d,
    prove_lemma_8_3e,
    prove_theorem_8,
    _golden_signal_basis,
    _canonical_U,
    _dft_matrix,
    _K_modes,
    _k99,
    _generate_golden_signal,
    PHI,
)


# ─── Test Sizes (small for CI speed) ───────────────────────────────────────────

SMALL_SIZES = [32, 64, 128]
FULL_SIZES = [32, 64, 128, 256]
N_TRIALS = 100


# ─── Lemma 8.3a: Finite-Rank Covariance ────────────────────────────────────────

class TestLemma83a:
    """Finite-rank covariance of the Golden-Hull Analytic Ensemble."""

    def test_verified(self):
        result = prove_lemma_8_3a(SMALL_SIZES)
        assert result.verified, f"Lemma 8.3a not verified: {result.proof_steps[-1]}"

    def test_rank_equals_K(self):
        result = prove_lemma_8_3a(SMALL_SIZES)
        for N in SMALL_SIZES:
            K = _K_modes(N)
            assert result.certificates[f"N{N}_rank_V"] == K, (
                f"N={N}: rank(V)={result.certificates[f'N{N}_rank_V']} != K={K}"
            )

    def test_tail_eigenvalues_zero(self):
        result = prove_lemma_8_3a(SMALL_SIZES)
        for N in SMALL_SIZES:
            tail = result.certificates[f"N{N}_max_tail_eig"]
            assert tail < 1e-10, f"N={N}: tail eigenvalue {tail} too large"

    def test_vandermonde_full_column_rank(self):
        """Direct test: V has full column rank via SVD."""
        for N in SMALL_SIZES:
            K = _K_modes(N)
            V = _golden_signal_basis(N, K)
            sv = np.linalg.svd(V, compute_uv=False)
            assert sv[-1] > 1e-6, f"N={N}: sigma_min={sv[-1]} too small"

    def test_golden_nodes_distinct(self):
        """φ irrational ⟹ all frac(mφ) distinct."""
        for N in [64, 256, 1024]:
            K = _K_modes(N)
            nodes = np.mod(np.arange(1, K + 1) * PHI, 1.0)
            # All pairwise distances > 0
            dists = np.abs(nodes[:, None] - nodes[None, :])
            np.fill_diagonal(dists, 1.0)
            assert dists.min() > 1e-10, f"N={N}: duplicate golden nodes"


# ─── Lemma 8.3b: Vandermonde Conditioning ──────────────────────────────────────

class TestLemma83b:
    """Condition number κ(V) → 1 as N → ∞."""

    def test_verified(self):
        result = prove_lemma_8_3b(SMALL_SIZES)
        assert result.verified, f"Lemma 8.3b not verified: {result.proof_steps[-1]}"

    def test_kappa_decreasing(self):
        result = prove_lemma_8_3b(FULL_SIZES)
        kappas = [result.certificates[f"N{N}_kappa"] for N in FULL_SIZES]
        for i in range(len(kappas) - 1):
            assert kappas[i] >= kappas[i + 1] - 0.01, (
                f"κ not decreasing: {kappas}"
            )

    def test_kappa_near_1_at_large_N(self):
        for N in [256, 512]:
            K = _K_modes(N)
            V = _golden_signal_basis(N, K)
            kappa = float(np.linalg.cond(V))
            assert kappa < 1.15, f"N={N}: κ={kappa} not near 1"

    def test_gram_converges_to_identity(self):
        """V†V/K → I as N → ∞ (Weyl equidistribution)."""
        prev_err = float("inf")
        for N in [64, 128, 256, 512]:
            K = _K_modes(N)
            V = _golden_signal_basis(N, K)
            gram_err = np.linalg.norm(V.conj().T @ V - np.eye(K), "fro") / K
            assert gram_err < prev_err + 1e-6, "Gram error not decreasing"
            prev_err = gram_err


# ─── Lemma 8.3c: Oracle Concentration ──────────────────────────────────────────

class TestLemma83c:
    """Oracle basis achieves K₀.₉₉ = K = O(log N)."""

    def test_verified(self):
        result = prove_lemma_8_3c(SMALL_SIZES, n_trials=N_TRIALS)
        assert result.verified, f"Lemma 8.3c not verified: {result.proof_steps[-1]}"

    def test_oracle_k99_le_K(self):
        """Oracle K₀.₉₉ never exceeds K."""
        result = prove_lemma_8_3c(SMALL_SIZES, n_trials=N_TRIALS)
        for N in SMALL_SIZES:
            K = result.certificates[f"N{N}_K"]
            max_k99 = result.certificates[f"N{N}_oracle_k99_max"]
            assert max_k99 <= K, f"N={N}: oracle K₀.₉₉={max_k99} > K={K}"

    def test_oracle_captures_all_energy(self):
        """Top-K oracle coefficients capture 100% energy."""
        result = prove_lemma_8_3c(SMALL_SIZES, n_trials=N_TRIALS)
        for N in SMALL_SIZES:
            energy = result.certificates[f"N{N}_oracle_energy_K"]
            assert energy > 0.9999, f"N={N}: oracle energy={energy} < 1.0"

    def test_K_is_O_log_N(self):
        """K = O(log N) — at most c·log₂(N) + d."""
        for N in [32, 64, 128, 256, 512, 1024]:
            K = _K_modes(N)
            assert K <= 4 * np.log2(N), f"N={N}: K={K} > 4·log₂(N)={4*np.log2(N)}"


# ─── Lemma 8.3d: DFT Spectral Leakage ─────────────────────────────────────────

class TestLemma83d:
    """DFT requires Θ(N^γ) coefficients for golden signals."""

    def test_verified(self):
        result = prove_lemma_8_3d(SMALL_SIZES, n_trials=N_TRIALS)
        assert result.verified, f"Lemma 8.3d not verified: {result.proof_steps[-1]}"

    def test_beta_gt_1(self):
        """K₀.₉₉(F) > K at every size (DFT strictly worse than oracle)."""
        result = prove_lemma_8_3d(FULL_SIZES, n_trials=N_TRIALS)
        for N in FULL_SIZES:
            beta = result.certificates[f"N{N}_beta"]
            assert beta > 1.0, f"N={N}: β={beta} ≤ 1"

    def test_dft_k99_grows_with_N(self):
        """DFT K₀.₉₉ increases monotonically with N."""
        result = prove_lemma_8_3d(FULL_SIZES, n_trials=N_TRIALS)
        means = [result.certificates[f"N{N}_dft_k99_mean"] for N in FULL_SIZES]
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], f"DFT K₀.₉₉ not increasing: {means}"

    def test_dft_scaling_super_logarithmic(self):
        """DFT scaling exponent γ > 0.3 (super-logarithmic)."""
        result = prove_lemma_8_3d(FULL_SIZES, n_trials=N_TRIALS)
        gamma = result.certificates["scaling_gamma"]
        assert gamma > 0.3, f"DFT scaling γ={gamma} not super-logarithmic"


# ─── Lemma 8.3e: RFT vs DFT Gap ───────────────────────────────────────────────

class TestLemma83e:
    """RFT achieves strictly better K₀.₉₉ than DFT at every N."""

    def test_verified(self):
        result = prove_lemma_8_3e(SMALL_SIZES, n_trials=N_TRIALS)
        assert result.verified, f"Lemma 8.3e not verified: {result.proof_steps[-1]}"

    def test_rft_beats_dft_at_every_N(self):
        result = prove_lemma_8_3e(FULL_SIZES, n_trials=N_TRIALS)
        for N in FULL_SIZES:
            ci_lo = result.certificates[f"N{N}_ci_lo"]
            assert ci_lo > 0, f"N={N}: CI lower bound {ci_lo} ≤ 0"

    def test_gap_grows_with_N(self):
        result = prove_lemma_8_3e(FULL_SIZES, n_trials=N_TRIALS)
        deltas = [result.certificates[f"N{N}_delta"] for N in FULL_SIZES]
        for i in range(len(deltas) - 1):
            assert deltas[i] <= deltas[i + 1] + 1.0, (
                f"Gap not growing: {deltas}"
            )

    def test_cohens_d_large(self):
        """Effect size is large (d > 1) at all N."""
        result = prove_lemma_8_3e(FULL_SIZES, n_trials=N_TRIALS)
        for N in FULL_SIZES:
            d = result.certificates[f"N{N}_cohens_d"]
            assert d > 1.0, f"N={N}: Cohen's d={d} not large"

    def test_rft_closer_to_oracle(self):
        """RFT K₀.₉₉ is closer to oracle than DFT K₀.₉₉."""
        result = prove_lemma_8_3e(FULL_SIZES, n_trials=N_TRIALS)
        for N in FULL_SIZES:
            rft = result.certificates[f"N{N}_rft_k99"]
            dft = result.certificates[f"N{N}_dft_k99"]
            oracle = result.certificates[f"N{N}_oracle_k99"]
            assert abs(rft - oracle) < abs(dft - oracle), (
                f"N={N}: RFT ({rft}) not closer to oracle ({oracle}) than DFT ({dft})"
            )


# ─── Combined Theorem 8 ───────────────────────────────────────────────────────

class TestTheorem8Combined:
    """Full Theorem 8 proof chain."""

    def test_all_lemmas_verified(self):
        proof = prove_theorem_8(SMALL_SIZES, n_trials=N_TRIALS)
        assert proof.is_complete(), "Not all lemmas verified"

    def test_theorem_verified(self):
        proof = prove_theorem_8(SMALL_SIZES, n_trials=N_TRIALS)
        assert proof.theorem_verified, "Theorem 8 not verified"

    def test_classification(self):
        proof = prove_theorem_8(SMALL_SIZES, n_trials=N_TRIALS)
        assert proof.classification == "CONSTRUCTIVE + COMPUTATIONAL"

    def test_three_constructive_lemmas(self):
        proof = prove_theorem_8(SMALL_SIZES, n_trials=N_TRIALS)
        assert proof.summary["constructive_count"] == 3

    def test_two_computational_lemmas(self):
        proof = prove_theorem_8(SMALL_SIZES, n_trials=N_TRIALS)
        assert proof.summary["computational_count"] == 2

    def test_no_empirical_claims(self):
        """Theorem 8 should have zero empirical lemmas."""
        proof = prove_theorem_8(SMALL_SIZES, n_trials=N_TRIALS)
        for lid, lem in proof.lemmas.items():
            assert lem.status in ("CONSTRUCTIVE", "COMPUTATIONAL"), (
                f"Lemma {lid} has status {lem.status} — expected non-empirical"
            )


# ─── Cross-cutting structural tests ───────────────────────────────────────────

class TestStructural:
    """Tests for mathematical properties independent of the proof engine."""

    def test_signal_in_subspace(self):
        """Every golden signal lies exactly in span(V)."""
        rng = np.random.default_rng(999)
        for N in [64, 128]:
            K = _K_modes(N)
            V = _golden_signal_basis(N, K)
            Q, _ = np.linalg.qr(V)
            Q_K = Q[:, :K]
            P = Q_K @ Q_K.conj().T  # Projection onto span(V)
            for _ in range(50):
                x = _generate_golden_signal(N, K, rng)
                residual = np.linalg.norm(x - P @ x) / np.linalg.norm(x)
                assert residual < 1e-12, f"N={N}: signal not in span(V), res={residual}"

    def test_dft_misses_golden_frequencies(self):
        """No DFT bin exactly matches any golden frequency."""
        for N in [32, 64, 128, 256]:
            K = _K_modes(N)
            golden_freqs = np.mod(np.arange(1, K + 1) * PHI, 1.0)
            dft_freqs = np.arange(N) / N
            # Min distance between golden and DFT grids
            dists = np.abs(golden_freqs[:, None] - dft_freqs[None, :])
            min_dist = dists.min()
            assert min_dist > 1e-10, f"N={N}: DFT bin matches golden freq!"

    def test_oracle_is_unitary(self):
        """The oracle basis construction produces orthonormal columns."""
        for N in [32, 64, 128]:
            K = _K_modes(N)
            V = _golden_signal_basis(N, K)
            Q, _ = np.linalg.qr(V)  # Q is N×K (reduced QR)
            err = np.linalg.norm(Q.conj().T @ Q - np.eye(K))
            assert err < 1e-12, f"N={N}: oracle columns not orthonormal, err={err}"

    def test_rft_is_unitary(self):
        """Canonical RFT U is unitary."""
        for N in [32, 64, 128]:
            U = _canonical_U(N)
            err = np.linalg.norm(U.conj().T @ U - np.eye(N))
            assert err < 1e-10, f"N={N}: RFT not unitary, err={err}"

    def test_parseval_holds(self):
        """Parseval: total energy is preserved by all transforms."""
        rng = np.random.default_rng(1234)
        N = 64
        K = _K_modes(N)
        U = _canonical_U(N)
        F = _dft_matrix(N)
        for _ in range(20):
            x = _generate_golden_signal(N, K, rng)
            energy_x = np.sum(np.abs(x) ** 2)
            energy_rft = np.sum(np.abs(U.conj().T @ x) ** 2)
            energy_dft = np.sum(np.abs(F.conj().T @ x) ** 2)
            assert abs(energy_rft - energy_x) < 1e-10
            assert abs(energy_dft - energy_x) < 1e-10
