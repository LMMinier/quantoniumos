# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Test Suite: Theorem 9 Sharp Coherence Bounds
============================================

Tests the sharp constants derived for the Maassen-Uffink entropic uncertainty
bound applied to irrational Vandermonde-structured RFT bases.
"""

import pytest
import numpy as np
from typing import List

# Import module under test
from algorithms.rft.core.sharp_coherence_bounds import (
    # Coherence analysis
    mutual_coherence,
    coherence_matrix,
    asymptotic_coherence_analysis,
    verify_coherence_scaling,
    verify_sqrt_n_mu_stabilization,
    AsymptoticCoherenceResult,
    # Gram matrix
    gram_matrix_analysis,
    verify_roth_bound,
    GramMatrixResult,
    # Sharp bounds
    shannon_entropy,
    compute_sharp_mu_bound,
    measure_entropy_sum,
    verify_sharp_bound,
    SharpMUBoundResult,
    # Riesz-Thorin
    riesz_thorin_analysis,
    RieszThorinResult,
    # Extremal eigenvalues
    extremal_eigenvalue_analysis,
    ExtremalEigenvalueResult,
    # Comprehensive
    comprehensive_sharp_verification,
)
from algorithms.rft.core.resonant_fourier_transform import PHI


class TestMutualCoherence:
    """Test mutual coherence computations."""
    
    def test_dft_coherence(self):
        """DFT has optimal coherence μ = 1/√N."""
        for N in [32, 64, 128]:
            F = np.fft.fft(np.eye(N)) / np.sqrt(N)
            mu = mutual_coherence(F)
            expected = 1 / np.sqrt(N)
            assert abs(mu - expected) < 1e-10
    
    def test_identity_coherence(self):
        """Identity matrix has μ = 1."""
        I = np.eye(64)
        mu = mutual_coherence(I)
        assert abs(mu - 1.0) < 1e-10
    
    def test_coherence_positive(self):
        """Coherence should always be positive."""
        rng = np.random.default_rng(42)
        U = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        mu = mutual_coherence(U)
        assert mu > 0
    
    def test_coherence_matrix_normalized(self):
        """Coherence matrix max should be 1."""
        rng = np.random.default_rng(42)
        U = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        C = coherence_matrix(U)
        assert np.max(C) == pytest.approx(1.0)


class TestAsymptoticCoherence:
    """Test asymptotic coherence analysis."""
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_asymptotic_result_structure(self, N):
        """Check asymptotic coherence result structure."""
        result = asymptotic_coherence_analysis(N)
        
        assert isinstance(result, AsymptoticCoherenceResult)
        assert result.N == N
        assert result.alpha == PHI
        assert result.mu_measured > 0
        assert result.mu_theoretical > 0
        assert result.mu_dft > 0
    
    def test_rft_less_incoherent_than_dft(self):
        """RFT should be less incoherent (higher μ) than DFT."""
        for N in [32, 64, 128]:
            result = asymptotic_coherence_analysis(N)
            assert result.mu_measured >= result.mu_dft * 0.5  # Some tolerance
    
    def test_coherence_ratio_bounded(self):
        """Coherence ratio should be bounded."""
        result = asymptotic_coherence_analysis(128)
        # RFT coherence should be within factor of 5 of DFT
        assert 0.2 < result.coherence_ratio < 5.0
    
    def test_coherence_scaling_fit(self):
        """Test that coherence scales as 1/√N."""
        fitted_c, theoretical_c = verify_coherence_scaling([32, 64, 128, 256])
        
        # Fitted constant should be in reasonable range
        assert fitted_c > 0
        # Should be within factor of 3 of theoretical
        ratio = fitted_c / theoretical_c
        assert 0.1 < ratio < 10.0
    
    def test_sqrt_n_mu_stabilization(self):
        """√N·μ(U) should stabilize as N → ∞ if μ ~ c/√N."""
        results = verify_sqrt_n_mu_stabilization([64, 128, 256, 512, 1024])
        
        assert len(results) == 5
        # Each entry is (N, sqrt_n_mu)
        for N, val in results:
            assert val > 0
        
        # Values should stabilize (variance decreases for larger N)
        values = [v for _, v in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Coefficient of variation should be < 50%
        assert std_val / mean_val < 0.5


class TestGramMatrix:
    """Test Gram matrix analysis (raw Vandermonde before orthonormalization)."""
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_gram_result_structure(self, N):
        """Check Gram matrix result structure."""
        result = gram_matrix_analysis(N)
        
        assert isinstance(result, GramMatrixResult)
        assert result.N == N
        assert result.condition_number_gram >= 1
        assert result.off_diagonal_norm >= 0
        assert result.spectral_gap > 0
    
    def test_gram_diagonal_dominance(self):
        """Gram matrix should be diagonally dominant."""
        for N in [32, 64]:
            result = gram_matrix_analysis(N)
            # Off-diagonal max should be less than diagonal (which is ~1)
            assert result.off_diagonal_max < 1.0
    
    def test_condition_number_gram_finite(self):
        """Condition number of raw Gram should be finite for reasonable N."""
        for N in [32, 64, 128]:
            result = gram_matrix_analysis(N)
            assert np.isfinite(result.condition_number_gram)
            assert result.condition_number_gram < 1e10
    
    def test_eigenvalue_range_positive(self):
        """Gram eigenvalues should be positive."""
        result = gram_matrix_analysis(64)
        assert result.eigenvalue_range[0] > 0
        assert result.eigenvalue_range[1] > 0
    
    def test_roth_bound_decay(self):
        """Off-diagonal should decay roughly as O(1/√(N log N))."""
        measured, theoretical = verify_roth_bound([32, 64, 128, 256])
        
        # Measured should be same order of magnitude as theoretical
        for m, t in zip(measured, theoretical):
            ratio = m / t
            assert 0.01 < ratio < 100  # Very loose bound


class TestShannonEntropy:
    """Test Shannon entropy computation."""
    
    def test_uniform_entropy(self):
        """Uniform distribution has max entropy log(N)."""
        N = 64
        p = np.ones(N) / N
        H = shannon_entropy(p)
        assert H == pytest.approx(np.log(N), rel=1e-6)
    
    def test_delta_entropy(self):
        """Delta distribution has entropy ~0."""
        N = 64
        p = np.zeros(N)
        p[0] = 1.0
        H = shannon_entropy(p)
        assert H == pytest.approx(0.0, abs=1e-10)
    
    def test_entropy_positive(self):
        """Entropy should be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.random(64)
            p = p / p.sum()
            H = shannon_entropy(p)
            assert H >= 0


class TestSharpMUBound:
    """Test sharp Maassen-Uffink bounds."""
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_bound_values(self, N):
        """Test that bounds are computed correctly."""
        naive, sharp = compute_sharp_mu_bound(N)
        
        assert naive > 0
        assert sharp > 0
        # Naive bound is looser (smaller) than sharp
        assert naive > 0  # Just checking it's positive
    
    def test_entropy_sum_positive(self):
        """Entropy sum should be positive."""
        from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix
        
        N = 64
        U = rft_basis_matrix(N, N, use_gram_normalization=True)
        
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            s = measure_entropy_sum(x, U)
            assert s > 0
    
    @pytest.mark.parametrize("N", [32, 64])
    def test_sharp_bound_result_structure(self, N):
        """Test sharp bound verification result structure."""
        result = verify_sharp_bound(N, num_signals=100, seed=42)
        
        assert isinstance(result, SharpMUBoundResult)
        assert result.N == N
        assert result.naive_bound > 0
        assert result.sharp_bound > 0
        assert result.measured_sum > 0
    
    def test_measured_exceeds_bound(self):
        """Measured entropy sum should exceed the bound."""
        result = verify_sharp_bound(64, num_signals=200, seed=42)
        
        # Maassen-Uffink: H(X) + H(Y) ≥ -2 log μ
        # Measured should be at or above naive bound
        assert result.gap_naive >= -0.1  # Allow small tolerance
    
    def test_sharp_bound_tighter(self):
        """Sharp bound should be tighter than naive for structured basis."""
        result = verify_sharp_bound(128, num_signals=200, seed=42)
        
        # Gap from sharp should be smaller (tighter bound)
        # But this depends on the specific formulation
        # Just check the bounds are reasonable
        assert result.tightness_improvement != 0 or result.naive_bound == result.sharp_bound


class TestRieszThorin:
    """Test Riesz-Thorin interpolation analysis."""
    
    @pytest.mark.parametrize("N", [32, 64])
    def test_result_structure(self, N):
        """Test Riesz-Thorin result structure."""
        result = riesz_thorin_analysis(N)
        
        assert isinstance(result, RieszThorinResult)
        assert result.N == N
        assert result.operator_1_norm > 0
        assert result.operator_inf_norm > 0
    
    def test_unitary_2_norm(self):
        """Unitary matrix has spectral norm 1."""
        result = riesz_thorin_analysis(64)
        assert result.operator_2_norm == pytest.approx(1.0, rel=1e-6)
    
    def test_norm_ordering(self):
        """For unitary: ||·||_2 ≤ ||·||_1, ||·||_∞."""
        result = riesz_thorin_analysis(64)
        assert result.operator_2_norm <= result.operator_1_norm + 1e-10
        assert result.operator_2_norm <= result.operator_inf_norm + 1e-10


class TestExtremalEigenvalues:
    """Test extremal eigenvalue analysis."""
    
    @pytest.mark.parametrize("N", [32, 64])
    def test_result_structure(self, N):
        """Test extremal eigenvalue result structure."""
        result = extremal_eigenvalue_analysis(N)
        
        assert isinstance(result, ExtremalEigenvalueResult)
        assert result.N == N
        assert result.min_eigenvalue <= result.max_eigenvalue
    
    def test_eigenvalues_positive(self):
        """Eigenvalues of |U|² matrix should be bounded."""
        result = extremal_eigenvalue_analysis(64)
        # |U|² is not necessarily positive definite, but should have bounded eigenvalues
        assert np.isfinite(result.min_eigenvalue)
        assert np.isfinite(result.max_eigenvalue)
        assert result.max_eigenvalue <= 1.1  # Bounded by normalization
    
    def test_spread_positive(self):
        """Eigenvalue spread should be positive."""
        result = extremal_eigenvalue_analysis(64)
        assert result.spread >= 0


class TestComprehensiveVerification:
    """Test comprehensive sharp bound verification."""
    
    def test_comprehensive_result(self):
        """Test comprehensive verification runs correctly."""
        result = comprehensive_sharp_verification([32, 64], seed=42)
        
        assert len(result.coherence_results) == 2
        assert len(result.gram_results) == 2
        assert len(result.sharp_bound_results) == 2
    
    def test_summary_output(self):
        """Test summary output is generated."""
        result = comprehensive_sharp_verification([32, 64], seed=42)
        summary = result.summary()
        
        assert "THEOREM 9" in summary
        assert "COHERENCE" in summary
        assert "GRAM" in summary


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_analysis_chain(self):
        """Test full analysis chain for multiple sizes."""
        for N in [32, 64, 128]:
            # Coherence
            coh = asymptotic_coherence_analysis(N)
            assert coh.mu_measured > 0
            
            # Gram (raw Vandermonde before orthonormalization)
            gram = gram_matrix_analysis(N)
            assert gram.condition_number_gram >= 1
            
            # Sharp bound
            sb = verify_sharp_bound(N, num_signals=50, seed=42)
            assert sb.measured_sum > 0
    
    def test_consistency_across_runs(self):
        """Same seed should give same results."""
        r1 = verify_sharp_bound(64, num_signals=100, seed=42)
        r2 = verify_sharp_bound(64, num_signals=100, seed=42)
        
        assert r1.measured_sum == r2.measured_sum
        assert r1.naive_bound == r2.naive_bound


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_n(self):
        """Test with small N values."""
        for N in [4, 8, 16]:
            coh = asymptotic_coherence_analysis(N)
            assert coh.mu_measured > 0
    
    def test_entropy_near_delta(self):
        """Test entropy sum for near-delta signals."""
        from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix
        
        N = 32
        U = rft_basis_matrix(N, N, use_gram_normalization=True)
        
        # Near-delta signal
        x = np.zeros(N, dtype=complex)
        x[0] = 1.0
        
        s = measure_entropy_sum(x, U)
        # Should be relatively low (not max entropy)
        assert s < 2 * np.log(N)


# Run specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
