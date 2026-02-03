# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Test Suite: Diophantine Irrational RFT Extension (Theorem 8 Universality)
=========================================================================

Tests the extension of Theorem 8 (K99 Energy Compaction) to general 
badly approximable (Diophantine) irrationals beyond the golden ratio.
"""

import pytest
import numpy as np
from typing import Tuple

# Import module under test
from algorithms.rft.core.diophantine_rft_extension import (
    # Constants
    SQRT2, SQRT3, SQRT5, SILVER_RATIO, PHI,
    DIOPHANTINE_CONSTANTS,
    # Continued fractions
    continued_fraction,
    convergents,
    diophantine_constant,
    # Basis construction
    diophantine_frequency_grid,
    diophantine_basis_matrix,
    # Discrepancy analysis  
    star_discrepancy,
    analyze_equidistribution,
    DiscrepancyResult,
    # Davis-Kahan analysis
    davis_kahan_analysis,
    companion_matrix_alpha,
    minimal_eigenvalue_gap,
    DavisKahanResult,
    # K99 and scaling law (renamed from universality)
    k99,
    diophantine_drift_ensemble,
    compare_k99_diophantine,
    verify_scaling_law,
    verify_universality,  # Backward compat alias
    verify_sharp_logn_bound,
    ScalingLawResult,
    UniversalityResult,  # Backward compat alias
    SharpBoundResult,
    DiophantineK99Result,
)


class TestContinuedFractions:
    """Test continued fraction utilities."""
    
    def test_golden_ratio_cf(self):
        """Golden ratio has CF [1; 1, 1, 1, ...]."""
        cf = continued_fraction(PHI, max_terms=10)
        assert all(c == 1 for c in cf)
    
    def test_sqrt2_cf(self):
        """√2 has CF [1; 2, 2, 2, ...]."""
        cf = continued_fraction(SQRT2, max_terms=10)
        assert cf[0] == 1
        assert all(c == 2 for c in cf[1:])
    
    def test_sqrt3_cf(self):
        """√3 has CF [1; 1, 2, 1, 2, ...]."""
        cf = continued_fraction(SQRT3, max_terms=10)
        assert cf[0] == 1
        # Pattern [1, 2, 1, 2, ...]
        for i, c in enumerate(cf[1:]):
            expected = 2 if (i % 2 == 1) else 1
            assert c == expected, f"Position {i+1}: expected {expected}, got {c}"
    
    def test_silver_ratio_cf(self):
        """Silver ratio 1+√2 has CF [2; 2, 2, 2, ...]."""
        cf = continued_fraction(SILVER_RATIO, max_terms=10)
        assert cf[0] == 2
        assert all(c == 2 for c in cf[1:])
    
    def test_convergents_approximate(self):
        """Convergents should approximate the irrational."""
        cf = continued_fraction(PHI, max_terms=8)
        convs = convergents(cf)
        
        # Later convergents should be closer
        for i in range(1, len(convs)):
            p_prev, q_prev = convs[i - 1]
            p, q = convs[i]
            err_prev = abs(p_prev / q_prev - PHI)
            err = abs(p / q - PHI)
            assert err <= err_prev * 1.1, f"Convergent {i} not closer than {i-1}"
    
    def test_diophantine_constant_golden(self):
        """Golden ratio has Diophantine constant c ≈ 1/√5."""
        c = diophantine_constant(PHI, num_tests=100)
        expected = 1 / np.sqrt(5)
        # Allow reasonable tolerance
        assert abs(c - expected) < 0.5, f"Expected ~{expected:.4f}, got {c:.4f}"


class TestDiophantineConstants:
    """Test Diophantine constants dictionary."""
    
    def test_constants_defined(self):
        """Check all expected constants exist."""
        assert SQRT2 == pytest.approx(np.sqrt(2))
        assert SQRT3 == pytest.approx(np.sqrt(3))
        assert SQRT5 == pytest.approx(np.sqrt(5))
        assert SILVER_RATIO == pytest.approx(1 + np.sqrt(2))
    
    def test_diophantine_constants_dict(self):
        """Check DIOPHANTINE_CONSTANTS has expected entries."""
        assert 'phi' in DIOPHANTINE_CONSTANTS
        assert 'sqrt2' in DIOPHANTINE_CONSTANTS
        assert 'sqrt3' in DIOPHANTINE_CONSTANTS


class TestDiophantineBasis:
    """Test Diophantine basis matrix construction."""
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3, SILVER_RATIO])
    def test_frequency_grid_shape(self, alpha):
        """Frequency grid should have correct length."""
        N = 64
        f = diophantine_frequency_grid(N, alpha)
        assert len(f) == N
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3, SILVER_RATIO])
    def test_basis_shape(self, alpha):
        """Basis matrix should have correct shape."""
        N = 64
        U = diophantine_basis_matrix(N, alpha)
        assert U.shape == (N, N)
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_basis_unitary(self, alpha):
        """With Gram normalization, basis should be unitary."""
        N = 32
        U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
        UhU = U.conj().T @ U
        identity = np.eye(N)
        error = np.linalg.norm(UhU - identity, ord='fro')
        assert error < 1e-10, f"Unitarity error: {error}"
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_basis_column_norms(self, alpha):
        """Columns should have unit norm."""
        N = 32
        U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
        col_norms = np.linalg.norm(U, axis=0)
        assert np.allclose(col_norms, 1.0, atol=1e-10)


class TestDiscrepancy:
    """Test discrepancy and equidistribution analysis."""
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_star_discrepancy_bounded(self, alpha):
        """Star discrepancy should be in [0, 1]."""
        N = 128
        seq = np.mod(np.arange(N) * alpha, 1.0)
        D = star_discrepancy(seq)
        assert 0 <= D <= 1
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_discrepancy_decreases(self, alpha):
        """Star discrepancy should generally decrease with N."""
        discrepancies = []
        for N in [64, 128, 256, 512]:
            seq = np.mod(np.arange(N) * alpha, 1.0)
            D = star_discrepancy(seq)
            discrepancies.append(D)
        
        # Should generally decrease (allow some noise)
        for i in range(1, len(discrepancies)):
            # Allow up to 30% increase due to fluctuations
            assert discrepancies[i] < discrepancies[i-1] * 1.3
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_analyze_equidistribution(self, alpha):
        """Equidistribution analysis should return valid result."""
        result = analyze_equidistribution(N=128, alpha=alpha)
        
        assert isinstance(result, DiscrepancyResult)
        assert result.N == 128
        assert 0 <= result.star_discrepancy <= 1
        assert result.expected_bound > 0


class TestDavisKahan:
    """Test Davis-Kahan perturbation analysis."""
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_davis_kahan_valid(self, alpha):
        """Davis-Kahan analysis should return valid bounds."""
        result = davis_kahan_analysis(N=64, alpha=alpha)
        
        assert isinstance(result, DavisKahanResult)
        assert result.N == 64
        assert result.perturbation_norm >= 0
        assert result.minimal_gap > 0
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_companion_matrix(self, alpha):
        """Companion matrix should have correct shape."""
        N = 10
        C = companion_matrix_alpha(N, alpha)
        assert C.shape == (N, N)
    
    def test_minimal_eigenvalue_gap_positive(self):
        """Minimal eigenvalue gap should be positive for distinct eigenvalues."""
        z = np.exp(2j * np.pi * np.arange(10) / 10)  # Roots of unity
        gap = minimal_eigenvalue_gap(z)
        assert gap > 0


class TestK99:
    """Test K99 energy compaction metric."""
    
    def test_k99_positive(self):
        """K99 should be a positive integer."""
        N = 64
        rng = np.random.default_rng(42)
        coeffs = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        
        k = k99(coeffs)
        assert k >= 1
        assert k <= N
        assert isinstance(k, (int, np.integer))
    
    def test_k99_sparse_signal(self):
        """K99 should be small for sparse signals."""
        N = 64
        coeffs = np.zeros(N, dtype=complex)
        coeffs[:5] = 1.0  # Only first 5 nonzero
        
        k = k99(coeffs)
        assert k <= 10  # Should capture most energy in first few
    
    def test_k99_uniform_signal(self):
        """K99 should be larger for uniform signals."""
        N = 64
        coeffs = np.ones(N, dtype=complex)
        
        k = k99(coeffs)
        assert k > N // 2  # Needs many coefficients


class TestEnsembleAnalysis:
    """Test ensemble-based K99 analysis."""
    
    def test_drift_ensemble_shape(self):
        """Drift ensemble should return correct shape."""
        N, M = 32, 50
        rng = np.random.default_rng(42)
        result = diophantine_drift_ensemble(N, M, PHI, rng)
        
        assert result.shape == (M, N)  # Returns (M, N) array of signals
    
    def test_compare_k99_structure(self):
        """Compare K99 should return valid structure."""
        result = compare_k99_diophantine(N=32, M=30, alpha=PHI, seed=42)
        
        assert isinstance(result, DiophantineK99Result)
        assert result.N == 32
        assert result.M == 30
        assert result.mean_k99_rft > 0
        assert result.mean_k99_dft > 0


class TestScalingLaw:
    """Test scaling law verification across irrationals (with statistics)."""
    
    def test_verify_scaling_law_structure(self):
        """Verify scaling law should return valid structure with CI."""
        result = verify_scaling_law(N=32, M=30, seed=42)
        
        assert isinstance(result, ScalingLawResult)
        assert len(result.results) >= 4
        assert result.results[0].N == 32
        # New statistical fields should exist
        assert hasattr(result.results[0], 'ci_delta_low')
        assert hasattr(result.results[0], 'ci_delta_high')
        assert hasattr(result.results[0], 'p_value')
    
    def test_backward_compat_alias(self):
        """verify_universality should still work as alias."""
        result = verify_universality(N=32, M=30, seed=42)
        assert isinstance(result, ScalingLawResult)
    
    def test_scaling_law_all_positive(self):
        """All K99 means should be positive."""
        result = verify_scaling_law(N=32, M=30, seed=42)
        
        for r in result.results:
            assert r.mean_k99_rft > 0, f"{r.alpha_name} has non-positive K99"
    
    def test_scaling_law_compaction(self):
        """All irrationals should achieve compaction (K99 < N)."""
        result = verify_scaling_law(N=64, M=50, seed=42)
        
        for r in result.results:
            assert r.mean_k99_rft < r.N, f"{r.alpha_name} K99={r.mean_k99_rft} >= N={r.N}"
    
    def test_ci_ordered(self):
        """CI lower should be <= upper for all results."""
        result = verify_scaling_law(N=64, M=100, seed=42)
        
        for r in result.results:
            assert r.ci_delta_low <= r.ci_delta_high, f"{r.alpha_name} CI inverted"
    
    def test_p_value_bounded(self):
        """p-value should be in [0, 1]."""
        result = verify_scaling_law(N=64, M=100, seed=42)
        
        for r in result.results:
            assert 0 <= r.p_value <= 1, f"{r.alpha_name} invalid p-value"
    
    def test_ci_includes_zero_flag(self):
        """ci_includes_zero should match CI bounds."""
        result = verify_scaling_law(N=64, M=100, seed=42)
        
        for r in result.results:
            manual_check = (r.ci_delta_low <= 0 <= r.ci_delta_high)
            assert r.ci_includes_zero == manual_check, f"{r.alpha_name} CI zero flag mismatch"


class TestSharpBounds:
    """Test sharp log(N) bounds for K99."""
    
    def test_sharp_bound_structure(self):
        """Verify sharp bound result structure."""
        result = verify_sharp_logn_bound(
            alpha=PHI,
            N_values=[32, 64, 128],
            M=20,
            seed=42
        )
        
        assert isinstance(result, SharpBoundResult)
        assert len(result.N_values) == 3
        assert len(result.k99_means) == 3
        assert result.fitted_slope != 0
    
    def test_sharp_bound_scaling(self):
        """K99 should scale roughly as O(log N)."""
        result = verify_sharp_logn_bound(
            alpha=PHI,
            N_values=[32, 64, 128, 256],
            M=30,
            seed=42
        )
        
        # R² should be reasonably high if log scaling holds
        assert result.r_squared > 0.3, f"R² = {result.r_squared} too low"
    
    @pytest.mark.parametrize("alpha", [PHI, SQRT2, SQRT3])
    def test_sharp_bound_different_alphas(self, alpha):
        """Sharp bound should work for different irrationals."""
        result = verify_sharp_logn_bound(
            alpha=alpha,
            N_values=[32, 64, 128],
            M=20,
            seed=42
        )
        
        assert result.fitted_slope > 0  # Should have positive log slope


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_analysis(self):
        """End-to-end test of Diophantine analysis."""
        N = 64
        for alpha in [PHI, SQRT2, SQRT3]:
            # Build basis
            U = diophantine_basis_matrix(N, alpha, use_gram_normalization=True)
            
            # Check unitarity
            error = np.linalg.norm(U.conj().T @ U - np.eye(N), ord='fro')
            assert error < 1e-10
            
            # Check discrepancy
            result = analyze_equidistribution(N, alpha)
            assert result.star_discrepancy < 1.0
    
    def test_phi_competitive(self):
        """Golden ratio should be competitive with other irrationals."""
        result = verify_scaling_law(N=64, M=50, seed=42)
        
        # Find phi result
        phi_k99 = None
        for r in result.results:
            if 'golden' in r.alpha_name or 'φ' in r.alpha_name:
                phi_k99 = r.mean_k99_rft
                break
        
        if phi_k99 is not None:
            min_k99 = min(r.mean_k99_rft for r in result.results)
            # Phi should be within factor of 2 of the best
            assert phi_k99 < 2.0 * min_k99


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_n(self):
        """Test with small N values."""
        for N in [4, 8, 16]:
            U = diophantine_basis_matrix(N, PHI, use_gram_normalization=True)
            assert U.shape == (N, N)
    
    def test_cf_convergence(self):
        """Test that CF convergents really converge."""
        for alpha in [PHI, SQRT2, SQRT3]:
            cf = continued_fraction(alpha, max_terms=15)
            convs = convergents(cf)
            
            # Last convergent should be very close
            p, q = convs[-1]
            error = abs(p / q - alpha)
            assert error < 1e-5, f"Convergent {p}/{q} error {error}"


# Run specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
