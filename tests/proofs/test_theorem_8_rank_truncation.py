"""
Test: Theorem 8 Rank Truncation - Kernel Decomposition (Model Validation)

**IMPORTANT CLARIFICATION (February 6, 2026):**
These tests validate that a **model kernel** with assumed exponential eigenvalue
decay behaves as expected. They do NOT prove that the actual golden quasi-periodic
ensemble's covariance has this decay.

The model kernel K = Φ D Φ^H is constructed with exponential eigenvalues by
definition. These tests verify that:
1. The code correctly implements the rank truncation machinery
2. If Assumption 8.3 holds (eigenvalue decay), then Theorem 8 follows

TO FULLY CLOSE THEOREM 8, one would need to prove that the sinc·Bessel kernel
from Lemma 8.1 has eigenvalues matching the model. That analysis (via Jacobi-Anger
expansion or Landau-Widom theory) is NOT performed here.

Tests verify:
    K_φ = K_M + E_M

where:
    - rank(K_M) = M exactly
    - ||E_M||_2 = λ_{M+1}(K_φ)

Author: QuantoniumOS Team
Date: February 2026
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from algorithms.rft.core.kernel_truncation import (
    build_covariance_kernel,
    build_truncated_kernel,
    verify_kernel_rank_truncation,
    golden_discrepancy,
    kernel_diagonal,
)


class TestKernelTruncation:
    """Tests for kernel rank truncation decomposition."""
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_kernel_is_symmetric_psd(self, N):
        """Verify K_φ is a symmetric positive semi-definite matrix."""
        K = build_covariance_kernel(N)
        
        # Check symmetry
        assert np.allclose(K, K.T), "K_φ should be symmetric"
        
        # Check PSD (all eigenvalues non-negative)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10), \
            f"K_φ should be PSD, but min eigenvalue = {eigvals.min()}"
    
    @pytest.mark.parametrize("N,M", [(32, 10), (64, 12), (128, 15)])
    def test_truncated_kernel_has_exact_rank(self, N, M):
        """Verify K_M has exactly rank M."""
        K_M, E_M, _ = build_truncated_kernel(N, M)
        
        # Check rank
        computed_rank = np.linalg.matrix_rank(K_M, tol=1e-10)
        assert computed_rank == M, \
            f"K_M should have rank {M}, got {computed_rank}"
    
    @pytest.mark.parametrize("N,M", [(32, 10), (64, 12), (128, 15), (256, 20)])
    def test_error_equals_tail_eigenvalue(self, N, M):
        """Verify ||E_M||_2 = λ_{M+1}(K) exactly."""
        K = build_covariance_kernel(N)
        K_M, E_M, bound = build_truncated_kernel(N, M)
        
        actual_opnorm = np.linalg.norm(E_M, ord=2)
        
        # The bound should be exact (it's λ_{M+1})
        assert np.isclose(actual_opnorm, bound, rtol=1e-10), \
            f"||E_M||_2 = {actual_opnorm:.6f} should equal λ_{M+1} = {bound:.6f}"
    
    @pytest.mark.parametrize("N", [64, 128, 256])
    def test_full_verification_passes(self, N):
        """Run full verification that K_φ admits rank-truncation decomposition."""
        M = int(np.ceil(3 * np.log(N)))  # M = O(log N)
        result = verify_kernel_rank_truncation(N, M)
        
        assert result['passes'], \
            f"Full verification failed for N={N}, M={M}: {result}"
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_eigenvalue_decay_exponential(self, N):
        """Verify eigenvalues decay exponentially."""
        K = build_covariance_kernel(N)
        eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
        
        # For exponential decay λ_k ≈ exp(-k/τ), check ratio
        # λ_k / λ_{k+1} should be roughly constant and bounded
        for k in range(min(15, N - 2)):  # Check first 15 eigenvalues
            if eigvals[k + 1] > 1e-10:
                ratio = eigvals[k] / eigvals[k + 1]
                # Ratio should be > 1 (decaying) and bounded
                assert ratio >= 0.99, f"Eigenvalues should not increase: λ_{k}/λ_{k+1} = {ratio:.4f}"
                # Allow ratio up to 3 for small N (heavier tail)
                assert ratio < 3, f"Decay ratio should be bounded: λ_{k}/λ_{k+1} = {ratio:.4f}"


class TestScalingBehavior:
    """Tests for scaling behavior of the decomposition."""
    
    def test_error_decreases_with_M(self):
        """Verify ||E_M||_2 decreases as M increases."""
        N = 128
        prev_opnorm = float('inf')
        
        for M in [5, 10, 20, 40]:
            _, E_M, _ = build_truncated_kernel(N, M)
            opnorm = np.linalg.norm(E_M, ord=2)
            
            assert opnorm < prev_opnorm, \
                f"||E_M||_2 should decrease with M: {opnorm:.6f} >= {prev_opnorm:.6f}"
            prev_opnorm = opnorm
    
    def test_m_log_n_sufficient_for_small_error(self):
        """Verify M = O(log N) gives small error."""
        for N in [64, 128, 256]:
            M = int(np.ceil(5 * np.log(N)))  # M = 5 log(N)
            _, E_M, _ = build_truncated_kernel(N, M)
            opnorm = np.linalg.norm(E_M, ord=2)
            
            # With M = O(log N), error should be O(√(N/log N)) 
            # which is still growing, but slowly
            threshold = np.sqrt(N / np.log(N))
            assert opnorm < threshold, \
                f"||E_M||_2 = {opnorm:.4f} with M=5logN should be < {threshold:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
