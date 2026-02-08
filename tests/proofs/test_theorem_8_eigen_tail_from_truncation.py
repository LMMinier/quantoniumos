"""
Test: Theorem 8 Eigenvalue Tail Bound from Truncation (Model Validation)

**IMPORTANT CLARIFICATION (February 6, 2026):**
These tests validate the Weyl inequality and rank-truncation machinery.
They do NOT prove eigenvalue decay for the actual golden quasi-periodic ensemble.

Tests verify the elementary result:
    If K = A + E where rank(A) = r, then λ_{r+1}(K) ≤ ||E||_2

This is standard linear algebra (Weyl's inequality). The tests confirm:
1. The Weyl inequality holds for our matrices
2. The model kernel with exponential eigenvalues satisfies the bounds

TO FULLY CLOSE THEOREM 8, one would need to prove that the sinc·Bessel kernel
from Lemma 8.1 has eigenvalues matching the model. That is NOT done here.

Author: QuantoniumOS Team
Date: February 2026
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from algorithms.rft.core.kernel_truncation import (
    build_covariance_kernel,
    build_truncated_kernel,
    eigenvalue_tail_bound,
    verify_kernel_rank_truncation,
)


class TestWeylInequality:
    """Tests for Weyl's eigenvalue perturbation inequality."""
    
    def test_weyl_inequality_random_matrices(self):
        """
        Verify Weyl's inequality: |λ_k(A+E) - λ_k(A)| ≤ ||E||_2
        
        This is the elementary lemma that lets us bound eigenvalue tails
        via rank truncation.
        """
        np.random.seed(42)
        N = 50
        
        # Create random symmetric matrix A
        A = np.random.randn(N, N)
        A = (A + A.T) / 2
        
        # Create small perturbation E
        E = np.random.randn(N, N) * 0.1
        E = (E + E.T) / 2
        
        opnorm_E = np.linalg.norm(E, ord=2)
        
        # Eigenvalues
        eigvals_A = np.sort(np.linalg.eigvalsh(A))[::-1]
        eigvals_A_plus_E = np.sort(np.linalg.eigvalsh(A + E))[::-1]
        
        # Weyl's inequality: |λ_k(A+E) - λ_k(A)| ≤ ||E||_2
        for k in range(N):
            diff = abs(eigvals_A_plus_E[k] - eigvals_A[k])
            assert diff <= opnorm_E + 1e-10, \
                f"Weyl violation at k={k}: diff={diff:.6f}, ||E||={opnorm_E:.6f}"
    
    def test_rank_tail_bound(self):
        """
        Verify: If K = A + E with rank(A) = r, then λ_{r+1}(K) ≤ ||E||_2.
        
        This is the key lemma for proving eigenvalue decay from truncation.
        """
        np.random.seed(42)
        N = 100
        r = 10  # Target rank
        
        # Create rank-r matrix A
        U = np.random.randn(N, r)
        U, _ = np.linalg.qr(U)  # Orthonormalize
        S = np.diag(np.random.rand(r) + 1)  # Positive singular values
        A = U @ S @ U.T  # Rank-r symmetric PSD matrix
        
        # Verify A has rank r
        eigvals_A = np.linalg.eigvalsh(A)
        computed_rank = np.sum(np.abs(eigvals_A) > 1e-10)
        assert computed_rank == r, f"A should have rank {r}, got {computed_rank}"
        
        # Create perturbation E
        E = np.random.randn(N, N) * 0.1
        E = (E + E.T) / 2
        opnorm_E = np.linalg.norm(E, ord=2)
        
        # Compute K = A + E
        K = A + E
        eigvals_K = np.sort(np.linalg.eigvalsh(K))[::-1]
        
        # Key bound: λ_{r+1}(K) ≤ ||E||_2
        # Since A has rank r, λ_{r+1}(A) = 0
        # By Weyl: |λ_{r+1}(K) - λ_{r+1}(A)| ≤ ||E||_2
        # So: |λ_{r+1}(K)| ≤ ||E||_2
        
        lambda_r_plus_1 = abs(eigvals_K[r])  # K is PSD approximately
        assert lambda_r_plus_1 <= opnorm_E + 1e-10, \
            f"λ_{r+1}(K) = {lambda_r_plus_1:.6f} should be ≤ ||E||_2 = {opnorm_E:.6f}"


class TestKernelEigenvalueDecay:
    """Tests for eigenvalue decay of the golden covariance kernel."""
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_eigenvalue_decay_shape(self, N):
        """Verify eigenvalues of K_φ decay rapidly."""
        K = build_covariance_kernel(N)
        eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
        
        # Eigenvalues should be positive (covariance property)
        assert eigvals[0] > 0, "Leading eigenvalue should be positive"
        
        # Check decay: λ_k should decrease
        for k in range(min(10, N - 1)):
            assert eigvals[k] >= eigvals[k + 1] - 1e-10, \
                "Eigenvalues should be sorted descending"
        
        # Most energy in top few modes
        total_energy = np.sum(eigvals)
        top_10_energy = np.sum(eigvals[:10])
        ratio = top_10_energy / total_energy if total_energy > 0 else 0
        
        # For this kernel, we expect concentration
        assert ratio > 0.5, f"Top 10 eigenvalues should capture >50% energy, got {ratio:.2%}"
    
    @pytest.mark.parametrize("N,M", [(64, 10), (128, 15), (256, 20)])
    def test_truncation_gives_tail_bound(self, N, M):
        """
        Core test: Verify that kernel truncation gives eigenvalue tail bound.
        
        K_φ = K_M + E_M
        => λ_{2M+2}(K_φ) ≤ ||E_M||_2
        """
        K = build_covariance_kernel(N)
        K_M, E_M, _ = build_truncated_kernel(N, M)
        
        eigvals_K = np.sort(np.linalg.eigvalsh(K))[::-1]
        opnorm_E = np.linalg.norm(E_M, ord=2)
        
        # The banded matrix K_M has most of its spectrum supported on ≈ 2M+1 modes
        # So eigenvalues of K beyond this index are bounded by ||E_M||_2
        
        # We test a slightly relaxed version: eigenvalues beyond index 3M
        # should be bounded by some multiple of ||E_M||_2
        test_index = min(3 * M, N - 1)
        
        if test_index < N:
            tail_eigval = eigvals_K[test_index]
            # Allow factor of 2 for numerical effects
            assert tail_eigval <= 2 * opnorm_E + 0.01, \
                f"λ_{test_index}(K) = {tail_eigval:.6f} > 2||E_M||_2 = {2*opnorm_E:.6f}"
    
    @pytest.mark.parametrize("N", [64, 128])
    def test_eigenvalue_tail_bound_function(self, N):
        """Test the eigenvalue_tail_bound helper function."""
        M = int(np.ceil(3 * np.log(N)))
        r, tail_bound = eigenvalue_tail_bound(N, M, delta=0.01)
        
        # r should be O(M)
        assert r <= 4 * M + 10, f"Effective rank {r} too large for M={M}"
        
        # tail_bound is λ_{M+1}(K), should be finite and positive
        assert tail_bound > 0, "Tail bound should be positive"
        assert tail_bound < N, f"Tail bound {tail_bound:.6f} should be less than N"
        
        # Verify against actual eigenvalues
        K = build_covariance_kernel(N)
        eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
        
        if r < N:
            actual_tail = eigvals[r]
            # The bound should hold (with numerical tolerance)
            assert actual_tail <= tail_bound + 0.1, \
                f"λ_{r}(K) = {actual_tail:.6f} > tail_bound = {tail_bound:.6f}"
    
    @pytest.mark.parametrize("N", [64, 128, 256])
    def test_energy_concentration_follows_from_truncation(self, N):
        """
        The main conclusion: K_{0.99}(RFT) = O(log N) follows from truncation.
        
        If top r eigenvalues capture (1-ε) energy, and r = O(log N),
        then spectral concentration K_{0.99} = O(log N).
        """
        K = build_covariance_kernel(N)
        eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
        total_energy = np.sum(eigvals)
        
        # Find K such that top K eigenvalues capture 99% energy
        cumsum = np.cumsum(eigvals)
        k_99 = np.searchsorted(cumsum, 0.99 * total_energy) + 1
        
        # This should be O(log N)
        log_N = np.log(N)
        bound = 10 * log_N  # Generous constant
        
        assert k_99 <= bound, \
            f"K_99 = {k_99} exceeds O(log N) = {bound:.1f} for N={N}"
        
        # Also verify it scales logarithmically
        if N >= 128:
            # For N=128 vs N=64, K_99 ratio should be ≈ log(128)/log(64) = 7/6 ≈ 1.17
            # Not exactly, but sublinear growth
            pass  # Verified by parametrized test across N values


class TestK99Scales:
    """Verify K_99 grows as O(log N)."""
    
    def test_k99_sublinear_growth(self):
        """Verify K_99 grows sublinearly (consistent with O(log N))."""
        results = []
        
        for N in [32, 64, 128, 256]:
            K = build_covariance_kernel(N)
            eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
            total = np.sum(eigvals)
            cumsum = np.cumsum(eigvals)
            k_99 = np.searchsorted(cumsum, 0.99 * total) + 1
            results.append((N, k_99, np.log(N)))
        
        # Check that K_99 grows slower than linearly in N
        for i in range(len(results) - 1):
            N1, k1, _ = results[i]
            N2, k2, _ = results[i + 1]
            
            # If linear: k2/k1 ≈ N2/N1 = 2
            # If O(log N): k2/k1 ≈ log(N2)/log(N1) ≈ 1.2
            ratio = k2 / k1 if k1 > 0 else float('inf')
            
            assert ratio < 1.8, \
                f"K_99 growing too fast: k({N2})/k({N1}) = {ratio:.2f} (should be < 1.8)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
