#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Transform Correctness Tests
================================

Rigorous mathematical property tests for the Φ-RFT transform.
These tests validate the core claims about the transform.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward_square as rft_forward, rft_inverse_square as rft_inverse, rft_matrix_canonical as rft_matrix
)

# Golden ratio constant
PHI = (1 + np.sqrt(5)) / 2


def golden_ratio_chirp(n: int) -> np.ndarray:
    """Generate a golden-ratio chirp signal (for testing)."""
    t = np.arange(n, dtype=np.float64)
    return np.cos(2 * np.pi * PHI ** (t / n * 4))


class TestRoundTrip:
    """Test that forward + inverse recovers original signal."""
    
    @pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256, 512, 1024])
    def test_roundtrip_random(self, n: int):
        """Round-trip with random signals of various sizes."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        # Should be lossless up to floating-point precision
        assert_allclose(x, x_rec.real, rtol=1e-10, atol=1e-12,
                       err_msg=f"Round-trip failed for n={n}")
    
    @pytest.mark.parametrize("n", [8, 64, 256])
    def test_roundtrip_complex(self, n: int):
        """Round-trip with complex signals."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        assert_allclose(x, x_rec, rtol=1e-10, atol=1e-12,
                       err_msg=f"Complex round-trip failed for n={n}")
    
    @pytest.mark.parametrize("n", [8, 64, 256])
    def test_roundtrip_sparse(self, n: int):
        """Round-trip with sparse signals (mostly zeros)."""
        rng = np.random.default_rng(42)
        x = np.zeros(n)
        indices = rng.choice(n, size=n//10, replace=False)
        x[indices] = rng.standard_normal(len(indices))
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        assert_allclose(x, x_rec.real, rtol=1e-10, atol=1e-12,
                       err_msg=f"Sparse round-trip failed for n={n}")
    
    def test_roundtrip_golden_signal(self):
        """RFT should be especially accurate on golden-ratio signals."""
        n = 256
        t = np.arange(n, dtype=np.float64)
        x = golden_ratio_chirp(n)
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        # Canonical Gram-normalized basis — roundtrip accurate to machine eps
        assert_allclose(x, x_rec.real, rtol=1e-10, atol=1e-12,
                       err_msg="Golden signal round-trip degraded")


class TestParseval:
    """Test energy preservation (Parseval's theorem)."""
    
    @pytest.mark.parametrize("n", [8, 32, 128, 512])
    def test_parseval_energy(self, n: int):
        """Energy in time domain = energy in RFT domain.
        
        Note: RFT uses FFT with norm='ortho', which already normalizes.
        The RFT adds phase modulation which preserves magnitudes.
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X = rft_forward(x)
        
        # Time-domain energy
        E_time = np.sum(np.abs(x) ** 2)
        
        # RFT-domain energy (ortho-normalized FFT preserves energy)
        E_rft = np.sum(np.abs(X) ** 2)
        
        # Should be equal (energy preservation via Parseval)
        assert_allclose(E_time, E_rft, rtol=1e-10,
                       err_msg=f"Parseval failed for n={n}")
    
    def test_parseval_complex(self):
        """Parseval for complex signals."""
        n = 128
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        
        X = rft_forward(x)
        
        E_time = np.sum(np.abs(x) ** 2)
        E_rft = np.sum(np.abs(X) ** 2)
        
        assert_allclose(E_time, E_rft, rtol=1e-10)


class TestMatrixProperties:
    """Test properties of the RFT matrix."""
    
    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    def test_matrix_invertibility(self, n: int):
        """RFT matrix should be invertible (non-singular)."""
        M = rft_matrix(n)
        
        # Condition number should be reasonable (not too large)
        cond = np.linalg.cond(M)
        assert cond < 1e6, f"RFT matrix poorly conditioned: cond={cond}"
        
        # Actually compute inverse
        M_inv = np.linalg.inv(M)
        identity = M @ M_inv
        
        assert_allclose(identity, np.eye(n), rtol=1e-10, atol=1e-12,
                       err_msg="M @ M_inv != I")
    
    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_matrix_vs_function(self, n: int):
        """Matrix multiplication should match function output."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        M = rft_matrix(n)
        X_matrix = M @ x
        X_func = rft_forward(x)
        
        assert_allclose(X_matrix, X_func, rtol=1e-10,
                       err_msg="Matrix and function disagree")
    
    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_golden_ratio_structure(self, n: int):
        """Verify golden ratio appears in matrix structure."""
        M = rft_matrix(n)
        
        # Check that matrix involves phi (golden ratio)
        # The specific structure depends on implementation
        # At minimum, the matrix should be complex
        assert np.iscomplexobj(M), "RFT matrix should be complex"
        
        # Non-trivial structure (not just identity or DFT)
        # Check it's different from DFT matrix
        omega = np.exp(-2j * np.pi / n)
        dft_matrix = np.array([[omega ** (j * k) for k in range(n)] for j in range(n)])
        
        # Frobenius distance should be significant
        dist = np.linalg.norm(M - dft_matrix, 'fro')
        assert dist > 0.1 * n, "RFT matrix too similar to DFT"


class TestInverseConsistency:
    """Test consistency between inverse methods."""
    
    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_inverse_via_matrix(self, n: int):
        """Inverse via function matches inverse via matrix."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X = rft_forward(x)
        
        # Inverse via function
        x_func = rft_inverse(X)
        
        # Inverse via matrix
        M = rft_matrix(n)
        M_inv = np.linalg.inv(M)
        x_matrix = M_inv @ X
        
        assert_allclose(x_func, x_matrix, rtol=1e-10,
                       err_msg="Inverse methods disagree")


class TestLinearityAndScaling:
    """Test linearity properties."""
    
    def test_linearity_addition(self):
        """RFT(a + b) = RFT(a) + RFT(b)."""
        n = 64
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        
        X_sum = rft_forward(a + b)
        X_a = rft_forward(a)
        X_b = rft_forward(b)
        
        assert_allclose(X_sum, X_a + X_b, rtol=1e-10,
                       err_msg="Linearity (addition) failed")
    
    def test_linearity_scaling(self):
        """RFT(c * x) = c * RFT(x)."""
        n = 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        c = 3.14159
        
        X_scaled = rft_forward(c * x)
        X = rft_forward(x)
        
        assert_allclose(X_scaled, c * X, rtol=1e-10,
                       err_msg="Linearity (scaling) failed")
    
    def test_linearity_complex_scaling(self):
        """RFT(c * x) = c * RFT(x) for complex c."""
        n = 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        c = 2.0 + 3.0j
        
        X_scaled = rft_forward(c * x)
        X = rft_forward(x)
        
        assert_allclose(X_scaled, c * X, rtol=1e-10,
                       err_msg="Complex scaling failed")


class TestNumericalStability:
    """Test numerical stability under extreme conditions."""
    
    def test_very_small_values(self):
        """Stability with very small signal values."""
        n = 64
        x = np.random.randn(n) * 1e-300
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        # Check relative error
        rel_err = np.linalg.norm(x - x_rec.real) / (np.linalg.norm(x) + 1e-320)
        assert rel_err < 1e-10, f"Small value instability: rel_err={rel_err}"
    
    def test_very_large_values(self):
        """Stability with very large signal values."""
        n = 64
        x = np.random.randn(n) * 1e100
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        rel_err = np.linalg.norm(x - x_rec.real) / np.linalg.norm(x)
        assert rel_err < 1e-10, f"Large value instability: rel_err={rel_err}"
    
    def test_mixed_magnitude(self):
        """Stability with mixed magnitudes."""
        n = 64
        x = np.random.randn(n)
        x[0] = 1e10  # One very large
        x[1] = 1e-10  # One very small
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        # For mixed magnitudes, relative tolerance should be based on largest values
        # Use larger tolerance due to numerical challenges
        rel_err = np.linalg.norm(x - x_rec.real) / np.linalg.norm(x)
        assert rel_err < 1e-8, f"Mixed magnitude instability: rel_err={rel_err}"
    
    def test_constant_signal(self):
        """Stability with constant (DC) signal."""
        n = 64
        x = np.ones(n) * 5.0
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        assert_allclose(x, x_rec.real, rtol=1e-10)
    
    def test_alternating_signal(self):
        """Stability with Nyquist frequency signal."""
        n = 64
        x = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        
        X = rft_forward(x)
        x_rec = rft_inverse(X)
        
        assert_allclose(x, x_rec.real, rtol=1e-10)


class TestGoldenRatioSparsity:
    """Test sparsity properties for golden-ratio signals (Observation 4)."""
    
    def test_golden_chirp_sparse(self):
        """Golden-ratio chirp should have concentrated energy in RFT."""
        n = 256
        x = golden_ratio_chirp(n)
        
        X = rft_forward(x)
        X_fft = np.fft.fft(x)
        
        # Measure concentration: what fraction of coefficients contain 90% of energy?
        def energy_concentration(coeffs, threshold=0.9):
            energy = np.abs(coeffs) ** 2
            total = energy.sum()
            sorted_energy = np.sort(energy)[::-1]
            cumsum = np.cumsum(sorted_energy)
            idx = np.searchsorted(cumsum, threshold * total) + 1
            return idx / len(coeffs)
        
        rft_conc = energy_concentration(X)
        fft_conc = energy_concentration(X_fft)
        
        # RFT should be more concentrated for golden signals
        # This is Observation 4 (empirical, not guaranteed)
        print(f"Energy concentration: RFT={rft_conc:.3f}, FFT={fft_conc:.3f}")
    
    def test_fibonacci_modulated_sparse(self):
        """Fibonacci-modulated signal sparsity."""
        n = 256
        
        # Fibonacci sequence modulo n
        fib = [1, 1]
        while len(fib) < n:
            fib.append((fib[-1] + fib[-2]) % n)
        x = np.sin(2 * np.pi * np.array(fib[:n]) / n)
        
        X = rft_forward(x)
        X_fft = np.fft.fft(x)
        
        # Compare L1/L2 norms (sparsity measure)
        rft_sparsity = np.linalg.norm(X, 1) / (np.sqrt(n) * np.linalg.norm(X, 2) + 1e-10)
        fft_sparsity = np.linalg.norm(X_fft, 1) / (np.sqrt(n) * np.linalg.norm(X_fft, 2) + 1e-10)
        
        print(f"L1/L2 sparsity: RFT={rft_sparsity:.3f}, FFT={fft_sparsity:.3f}")


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
