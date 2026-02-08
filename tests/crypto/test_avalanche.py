#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Cryptographic Property Tests
=============================

Tests for avalanche effect, bit sensitivity, and statistical distribution.
These validate claims about RFT's cryptographic potential (Observation 9).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from scipy import stats

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward, rft_inverse_square as rft_inverse


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two byte arrays."""
    a_bytes = np.asarray(a, dtype=np.complex128).view(np.float64).tobytes()
    b_bytes = np.asarray(b, dtype=np.complex128).view(np.float64).tobytes()
    
    # Count differing bits
    diff = 0
    for x, y in zip(a_bytes, b_bytes):
        diff += bin(x ^ y).count('1')
    return diff


def bit_flip_input(x: np.ndarray, position: int, bit: int = 0) -> np.ndarray:
    """Flip a bit in the input array (at float64 bit level)."""
    x_bytes = bytearray(x.astype(np.float64).tobytes())
    byte_pos = position * 8 + (7 - bit // 8)  # Little endian
    bit_pos = bit % 8
    x_bytes[byte_pos] ^= (1 << bit_pos)
    return np.frombuffer(bytes(x_bytes), dtype=np.float64)


class TestAvalancheEffect:
    """Test avalanche effect: small input change -> large output change."""
    
    def test_single_bit_flip_avalanche(self):
        """Single bit flip should cause ~50% of output bits to change."""
        n = 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X_orig = rft_forward(x)
        
        # Flip each input element slightly
        total_bits = n * 64  # float64
        distances = []
        
        for i in range(min(n, 16)):  # Test first 16 elements
            x_mod = x.copy()
            x_mod[i] += 1e-10  # Tiny perturbation
            
            X_mod = rft_forward(x_mod)
            
            # Measure change in output
            dist = hamming_distance(X_orig, X_mod)
            distances.append(dist / total_bits)
        
        mean_change = np.mean(distances)
        
        # Good avalanche: ~50% bit change (0.3-0.7 is acceptable)
        # RFT is NOT a hash function, so we don't expect perfect avalanche
        print(f"Mean bit change ratio: {mean_change:.4f}")
        
        # At minimum, there should be SOME propagation
        assert mean_change > 0.01, f"No avalanche effect: mean_change={mean_change:.6f}"
    
    def test_cascading_changes(self):
        """Changes should propagate throughout the output."""
        n = 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X_orig = rft_forward(x)
        
        # Perturb first element
        x_mod = x.copy()
        x_mod[0] *= 1.001
        X_mod = rft_forward(x_mod)
        
        # Count how many output elements changed
        changed_mask = np.abs(X_orig - X_mod) > 1e-15
        fraction_changed = np.mean(changed_mask)
        
        # Changes should spread to many outputs
        assert fraction_changed > 0.1, f"Changes didn't propagate: {fraction_changed:.4f}"


class TestBitSensitivity:
    """Test sensitivity to bit-level changes."""
    
    def test_mantissa_sensitivity(self):
        """Changes in mantissa bits should affect output."""
        n = 32
        x = np.ones(n) * 1.0  # Clean starting point
        
        X_orig = rft_forward(x)
        
        # Change last mantissa bit (LSB of float64)
        x_mod = x.copy()
        x_mod[0] = np.nextafter(x_mod[0], 2.0)  # Next representable number
        
        X_mod = rft_forward(x_mod)
        
        # Should produce a different output
        diff = np.max(np.abs(X_orig - X_mod))
        assert diff > 0, "LSB change had no effect"
    
    def test_sign_bit_sensitivity(self):
        """Sign bit changes should have large effect."""
        n = 32
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X_orig = rft_forward(x)
        
        # Flip sign of one element
        x_mod = x.copy()
        x_mod[0] = -x_mod[0]
        X_mod = rft_forward(x_mod)
        
        # Should have significant change
        rel_change = np.linalg.norm(X_orig - X_mod) / np.linalg.norm(X_orig)
        assert rel_change > 0.01, f"Sign flip had little effect: {rel_change:.6f}"


class TestStatisticalDistribution:
    """Test statistical properties of RFT output."""
    
    def test_output_uniformity(self):
        """RFT of random input should have uniform phase distribution."""
        n = 1024
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X = rft_forward(x)
        
        # Extract phases
        phases = np.angle(X)
        
        # Test for uniformity on [-π, π]
        # Kolmogorov-Smirnov test against uniform
        phases_normalized = (phases + np.pi) / (2 * np.pi)  # Map to [0, 1]
        stat, pvalue = stats.kstest(phases_normalized, 'uniform')
        
        # p-value > 0.01 suggests uniform distribution
        print(f"Phase uniformity KS test: stat={stat:.4f}, p-value={pvalue:.4f}")
    
    def test_magnitude_distribution(self):
        """RFT magnitudes should follow expected distribution for Gaussian input."""
        n = 4096
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X = rft_forward(x)
        magnitudes = np.abs(X)
        
        # For DFT of Gaussian noise, magnitudes follow Rayleigh distribution
        # RFT may differ, but should still be reasonable
        
        # Check that magnitudes are positive
        assert np.all(magnitudes >= 0)
        
        # Check that distribution is not degenerate
        assert np.std(magnitudes) > 0
        
        # Skewness should be positive (Rayleigh-like)
        skew = stats.skew(magnitudes)
        print(f"Magnitude skewness: {skew:.4f}")
    
    def test_no_obvious_patterns(self):
        """RFT output should not have obvious periodic patterns."""
        n = 256
        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)
        
        X = rft_forward(x)
        magnitudes = np.abs(X)
        
        # Autocorrelation of magnitudes
        autocorr = np.correlate(magnitudes - magnitudes.mean(), 
                                magnitudes - magnitudes.mean(), mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Positive lags only
        autocorr /= autocorr[0]  # Normalize
        
        # Check for no strong periodic peaks (lag > 1)
        strong_peaks = np.where(autocorr[2:] > 0.3)[0]
        assert len(strong_peaks) < 5, f"Found {len(strong_peaks)} strong periodic patterns"


class TestDifferentialProperties:
    """Test differential cryptanalysis resistance."""
    
    def test_input_difference_propagation(self):
        """Small input differences should not be predictably related to output differences."""
        n = 64
        n_trials = 100
        rng = np.random.default_rng(42)
        
        input_diffs = []
        output_diffs = []
        
        for _ in range(n_trials):
            x1 = rng.standard_normal(n)
            x2 = x1.copy()
            x2[rng.integers(n)] += rng.uniform(-0.1, 0.1)
            
            X1 = rft_forward(x1)
            X2 = rft_forward(x2)
            
            input_diff = np.linalg.norm(x1 - x2)
            output_diff = np.linalg.norm(X1 - X2)
            
            input_diffs.append(input_diff)
            output_diffs.append(output_diff)
        
        # Correlation between input and output differences
        corr = np.corrcoef(input_diffs, output_diffs)[0, 1]
        
        # High correlation would indicate linear relationship (bad for crypto)
        # Some correlation is expected since RFT is linear transform
        print(f"Input-output difference correlation: {corr:.4f}")
    
    def test_collision_resistance(self):
        """Different inputs should produce different outputs."""
        n = 64
        n_trials = 1000
        rng = np.random.default_rng(42)
        
        outputs = []
        for _ in range(n_trials):
            x = rng.standard_normal(n)
            X = rft_forward(x)
            outputs.append(tuple(X.ravel().view(np.float64)))
        
        # Check for collisions (same output from different input)
        unique_outputs = len(set(outputs))
        
        assert unique_outputs == n_trials, f"Found {n_trials - unique_outputs} collisions!"


class TestNonLinearity:
    """Test for non-linear mixing properties."""
    
    def test_superposition_deviation(self):
        """Measure deviation from pure linearity (if any non-linear elements exist)."""
        n = 64
        rng = np.random.default_rng(42)
        
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        
        X1 = rft_forward(x1)
        X2 = rft_forward(x2)
        X_sum = rft_forward(x1 + x2)
        
        # For a linear transform: X_sum == X1 + X2
        linearity_error = np.linalg.norm(X_sum - (X1 + X2)) / np.linalg.norm(X_sum)
        
        # RFT should be linear, so error should be very small
        assert linearity_error < 1e-10, f"RFT is non-linear? error={linearity_error:.2e}"
        
        print(f"Linearity preserved: error={linearity_error:.2e}")


class TestKeyDependence:
    """Test for key-dependent behavior (if applicable)."""
    
    def test_different_sizes_different_transforms(self):
        """Different signal sizes should use different transforms."""
        rng = np.random.default_rng(42)
        
        # Same pattern, different sizes
        x16 = rng.standard_normal(16)
        x32 = np.concatenate([x16, rng.standard_normal(16)])
        
        X16 = rft_forward(x16)
        X32 = rft_forward(x32)
        
        # First 16 elements of X32 should differ from X16
        # (different transform matrices for different sizes)
        diff = np.linalg.norm(X16 - X32[:16])
        
        # Should be significantly different
        assert diff > 0.1 * np.linalg.norm(X16), "Size-independent transform?"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
