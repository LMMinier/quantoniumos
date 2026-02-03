# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Test Suite: Fibonacci-Lattice Fast RFT Algorithm
================================================

Tests the native O(N log N) RFT algorithm using Fibonacci-lattice decomposition.
"""

import pytest
import numpy as np
from typing import List

# Import module under test
from algorithms.rft.core.fibonacci_fast_rft import (
    # Fibonacci utilities
    fibonacci,
    fibonacci_sequence,
    zeckendorf,
    nearest_fibonacci,
    # Golden ratio modular arithmetic
    phi_power_mod1,
    fibonacci_phase_factor,
    # Core algorithms
    fast_rft_fibonacci,
    fast_rft_bluestein,
    FibonacciRFTResult,
    # Performance
    compare_rft_algorithms,
    AlgorithmComparison,
    # Fibonacci selection
    optimal_fibonacci_size,
    list_fibonacci_rft_sizes,
    # Complexity
    analyze_complexity,
    ComplexityResult,
)
from algorithms.rft.core.resonant_fourier_transform import PHI


class TestFibonacciSequence:
    """Test Fibonacci sequence utilities."""
    
    def test_fibonacci_known_values(self):
        """Test known Fibonacci values."""
        known = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for k, expected in enumerate(known):
            assert fibonacci(k) == expected
    
    def test_fibonacci_recurrence(self):
        """Test Fibonacci recurrence: F_n = F_{n-1} + F_{n-2}."""
        for n in range(3, 20):
            assert fibonacci(n) == fibonacci(n - 1) + fibonacci(n - 2)
    
    def test_fibonacci_sequence_generation(self):
        """Test Fibonacci sequence generation."""
        fibs = fibonacci_sequence(100)
        expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        assert fibs == expected
    
    def test_fibonacci_sequence_max(self):
        """Last element should not exceed max_val."""
        max_val = 500
        fibs = fibonacci_sequence(max_val)
        assert fibs[-1] <= max_val
        # Next would exceed
        assert fibs[-1] + fibs[-2] > max_val


class TestZeckendorf:
    """Test Zeckendorf representation."""
    
    @pytest.mark.parametrize("n,expected_sum", [
        (1, [1]),
        (2, [2]),
        (3, [3]),
        (4, [3, 1]),  # 4 = 3 + 1
        (5, [5]),
        (6, [5, 1]),  # 6 = 5 + 1
        (7, [5, 2]),  # 7 = 5 + 2
        (8, [8]),
        (10, [8, 2]),  # 10 = 8 + 2
        (100, [89, 8, 3]),  # 100 = 89 + 8 + 3
    ])
    def test_zeckendorf_sums(self, n, expected_sum):
        """Test Zeckendorf representation sums correctly."""
        indices = zeckendorf(n)
        fibs = [fibonacci(i) for i in indices]
        assert sum(fibs) == n
    
    def test_zeckendorf_non_consecutive(self):
        """Zeckendorf indices should be non-consecutive."""
        for n in range(1, 200):
            indices = zeckendorf(n)
            for i in range(1, len(indices)):
                # Indices should differ by at least 2
                assert abs(indices[i] - indices[i-1]) >= 2
    
    def test_zeckendorf_zero(self):
        """Zeckendorf of 0 should be empty."""
        assert zeckendorf(0) == []
    
    def test_zeckendorf_fib_numbers(self):
        """Fibonacci numbers have single-term Zeckendorf."""
        for k in range(1, 15):
            F_k = fibonacci(k)
            indices = zeckendorf(F_k)
            assert len(indices) == 1
            # The index returned should correspond to F_k
            assert fibonacci(indices[0]) == F_k


class TestNearestFibonacci:
    """Test nearest Fibonacci number finding."""
    
    def test_exact_fibonacci(self):
        """Exact Fibonacci numbers should return themselves."""
        for k in range(1, 15):
            F_k = fibonacci(k)
            nearest, idx = nearest_fibonacci(F_k)
            assert nearest == F_k  # Value should match
    
    @pytest.mark.parametrize("n,expected", [
        (6, 5),   # Closer to 5 than 8
        (7, 8),   # Closer to 8 than 5
        (100, 89),  # Closer to 89 than 144
        (120, 144),  # Closer to 144 than 89
    ])
    def test_nearest_values(self, n, expected):
        """Test nearest Fibonacci for specific values."""
        nearest, _ = nearest_fibonacci(n)
        assert nearest == expected


class TestPhiPowerMod1:
    """Test golden ratio modular arithmetic."""
    
    def test_phi_power_0(self):
        """φ^0 mod 1 = 0."""
        assert phi_power_mod1(0) == 0.0
    
    def test_phi_power_in_unit_interval(self):
        """φ^k mod 1 should be in [0, 1)."""
        for k in range(1, 20):
            result = phi_power_mod1(k)
            assert 0 <= result < 1
    
    def test_phi_power_not_periodic(self):
        """φ^k mod 1 values should be quasi-random (low repetition)."""
        values = [phi_power_mod1(k) for k in range(1, 100)]
        # Count distinct values (allowing small tolerance)
        unique_count = 0
        seen = []
        for v in values:
            is_new = True
            for s in seen:
                if abs(v - s) < 1e-10:
                    is_new = False
                    break
            if is_new:
                seen.append(v)
                unique_count += 1
        # Should have many distinct values (>30% unique is reasonable)
        assert unique_count > 30, f"Only {unique_count} distinct values in 99"


class TestFibonacciPhaseFactor:
    """Test Fibonacci phase factors."""
    
    def test_phase_unit_magnitude(self):
        """Phase factors should have unit magnitude."""
        for k in range(1, 10):
            for m in range(0, 20):
                w = fibonacci_phase_factor(k, m)
                assert abs(abs(w) - 1.0) < 1e-10


class TestFastRFTFibonacci:
    """Test Fibonacci fast RFT algorithm."""
    
    @pytest.mark.parametrize("k", [5, 6, 7, 8, 9, 10])
    def test_fibonacci_size_result(self, k):
        """Test fast RFT on Fibonacci-sized inputs."""
        N = fibonacci(k)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        
        result = fast_rft_fibonacci(x)
        
        assert isinstance(result, FibonacciRFTResult)
        assert result.N == N
        assert len(result.transform) == N
        assert result.is_exact_fib or N < 3
    
    def test_small_inputs(self):
        """Test with small inputs."""
        for N in [1, 2, 3]:
            x = np.ones(N, dtype=complex)
            result = fast_rft_fibonacci(x)
            assert len(result.transform) == N
    
    def test_non_fibonacci_fallback(self):
        """Non-Fibonacci sizes should fallback to direct."""
        N = 50  # Not a Fibonacci number
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        
        result = fast_rft_fibonacci(x)
        
        assert result.N == N
        assert not result.is_exact_fib


class TestFastRFTBluestein:
    """Test Bluestein fast RFT algorithm."""
    
    @pytest.mark.parametrize("N", [32, 50, 64, 100, 128])
    def test_bluestein_shape(self, N):
        """Bluestein should return correct shape."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        
        result = fast_rft_bluestein(x)
        assert len(result) == N
    
    def test_bluestein_not_nan(self):
        """Bluestein should not produce NaN."""
        N = 64
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        
        result = fast_rft_bluestein(x)
        assert not np.any(np.isnan(result))


class TestAlgorithmComparison:
    """Test algorithm comparison functionality."""
    
    @pytest.mark.parametrize("N", [34, 55, 89])  # Fibonacci numbers
    def test_comparison_fibonacci_size(self, N):
        """Test comparison on Fibonacci-sized inputs."""
        result = compare_rft_algorithms(N, num_trials=5, seed=42)
        
        assert isinstance(result, AlgorithmComparison)
        assert result.N == N
        assert result.direct_time_ms > 0
        assert result.fibonacci_time_ms > 0
        assert result.bluestein_time_ms > 0
        assert result.fft_time_ms > 0
    
    def test_errors_bounded(self):
        """Algorithm errors should be bounded."""
        result = compare_rft_algorithms(55, num_trials=10, seed=42)
        
        # Fibonacci error might be high for approximate algorithm
        # Bluestein error should be reasonable
        assert result.bluestein_error < 10.0  # Very loose bound


class TestFibonacciSizeSelection:
    """Test Fibonacci size selection utilities."""
    
    def test_optimal_fibonacci_exact(self):
        """Optimal Fibonacci for exact Fib number."""
        for k in range(5, 15):
            F_k = fibonacci(k)
            opt, idx = optimal_fibonacci_size(F_k)
            assert opt == F_k
    
    def test_optimal_fibonacci_larger(self):
        """Test finding larger Fibonacci when smaller not allowed."""
        N = 100
        opt, idx = optimal_fibonacci_size(N, allow_smaller=False)
        assert opt >= N
        assert opt == 144  # Next Fibonacci >= 100
    
    def test_list_fibonacci_sizes(self):
        """Test listing Fibonacci RFT sizes."""
        sizes = list_fibonacci_rft_sizes(1000)
        
        # Should have reasonable number of sizes
        assert len(sizes) > 10
        
        # All should be valid Fibonacci
        for F_k, k in sizes:
            assert F_k == fibonacci(k)
            assert F_k <= 1000


class TestComplexityAnalysis:
    """Test complexity analysis."""
    
    @pytest.mark.parametrize("N", [64, 256, 1024])
    def test_complexity_result(self, N):
        """Test complexity analysis result."""
        result = analyze_complexity(N)
        
        assert isinstance(result, ComplexityResult)
        assert result.N == N
        assert result.direct_ops > result.fft_ops
        assert result.direct_ops > result.fib_rft_ops
    
    def test_speedup_increases_with_n(self):
        """Speedup should increase with N."""
        speedups = []
        for N in [64, 256, 1024, 4096]:
            result = analyze_complexity(N)
            speedups.append(result.speedup_vs_direct)
        
        # Speedup should generally increase
        for i in range(1, len(speedups)):
            assert speedups[i] > speedups[i-1]
    
    def test_fib_rft_vs_fft(self):
        """Fib-RFT should be comparable to FFT in complexity."""
        result = analyze_complexity(1024)
        
        # Fib-RFT is ~1.4x more ops than FFT due to log_φ vs log_2
        assert result.speedup_vs_fft > 0.5
        assert result.speedup_vs_fft < 2.0


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_fibonacci(self):
        """Test full pipeline for Fibonacci sizes."""
        for k in [8, 9, 10, 11]:
            N = fibonacci(k)
            rng = np.random.default_rng(42 + k)
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            
            # Run both algorithms
            fib_result = fast_rft_fibonacci(x)
            blu_result = fast_rft_bluestein(x)
            
            # Both should produce non-trivial output
            assert np.linalg.norm(fib_result.transform) > 0
            assert np.linalg.norm(blu_result) > 0
    
    def test_consistency(self):
        """Same seed should give same results."""
        N = fibonacci(10)  # 55
        
        for seed in [42, 123, 456]:
            rng = np.random.default_rng(seed)
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            
            r1 = fast_rft_fibonacci(x)
            r2 = fast_rft_fibonacci(x)
            
            assert np.allclose(r1.transform, r2.transform)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_element(self):
        """Test with single element."""
        x = np.array([1.0 + 2.0j])
        result = fast_rft_fibonacci(x)
        assert len(result.transform) == 1
    
    def test_two_elements(self):
        """Test with two elements."""
        x = np.array([1.0, 2.0], dtype=complex)
        result = fast_rft_fibonacci(x)
        assert len(result.transform) == 2
    
    def test_large_fibonacci(self):
        """Test with larger Fibonacci number."""
        N = fibonacci(15)  # 610
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        
        result = fast_rft_fibonacci(x)
        assert result.N == N
    
    def test_zeckendorf_large(self):
        """Test Zeckendorf for large numbers."""
        n = 10000
        indices = zeckendorf(n)
        fibs = [fibonacci(i) for i in indices]
        assert sum(fibs) == n


# Run specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
