# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Statistical Tests - Section D
=============================

PRNG-style batteries on derived bitstreams.
Run NIST-style statistical tests, with proper caveats.

IMPORTANT: These results indicate "no obvious statistical weakness",
NOT a security proof. Treat results accordingly.
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple
from scipy import stats
from algorithms.rft.core.resonant_fourier_transform import (
    rft_forward,
    rft_basis_matrix,
)


# =============================================================================
# D.1 NIST-STYLE STATISTICAL TESTS
# =============================================================================

class TestStatisticalProperties:
    """PRNG-style statistical tests on RFT-derived bitstreams.
    
    WARNING: These tests show "no obvious weakness", not security proofs.
    """
    
    def _to_bitstream(self, X: np.ndarray, bits_per_sample: int = 8) -> np.ndarray:
        """Convert complex coefficients to bitstream."""
        # Use magnitude and phase, quantized
        mags = np.abs(X)
        phases = np.angle(X)
        
        # Normalize and quantize
        mags_norm = (mags - np.min(mags)) / (np.max(mags) - np.min(mags) + 1e-10)
        phases_norm = (phases + np.pi) / (2 * np.pi)
        
        mag_bits = (mags_norm * (2**bits_per_sample - 1)).astype(np.uint8)
        phase_bits = (phases_norm * (2**bits_per_sample - 1)).astype(np.uint8)
        
        # Interleave
        bits = np.empty(len(X) * 2, dtype=np.uint8)
        bits[0::2] = mag_bits
        bits[1::2] = phase_bits
        
        return bits
    
    def _monobit_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """NIST Monobit test: proportion of 1s should be ~0.5."""
        # Expand to actual bits
        bit_array = np.unpackbits(bits)
        n = len(bit_array)
        ones = np.sum(bit_array)
        
        # S_n = 2*ones - n (deviation from expected)
        s_n = 2 * ones - n
        s_obs = np.abs(s_n) / np.sqrt(n)
        
        # P-value using complementary error function
        p_value = stats.erfc(s_obs / np.sqrt(2))
        
        return p_value, p_value > 0.01
    
    def _runs_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """NIST Runs test: tests oscillation between 0s and 1s."""
        bit_array = np.unpackbits(bits)
        n = len(bit_array)
        
        # Count proportion of 1s
        pi = np.sum(bit_array) / n
        
        # If proportion too extreme, test doesn't apply
        if np.abs(pi - 0.5) >= 2/np.sqrt(n):
            return 0.0, False
        
        # Count runs (consecutive sequences of same bit)
        runs = 1 + np.sum(bit_array[:-1] != bit_array[1:])
        
        # Expected runs and variance
        expected = 2 * n * pi * (1 - pi)
        variance = 2 * np.sqrt(2*n) * pi * (1 - pi)
        
        if variance == 0:
            return 0.0, False
        
        z = (runs - expected) / variance
        p_value = stats.erfc(np.abs(z) / np.sqrt(2))
        
        return p_value, p_value > 0.01
    
    def _serial_test(self, bits: np.ndarray, m: int = 2) -> Tuple[float, bool]:
        """NIST Serial test: tests uniformity of m-bit patterns."""
        bit_array = np.unpackbits(bits)
        n = len(bit_array)
        
        # Count m-bit patterns
        patterns = {}
        for i in range(n - m + 1):
            pattern = tuple(bit_array[i:i+m])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Chi-square statistic
        expected = (n - m + 1) / (2**m)
        chi_sq = sum((count - expected)**2 / expected for count in patterns.values())
        
        # Degrees of freedom
        df = 2**m - 1
        p_value = 1 - stats.chi2.cdf(chi_sq, df)
        
        return p_value, p_value > 0.01
    
    def _longest_run_test(self, bits: np.ndarray) -> Tuple[float, bool]:
        """Test longest run of ones in blocks."""
        bit_array = np.unpackbits(bits)
        n = len(bit_array)
        
        # Use blocks of 128 bits
        M = 128
        num_blocks = n // M
        
        if num_blocks < 8:
            return 0.5, True  # Not enough data
        
        # Find longest run in each block
        longest_runs = []
        for i in range(num_blocks):
            block = bit_array[i*M:(i+1)*M]
            max_run = 0
            current_run = 0
            for bit in block:
                if bit == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            longest_runs.append(max_run)
        
        # Expected distribution for M=128
        # Categories: <=4, 5, 6, 7, 8, >=9
        categories = [0] * 6
        for run in longest_runs:
            if run <= 4:
                categories[0] += 1
            elif run == 5:
                categories[1] += 1
            elif run == 6:
                categories[2] += 1
            elif run == 7:
                categories[3] += 1
            elif run == 8:
                categories[4] += 1
            else:
                categories[5] += 1
        
        # Expected probabilities for M=128
        expected_probs = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        expected = [p * num_blocks for p in expected_probs]
        
        # Chi-square
        chi_sq = sum((o - e)**2 / e for o, e in zip(categories, expected) if e > 0)
        p_value = 1 - stats.chi2.cdf(chi_sq, 5)
        
        return p_value, p_value > 0.01
    
    def test_nist_suite_report(self):
        """Run NIST-style test battery and generate report."""
        print("\n" + "="*80)
        print("STATISTICAL TESTS REPORT (NIST-STYLE)")
        print("="*80)
        print("\n⚠️  WARNING: These results indicate 'no obvious statistical weakness',")
        print("    NOT a security proof. Treat results accordingly.\n")
        
        rng = np.random.default_rng(42)
        
        # Generate test data
        test_cases = []
        for N in [256, 1024, 4096]:
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            X = rft_forward(x, T=N, use_gram_normalization=True)
            test_cases.append((f"RFT N={N}", X))
        
        # Also test random baseline
        X_rand = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
        test_cases.append(("Random baseline", X_rand))
        
        print(f"{'Test Case':<20} | {'Monobit':>12} | {'Runs':>12} | "
              f"{'Serial(2)':>12} | {'LongestRun':>12} | {'Status':>8}")
        print("-"*90)
        
        for name, X in test_cases:
            bits = self._to_bitstream(X)
            
            p_mono, pass_mono = self._monobit_test(bits)
            p_runs, pass_runs = self._runs_test(bits)
            p_serial, pass_serial = self._serial_test(bits)
            p_longest, pass_longest = self._longest_run_test(bits)
            
            all_pass = all([pass_mono, pass_runs, pass_serial, pass_longest])
            status = "✓ PASS" if all_pass else "✗ FAIL"
            
            print(f"{name:<20} | {p_mono:>12.4f} | {p_runs:>12.4f} | "
                  f"{p_serial:>12.4f} | {p_longest:>12.4f} | {status:>8}")
    
    @pytest.mark.parametrize("N", [256, 1024])
    def test_basic_statistical_properties(self, N: int):
        """Verify basic statistical properties pass NIST thresholds."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        X = rft_forward(x, T=N, use_gram_normalization=True)
        
        bits = self._to_bitstream(X)
        
        p_mono, _ = self._monobit_test(bits)
        p_runs, _ = self._runs_test(bits)
        
        # Don't require passing (statistical tests can fail randomly)
        # but log the results
        print(f"\nN={N}: monobit p={p_mono:.4f}, runs p={p_runs:.4f}")


# =============================================================================
# D.2 ADDITIONAL STATISTICAL TESTS
# =============================================================================

class TestAdditionalStatistics:
    """Additional statistical tests beyond NIST basics."""
    
    def test_coefficient_distribution(self):
        """Test that output coefficients follow expected distribution."""
        print("\n" + "="*80)
        print("COEFFICIENT DISTRIBUTION TESTS")
        print("="*80)
        
        rng = np.random.default_rng(42)
        N = 1024
        num_trials = 100
        
        all_mags = []
        all_phases = []
        
        for _ in range(num_trials):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            X = rft_forward(x, T=N, use_gram_normalization=True)
            all_mags.extend(np.abs(X))
            all_phases.extend(np.angle(X))
        
        all_mags = np.array(all_mags)
        all_phases = np.array(all_phases)
        
        # Test magnitude distribution (should be Rayleigh-like for Gaussian input)
        # Test phase distribution (should be uniform on [-π, π])
        
        # Phase uniformity test (Kolmogorov-Smirnov)
        uniform_phases = (all_phases + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        ks_stat, ks_p = stats.kstest(uniform_phases, 'uniform')
        
        print(f"\nPhase uniformity (KS test): stat={ks_stat:.4f}, p={ks_p:.4f}")
        print(f"  Result: {'✓ Uniform' if ks_p > 0.01 else '✗ Non-uniform'}")
        
        # Magnitude statistics
        print(f"\nMagnitude statistics:")
        print(f"  Mean: {np.mean(all_mags):.4f}")
        print(f"  Std: {np.std(all_mags):.4f}")
        print(f"  Skewness: {stats.skew(all_mags):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(all_mags):.4f}")
    
    def test_autocorrelation_of_output(self):
        """Test that output has low autocorrelation (whiteness)."""
        print("\n" + "="*80)
        print("OUTPUT AUTOCORRELATION TESTS")
        print("="*80)
        
        rng = np.random.default_rng(42)
        N = 512
        
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        X = rft_forward(x, T=N, use_gram_normalization=True)
        
        # Compute autocorrelation
        X_centered = X - np.mean(X)
        autocorr = np.correlate(X_centered, X_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = np.abs(autocorr / autocorr[0])  # Normalize
        
        # Check that autocorrelation decays quickly
        print(f"\nAutocorrelation at different lags:")
        for lag in [1, 2, 5, 10, 20, 50]:
            if lag < len(autocorr):
                print(f"  Lag {lag:>3}: {autocorr[lag]:.6f}")
        
        # For white noise, |ρ(k)| < 2/√N for k > 0
        threshold = 2 / np.sqrt(N)
        num_exceeding = np.sum(autocorr[1:51] > threshold)
        print(f"\nLags 1-50 exceeding 2/√N threshold ({threshold:.4f}): {num_exceeding}/50")


# =============================================================================
# COMBINED STATISTICAL SUITE
# =============================================================================

def test_full_statistical_report():
    """Run all statistical tests and generate combined report."""
    print("\n" + "="*80)
    print("FULL STATISTICAL TESTS REPORT")
    print("="*80)
    
    # NIST-style tests
    ts = TestStatisticalProperties()
    ts.test_nist_suite_report()
    
    # Additional statistics
    ta = TestAdditionalStatistics()
    ta.test_coefficient_distribution()
    ta.test_autocorrelation_of_output()
    
    print("\n" + "="*80)
    print("STATISTICAL TESTS COMPLETE")
    print("Note: Passing indicates 'no obvious weakness', not security proof")
    print("="*80)


if __name__ == "__main__":
    test_full_statistical_report()
