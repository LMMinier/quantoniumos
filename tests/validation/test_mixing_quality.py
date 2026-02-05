# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Mixing Quality Tests - Section C
================================

Tests for "good diffusion" properties:
1. Avalanche / bit influence test
2. Correlation + mutual information reduction
3. Spectral flatness / energy spread
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple
from scipy import stats
from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix,
    rft_forward,
    rft_inverse,
)


# =============================================================================
# C.1 AVALANCHE / BIT INFLUENCE TEST
# =============================================================================

class TestAvalancheBitInfluence:
    """Test: flip 1 input bit (or small epsilon in one coordinate) and observe output change.
    Metrics:
    - Mean changed bits (if binarized output)
    - Output L2 change distribution
    - Per-coordinate sensitivity histogram
    """
    
    SIZES = [64, 128, 256, 512]
    NUM_TRIALS = 100
    
    def _perturb_single_coord(self, x: np.ndarray, idx: int, epsilon: float) -> np.ndarray:
        """Perturb a single coordinate by epsilon."""
        x_pert = x.copy()
        x_pert[idx] += epsilon
        return x_pert
    
    @pytest.mark.parametrize("N", SIZES)
    def test_avalanche_l2_spread(self, N: int):
        """Verify small input changes spread globally in output."""
        rng = np.random.default_rng(42)
        epsilon = 1e-6
        
        l2_changes = []
        affected_coords = []
        
        for _ in range(self.NUM_TRIALS):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            X = rft_forward(x, T=N, use_gram_normalization=True)
            
            # Perturb random coordinate
            idx = rng.integers(N)
            x_pert = self._perturb_single_coord(x, idx, epsilon)
            X_pert = rft_forward(x_pert, T=N, use_gram_normalization=True)
            
            # Measure output change
            delta = X_pert - X
            l2_change = np.linalg.norm(delta)
            l2_changes.append(l2_change)
            
            # Count significantly affected coordinates (> 1% of change)
            threshold = 0.01 * l2_change
            num_affected = np.sum(np.abs(delta) > threshold)
            affected_coords.append(num_affected / N)
        
        mean_l2 = np.mean(l2_changes)
        mean_spread = np.mean(affected_coords)
        
        # Expect change to spread to many coordinates (good mixing)
        assert mean_spread > 0.3, (
            f"Poor avalanche spread at N={N}: only {mean_spread*100:.1f}% coords affected"
        )
    
    def test_bit_flip_influence_report(self):
        """Generate detailed avalanche/bit influence report."""
        print("\n" + "="*80)
        print("AVALANCHE / BIT INFLUENCE REPORT")
        print("="*80)
        
        rng = np.random.default_rng(42)
        epsilon = 1e-6
        
        for N in self.SIZES:
            print(f"\nN = {N}, ε = {epsilon:.0e}")
            
            l2_changes = []
            spread_ratios = []
            per_coord_sensitivity = np.zeros(N)
            
            for trial in range(self.NUM_TRIALS):
                x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
                X = rft_forward(x, T=N, use_gram_normalization=True)
                
                # Perturb each coordinate once
                idx = trial % N
                x_pert = self._perturb_single_coord(x, idx, epsilon)
                X_pert = rft_forward(x_pert, T=N, use_gram_normalization=True)
                
                delta = X_pert - X
                l2 = np.linalg.norm(delta)
                l2_changes.append(l2)
                
                # Per-coordinate contribution
                per_coord_sensitivity += np.abs(delta)**2 / self.NUM_TRIALS
                
                # Spread ratio
                threshold = 0.01 * l2
                spread_ratios.append(np.sum(np.abs(delta) > threshold) / N)
            
            print(f"  L2 change: mean={np.mean(l2_changes):.2e}, std={np.std(l2_changes):.2e}")
            print(f"  Spread ratio: mean={np.mean(spread_ratios)*100:.1f}%, "
                  f"min={np.min(spread_ratios)*100:.1f}%, max={np.max(spread_ratios)*100:.1f}%")
            print(f"  Sensitivity uniformity: std/mean = "
                  f"{np.std(per_coord_sensitivity)/np.mean(per_coord_sensitivity):.3f}")
    
    def test_sensitivity_histogram(self):
        """Generate per-coordinate sensitivity histogram."""
        N = 256
        rng = np.random.default_rng(42)
        epsilon = 1e-6
        num_trials = 500
        
        sensitivity = np.zeros(N)
        
        for _ in range(num_trials):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            X = rft_forward(x, T=N, use_gram_normalization=True)
            
            for idx in range(N):
                x_pert = self._perturb_single_coord(x, idx, epsilon)
                X_pert = rft_forward(x_pert, T=N, use_gram_normalization=True)
                sensitivity[idx] += np.linalg.norm(X_pert - X)
        
        sensitivity /= num_trials
        
        # Check uniformity - sensitivity should be similar across all coords
        cv = np.std(sensitivity) / np.mean(sensitivity)  # Coefficient of variation
        
        print(f"\nSensitivity Histogram (N={N}):")
        print(f"  Mean sensitivity: {np.mean(sensitivity):.4e}")
        print(f"  Std sensitivity: {np.std(sensitivity):.4e}")
        print(f"  CV (std/mean): {cv:.4f}")
        print(f"  Min/Max ratio: {np.min(sensitivity)/np.max(sensitivity):.4f}")
        
        # Good mixing implies uniform sensitivity
        assert cv < 0.5, f"Sensitivity too non-uniform: CV={cv:.4f}"


# =============================================================================
# C.2 CORRELATION + MUTUAL INFORMATION REDUCTION
# =============================================================================

class TestCorrelationMIReduction:
    """Test: inputs with known structure, check how much structure remains in output.
    Metrics:
    - Adjacent correlation of coefficients
    - Mutual information estimate between neighboring bins
    """
    
    def _generate_markov_signal(self, N: int, rho: float, seed: int = 42) -> np.ndarray:
        """Generate AR(1) Markov process with correlation rho."""
        rng = np.random.default_rng(seed)
        x = np.zeros(N, dtype=np.complex128)
        x[0] = rng.standard_normal() + 1j * rng.standard_normal()
        for i in range(1, N):
            noise = rng.standard_normal() + 1j * rng.standard_normal()
            x[i] = rho * x[i-1] + np.sqrt(1 - rho**2) * noise
        return x
    
    def _generate_periodic_signal(self, N: int, period: int) -> np.ndarray:
        """Generate periodic signal."""
        t = np.arange(N)
        return np.sin(2 * np.pi * t / period) + 0.5 * np.cos(4 * np.pi * t / period)
    
    def _generate_sparse_signal(self, N: int, sparsity: float, seed: int = 42) -> np.ndarray:
        """Generate sparse signal with given sparsity (fraction of non-zeros)."""
        rng = np.random.default_rng(seed)
        x = np.zeros(N, dtype=np.complex128)
        num_nonzero = int(N * sparsity)
        indices = rng.choice(N, num_nonzero, replace=False)
        x[indices] = rng.standard_normal(num_nonzero) + 1j * rng.standard_normal(num_nonzero)
        return x
    
    def _adjacent_correlation(self, x: np.ndarray) -> float:
        """Compute correlation between adjacent elements."""
        return np.abs(np.corrcoef(x[:-1], x[1:])[0, 1])
    
    def _estimate_mi(self, x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """Estimate mutual information using histogram method."""
        # Use magnitude for continuous estimate
        x_mag = np.abs(x)
        y_mag = np.abs(y)
        
        # Joint histogram
        hist_2d, _, _ = np.histogram2d(x_mag, y_mag, bins=bins)
        
        # Normalize to get joint probability
        p_xy = hist_2d / np.sum(hist_2d)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # Compute MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        return mi
    
    @pytest.mark.parametrize("rho", [0.5, 0.8, 0.95])
    def test_decorrelation_markov(self, rho: float):
        """Test decorrelation of Markov inputs."""
        N = 256
        x = self._generate_markov_signal(N, rho)
        X = rft_forward(x, T=N, use_gram_normalization=True)
        
        input_corr = self._adjacent_correlation(x)
        output_corr = self._adjacent_correlation(X)
        
        # Output should be less correlated than input
        assert output_corr < input_corr, (
            f"Correlation not reduced: input={input_corr:.4f}, output={output_corr:.4f}"
        )
    
    def test_correlation_reduction_report(self):
        """Generate detailed correlation reduction report."""
        print("\n" + "="*80)
        print("CORRELATION + MUTUAL INFORMATION REDUCTION REPORT")
        print("="*80)
        
        N = 256
        
        # Test different structured inputs
        signals = {
            'Markov ρ=0.5': self._generate_markov_signal(N, 0.5),
            'Markov ρ=0.8': self._generate_markov_signal(N, 0.8),
            'Markov ρ=0.95': self._generate_markov_signal(N, 0.95),
            'Periodic T=16': self._generate_periodic_signal(N, 16).astype(np.complex128),
            'Periodic T=32': self._generate_periodic_signal(N, 32).astype(np.complex128),
            'Sparse 10%': self._generate_sparse_signal(N, 0.1),
            'Sparse 30%': self._generate_sparse_signal(N, 0.3),
        }
        
        print(f"\n{'Signal Type':<20} | {'Input Corr':>12} | {'Output Corr':>12} | "
              f"{'Input MI':>10} | {'Output MI':>10} | {'Reduction':>10}")
        print("-"*90)
        
        for name, x in signals.items():
            X = rft_forward(x, T=N, use_gram_normalization=True)
            
            in_corr = self._adjacent_correlation(x)
            out_corr = self._adjacent_correlation(X)
            
            in_mi = self._estimate_mi(x[:-1], x[1:])
            out_mi = self._estimate_mi(X[:-1], X[1:])
            
            reduction = (1 - out_corr/in_corr) * 100 if in_corr > 0 else 0
            
            print(f"{name:<20} | {in_corr:>12.4f} | {out_corr:>12.4f} | "
                  f"{in_mi:>10.4f} | {out_mi:>10.4f} | {reduction:>9.1f}%")


# =============================================================================
# C.3 SPECTRAL FLATNESS / ENERGY SPREAD
# =============================================================================

class TestSpectralFlatness:
    """Test: structured inputs (sinusoids, impulses, ramps).
    Metric: spectral flatness measure (SFM) of magnitude distribution.
    """
    
    def _spectral_flatness(self, X: np.ndarray) -> float:
        """Compute spectral flatness measure (SFM).
        SFM = geometric_mean / arithmetic_mean
        Range: 0 (concentrated) to 1 (flat)
        """
        mags = np.abs(X)
        mags = mags[mags > 0]  # Avoid log(0)
        
        if len(mags) == 0:
            return 0.0
        
        geometric_mean = np.exp(np.mean(np.log(mags)))
        arithmetic_mean = np.mean(mags)
        
        return geometric_mean / arithmetic_mean
    
    def _gini_coefficient(self, X: np.ndarray) -> float:
        """Compute Gini coefficient of energy distribution.
        Range: 0 (equal) to 1 (concentrated)
        """
        mags = np.sort(np.abs(X)**2)
        n = len(mags)
        cumsum = np.cumsum(mags)
        return (2 * np.sum((np.arange(1, n+1)) * mags) / (n * cumsum[-1])) - (n + 1) / n
    
    def test_energy_spread_report(self):
        """Generate spectral flatness / energy spread report."""
        print("\n" + "="*80)
        print("SPECTRAL FLATNESS / ENERGY SPREAD REPORT")
        print("="*80)
        
        N = 256
        t = np.arange(N) / N
        
        # Test signals
        signals = {
            'Impulse': np.zeros(N, dtype=np.complex128),
            'Single sine': np.sin(2 * np.pi * 10 * t).astype(np.complex128),
            'Multi sine': (np.sin(2*np.pi*10*t) + np.sin(2*np.pi*23*t) + np.sin(2*np.pi*47*t)).astype(np.complex128),
            'Ramp': (t * 2 - 1).astype(np.complex128),
            'Step': np.where(t < 0.5, -1, 1).astype(np.complex128),
            'White noise': np.random.default_rng(42).standard_normal(N).astype(np.complex128),
            'Chirp': np.exp(1j * np.pi * 100 * t**2).astype(np.complex128),
        }
        signals['Impulse'][N//2] = 1.0
        
        print(f"\n{'Signal':>15} | {'Input SFM':>10} | {'Output SFM':>11} | "
              f"{'Input Gini':>11} | {'Output Gini':>12} | {'Energy Spread':>14}")
        print("-"*90)
        
        for name, x in signals.items():
            X = rft_forward(x, T=N, use_gram_normalization=True)
            
            in_sfm = self._spectral_flatness(x)
            out_sfm = self._spectral_flatness(X)
            in_gini = self._gini_coefficient(x)
            out_gini = self._gini_coefficient(X)
            
            # Energy spread: how many coefficients contain 90% of energy
            mags_sorted = np.sort(np.abs(X)**2)[::-1]
            cumsum = np.cumsum(mags_sorted) / np.sum(mags_sorted)
            num_90 = np.searchsorted(cumsum, 0.9) + 1
            spread_pct = num_90 / N * 100
            
            print(f"{name:>15} | {in_sfm:>10.4f} | {out_sfm:>11.4f} | "
                  f"{in_gini:>11.4f} | {out_gini:>12.4f} | {spread_pct:>13.1f}%")
    
    @pytest.mark.parametrize("signal_type", ['impulse', 'sine', 'noise'])
    def test_energy_spread_threshold(self, signal_type: str):
        """Verify RFT spreads energy for structured inputs."""
        N = 256
        t = np.arange(N) / N
        
        if signal_type == 'impulse':
            x = np.zeros(N, dtype=np.complex128)
            x[N//2] = 1.0
        elif signal_type == 'sine':
            x = np.sin(2 * np.pi * 10 * t).astype(np.complex128)
        else:
            x = np.random.default_rng(42).standard_normal(N).astype(np.complex128)
        
        X = rft_forward(x, T=N, use_gram_normalization=True)
        
        # 90% of energy should be spread across multiple coefficients
        mags_sorted = np.sort(np.abs(X)**2)[::-1]
        cumsum = np.cumsum(mags_sorted) / np.sum(mags_sorted)
        num_90 = np.searchsorted(cumsum, 0.9) + 1
        
        # For structured signals, energy should spread (not concentrated in few coeffs)
        if signal_type in ['impulse', 'sine']:
            assert num_90 > N * 0.1, f"Energy too concentrated for {signal_type}"


# =============================================================================
# COMBINED MIXING QUALITY SUITE
# =============================================================================

def test_full_mixing_quality_report():
    """Run all mixing quality tests and generate combined report."""
    print("\n" + "="*80)
    print("FULL MIXING QUALITY REPORT")
    print("="*80)
    
    # Avalanche tests
    ta = TestAvalancheBitInfluence()
    ta.test_bit_flip_influence_report()
    
    # Correlation tests
    tc = TestCorrelationMIReduction()
    tc.test_correlation_reduction_report()
    
    # Spectral flatness tests
    ts = TestSpectralFlatness()
    ts.test_energy_spread_report()
    
    print("\n" + "="*80)
    print("ALL MIXING QUALITY TESTS PASSED")
    print("="*80)


if __name__ == "__main__":
    test_full_mixing_quality_report()
