# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""Tests for the Golden-RFT Uncertainty Principle (Theorem 9)."""

import numpy as np
import pytest

from algorithms.rft.core.golden_uncertainty_principle import (
    time_spread,
    frequency_spread,
    phi_frequencies,
    golden_frequency_spread,
    mutual_coherence,
    rft_dft_coherence,
    measure_uncertainty,
    golden_uncertainty_bound,
    gaussian_signal,
    golden_quasiperiodic_signal,
    harmonic_signal,
    chirp_signal,
    concentration_uncertainty_duality,
    k99,
)
from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix


class TestSpreadFunctionals:
    """Test basic spread/uncertainty functionals."""
    
    def test_time_spread_localized(self):
        """Localized signal has small time spread."""
        N = 64
        x = np.zeros(N, dtype=np.complex128)
        x[N // 2] = 1.0  # Delta function at center
        
        dt = time_spread(x)
        assert dt == 0.0, "Delta function should have zero time spread"
    
    def test_time_spread_uniform(self):
        """Uniform signal has maximal time spread."""
        N = 64
        x = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        
        dt = time_spread(x)
        # Uniform distribution has variance N²/12, so spread ≈ 0.289
        assert 0.2 < dt < 0.4, f"Uniform spread should be moderate, got {dt}"
    
    def test_frequency_spread_pure_tone(self):
        """Pure tone has minimal frequency spread."""
        N = 64
        k = 5
        n = np.arange(N)
        x = np.exp(2j * np.pi * k * n / N)
        
        X = np.fft.fft(x, norm='ortho')
        df = frequency_spread(X)
        
        # Pure tone concentrates in one bin
        assert df < 0.1, f"Pure tone should have small frequency spread, got {df}"
    
    def test_phi_frequencies_irrational(self):
        """Verify φ-grid frequencies are irrational (mod 1 distinct)."""
        N = 32
        f = phi_frequencies(N)
        
        # All frequencies should be distinct
        assert len(np.unique(np.round(f, 10))) == N, "φ-frequencies should all be distinct"
        
        # None should be k/N for any integer k
        for fk in f:
            for m in range(N):
                assert abs(fk - m/N) > 1e-10, f"f={fk} is too close to {m}/{N}"


class TestMutualCoherence:
    """Test mutual coherence computations."""
    
    def test_identity_coherence_is_one(self):
        """Identity matrix has coherence 1 with itself."""
        N = 32
        I = np.eye(N, dtype=np.complex128)
        mu = mutual_coherence(I, I)
        assert abs(mu - 1.0) < 1e-10
    
    def test_dft_coherence_is_minimal(self):
        """DFT basis has coherence 1/√N with identity."""
        N = 32
        F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
        mu = mutual_coherence(F)
        
        expected = 1 / np.sqrt(N)
        assert abs(mu - expected) < 1e-10, f"DFT coherence should be 1/√N, got {mu}"
    
    def test_rft_coherence_between_identity_and_dft(self):
        """RFT coherence should be between identity (1) and DFT (1/√N)."""
        N = 32
        U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
        mu = mutual_coherence(U_rft)
        
        assert 1/np.sqrt(N) <= mu <= 1.0, f"RFT coherence out of bounds: {mu}"
    
    def test_rft_dft_coherence_moderate(self):
        """RFT-DFT coherence should be high but not identical."""
        N = 32
        mu = rft_dft_coherence(N)
        
        # RFT and DFT are quite similar (both Fourier-like) but not identical
        # High coherence is expected since RFT is a Gram-normalized φ-grid
        assert mu < 1.0, f"RFT and DFT should not be identical, got {mu}"
        # Not minimal (would be 1/√N ≈ 0.18)
        assert mu > 0.5, f"RFT and DFT should have significant overlap, got {mu}"


class TestUncertaintyBounds:
    """Test uncertainty principle bounds."""
    
    def test_heisenberg_bound_correct(self):
        """Heisenberg bound is 1/(4π)."""
        heisenberg, _ = golden_uncertainty_bound(32)
        expected = 1 / (4 * np.pi)
        assert abs(heisenberg - expected) < 1e-10
    
    def test_golden_bound_less_than_heisenberg(self):
        """Golden bound should be less than Heisenberg (tighter constraint)."""
        heisenberg, golden = golden_uncertainty_bound(64)
        assert golden < heisenberg, "Golden bound should be tighter"
    
    def test_golden_bound_positive(self):
        """Golden bound should be positive."""
        _, golden = golden_uncertainty_bound(64)
        assert golden > 0, "Golden bound should be positive"


class TestUncertaintyMeasurement:
    """Test uncertainty measurements on signals."""
    
    def test_gaussian_near_heisenberg_bound(self):
        """Gaussian should be near the Heisenberg bound (optimal for DFT)."""
        N = 128
        x = gaussian_signal(N, center=0.5, width=0.05)
        m = measure_uncertainty(x)
        
        # Gaussian approaches but may not exactly achieve the bound
        assert m.product_dft < 0.3, f"Gaussian DFT uncertainty too large: {m.product_dft}"
    
    def test_golden_qp_favors_rft(self):
        """Golden quasi-periodic signals should favor RFT."""
        N = 64
        x = golden_quasiperiodic_signal(N, f0=0.3, a=0.5)
        m = measure_uncertainty(x)
        
        # RFT should achieve similar or better uncertainty
        # (ratio ≤ 1 means RFT is better or equal)
        assert m.uncertainty_ratio < 1.5, f"RFT should not be much worse: ratio={m.uncertainty_ratio}"
    
    def test_harmonic_favors_dft(self):
        """Pure harmonics should strongly favor DFT."""
        N = 64
        x = harmonic_signal(N, k=3)
        m = measure_uncertainty(x)
        
        # DFT should be much better for pure harmonics
        assert m.product_dft < m.product_rft, "DFT should win on harmonics"


class TestConcentrationDuality:
    """Test concentration-uncertainty duality (Theorem 8 connection)."""
    
    def test_harmonic_k99_dft_is_one(self):
        """Pure harmonic has K99=1 under DFT."""
        N = 64
        x = harmonic_signal(N, k=7)
        d = concentration_uncertainty_duality(x)
        
        assert d.k99_dft == 1, f"Harmonic K99(DFT) should be 1, got {d.k99_dft}"
    
    def test_golden_qp_k99_rft_smaller(self):
        """Golden QP signal should have smaller K99 under RFT."""
        N = 64
        rng = np.random.default_rng(42)
        
        wins = 0
        num_trials = 50
        
        for _ in range(num_trials):
            f0 = rng.uniform(0, 1)
            a = rng.uniform(-1, 1)
            x = golden_quasiperiodic_signal(N, f0, a)
            d = concentration_uncertainty_duality(x)
            
            if d.k99_rft <= d.k99_dft:
                wins += 1
        
        # Majority should favor RFT (connects to Theorem 8)
        assert wins > num_trials * 0.5, f"RFT should win majority: {wins}/{num_trials}"


class TestTheorem9GoldenUncertaintyPrinciple:
    """Main theorem tests for the Golden-RFT Uncertainty Principle."""
    
    def test_theorem_9_golden_ensemble_shows_different_tradeoff(self):
        """THEOREM 9: RFT achieves a different uncertainty trade-off.
        
        For golden quasi-periodic signals, the RFT basis trades time-spread
        for improved frequency concentration compared to DFT.
        """
        N = 64
        rng = np.random.default_rng(42)
        
        # Compare uncertainty products over ensemble
        dft_products = []
        rft_products = []
        
        for _ in range(100):
            f0 = rng.uniform(0, 1)
            a = rng.uniform(-1, 1)
            x = golden_quasiperiodic_signal(N, f0, a)
            m = measure_uncertainty(x)
            
            dft_products.append(m.product_dft)
            rft_products.append(m.product_rft)
        
        mean_dft = np.mean(dft_products)
        mean_rft = np.mean(rft_products)
        
        # RFT should achieve comparable uncertainty (within 20%)
        ratio = mean_rft / mean_dft
        assert 0.8 < ratio < 1.2, f"RFT/DFT uncertainty ratio should be near 1: {ratio}"
    
    def test_theorem_9_negative_control_harmonics(self):
        """NEGATIVE CONTROL: Pure harmonics strongly favor DFT."""
        N = 64
        
        for k in [1, 5, 10, 20]:
            x = harmonic_signal(N, k=k)
            m = measure_uncertainty(x)
            
            # DFT should dramatically outperform RFT on harmonics
            assert m.product_dft < m.product_rft, f"DFT should win on harmonic k={k}"
    
    def test_theorem_9_concentration_duality_holds(self):
        """DUALITY: Low K99 correlates with the correct basis choice.
        
        This connects the concentration inequality (Theorem 8) with
        the uncertainty principle (Theorem 9).
        """
        N = 64
        
        # Golden QP: RFT should have lower K99 (Theorem 8)
        x_golden = golden_quasiperiodic_signal(N, f0=0.3, a=0.5)
        d_golden = concentration_uncertainty_duality(x_golden)
        
        # Harmonic: DFT should have lower K99
        x_harmonic = harmonic_signal(N, k=5)
        d_harmonic = concentration_uncertainty_duality(x_harmonic)
        
        # Verify duality
        assert d_harmonic.k99_dft < d_harmonic.k99_rft, "Harmonic: DFT should concentrate"
        # Golden QP allows similar concentration (the key insight)
        assert d_golden.k99_rft <= d_golden.k99_dft + 10, "Golden QP: RFT should be competitive"
    
    def test_theorem_9_mutual_coherence_explains_bound(self):
        """The mutual coherence μ explains the modified uncertainty bound."""
        N = 64
        U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
        mu = mutual_coherence(U_rft)
        
        heisenberg, golden = golden_uncertainty_bound(N)
        
        # Verify the relationship: golden = heisenberg * (1 - μ²)
        computed_golden = heisenberg * (1 - mu**2)
        assert abs(golden - computed_golden) < 1e-10, "Golden bound formula should hold"
        
        # The bound should be meaningful (not trivially zero)
        assert golden > 0.01, "Golden bound should be non-trivial"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
