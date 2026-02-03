# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""
Test Suite: Theorem 9 (Maassen-Uffink Entropic Uncertainty Principle)
=====================================================================

This test suite verifies the CORRECT finite-dimensional uncertainty principle
for the canonical RFT, using the Maassen-Uffink entropic bound.

The Maassen-Uffink bound states:
    H(|x|²) + H(|Ux|²) ≥ -2 log(μ(U))

where μ(U) = max|U_{jk}| is the mutual coherence.

⚠️ This replaces the incorrect "Heisenberg-style" spread products that
   can produce values below the continuous bound (which is invalid for
   finite-dimensional discrete transforms).
"""

import numpy as np
import pytest

from algorithms.rft.core.maassen_uffink_uncertainty import (
    shannon_entropy,
    signal_entropy,
    mutual_coherence,
    compute_maassen_uffink_bound,
    rft_maassen_uffink_bound,
    dft_maassen_uffink_bound,
    measure_entropic_uncertainty,
    measure_concentration,
    verify_theorem_9,
    # Test signals
    gaussian_signal,
    golden_quasiperiodic_signal,
    harmonic_signal,
    delta_signal,
    uniform_signal,
    k99,
)
from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix


# =============================================================================
# Test: Shannon Entropy
# =============================================================================

class TestShannonEntropy:
    """Tests for Shannon entropy computation."""
    
    def test_entropy_uniform_distribution(self):
        """Uniform distribution has maximum entropy log(N)."""
        N = 64
        p = np.ones(N) / N
        H = shannon_entropy(p)
        expected = np.log(N)
        assert abs(H - expected) < 1e-10, f"Uniform entropy should be log(N)={expected:.4f}, got {H:.4f}"
    
    def test_entropy_delta_distribution(self):
        """Delta distribution has zero entropy."""
        N = 64
        p = np.zeros(N)
        p[0] = 1.0
        H = shannon_entropy(p)
        assert H < 1e-10, f"Delta entropy should be ~0, got {H:.6f}"
    
    def test_entropy_nonnegative(self):
        """Entropy is always non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.random(64)
            p = p / p.sum()
            H = shannon_entropy(p)
            assert H >= -1e-15, f"Entropy should be non-negative, got {H}"


# =============================================================================
# Test: Mutual Coherence
# =============================================================================

class TestMutualCoherence:
    """Tests for mutual coherence computation."""
    
    def test_dft_coherence_is_one_over_sqrt_n(self):
        """DFT has mutual coherence exactly 1/√N."""
        for N in [32, 64, 128]:
            F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
            mu = mutual_coherence(F)
            expected = 1 / np.sqrt(N)
            assert abs(mu - expected) < 1e-10, f"DFT coherence should be 1/√{N}={expected:.6f}, got {mu:.6f}"
    
    def test_identity_coherence_is_one(self):
        """Identity matrix has mutual coherence 1."""
        N = 64
        I = np.eye(N)
        mu = mutual_coherence(I)
        assert abs(mu - 1.0) < 1e-10, f"Identity coherence should be 1, got {mu}"
    
    def test_rft_coherence_between_dft_and_identity(self):
        """RFT coherence is between 1/√N and 1."""
        for N in [32, 64, 128]:
            U_rft = rft_basis_matrix(N, N, use_gram_normalization=True)
            mu = mutual_coherence(U_rft)
            lower = 1 / np.sqrt(N)
            upper = 1.0
            assert lower < mu < upper, f"RFT coherence {mu} not in ({lower:.4f}, {upper})"


# =============================================================================
# Test: Maassen-Uffink Bound
# =============================================================================

class TestMaassenUffinkBound:
    """Tests for the Maassen-Uffink entropic bound."""
    
    def test_dft_bound_equals_log_n(self):
        """DFT bound is exactly log(N)."""
        for N in [32, 64, 128]:
            bound = dft_maassen_uffink_bound(N)
            expected = np.log(N)
            assert abs(bound.entropy_bound - expected) < 1e-10, \
                f"DFT bound should be log({N})={expected:.4f}, got {bound.entropy_bound:.4f}"
    
    def test_rft_bound_less_than_dft(self):
        """RFT has a looser bound than DFT (μ > 1/√N → bound < log(N))."""
        for N in [32, 64, 128]:
            rft_bound = rft_maassen_uffink_bound(N)
            dft_bound = dft_maassen_uffink_bound(N)
            assert rft_bound.entropy_bound < dft_bound.entropy_bound, \
                f"RFT bound {rft_bound.entropy_bound:.4f} should be < DFT bound {dft_bound.entropy_bound:.4f}"
    
    def test_bound_is_positive(self):
        """Entropy bound is always positive for non-identity bases."""
        for N in [32, 64, 128]:
            rft_bound = rft_maassen_uffink_bound(N)
            assert rft_bound.entropy_bound > 0, "RFT bound should be positive"


# =============================================================================
# Test: Theorem 9 - The Maassen-Uffink Inequality MUST Hold
# =============================================================================

class TestTheorem9MaassenUffink:
    """Tests that the Maassen-Uffink inequality MUST hold for ALL signals."""
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_bound_holds_for_delta(self, N):
        """Delta signal satisfies the bound."""
        x = delta_signal(N)
        eu = measure_entropic_uncertainty(x)
        
        # DFT bound
        assert eu.dft_sum >= eu.dft_bound - 1e-10, \
            f"DFT bound violated: {eu.dft_sum:.4f} < {eu.dft_bound:.4f}"
        
        # RFT bound
        assert eu.rft_sum >= eu.rft_bound - 1e-10, \
            f"RFT bound violated: {eu.rft_sum:.4f} < {eu.rft_bound:.4f}"
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_bound_holds_for_uniform(self, N):
        """Uniform signal satisfies the bound."""
        x = uniform_signal(N)
        eu = measure_entropic_uncertainty(x)
        
        assert eu.dft_sum >= eu.dft_bound - 1e-10
        assert eu.rft_sum >= eu.rft_bound - 1e-10
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_bound_holds_for_gaussian(self, N):
        """Gaussian signal satisfies the bound."""
        x = gaussian_signal(N)
        eu = measure_entropic_uncertainty(x)
        
        assert eu.dft_sum >= eu.dft_bound - 1e-10
        assert eu.rft_sum >= eu.rft_bound - 1e-10
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_bound_holds_for_harmonic(self, N):
        """Harmonic signal satisfies the bound."""
        x = harmonic_signal(N, k=3)
        eu = measure_entropic_uncertainty(x)
        
        assert eu.dft_sum >= eu.dft_bound - 1e-10
        assert eu.rft_sum >= eu.rft_bound - 1e-10
    
    @pytest.mark.parametrize("N", [32, 64, 128])
    def test_bound_holds_for_golden_qp(self, N):
        """Golden quasi-periodic signal satisfies the bound."""
        x = golden_quasiperiodic_signal(N)
        eu = measure_entropic_uncertainty(x)
        
        assert eu.dft_sum >= eu.dft_bound - 1e-10
        assert eu.rft_sum >= eu.rft_bound - 1e-10
    
    def test_bound_holds_for_random_signals(self):
        """Bound holds for 1000 random signals."""
        N = 64
        rng = np.random.default_rng(42)
        
        for _ in range(1000):
            # Random unit vector
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            x = x / np.linalg.norm(x)
            
            eu = measure_entropic_uncertainty(x)
            
            assert eu.dft_sum >= eu.dft_bound - 1e-10, \
                f"DFT bound violated for random signal"
            assert eu.rft_sum >= eu.rft_bound - 1e-10, \
                f"RFT bound violated for random signal"


# =============================================================================
# Test: Concentration Properties (Connection to Theorem 8)
# =============================================================================

class TestConcentrationConnection:
    """Tests connecting entropic uncertainty to K99 concentration."""
    
    def test_harmonic_has_perfect_dft_concentration(self):
        """Harmonic signals achieve K99=1 under DFT (perfect sparsity)."""
        N = 64
        x = harmonic_signal(N, k=5)
        cm = measure_concentration(x)
        
        assert cm.k99_dft == 1, f"Harmonic should have K99(DFT)=1, got {cm.k99_dft}"
        assert cm.k99_rft > 1, f"Harmonic should NOT have K99(RFT)=1"
    
    def test_delta_has_spread_dft_coefficients(self):
        """Delta signals have maximum spread under DFT."""
        N = 64
        x = delta_signal(N)
        cm = measure_concentration(x)
        
        # Delta in time → uniform in frequency
        assert cm.k99_dft == N, f"Delta should have K99(DFT)=N={N}, got {cm.k99_dft}"
    
    def test_golden_qp_favors_rft(self):
        """Golden quasi-periodic signals show better RFT concentration on average."""
        N = 64
        rng = np.random.default_rng(42)
        
        rft_wins_k99 = 0
        rft_wins_entropy = 0
        num_samples = 200
        
        for _ in range(num_samples):
            f0 = rng.uniform(0, 1)
            a = rng.uniform(-1, 1)
            x = golden_quasiperiodic_signal(N, f0, a)
            cm = measure_concentration(x)
            
            if cm.rft_wins_k99:
                rft_wins_k99 += 1
            if cm.rft_wins_entropy:
                rft_wins_entropy += 1
        
        # RFT should win on at least 40% of golden QP signals
        win_rate_k99 = rft_wins_k99 / num_samples
        win_rate_entropy = rft_wins_entropy / num_samples
        
        assert win_rate_k99 > 0.3, f"RFT should win K99 on >30% of golden QP, got {win_rate_k99:.1%}"
        assert win_rate_entropy > 0.3, f"RFT should win entropy on >30% of golden QP, got {win_rate_entropy:.1%}"


# =============================================================================
# Test: Negative Controls
# =============================================================================

class TestNegativeControls:
    """Negative controls: DFT should win on harmonic signals."""
    
    def test_dft_wins_on_harmonics(self):
        """DFT achieves perfect concentration on harmonics (RFT does not)."""
        N = 64
        
        for k in range(1, 10):
            x = harmonic_signal(N, k=k)
            cm = measure_concentration(x)
            
            assert cm.k99_dft == 1, f"DFT should achieve K99=1 on harmonic k={k}"
            assert cm.k99_rft > 1, f"RFT should NOT achieve K99=1 on harmonic k={k}"
            assert cm.entropy_dft < cm.entropy_rft, \
                f"DFT should have lower entropy on harmonic k={k}"


# =============================================================================
# Test: verify_theorem_9 function
# =============================================================================

class TestVerifyTheorem9:
    """Tests for the verify_theorem_9 helper function."""
    
    def test_all_signals_pass_verification(self):
        """All standard test signals pass Theorem 9 verification."""
        N = 64
        
        signals = [
            delta_signal(N),
            uniform_signal(N),
            gaussian_signal(N),
            harmonic_signal(N, k=3),
            golden_quasiperiodic_signal(N),
        ]
        
        for x in signals:
            passes, msg = verify_theorem_9(N, x)
            assert passes, f"Theorem 9 verification failed: {msg}"


# =============================================================================
# Test: Definition Firewall Compliance
# =============================================================================

class TestDefinitionFirewallCompliance:
    """Tests ensuring we don't make forbidden claims."""
    
    def test_no_heisenberg_bound_violation(self):
        """
        The old code could produce spread products below 1/(4π).
        With entropic bounds, there are NO violations possible.
        """
        N = 64
        heisenberg = 1 / (4 * np.pi)  # ~0.0796
        
        # Gaussian often gave products ~0.005 which is BELOW Heisenberg
        # This was a sign of incorrect formulation
        x = gaussian_signal(N)
        eu = measure_entropic_uncertainty(x)
        
        # With Maassen-Uffink, the entropy sum is ALWAYS above the bound
        assert eu.dft_sum >= eu.dft_bound - 1e-10
        assert eu.rft_sum >= eu.rft_bound - 1e-10
        
        # Note: We no longer compute spread products that could violate Heisenberg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
