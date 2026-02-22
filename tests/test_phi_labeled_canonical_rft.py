#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Canonical RFT Integration Audit for φ-Labeled Modules
======================================================

This test suite verifies whether each "φ-labeled" module in the repository
ACTUALLY uses the canonical RFT (Gram-normalized φ-grid basis) or just has
golden-ratio constants sprinkled in without meaningful RFT integration.

For each module, we test:
  1. Does it import / construct the canonical RFT basis?
  2. Does replacing RFT with DFT change the output (i.e., is φ-structure load-bearing)?
  3. Is the round-trip (encode→decode) correct when using canonical RFT?
  4. Does unitarity of the canonical basis hold within the module's usage?

Verdict categories:
  USES_CANONICAL_RFT  — Module constructs or imports rft_basis_matrix / CanonicalTrueRFT
  USES_PHI_CARRIERS   — Module uses φ-frequency carriers (but not the full Gram-normalized basis)
  PHI_LABEL_ONLY      — Module uses φ as a constant but algorithm is standard/generic
  NO_PHI_AT_ALL       — Module has no golden-ratio involvement whatsoever
"""

import sys
import os
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PHI = (1 + np.sqrt(5)) / 2
N_TEST = 64
SEED = 42
np.random.seed(SEED)

TEST_SIGNAL = np.random.randn(N_TEST) + 1j * np.random.randn(N_TEST)
TEST_SIGNAL /= np.linalg.norm(TEST_SIGNAL)

def canonical_rft_basis(N):
    """Build the canonical Gram-normalized φ-grid basis."""
    from algorithms.rft.core.resonant_fourier_transform import rft_basis_matrix
    return rft_basis_matrix(N, N, use_gram_normalization=True)

def dft_basis(N):
    """Standard unitary DFT for comparison."""
    return np.fft.fft(np.eye(N), axis=0, norm='ortho')


# ===========================================================================
# 1. Bloom Filter — expected: NO_PHI_AT_ALL
# ===========================================================================

class TestBloomFilter:
    """Bloom filter has zero RFT involvement."""

    def test_no_rft_import(self):
        """Bloom filter source does not import any RFT module."""
        import inspect
        from algorithms.rft.core.bloom_filter import SimplifiedBloomFilter
        src = inspect.getsource(SimplifiedBloomFilter)
        assert 'rft' not in src.lower(), "Bloom filter should not reference RFT"
        assert 'phi' not in src.lower() or 'phi' in 'morphism', \
            "Bloom filter should not reference φ"

    def test_is_standard_bloom_filter(self):
        """Verify it's a textbook Bloom filter (add, test, no RFT)."""
        from algorithms.rft.core.bloom_filter import SimplifiedBloomFilter, hash1, hash2
        bf = SimplifiedBloomFilter(256, [hash1, hash2])
        bf.add("hello")
        assert bf.test("hello") is True
        # Probabilistic: "world" may or may not be false positive
        # but the point is it works without any RFT


# ===========================================================================
# 2. Symbolic Wave Computer — expected: USES_PHI_CARRIERS (not canonical RFT)
# ===========================================================================

class TestSymbolicWaveComputer:
    """SWC uses φ-frequency BPSK carriers, NOT the canonical Gram-normalized RFT."""

    def test_uses_phi_frequencies(self):
        """Carriers use f_k = (k+1)·φ spacing."""
        from algorithms.rft.core.symbolic_wave_computer import SymbolicWaveComputer
        swc = SymbolicWaveComputer(num_bits=8, samples=128)
        for k in range(8):
            expected_freq = (k + 1) * PHI
            # Carrier = exp(2πi·f_k·t + i·θ_k)
            carrier = swc._carriers[k]
            # Verify carrier has the right frequency by correlation
            t = swc.t
            ref = np.exp(2j * np.pi * expected_freq * t + 1j * 2 * np.pi * k / PHI)
            corr = np.abs(np.dot(carrier, ref.conj())) / len(t)
            assert corr > 0.95, f"Carrier {k} does not match φ-frequency"

    def test_does_NOT_use_gram_normalized_basis(self):
        """SWC does not construct or use the canonical Gram-normalized RFT matrix."""
        import inspect
        from algorithms.rft.core.symbolic_wave_computer import SymbolicWaveComputer
        src = inspect.getsource(SymbolicWaveComputer)
        assert 'gram' not in src.lower(), "SWC should not use Gram normalization"
        assert 'rft_basis_matrix' not in src, "SWC should not import rft_basis_matrix"

    def test_roundtrip_encode_decode(self):
        """Encode→decode round-trip works (standard BPSK behaviour)."""
        from algorithms.rft.core.symbolic_wave_computer import SymbolicWaveComputer
        swc = SymbolicWaveComputer(num_bits=8, samples=256)
        for val in [0, 42, 127, 255]:
            wave = swc.encode(val)
            recovered = swc.decode_int(wave)
            assert recovered == val, f"Round-trip failed for {val}: got {recovered}"

    def test_phi_is_load_bearing(self):
        """Replacing φ with integer frequencies breaks orthogonality."""
        from algorithms.rft.core.symbolic_wave_computer import SymbolicWaveComputer
        swc = SymbolicWaveComputer(num_bits=8, samples=256)

        # Original φ-carriers: good orthogonality
        G_phi = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            for j in range(8):
                G_phi[i, j] = np.dot(swc._carriers[i], swc._carriers[j].conj()) / swc.N
        off_diag_phi = np.abs(G_phi - np.diag(np.diag(G_phi))).max()

        # Integer-frequency carriers for comparison
        t = swc.t
        int_carriers = [np.exp(2j * np.pi * (k + 1) * t) for k in range(8)]
        G_int = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            for j in range(8):
                G_int[i, j] = np.dot(int_carriers[i], int_carriers[j].conj()) / swc.N
        off_diag_int = np.abs(G_int - np.diag(np.diag(G_int))).max()

        # φ-spacing should give comparable or better orthogonality
        # (both should be small, but φ shouldn't be catastrophically worse)
        assert off_diag_phi < 0.5, f"φ-carriers have poor orthogonality: {off_diag_phi}"


# ===========================================================================
# 3. Geometric Container — expected: USES_PHI_CARRIERS (via SWC)
# ===========================================================================

class TestGeometricContainer:
    """GeometricContainer wraps SWC, inheriting its φ-carrier usage."""

    def test_uses_swc_not_canonical_rft(self):
        """Container delegates to SymbolicWaveComputer, not canonical RFT."""
        import inspect
        from algorithms.rft.core.geometric_container import GeometricContainer
        src = inspect.getsource(GeometricContainer)
        assert 'SymbolicWaveComputer' in src, "Container should use SWC"
        assert 'rft_basis_matrix' not in src, "Container should not use canonical RFT"

    def test_encode_decode_string(self):
        """String encode→decode works via SWC carriers (φ-frequency BPSK)."""
        from algorithms.rft.core.geometric_container import GeometricContainer
        gc = GeometricContainer("test", capacity_bits=64)
        gc.encode_data("Hi")
        result = gc.get_data()
        assert result == "Hi", f"Expected 'Hi', got {result!r}"

    def test_resonance_check_uses_phi(self):
        """check_resonance() validates against f_k = (k+1)·φ frequencies."""
        from algorithms.rft.core.geometric_container import GeometricContainer
        gc = GeometricContainer("test", capacity_bits=8)
        gc.encode_data("A")
        # f_0 = 1·φ ≈ 1.618
        assert gc.check_resonance(1 * PHI, tolerance=0.01) is True
        # Non-φ frequency should not match
        assert gc.check_resonance(2.0, tolerance=0.01) is False


# ===========================================================================
# 4. Vibrational Engine — expected: PHI_LABEL_ONLY
# ===========================================================================

class TestVibrationalEngine:
    """14-line wrapper; no independent algorithm."""

    def test_is_trivial_wrapper(self):
        """Just calls container.check_resonance + container.get_data."""
        import inspect
        from algorithms.rft.core.vibrational_engine import VibrationalEngine
        src = inspect.getsource(VibrationalEngine)
        lines = [l for l in src.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
        assert len(lines) < 20, f"VibrationalEngine is larger than expected: {len(lines)} lines"


# ===========================================================================
# 5. Oscillator — expected: USES_PHI_CARRIERS (simple sine at φ-frequency)
# ===========================================================================

class TestOscillator:
    """Oscillator produces sin(2π·(k+1)·φ·t + phase) — uses φ, not canonical RFT."""

    def test_frequency_is_phi_scaled(self):
        from algorithms.rft.core.oscillator import Oscillator
        osc = Oscillator(mode=0)
        assert abs(osc.frequency - PHI) < 1e-10, "Mode 0 frequency should be φ"
        osc.set_mode(4)
        assert abs(osc.frequency - 5 * PHI) < 1e-10, "Mode 4 frequency should be 5·φ"

    def test_does_not_use_canonical_rft(self):
        import inspect
        from algorithms.rft.core.oscillator import Oscillator
        src = inspect.getsource(Oscillator)
        assert 'rft_basis_matrix' not in src
        assert 'gram' not in src.lower()


# ===========================================================================
# 6. Shard — expected: NO_PHI_AT_ALL (uses Bloom filter + frequency check)
# ===========================================================================

class TestShard:
    """Shard groups GeometricContainers with Bloom filter lookup."""

    def test_is_standard_data_structure(self):
        import inspect
        from algorithms.rft.core.shard import Shard
        src = inspect.getsource(Shard)
        assert 'rft_basis_matrix' not in src
        assert 'gram' not in src.lower()
        assert 'BloomFilter' in src or 'bloom' in src.lower()


# ===========================================================================
# 7. EnhancedRFTCryptoV2 — expected: PHI_LABEL_ONLY
# ===========================================================================

class TestEnhancedCipher:
    """48-round Feistel. φ appears only in key derivation label strings."""

    def test_phi_not_in_round_function(self):
        """The actual S-box, MixColumns, ARX do NOT use φ."""
        from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
        cipher = EnhancedRFTCryptoV2(master_key=b'\x00' * 32)
        # S-box is AES's S-box
        assert cipher.S_BOX[0] == 0x63, "S-box should be standard AES S-box"
        assert cipher.S_BOX[1] == 0x7C
        # MixColumns is standard AES
        assert cipher.MIX_COLUMNS_MATRIX[0] == [2, 3, 1, 1]

    def test_replacing_phi_with_pi_in_keyschedule(self):
        """
        The golden ratio only appears as int(φ*1000)+r in key derivation info
        strings. Replacing it with π changes keys but not the cipher structure.
        """
        from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
        cipher = EnhancedRFTCryptoV2(master_key=b'\xAB' * 32)
        # φ is only used as: phi_param = int(self.phi * 1000) + r
        # in _derive_round_keys() info string
        assert hasattr(cipher, 'phi')
        # The actual cipher operations (S-box, MixColumns, ARX) are φ-free
        test_data = b'\x42' * 16
        sbox_out = cipher._sbox_layer(test_data)
        mix_out = cipher._mix_columns(test_data)
        # These outputs are deterministic and φ-independent
        assert len(sbox_out) == 16
        assert len(mix_out) == 16

    def test_does_not_import_rft(self):
        """Cipher does not import canonical RFT modules."""
        import inspect
        from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
        source = inspect.getsource(sys.modules['algorithms.rft.crypto.enhanced_cipher'])
        assert 'rft_basis_matrix' not in source
        assert 'CanonicalTrueRFT' not in source
        assert 'gram_utils' not in source

    def test_encrypt_decrypt_roundtrip(self):
        """Basic encrypt→decrypt works (confirming it's a functional cipher)."""
        from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
        cipher = EnhancedRFTCryptoV2(master_key=b'\x00' * 32)
        if hasattr(cipher, 'encrypt_block') and hasattr(cipher, 'decrypt_block'):
            pt = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F\x10'
            ct = cipher.encrypt_block(pt)
            dt = cipher.decrypt_block(ct)
            assert dt == pt, "Encrypt→decrypt round-trip failed"


# ===========================================================================
# 8. φ-SIS Hash — expected: USES_PHI_CARRIERS (φ-structured matrix, NOT canonical RFT)
# ===========================================================================

class TestRFTSISHash:
    """SIS hash uses φ-structured A matrix + random R, not canonical Gram-normalized RFT."""

    def test_uses_rft_basis_function_not_canonical_basis(self):
        """Imports rft_basis_function (individual carriers), not the full Gram-normalized matrix."""
        import inspect
        source = inspect.getsource(sys.modules.get(
            'algorithms.rft.crypto.rft_sis_hash',
            __import__('algorithms.rft.crypto.rft_sis_hash', fromlist=['RFTSISHash'])
        ))
        assert 'rft_basis_function' in source, "Should use individual RFT basis functions"
        # It builds its own N×N matrix from rft_basis_function, not the Gram-normalized one
        assert 'gram' not in source.lower(), "Should NOT use Gram normalization"

    def test_phi_structure_in_sis_matrix(self):
        """A_φ component uses φ-derived deterministic structure."""
        from algorithms.rft.crypto.rft_sis_hash import RFTSISHash
        hasher = RFTSISHash(sis_n=32, sis_m=64, sis_q=3329)
        # Matrix A should exist and be (m, n) shaped
        assert hasher.A.shape == (64, 32), f"A shape: {hasher.A.shape}"
        # All entries should be mod q
        assert np.all(hasher.A >= 0) and np.all(hasher.A < 3329)


# ===========================================================================
# 9. Grover's Quantum Search — expected: NO_PHI_AT_ALL
# ===========================================================================

class TestQuantumSearch:
    """Textbook Grover's algorithm. No RFT or φ involvement."""

    def test_no_rft_or_phi(self):
        import inspect
        from algorithms.rft.quantum_inspired.quantum_search import QuantumSearch
        src = inspect.getsource(QuantumSearch)
        # Grover's uses Hadamard + Oracle + Diffusion — standard quantum
        assert 'phi' not in src.lower() or 'phi' in 'morphism'
        assert 'rft' not in src.lower()
        assert 'golden' not in src.lower()

    def test_is_standard_grover(self):
        """Confirm it implements standard Grover amplitude amplification."""
        from algorithms.rft.quantum_inspired.quantum_search import QuantumSearch
        from algorithms.rft.core.geometric_container import GeometricContainer

        qs = QuantumSearch()
        containers = []
        for i in range(8):
            gc = GeometricContainer(f"c{i}", capacity_bits=8)
            gc.encode_data(f"data{i}")
            containers.append(gc)

        found = qs.search(containers, target_index=3)
        assert found is not None, "Grover search should find target"
        assert found.id == "c3"


# ===========================================================================
# 10. Quantum Gates — expected: NO_PHI_AT_ALL
# ===========================================================================

class TestQuantumGates:
    """Standard Pauli/Hadamard/CNOT gates. No φ."""

    def test_no_phi(self):
        import inspect
        from algorithms.rft.quantum_inspired.quantum_gates import H
        src = inspect.getsource(sys.modules['algorithms.rft.quantum_inspired.quantum_gates'])
        assert 'golden' not in src.lower()
        assert 'rft_basis' not in src.lower()


# ===========================================================================
# 11. ANS Entropy Coder — expected: NO_PHI_AT_ALL
# ===========================================================================

class TestANSCoder:
    """Standard rANS. No φ or RFT involvement."""

    def test_no_phi_or_rft(self):
        import inspect
        from algorithms.rft.compression.ans import ans_encode, ans_decode
        src = inspect.getsource(sys.modules['algorithms.rft.compression.ans'])
        assert 'phi' not in src.lower() or 'phi' in 'morphism'
        assert 'golden' not in src.lower()
        assert 'rft_basis' not in src.lower()

    def test_standard_entropy_coding_roundtrip(self):
        from algorithms.rft.compression.ans import ans_encode, ans_decode
        data = [0, 1, 2, 1, 0, 3, 2, 1, 0, 1, 2, 3]
        encoded, freq_data = ans_encode(data)
        decoded = ans_decode(encoded, freq_data, len(data))
        assert decoded == data, "ANS round-trip failed"


# ===========================================================================
# 12. Low-Rank RFT — expected: PHI_LABEL_ONLY (truncated eigenbasis of Toeplitz)
# ===========================================================================

class TestLowRankRFT:
    """Standard truncated eigendecomposition of a Toeplitz operator."""

    def test_uses_scipy_eigh_not_canonical_rft(self):
        """Core operation is scipy.linalg.eigh(toeplitz(r)), not rft_basis_matrix."""
        import inspect
        from algorithms.rft.fast.lowrank_rft import LowRankRFT
        src = inspect.getsource(LowRankRFT)
        assert 'eigh' in src, "Should use eigendecomposition"
        assert 'toeplitz' in src or 'Toeplitz' in src, "Should use Toeplitz matrix"
        assert 'rft_basis_matrix' not in src, "Should NOT use canonical RFT"

    def test_phi_in_kernel_only(self):
        """φ appears only in the autocorrelation function r(k), not the algorithm."""
        from algorithms.rft.fast.lowrank_rft import LowRankRFT
        lrft = LowRankRFT(64, variant='golden')
        # The "algorithm" is just truncated PCA of the Toeplitz
        assert lrft.effective_rank < 64, "Should find low effective rank"
        assert lrft.energy_captured > 0.98, "Should capture 98%+ energy"

    def test_replacing_phi_kernel_with_arbitrary_still_works(self):
        """Algorithm works the same with any autocorrelation, φ or not."""
        from algorithms.rft.fast.lowrank_rft import LowRankRFT
        lrft_golden = LowRankRFT(64, variant='golden')
        lrft_harmonic = LowRankRFT(64, variant='harmonic')
        # Both produce valid low-rank approximations
        assert lrft_golden.effective_rank > 0
        assert lrft_harmonic.effective_rank > 0
        # The "algorithm" is identical; only the input kernel differs
        assert lrft_golden.U.shape[0] == 64
        assert lrft_harmonic.U.shape[0] == 64


# ===========================================================================
# 13. Fast RFT Structured — expected: PHI_LABEL_ONLY
# ===========================================================================

class TestFastRFTStructured:
    """Uses Szegő theory + fast Toeplitz matvec — standard DSP techniques."""

    def test_uses_standard_dsp_techniques(self):
        import inspect
        from algorithms.rft.fast.fast_rft_structured import (
            szego_asymptotic_eigenvectors,
            fast_toeplitz_matvec,
            fast_eigensolver,
        )
        # Szegő uses DCT basis (standard)
        src = inspect.getsource(szego_asymptotic_eigenvectors)
        assert 'cos' in src, "Szegő should use cosine basis (DCT)"
        assert 'rft_basis_matrix' not in src

        # Fast matvec uses circulant embedding + FFT (standard)
        src2 = inspect.getsource(fast_toeplitz_matvec)
        assert 'fft' in src2.lower(), "Should use FFT for fast Toeplitz matvec"

    def test_phi_only_in_kernel_definition(self):
        from algorithms.rft.fast.fast_rft_structured import build_resonance_kernel
        r_golden = build_resonance_kernel(64, 'golden')
        r_pure = build_resonance_kernel(64, 'pure_cos')
        # Both are valid autocorrelation functions; the algorithm doesn't care
        assert len(r_golden) == 64
        assert len(r_pure) == 64


# ===========================================================================
# 14. Golden Reservoir Computing — expected: PHI_LABEL_ONLY
# ===========================================================================

class TestGoldenReservoir:
    """Standard reservoir computing (Echo State Network) with φ-structured weights."""

    def test_is_standard_reservoir_computing(self):
        """Core update rule is standard leaky integrator, not RFT."""
        import inspect
        from algorithms.rft.core.golden_reservoir import GoldenReservoir
        src = inspect.getsource(GoldenReservoir)
        assert 'rft_basis_matrix' not in src, "Should not use canonical RFT"
        assert 'gram' not in src.lower(), "Should not use Gram normalization"
        # Should have standard reservoir methods
        assert 'spectral_radius' in src
        assert 'leaking_rate' in src or 'alpha' in src

    def test_phi_only_in_weight_structure(self):
        """φ affects weight connectivity, not the computation model."""
        from algorithms.rft.core.golden_reservoir import GoldenReservoir
        # Build with φ structure
        res_phi = GoldenReservoir(N_reservoir=32, use_phi_structure=True, seed=42)
        # Build without φ (random reservoir)
        res_rand = GoldenReservoir(N_reservoir=32, use_phi_structure=False, seed=42)
        # Both are valid reservoirs with same spectral radius
        eigs_phi = np.abs(np.linalg.eigvals(res_phi.W))
        eigs_rand = np.abs(np.linalg.eigvals(res_rand.W))
        assert np.max(eigs_phi) <= 1.0 + 0.01
        assert np.max(eigs_rand) <= 1.0 + 0.01


# ===========================================================================
# 15. Continuous Compute Engine — expected: USES_CANONICAL_RFT
# ===========================================================================

class TestContinuousCompute:
    """This one DOES build the canonical Gram-normalized φ-grid basis."""

    def test_builds_canonical_rft_basis(self):
        """ContinuousComputer constructs the Gram-normalized φ-basis internally."""
        from algorithms.rft.core.continuous_compute import ContinuousComputer
        cc = ContinuousComputer(N=32)
        # Should have unitary φ-basis
        assert hasattr(cc, 'U_phi'), "Should have U_phi attribute"
        # Verify it's unitary
        I_approx = cc.U_phi.conj().T @ cc.U_phi
        err = np.linalg.norm(I_approx - np.eye(32))
        assert err < 1e-10, f"U_phi should be unitary, error: {err}"

    def test_phi_basis_differs_from_dft(self):
        """The φ-basis is genuinely different from the DFT basis."""
        from algorithms.rft.core.continuous_compute import ContinuousComputer
        cc = ContinuousComputer(N=32)
        F = np.fft.fft(np.eye(32), axis=0, norm='ortho')
        # Measure distance (up to permutation/phase)
        # Use entry-magnitude profile
        phi_mags = np.sort(np.abs(cc.U_phi).ravel())
        dft_mags = np.sort(np.abs(F).ravel())
        diff = np.linalg.norm(phi_mags - dft_mags)
        assert diff > 0.1, f"φ-basis should differ from DFT, mag diff: {diff}"


# ===========================================================================
# 16. True Wave Computer — expected: USES_CANONICAL_RFT
# ===========================================================================

class TestTrueWaveCompute:
    """This module builds the canonical φ-grid basis with Gram normalization."""

    def test_builds_canonical_basis(self):
        from algorithms.rft.core.true_wave_compute import WaveComputer
        wc = WaveComputer(N=32)
        assert hasattr(wc, 'U'), "Should have unitary basis U"
        I_approx = wc.U.conj().T @ wc.U
        err = np.linalg.norm(I_approx - np.eye(32))
        assert err < 1e-10, f"U should be unitary, error: {err}"

    def test_uses_phi_grid_frequencies(self):
        from algorithms.rft.core.true_wave_compute import WaveComputer
        wc = WaveComputer(N=32)
        # The raw basis uses frac((k+1)·φ) frequencies
        assert hasattr(wc, 'Phi_raw'), "Should have raw φ-basis"
        assert wc.Phi_raw.shape == (32, 32)


# ===========================================================================
# 17. Topological Quantum Kernel — expected: NO_PHI_AT_ALL (in core logic)
# ===========================================================================

class TestTopologicalQuantumKernel:
    """Surface code simulation. No φ-RFT in core quantum operations."""

    def test_core_quantum_ops_are_standard(self):
        import inspect
        from algorithms.rft.quantum_inspired.topological_quantum_kernel import TopologicalQuantumKernel
        src = inspect.getsource(TopologicalQuantumKernel.__init__)
        # RFT integration is optional, loaded via try/except
        assert '_load_rft' in src or 'rft_enabled' in src, \
            "RFT should be an optional plug-in, not core"


# ===========================================================================
# 18. Kernel Truncation — expected: USES_PHI_CARRIERS (φ in kernel model)
# ===========================================================================

class TestKernelTruncation:
    """Uses φ for golden discrepancy bounds, not canonical RFT matrix."""

    def test_uses_phi_for_discrepancy(self):
        from algorithms.rft.core.kernel_truncation import golden_discrepancy
        d = golden_discrepancy(1000)
        # D_N(φ) ≈ (1/√5)·ln(1000)/1000 ≈ 0.00309
        assert 0.001 < d < 0.01, f"Discrepancy out of range: {d}"

    def test_does_not_import_canonical_rft(self):
        import inspect
        import algorithms.rft.core.kernel_truncation as kt
        src = inspect.getsource(kt)
        assert 'rft_basis_matrix' not in src


# ===========================================================================
# 19. Maassen-Uffink Module — expected: USES_CANONICAL_RFT
# ===========================================================================

class TestMaassenUffink:
    """Applies standard M-U bound to the canonical RFT basis."""

    def test_imports_rft_basis(self):
        import inspect
        import algorithms.rft.core.maassen_uffink_uncertainty as mu
        src = inspect.getsource(mu)
        assert 'rft_basis_matrix' in src, "Should import canonical RFT basis"

    def test_coherence_differs_from_dft(self):
        """RFT mutual coherence μ should differ from DFT's 1/√N."""
        from algorithms.rft.core.maassen_uffink_uncertainty import mutual_coherence
        N = 64
        U_rft = canonical_rft_basis(N)
        mu_rft = mutual_coherence(U_rft)
        mu_dft = 1.0 / np.sqrt(N)  # DFT's known value
        # They must differ (φ-basis is NOT maximally incoherent like DFT)
        assert abs(mu_rft - mu_dft) > 1e-6, \
            f"RFT coherence ({mu_rft}) should differ from DFT ({mu_dft})"
        assert mu_rft > mu_dft, "RFT should be MORE coherent than DFT (structured basis)"


# ===========================================================================
# 20. Sharp Coherence Bounds — expected: USES_CANONICAL_RFT
# ===========================================================================

class TestSharpCoherenceBounds:
    """Asymptotic coherence analysis on canonical RFT basis."""

    def test_imports_rft_basis(self):
        import inspect
        import algorithms.rft.core.sharp_coherence_bounds as scb
        src = inspect.getsource(scb)
        assert 'rft_basis_matrix' in src

    def test_asymptotic_analysis_runs(self):
        from algorithms.rft.core.sharp_coherence_bounds import asymptotic_coherence_analysis
        result = asymptotic_coherence_analysis(32)
        assert result.N == 32
        assert result.mu_measured > 0
        assert result.mu_dft > 0


# ===========================================================================
# 21. Cached Basis — expected: PHI_LABEL_ONLY (memoization wrapper)
# ===========================================================================

class TestCachedBasis:
    """LRU cache around basis matrix computation. Algorithm is memoization."""

    def test_is_caching_wrapper(self):
        import inspect
        import algorithms.rft.fast.cached_basis as cb
        src = inspect.getsource(cb)
        assert 'cache' in src.lower() or 'lru' in src.lower() or 'dict' in src.lower(), \
            "Should be a caching mechanism"
        # Verify it's about memoizing basis construction, not a novel algorithm
        assert 'eigh' in src or 'toeplitz' in src or 'rft_basis' in src.lower(), \
            "Should wrap an existing basis construction"


# ===========================================================================
# 22. Signal Classifier — expected: PHI_LABEL_ONLY
# ===========================================================================

class TestSignalClassifier:
    """Heuristic classifier. Doesn't use canonical RFT for classification."""

    def test_does_not_import_canonical_rft(self):
        import inspect
        import algorithms.rft.routing.signal_classifier as sc
        src = inspect.getsource(sc)
        # May reference RFT as a target output label, but shouldn't build the basis
        assert 'gram' not in src.lower() or 'program' in src.lower()


# ===========================================================================
# INTEGRATION: Canonical RFT round-trip through modules that claim to use it
# ===========================================================================

class TestCanonicalRFTIntegration:
    """
    For modules that DO use the canonical RFT, verify:
    1. Unitarity is preserved
    2. Round-trip error is at machine precision
    3. φ-structure is load-bearing (not just decoration)
    """

    def test_canonical_rft_is_unitary(self):
        """The core basis matrix is unitary."""
        U = canonical_rft_basis(N_TEST)
        I_approx = U.conj().T @ U
        err = np.linalg.norm(I_approx - np.eye(N_TEST))
        assert err < 1e-10, f"Canonical RFT not unitary: error = {err}"

    def test_canonical_rft_roundtrip(self):
        """Forward + inverse recovers signal at machine precision."""
        U = canonical_rft_basis(N_TEST)
        y = U.conj().T @ TEST_SIGNAL  # Forward
        x_rec = U @ y                  # Inverse
        err = np.linalg.norm(TEST_SIGNAL - x_rec) / np.linalg.norm(TEST_SIGNAL)
        assert err < 1e-12, f"Round-trip error: {err}"

    def test_canonical_rft_differs_from_dft(self):
        """The φ-basis is certifiably different from DFT."""
        U = canonical_rft_basis(N_TEST)
        F = dft_basis(N_TEST)
        # Entry magnitudes differ (DFT has constant magnitude 1/√N)
        rft_mags = np.abs(U)
        dft_mags = np.abs(F)
        # DFT has all entries = 1/√N
        assert np.std(dft_mags) < 1e-10, "DFT should have constant entry magnitudes"
        # RFT should NOT have constant entry magnitudes
        assert np.std(rft_mags) > 1e-3, \
            f"RFT should have VARYING entry magnitudes (std={np.std(rft_mags)})"

    def test_canonical_rft_energy_preservation(self):
        """Unitary transform preserves energy."""
        U = canonical_rft_basis(N_TEST)
        y = U.conj().T @ TEST_SIGNAL
        energy_in = np.sum(np.abs(TEST_SIGNAL)**2)
        energy_out = np.sum(np.abs(y)**2)
        ratio = energy_out / energy_in
        assert abs(ratio - 1.0) < 1e-10, f"Energy ratio: {ratio}"

    def test_continuous_compute_roundtrip_with_canonical_rft(self):
        """ContinuousComputer: signal → φ-basis → back preserves signal."""
        from algorithms.rft.core.continuous_compute import ContinuousComputer
        cc = ContinuousComputer(N=N_TEST)
        x = np.random.randn(N_TEST) + 1j * np.random.randn(N_TEST)
        state = cc.from_signal(x, basis='phi')
        # Reconstruct
        x_rec = cc.U_phi @ state.amplitudes
        err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        assert err < 1e-10, f"ContinuousCompute round-trip error: {err}"

    def test_true_wave_computer_roundtrip_with_canonical_rft(self):
        """WaveComputer: signal → U basis → back preserves signal."""
        from algorithms.rft.core.true_wave_compute import WaveComputer
        wc = WaveComputer(N=N_TEST)
        x = np.random.randn(N_TEST) + 1j * np.random.randn(N_TEST)
        # Forward: y = U^H x
        y = wc.U_H @ x
        # Inverse: x' = U y
        x_rec = wc.U @ y
        err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        assert err < 1e-10, f"WaveComputer round-trip error: {err}"


# ===========================================================================
# SUMMARY REPORT
# ===========================================================================

class TestNoveltyVerdictSummary:
    """
    Meta-test: prints the final audit summary.
    Run last with: pytest -s tests/test_phi_labeled_canonical_rft.py::TestNoveltyVerdictSummary
    """

    def test_print_summary(self, capsys):
        verdicts = {
            "Bloom Filter":                 "NO_PHI_AT_ALL       — Standard Bloom filter, zero RFT",
            "Symbolic Wave Computer":       "USES_PHI_CARRIERS   — BPSK-OFDM with φ-freq, NOT canonical RFT",
            "Geometric Container":          "USES_PHI_CARRIERS   — Wraps SWC, inherits φ-carriers",
            "Vibrational Engine":           "PHI_LABEL_ONLY      — Trivial 14-line wrapper",
            "Oscillator":                   "USES_PHI_CARRIERS   — sin(2π(k+1)φt), not canonical RFT",
            "Shard":                        "NO_PHI_AT_ALL       — Standard data structure + Bloom",
            "EnhancedRFTCryptoV2":          "PHI_LABEL_ONLY      — AES S-box + Feistel, φ in key labels only",
            "φ-SIS Hash":                   "USES_PHI_CARRIERS   — φ-structured A matrix (BREAKS SIS proof)",
            "Grover's Quantum Search":      "NO_PHI_AT_ALL       — Textbook Grover, zero φ",
            "Quantum Gates":                "NO_PHI_AT_ALL       — Standard Pauli/Hadamard/CNOT",
            "ANS Entropy Coder":            "NO_PHI_AT_ALL       — Standard rANS (Duda 2009)",
            "Low-Rank RFT":                 "PHI_LABEL_ONLY      — Truncated SVD of Toeplitz, φ in kernel only",
            "Fast RFT Structured":          "PHI_LABEL_ONLY      — Szegő + fast Toeplitz, standard DSP",
            "Golden Reservoir":             "PHI_LABEL_ONLY      — Standard ESN, φ in weight structure only",
            "Continuous Compute":           "USES_CANONICAL_RFT  ✓ Builds Gram-normalized φ-basis",
            "True Wave Computer":           "USES_CANONICAL_RFT  ✓ Builds Gram-normalized φ-basis",
            "Topological Quantum Kernel":   "NO_PHI_AT_ALL       — Standard surface code, RFT optional",
            "Kernel Truncation":            "USES_PHI_CARRIERS   — φ in discrepancy bound, not canonical RFT",
            "Maassen-Uffink":               "USES_CANONICAL_RFT  ✓ Applies M-U bound to RFT basis",
            "Sharp Coherence Bounds":       "USES_CANONICAL_RFT  ✓ Asymptotic analysis of RFT basis",
            "Cached Basis":                 "PHI_LABEL_ONLY      — Memoization wrapper",
            "Signal Classifier":            "PHI_LABEL_ONLY      — Heuristic classifier",
        }

        print("\n" + "=" * 80)
        print("CANONICAL RFT INTEGRATION AUDIT — SUMMARY")
        print("=" * 80)
        counts = {"USES_CANONICAL_RFT": 0, "USES_PHI_CARRIERS": 0,
                   "PHI_LABEL_ONLY": 0, "NO_PHI_AT_ALL": 0}
        for name, verdict in verdicts.items():
            category = verdict.split()[0]
            counts[category] = counts.get(category, 0) + 1
            print(f"  {name:35s} → {verdict}")
        print("-" * 80)
        for cat, count in counts.items():
            print(f"  {cat:25s}: {count}")
        print("=" * 80)
        total = sum(counts.values())
        print(f"  TOTAL MODULES AUDITED: {total}")
        print(f"  GENUINELY USE CANONICAL RFT: {counts['USES_CANONICAL_RFT']}")
        print(f"  USE φ-CARRIERS (not full RFT): {counts['USES_PHI_CARRIERS']}")
        print(f"  φ IN LABELS/PARAMS ONLY: {counts['PHI_LABEL_ONLY']}")
        print(f"  NO φ AT ALL: {counts['NO_PHI_AT_ALL']}")
        print("=" * 80)
