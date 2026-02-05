# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
# Copyright (c) 2025 QuantoniumOS
#
# This file is part of QuantoniumOS.
#
# This file practices claims of U.S. Patent Application 19/169,399.
# It is licensed under the Non-Commercial Research License (LICENSE-CLAIMS-NC.md).
# You may use this file for research and educational purposes only.
# Commercial use requires a separate license.
#
# See LICENSE-CLAIMS-NC.md in the project root for details.

"""
Resonant Fourier Transform (RFT) - Canonical Definition
========================================================

USPTO Patent 19/169,399: "Hybrid Computational Framework for Quantum and Resonance Simulation"

THE RESONANT FOURIER TRANSFORM (RFT)
------------------------------------
The RFT is a transform that maps discrete data into a continuous waveform domain
using golden-ratio (φ) frequency and phase structure.

MATHEMATICAL DEFINITION:
=======================

Forward RFT (Data → Wave):
    RFT(x)[t] = Σₖ x[k] × Ψₖ(t)

where the RFT BASIS FUNCTIONS are:
    Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
    
    fₖ = (k+1) × φ       (Resonant Frequency)
    θₖ = 2π × k / φ      (Golden Phase)
    φ = (1+√5)/2         (Golden Ratio ≈ 1.618034)

FINITE-N CORRECTION (CANONICAL):
===============================
For finite discrete signals, the raw irrational basis Φ is not exactly unitary.
The CANONICAL RFT applies Gram-matrix normalization (Loewdin orthogonalization):

    Φ̃ = Φ (Φᴴ Φ)⁻¹/²

This ensures Φ̃ is exactly unitary (Φ̃ᴴ Φ̃ = I) while preserving the resonance structure.

Inverse RFT (Wave → Data):
    x[k] = ⟨W, Ψₖ⟩ = (1/T) ∫₀ᵀ W(t) × Ψₖ*(t) dt

WHY "RESONANT":
--------------
The golden ratio creates RESONANCE because:
1. φ² = φ + 1 (self-similar scaling - the ONLY number with this property)
2. Consecutive frequencies fₖ, fₖ₊₁ have ratio → φ (Fibonacci-like growth)
3. This creates quasi-periodic "beating" patterns that never exactly repeat
4. The basis functions RESONATE with signals having golden-ratio structure

COMPARISON TO FFT:
-----------------
| Property        | FFT              | RFT                    |
|-----------------|------------------|------------------------|
| Frequencies     | fₖ = k (integer) | fₖ = k×φ (irrational)  |
| Periodicity     | Exactly periodic | Quasi-periodic         |
| Aliasing        | At N boundaries  | No exact aliasing      |
| Basis           | e^(2πikn/N)      | e^(2πi(k+1)φt + iθₖ)   |
| Computation     | O(N log N)       | O(N²) naive, O(N) per coeff |

KEY INNOVATION:
--------------
The RFT enables COMPUTATION IN THE WAVE DOMAIN:
- Binary data encodes as amplitude/phase
- Logic operations work directly on waveforms
- No need to decode back for intermediate results
"""

import numpy as np
from typing import Union, Tuple, Optional
from functools import lru_cache

from .gram_utils import gram_matrix, gram_inverse_sqrt

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio ≈ 1.618033988749895
PHI_INV = PHI - 1           # 1/φ ≈ 0.618033988749895 (also = φ - 1)


# =============================================================================
# CORE RFT FUNCTIONS
# =============================================================================

def rft_frequency(k: int) -> float:
    """
    Compute the k-th RFT resonant frequency.
    
    fₖ = (k+1) × φ
    
    These are the golden-ratio spaced frequencies that define
    the RFT basis functions.
    """
    return (k + 1) * PHI


def rft_phase(k: int) -> float:
    """
    Compute the k-th RFT golden phase offset.
    
    θₖ = 2π × k / φ
    
    This phase offset creates the quasi-periodic structure
    that prevents aliasing.
    """
    return 2 * np.pi * k / PHI


def rft_basis_function(k: int, t: np.ndarray) -> np.ndarray:
    """
    Compute the k-th RFT basis function Ψₖ(t).
    
    Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)
    
    Args:
        k: Basis function index (0, 1, 2, ...)
        t: Time array (normalized to [0, 1])
    
    Returns:
        Complex waveform of shape t.shape
    """
    f_k = rft_frequency(k)
    theta_k = rft_phase(k)
    return np.exp(2j * np.pi * f_k * t + 1j * theta_k)


@lru_cache(maxsize=32)
def rft_basis_matrix(
    N: int,
    T: Optional[int] = None,
    use_gram_normalization: bool = True,
) -> np.ndarray:
    """Build an RFT basis matrix.

    Backward-compatible behavior:
    - If `T` is provided and `T != N`, this returns the historical *waveform* basis
      Ψ of shape (N, T): Ψ[k,t] = exp(2πi f_k (t/T) + i θ_k).
      This matches the original implementation used for data→wave synthesis.

    Mathematically complete square-kernel behavior:
    - If `T` is None (or `T == N`), this returns a square (N×N) φ-grid exponential
      basis Φ with entries:

          Φ[k,n] = exp(j 2π φ k n / N) / √N

      If `use_gram_normalization=True`, the basis is orthonormalized via:

          Φ̃ = Φ (ΦᴴΦ)^{-1/2}

      yielding a unitary Φ̃ so that Φ̃ᴴ Φ̃ = I.

    References (orientation):
    - Oppenheim & Schafer, Discrete-Time Signal Processing (DFT orthogonality)
    - Christensen, An Introduction to Frames and Riesz Bases (Gram/frame operators)
    - Encyclopaedia Britannica, Fourier analysis
    """

    if T is None:
        T = int(N)
    N = int(N)
    T = int(T)

    # Square φ-grid exponential basis for rigorous finite-N inversion.
    #
    # Important: for discrete-time exponentials, frequency is periodic (defined mod 1
    # in cycles/sample). Folding to the fundamental band avoids severe finite-N
    # ill-conditioning that occurs when using large unwrapped frequencies.
    if T == N:
        n = np.arange(N, dtype=np.float64)
        k = np.arange(N, dtype=np.float64)
        # Normalized frequencies in [0, 1): f_k = frac((k+1) φ)
        f = np.mod((k + 1.0) * PHI, 1.0)
        # Φ[n,k] = exp(j 2π f_k n) / √N
        Phi = np.exp(1j * 2.0 * np.pi * np.outer(n, f)) / np.sqrt(float(N))
        if use_gram_normalization:
            G = gram_matrix(Phi)
            Phi = Phi @ gram_inverse_sqrt(G)
        return Phi.astype(np.complex128, copy=False)

    # Legacy waveform basis (non-square synthesis operator).
    t = np.arange(T) / T  # Normalized time [0, 1)
    Psi = np.zeros((N, T), dtype=np.complex128)
    for k in range(N):
        Psi[k, :] = rft_basis_function(k, t)
    return Psi


def rft_forward_frame(x: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Frame-correct coefficient extraction for a (generally non-orthogonal) basis.

    Given a full-rank square basis Φ, the dual-frame (exact) coefficients are:

        X = (Φᴴ Φ)^{-1} Φᴴ x

    This reduces to X = Φᴴ x when Φ is unitary.

    Reference: Christensen, An Introduction to Frames and Riesz Bases.
    """

    x = np.asarray(x, dtype=np.complex128)
    Phi = np.asarray(Phi, dtype=np.complex128)
    G = gram_matrix(Phi)
    return np.linalg.solve(G, Phi.conj().T @ x)


def rft_inverse_frame(X: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Synthesis for coefficients X given basis Φ: x = Φ X."""

    X = np.asarray(X, dtype=np.complex128)
    Phi = np.asarray(Phi, dtype=np.complex128)
    return Phi @ X


# =============================================================================
# FORWARD AND INVERSE RFT
# =============================================================================

def rft_forward(
    x: np.ndarray,
    T: Optional[int] = None,
    *,
    use_gram_normalization: bool = True,
    frame_correct: bool = False,
) -> np.ndarray:
    """
    Forward Resonant Fourier Transform.
    
    RFT(x)[t] = Σₖ x[k] × Ψₖ(t)
    
    Transforms discrete data into a continuous waveform.
    
    Args:
        x: Input data array of length N
          T: Number of output time samples (default: N * 16) for waveform synthesis.
              For a square (N×N) canonical transform, pass T=N and
              set `use_gram_normalization=True` (or `frame_correct=True`).
        use_gram_normalization: If True (square mode), uses Φ̃ = Φ(ΦᴴΦ)^{-1/2}.
        frame_correct: If True (square mode), uses dual-frame coefficients
           X = (ΦᴴΦ)^{-1} Φᴴ x.
    
    Returns:
        Complex waveform of length T
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    
    if T is None:
        T = N * 16

    if T == N and (use_gram_normalization or frame_correct):
        Phi = rft_basis_matrix(N, N, use_gram_normalization=use_gram_normalization)
        if frame_correct:
            return rft_forward_frame(x, Phi)
        return Phi.conj().T @ x

    Psi = rft_basis_matrix(N, T)
    return x @ Psi


def rft_inverse(
    W: np.ndarray,
    N: Optional[int] = None,
    *,
    use_gram_normalization: bool = True,
    frame_correct: bool = False,
) -> np.ndarray:
    """
    Inverse Resonant Fourier Transform.
    
    x[k] = ⟨W, Ψₖ⟩ = (1/T) Σₜ W[t] × Ψₖ*[t]
    
    Recovers discrete data from a waveform.
    
    Args:
        W: Complex waveform of length T (legacy waveform mode), or coefficient
           vector of length N (square-kernel mode).
        N: Number of data points to recover (defaults to len(W) in square mode)
        use_gram_normalization: If True (square mode), uses Φ̃ = Φ(ΦᴴΦ)^{-1/2}.
        frame_correct: If True (square mode), treats W as dual-frame coefficients.
    
    Returns:
        Recovered data array of length N
    """
    W = np.asarray(W, dtype=np.complex128)

    if N is None:
        N = int(W.shape[0])

    # Square-kernel reconstruction (coefficients → signal).
    if W.shape[0] == int(N) and (use_gram_normalization or frame_correct):
        Phi = rft_basis_matrix(int(N), int(N), use_gram_normalization=use_gram_normalization)
        return rft_inverse_frame(W, Phi)

    # Legacy waveform inverse via correlation.
    T = len(W)
    Psi = rft_basis_matrix(int(N), int(T))
    return (Psi @ np.conj(W)) / T


# =============================================================================
# BINARY RFT (for computation)
# =============================================================================

class BinaryRFT:
    """
    RFT specialized for binary data and wave-domain computation.
    
    Uses BPSK encoding: bit 0 → -1, bit 1 → +1
    This enables logic operations directly on waveforms.
    
    Usage:
        rft = BinaryRFT(num_bits=8)
        
        # Encode binary to wave
        wave = rft.encode(0b10101010)
        
        # Compute in wave domain
        result_wave = rft.wave_xor(wave_a, wave_b)
        
        # Decode back to binary
        result = rft.decode(result_wave)
    """
    
    def __init__(self, num_bits: int = 8, samples_per_bit: int = 16):
        """
        Initialize Binary RFT.
        
        Args:
            num_bits: Number of bits to encode
            samples_per_bit: Time samples per bit (affects wave resolution)
        """
        self.num_bits = num_bits
        self.T = num_bits * samples_per_bit
        self.t = np.arange(self.T) / self.T
        
        # Pre-compute basis functions (carriers)
        self._carriers = [rft_basis_function(k, self.t) for k in range(num_bits)]
    
    def encode(self, value: Union[int, bytes]) -> np.ndarray:
        """
        Encode binary data as RFT waveform.
        
        Args:
            value: Integer or bytes to encode
        
        Returns:
            Complex waveform
        """
        # Convert to bit array
        if isinstance(value, int):
            bits = [(value >> k) & 1 for k in range(self.num_bits)]
        elif isinstance(value, bytes):
            bits = []
            for byte in value:
                bits.extend([(byte >> k) & 1 for k in range(8)])
            bits = bits[:self.num_bits]
        else:
            bits = list(value)[:self.num_bits]
        
        # Pad if needed
        while len(bits) < self.num_bits:
            bits.append(0)
        
        # BPSK encoding: 0 → -1, 1 → +1
        waveform = np.zeros(self.T, dtype=np.complex128)
        for k in range(self.num_bits):
            symbol = 2 * bits[k] - 1
            waveform += symbol * self._carriers[k]
        
        return waveform / np.sqrt(self.num_bits)
    
    def decode(self, waveform: np.ndarray) -> int:
        """
        Decode RFT waveform back to integer.
        
        Args:
            waveform: Complex waveform
        
        Returns:
            Decoded integer value
        """
        bits = []
        for k in range(self.num_bits):
            # Correlate with carrier (matched filter)
            corr = np.sum(waveform * np.conj(self._carriers[k])) / self.T
            # BPSK decision
            bits.append(1 if np.real(corr) > 0 else 0)
        
        return sum(b << k for k, b in enumerate(bits))
    
    def _get_symbol(self, waveform: np.ndarray, k: int) -> float:
        """Extract BPSK symbol for bit k."""
        corr = np.sum(waveform * np.conj(self._carriers[k]))
        return np.sign(np.real(corr))
    
    # =========================================================================
    # WAVE-DOMAIN LOGIC OPERATIONS
    # =========================================================================
    
    def wave_xor(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """XOR operation in wave domain."""
        result = np.zeros(self.T, dtype=np.complex128)
        for k in range(self.num_bits):
            sym1 = self._get_symbol(w1, k)
            sym2 = self._get_symbol(w2, k)
            sym_xor = -sym1 * sym2  # XOR = negate product in BPSK
            result += sym_xor * self._carriers[k]
        return result / np.sqrt(self.num_bits)
    
    def wave_and(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """AND operation in wave domain."""
        result = np.zeros(self.T, dtype=np.complex128)
        for k in range(self.num_bits):
            sym1 = self._get_symbol(w1, k)
            sym2 = self._get_symbol(w2, k)
            sym_and = 1.0 if (sym1 > 0 and sym2 > 0) else -1.0
            result += sym_and * self._carriers[k]
        return result / np.sqrt(self.num_bits)
    
    def wave_or(self, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
        """OR operation in wave domain."""
        result = np.zeros(self.T, dtype=np.complex128)
        for k in range(self.num_bits):
            sym1 = self._get_symbol(w1, k)
            sym2 = self._get_symbol(w2, k)
            sym_or = 1.0 if (sym1 > 0 or sym2 > 0) else -1.0
            result += sym_or * self._carriers[k]
        return result / np.sqrt(self.num_bits)
    
    def wave_not(self, w: np.ndarray) -> np.ndarray:
        """NOT operation in wave domain."""
        return -w
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def frequencies(self) -> np.ndarray:
        """Return the RFT resonant frequencies."""
        return np.array([rft_frequency(k) for k in range(self.num_bits)])
    
    @property 
    def phases(self) -> np.ndarray:
        """Return the RFT golden phases."""
        return np.array([rft_phase(k) for k in range(self.num_bits)])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def rft(x: np.ndarray) -> np.ndarray:
    """Quick forward RFT."""
    return rft_forward(x)


def irft(W: np.ndarray, N: int) -> np.ndarray:
    """Quick inverse RFT."""
    return rft_inverse(W, N)


# =============================================================================
# RFT-SIS CRYPTOGRAPHIC HASH
# =============================================================================

class RFTSISHash:
    """
    Hybrid RFT-SIS hash (research prototype): φ-structured + pseudo-random matrix.

    Construction: A = A_φ + R (mod q)
    - A_φ: Deterministic φ-structured matrix (golden ratio equidistribution)
    - R: Pseudo-random matrix from NumPy RNG (seeded for reproducibility)

    Security notes:
    - Any SIS-style argument depends on R being uniform and unpredictable.
        This implementation uses NumPy RNG for reproducible experiments, which
        is NOT a CSPRNG and should not be used for security claims.
    - Avalanche results are empirical only.
    - Per-domain salting is a research convenience, not a formal guarantee.

    Process:
    1. Expand input via SHA3 hash chain (amplifies tiny differences)
    2. Transform through RFT (golden-ratio structure)
    3. Quantize to SIS short vector
    4. Compute A @ s mod q (lattice point)
    5. Final SHA3 compression

    WARNING: Research implementation only. Not audited for production use.
    For production, use standard, reviewed primitives.
    """

    def __init__(self, 
                 sis_n: int = 512,
                 sis_m: int = 1024,
                 sis_q: int = 3329,
                 sis_beta: int = 100,
                 domain_salt: bytes = b"RFT_SIS_DEFAULT_DOMAIN_2026"):
        import hashlib
        
        self.sis_n = sis_n
        self.sis_m = sis_m
        self.sis_q = sis_q
        self.sis_beta = sis_beta
        self.domain_salt = domain_salt
        
        # Build RFT basis matrix
        self._build_rft_matrix()
        
        # Build hybrid SIS matrix: A = A_φ + R (mod q)
        self.A = self._build_hybrid_sis_matrix()
    
    def _build_rft_matrix(self):
        """Build N×N RFT basis matrix."""
        N = self.sis_n
        self._rft_matrix = np.zeros((N, N), dtype=np.complex128)
        t = np.arange(N) / N
        for k in range(N):
            self._rft_matrix[k, :] = rft_basis_function(k, t)
    
    def _build_hybrid_sis_matrix(self) -> np.ndarray:
        """
        Build hybrid SIS matrix: A = A_φ + R (mod q)
        
        This construction provides:
        - A_φ: Deterministic φ-structured matrix (equidistribution from golden ratio)
        - R: Pseudo-random matrix from a reproducible NumPy RNG seed
        
        Security note:
        Any reduction to standard SIS assumes R is uniform and unpredictable.
        Here R is generated for research reproducibility and should not be
        treated as a cryptographic randomness source.
        
        The φ-structure in A_φ is retained for mathematical aesthetics and
        potential future algorithmic applications; security assumptions are
        explicitly out of scope for this implementation.
        """
        import hashlib
        
        m, n, q = self.sis_m, self.sis_n, self.sis_q
        
        # Build A_φ: deterministic φ-structured matrix
        # Use (i+1)*(j+1) to avoid zero row/column at i=0 or j=0
        A_phi = np.zeros((m, n), dtype=np.int64)
        for i in range(m):
            for j in range(n):
                # φ equidistribution: floor(q * frac((i+1)*(j+1) * φ))
                val = (i + 1) * (j + 1) * PHI
                A_phi[i, j] = int((val % 1.0) * q)
        
        # Build R: random matrix from salted PRNG
        # Derive seed from domain salt via SHA3 to get reproducible but unique per-domain
        seed_hash = hashlib.sha3_256(self.domain_salt).digest()
        seed_int = int.from_bytes(seed_hash[:4], 'little')
        rng = np.random.default_rng(seed_int)
        R = rng.integers(0, q, size=(m, n), dtype=np.int64)
        
        # Combine: A = A_φ + R (mod q)
        A = (A_phi + R) % q
        
        return A.astype(np.int32)
    
    def _expand_input(self, data: bytes) -> np.ndarray:
        """
        Expand input bytes to sis_n floats via hash chain.
        This amplifies small differences before they get lost in quantization.
        """
        import hashlib
        
        expanded = np.zeros(self.sis_n, dtype=np.float64)
        
        # Hash chain expansion
        current_hash = data
        idx = 0
        
        while idx < self.sis_n:
            h = hashlib.sha3_256(current_hash if isinstance(current_hash, bytes) else current_hash.encode())
            digest = h.digest()
            
            # Convert hash bytes to floats in [-1, 1]
            for i in range(0, min(32, self.sis_n - idx), 4):
                uint_val = int.from_bytes(digest[i:i+4], 'little')
                float_val = (uint_val / (2**32)) * 2 - 1
                expanded[idx] = float_val
                idx += 1
                if idx >= self.sis_n:
                    break
            
            # Chain with original data
            current_hash = digest + (data if isinstance(data, bytes) else data.encode())
        
        return expanded
    
    def _rft_transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply RFT and extract magnitude/phase features."""
        transformed = self._rft_matrix @ signal.astype(np.complex128)
        
        magnitude = np.abs(transformed)
        phase = np.angle(transformed)
        
        # Combine magnitude and phase with golden ratio
        return magnitude * (1 + PHI * np.cos(phase))
    
    def _quantize(self, v: np.ndarray) -> np.ndarray:
        """Quantize to short integer vector for SIS."""
        max_val = np.max(np.abs(v))
        if max_val < 1e-15:
            max_val = 1e-15
        
        scaled = v * (self.sis_beta * 0.95) / max_val
        s = np.round(scaled).astype(np.int32)
        return np.clip(s, -self.sis_beta, self.sis_beta)
    
    def hash(self, data: bytes) -> bytes:
        """
        Hash arbitrary bytes using RFT-SIS.
        
        Args:
            data: Input bytes
        
        Returns:
            32-byte (256-bit) hash
        """
        import hashlib
        
        # 1. Expand input
        expanded = self._expand_input(data)
        
        # 2. RFT transform
        rft_output = self._rft_transform(expanded)
        
        # 3. Quantize to SIS vector
        s = self._quantize(rft_output)
        
        # 4. SIS computation: A @ s mod q
        lattice_point = self.A.astype(np.int64) @ s.astype(np.int64)
        lattice_point = lattice_point % self.sis_q
        lattice_point[lattice_point > self.sis_q // 2] -= self.sis_q
        
        # 5. Final hash
        h = hashlib.sha3_256()
        h.update(lattice_point.tobytes())
        h.update(b"RFT_SIS_CANONICAL")
        
        return h.digest()
    
    def hash_hex(self, data: bytes) -> str:
        """Return hash as hex string."""
        return self.hash(data).hex()


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RESONANT FOURIER TRANSFORM (RFT) - Test Suite")
    print("=" * 60)
    
    # Test 1: Forward/Inverse
    print("\n1. Forward/Inverse RFT")
    print("-" * 40)
    x = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    W = rft_forward(x)
    x_rec = rft_inverse(W, len(x))
    print(f"Original:  {x}")
    print(f"Recovered: {np.round(np.real(x_rec), 2)}")
    
    # Test 2: Binary RFT
    print("\n2. Binary RFT (Wave-Domain Computation)")
    print("-" * 40)
    brft = BinaryRFT(num_bits=8)
    
    for val in [0b10101010, 0b11110000, 0xFF, 0x00]:
        wave = brft.encode(val)
        recovered = brft.decode(wave)
        status = "✓" if val == recovered else "✗"
        print(f"{val:08b} → encode → decode → {recovered:08b} {status}")
    
    # Test 3: Wave Logic
    print("\n3. Wave-Domain Logic Operations")
    print("-" * 40)
    a, b = 0b10101010, 0b11001100
    wa, wb = brft.encode(a), brft.encode(b)
    
    xor_result = brft.decode(brft.wave_xor(wa, wb))
    and_result = brft.decode(brft.wave_and(wa, wb))
    or_result = brft.decode(brft.wave_or(wa, wb))
    
    print(f"XOR: {a:08b} ^ {b:08b} = {a^b:08b}, got {xor_result:08b} {'✓' if xor_result == a^b else '✗'}")
    print(f"AND: {a:08b} & {b:08b} = {a&b:08b}, got {and_result:08b} {'✓' if and_result == a&b else '✗'}")
    print(f"OR:  {a:08b} | {b:08b} = {a|b:08b}, got {or_result:08b} {'✓' if or_result == a|b else '✗'}")
    
    print("\n" + "=" * 60)
    print("✓ RFT Tests Passed")
    print("=" * 60)
