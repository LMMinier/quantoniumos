# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
True Wave-Domain Computation
============================

This module implements computations that GENUINELY operate in the wave/spectral
domain WITHOUT decoding to binary. These are mathematically valid continuous
operations.

KEY INSIGHT: Not all computation needs to be binary. Many useful operations
are naturally continuous:

✓ WAVE-NATIVE (no decoding needed):
  - Linear combinations (superposition)
  - Convolution (via spectral multiply)
  - Filtering (spectral masking)
  - Inner products / correlations
  - Matrix-vector products
  - Fourier/wavelet analysis

✗ BINARY-ONLY (requires thresholding):
  - XOR, AND, OR, NOT on bits
  - Comparison (>, <, ==)
  - Branching (if/else)

The φ-grid basis provides a specific advantage for CERTAIN signal classes
(quasi-periodic with golden-ratio structure) per Theorem 8.

This is NOT "post-binary" in the sense of replacing digital logic.
This IS legitimate analog/spectral signal processing.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from functools import lru_cache

PHI = (1 + np.sqrt(5)) / 2


class WaveComputer:
    """
    Genuine wave-domain computation engine.
    
    All operations here work on continuous complex amplitudes.
    NO bit extraction. NO thresholding (until final readout if needed).
    
    Mathematical basis: Linear algebra over ℂ^N
    """
    
    def __init__(self, N: int = 256):
        """
        Initialize wave computer.
        
        Args:
            N: Dimension of wave space
        """
        self.N = N
        self._build_phi_basis()
    
    def _build_phi_basis(self):
        """Build the φ-grid exponential basis and its Gram normalization."""
        N = self.N
        n = np.arange(N)
        
        # Frequencies: f_k = frac((k+1)φ)
        k = np.arange(N)
        f = np.mod((k + 1) * PHI, 1.0)
        
        # Raw basis Φ[n,k] = exp(i2π f_k n) / √N
        self.Phi_raw = np.exp(2j * np.pi * np.outer(n, f)) / np.sqrt(N)
        
        # Gram matrix and its inverse sqrt
        G = self.Phi_raw.conj().T @ self.Phi_raw
        G_inv_sqrt = np.linalg.inv(np.linalg.cholesky(G)).conj().T
        
        # Canonical unitary basis U = Φ(Φ†Φ)^{-1/2}
        self.U = self.Phi_raw @ G_inv_sqrt
        self.U_H = self.U.conj().T
        
        # Also keep FFT for comparison
        self.F = np.fft.fft(np.eye(N), axis=0, norm='ortho')
        self.F_H = self.F.conj().T
    
    # =========================================================================
    # ENCODING: Signal → Wave coefficients
    # =========================================================================
    
    def encode_rft(self, x: np.ndarray) -> np.ndarray:
        """Transform signal to RFT coefficients (stays in ℂ)."""
        return self.U_H @ x
    
    def decode_rft(self, c: np.ndarray) -> np.ndarray:
        """Transform RFT coefficients back to signal."""
        return self.U @ c
    
    def encode_fft(self, x: np.ndarray) -> np.ndarray:
        """Transform signal to FFT coefficients."""
        return self.F_H @ x
    
    def decode_fft(self, c: np.ndarray) -> np.ndarray:
        """Transform FFT coefficients back to signal."""
        return self.F @ c
    
    # =========================================================================
    # TRUE WAVE-NATIVE OPERATIONS (no decoding!)
    # =========================================================================
    
    def superposition(self, c1: np.ndarray, c2: np.ndarray, 
                     alpha: complex = 1.0, beta: complex = 1.0) -> np.ndarray:
        """
        Linear superposition in coefficient space.
        
        c_out = α·c1 + β·c2
        
        This is GENUINELY wave-native: no decoding, no thresholding.
        Physical analog: wave interference.
        """
        return alpha * c1 + beta * c2
    
    def spectral_multiply(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """
        Pointwise multiplication in spectral domain.
        
        c_out[k] = c1[k] · c2[k]
        
        This implements CONVOLUTION in the signal domain:
            decode(spectral_multiply(encode(x), encode(h))) = x ⊛ h
        
        GENUINELY wave-native: standard DSP, not fake "wave logic".
        """
        return c1 * c2
    
    def spectral_filter(self, c: np.ndarray, 
                        mask: np.ndarray) -> np.ndarray:
        """
        Apply spectral mask/filter.
        
        c_out = c ⊙ mask
        
        GENUINELY wave-native: frequency-domain filtering.
        """
        return c * mask
    
    def inner_product(self, c1: np.ndarray, c2: np.ndarray) -> complex:
        """
        Inner product in coefficient space: ⟨c1, c2⟩ = c1† · c2
        
        GENUINELY wave-native: correlation/similarity measure.
        """
        return np.vdot(c1, c2)
    
    def projection(self, c: np.ndarray, 
                   subspace_indices: np.ndarray) -> np.ndarray:
        """
        Project onto subspace (keep only certain coefficients).
        
        This is dimensionality reduction in spectral domain.
        GENUINELY wave-native.
        """
        result = np.zeros_like(c)
        result[subspace_indices] = c[subspace_indices]
        return result
    
    def energy(self, c: np.ndarray) -> float:
        """
        Total energy: ||c||² = Σ|c_k|²
        
        By Parseval: same as signal energy ||x||².
        """
        return float(np.sum(np.abs(c)**2))
    
    def phase_shift(self, c: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """
        Apply phase shifts to each coefficient.
        
        c_out[k] = c[k] · exp(i·phases[k])
        
        GENUINELY wave-native: phase modulation.
        """
        return c * np.exp(1j * phases)
    
    def amplitude_scale(self, c: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        Scale amplitudes of each coefficient.
        
        c_out[k] = c[k] · scales[k]  (scales are real)
        
        GENUINELY wave-native: amplitude modulation.
        """
        return c * scales

    def spectral_shift(self, c: np.ndarray, bins: int) -> np.ndarray:
        """Circularly shift coefficients by `bins`.

        This is a wave-space composition primitive (works for either FFT or RFT
        coefficient vectors).
        """

        return np.roll(c, int(bins))

    def spectral_nonlinearity_power(self, c: np.ndarray, power: float = 1.0) -> np.ndarray:
        """Apply a magnitude-only nonlinearity, preserving phase.

        c_out[k] = |c[k]|**power * exp(i*angle(c[k]))

        Note: This is nonlinear; it is wave-domain but not unitary.
        """

        c = np.asarray(c, dtype=np.complex128)
        mag = np.abs(c)
        phase = np.exp(1j * np.angle(c))
        return (mag**float(power)) * phase

    def conditional_select_soft(
        self,
        condition: np.ndarray,
        then_branch: np.ndarray,
        else_branch: np.ndarray,
        *,
        dc_index: int = 0,
    ) -> np.ndarray:
        """Soft, differentiable conditional selection based on DC energy.

        This is NOT a binary IF/ELSE. It interpolates between branches using a
        scalar derived from the condition spectrum.
        """

        condition = np.asarray(condition, dtype=np.complex128)
        then_branch = np.asarray(then_branch, dtype=np.complex128)
        else_branch = np.asarray(else_branch, dtype=np.complex128)

        denom = float(np.sum(np.abs(condition) ** 2))
        if denom <= 0.0:
            weight_then = 0.0
        else:
            dc_energy = float(np.abs(condition[int(dc_index)]) ** 2)
            weight_then = dc_energy / denom

        weight_then = float(np.clip(weight_then, 0.0, 1.0))
        weight_else = 1.0 - weight_then
        return weight_then * then_branch + weight_else * else_branch

    # ---------------------------------------------------------------------
    # Heuristic “soft logic” (masking-based). These are NOT true bitwise gates.
    # ---------------------------------------------------------------------

    def spectral_support_mask(self, c: np.ndarray, *, frac_of_max: float = 0.1) -> np.ndarray:
        """Return a boolean support mask based on magnitude thresholding."""

        c = np.asarray(c)
        mag = np.abs(c)
        thr = float(frac_of_max) * float(np.max(mag) if mag.size else 0.0)
        return mag > thr

    def wave_and_soft(self, c1: np.ndarray, c2: np.ndarray, *, frac_of_max: float = 0.1) -> np.ndarray:
        """Heuristic AND via overlap of spectral support masks."""

        c1 = np.asarray(c1, dtype=np.complex128)
        c2 = np.asarray(c2, dtype=np.complex128)
        m1 = self.spectral_support_mask(c1, frac_of_max=frac_of_max)
        m2 = self.spectral_support_mask(c2, frac_of_max=frac_of_max)
        mask = m1 & m2
        return mask.astype(np.complex128) * (c1 + c2) / 2.0

    def wave_or_soft(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """Heuristic OR via linear superposition."""

        return np.asarray(c1, dtype=np.complex128) + np.asarray(c2, dtype=np.complex128)

    def wave_not_soft(self, c: np.ndarray) -> np.ndarray:
        """Heuristic NOT via phase flip (DC treated specially)."""

        c = np.asarray(c, dtype=np.complex128)
        out = c.copy()
        if out.size:
            out[0] = -out[0]
            out[1:] = -out[1:]
        return out
    
    # =========================================================================
    # WAVE-DOMAIN LINEAR ALGEBRA
    # =========================================================================
    
    def matvec_spectral(self, c: np.ndarray, 
                        diagonal: np.ndarray) -> np.ndarray:
        """
        Matrix-vector product where matrix is diagonal in spectral basis.
        
        If A = U · diag(d) · U†, then:
            A @ x = U @ (d ⊙ (U† @ x))
                  = decode(d ⊙ encode(x))
        
        This is O(N) in spectral domain vs O(N²) in signal domain!
        GENUINELY wave-native: eigenvalue-weighted transform.
        """
        return diagonal * c
    
    def solve_diagonal(self, c: np.ndarray, 
                       diagonal: np.ndarray, 
                       regularization: float = 1e-10) -> np.ndarray:
        """
        Solve Ax = b where A is diagonal in spectral basis.
        
        x = A⁻¹b becomes trivial division in spectral domain.
        GENUINELY wave-native: spectral inversion.
        """
        safe_diag = np.where(np.abs(diagonal) > regularization, 
                            diagonal, regularization)
        return c / safe_diag
    
    # =========================================================================
    # WAVE-DOMAIN NEURAL NETWORK PRIMITIVES
    # =========================================================================
    
    def spectral_linear_layer(self, c: np.ndarray, 
                              weights: np.ndarray, 
                              bias: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Linear layer operating on spectral coefficients.
        
        y = W @ c + b
        
        If W is structured (e.g., circulant → diagonal in Fourier),
        this can be very efficient.
        """
        result = weights @ c
        if bias is not None:
            result = result + bias
        return result
    
    def spectral_activation(self, c: np.ndarray, 
                            activation: str = 'modulus') -> np.ndarray:
        """
        Nonlinear activation in spectral domain.
        
        Options:
        - 'modulus': |c| (amplitude only, loses phase)
        - 'phase': c/|c| (phase only, loses amplitude)  
        - 'softshrink': soft thresholding for sparsity
        - 'cardioid': Re(c) + |c| (keeps some phase info)
        
        NOTE: This IS a nonlinearity, so information is lost.
        But it's a wave-native nonlinearity, not binary thresholding.
        """
        if activation == 'modulus':
            return np.abs(c)
        elif activation == 'phase':
            return c / (np.abs(c) + 1e-10)
        elif activation == 'softshrink':
            # Soft thresholding: sign(c) * max(|c| - τ, 0)
            tau = 0.1 * np.max(np.abs(c))
            magnitude = np.maximum(np.abs(c) - tau, 0)
            phase = np.angle(c)
            return magnitude * np.exp(1j * phase)
        elif activation == 'cardioid':
            # Cardioid: maps ℂ → ℝ while partially preserving phase
            return np.real(c) + np.abs(c)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    # =========================================================================
    # CONVENIENCE: Full signal-domain operations via spectral
    # =========================================================================
    
    def convolve(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Convolve two signals using spectral multiplication.
        
        x ⊛ h = IFFT(FFT(x) · FFT(h))
        
        This is the classic FFT convolution theorem.
        """
        cx = self.encode_fft(x)
        ch = self.encode_fft(h)
        return self.decode_fft(self.spectral_multiply(cx, ch))
    
    def filter_signal(self, x: np.ndarray, 
                      freq_response: np.ndarray) -> np.ndarray:
        """
        Filter signal using frequency response.
        
        y = IFFT(FFT(x) · H)
        """
        cx = self.encode_fft(x)
        return self.decode_fft(self.spectral_filter(cx, freq_response))


# =============================================================================
# DEMONSTRATION: What's genuinely wave-native vs what requires decoding
# =============================================================================

def demonstrate_true_wave_compute():
    """Show what TRUE wave-domain computation looks like."""
    
    print("=" * 70)
    print("TRUE WAVE-DOMAIN COMPUTATION DEMONSTRATION")
    print("=" * 70)
    
    N = 64
    wc = WaveComputer(N)
    
    # Create two test signals
    t = np.linspace(0, 1, N, endpoint=False)
    x1 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    x2 = np.cos(2 * np.pi * 5 * t)
    
    # Encode to spectral domain
    c1 = wc.encode_fft(x1)
    c2 = wc.encode_fft(x2)
    
    print("\n1. SUPERPOSITION (wave-native)")
    print("   c_out = 0.7·c1 + 0.3·c2")
    c_super = wc.superposition(c1, c2, alpha=0.7, beta=0.3)
    x_super = wc.decode_fft(c_super)
    print(f"   Energy in: {wc.energy(c1):.2f}, {wc.energy(c2):.2f}")
    print(f"   Energy out: {wc.energy(c_super):.2f}")
    print("   ✓ No decoding to binary. Pure linear combination.")
    
    print("\n2. CONVOLUTION via spectral multiply (wave-native)")
    h = np.zeros(N)
    h[:5] = 1/5  # Simple averaging filter
    ch = wc.encode_fft(h)
    c_conv = wc.spectral_multiply(c1, ch)
    x_conv = wc.decode_fft(c_conv)
    print(f"   Computed x1 ⊛ h in spectral domain")
    print(f"   Original energy: {np.sum(x1**2):.2f}")
    print(f"   Filtered energy: {np.sum(np.abs(x_conv)**2):.2f}")
    print("   ✓ No decoding to binary. Standard DSP.")
    
    print("\n3. INNER PRODUCT (wave-native)")
    similarity = wc.inner_product(c1, c2)
    print(f"   ⟨c1, c2⟩ = {similarity:.4f}")
    print("   ✓ Correlation computed entirely in spectral domain.")
    
    print("\n4. PROJECTION to sparse subspace (wave-native)")
    # Keep only top 10 coefficients by magnitude
    magnitudes = np.abs(c1)
    top_k = np.argsort(magnitudes)[-10:]
    c_sparse = wc.projection(c1, top_k)
    x_sparse = wc.decode_fft(c_sparse)
    reconstruction_error = np.linalg.norm(x1 - x_sparse) / np.linalg.norm(x1)
    print(f"   Kept 10/{N} coefficients")
    print(f"   Reconstruction error: {reconstruction_error:.2%}")
    print("   ✓ Dimensionality reduction in spectral domain.")
    
    print("\n5. PHASE SHIFT (wave-native)")
    random_phases = np.random.uniform(0, 2*np.pi, N)
    c_shifted = wc.phase_shift(c1, random_phases)
    print(f"   Applied random phase shifts")
    print(f"   Energy preserved: {wc.energy(c1):.4f} → {wc.energy(c_shifted):.4f}")
    print("   ✓ Unitary operation, no information loss.")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
These operations are GENUINELY wave-native because:
1. They operate on continuous complex amplitudes
2. No thresholding or bit extraction
3. Mathematically valid linear/spectral operations
4. Used in real DSP, communications, and scientific computing

This is NOT "post-binary" in the sense of replacing digital logic.
This IS legitimate signal processing in the frequency domain.

The φ-grid RFT adds value for SPECIFIC signal classes (Theorem 8):
- Golden quasi-periodic signals concentrate better in RFT than FFT
- This is a proven mathematical advantage for that signal family
- It does NOT replace general computation
""")


if __name__ == "__main__":
    demonstrate_true_wave_compute()
