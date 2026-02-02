# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
H3-ARFT Integration
Extends the H3 Hierarchical Cascade to use the Operator-Based ARFT for the texture component.
This replaces the standard RFT with the signal-adaptive ARFT kernel.

Note on "Coherence":
    The coherence metric measures the normalized correlation between the
    structure and texture components. For an ideal decomposition, this
    should be close to zero (orthogonal components).
    
    coherence = |<structure, texture>| / (||structure|| * ||texture||)
    
    This is NOT the same as quantum coherence or signal-processing coherence
    functions. It's a simple measure of decomposition quality.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import time
from scipy.fft import dct, idct

# Import existing H3 for inheritance/reference
from algorithms.rft.hybrids.cascade_hybrids import H3HierarchicalCascade, CascadeResult
from algorithms.rft.kernels.operator_arft_kernel import build_operator_kernel, arft_forward


def compute_decomposition_coherence(structure: np.ndarray, texture: np.ndarray) -> float:
    """
    Compute normalized correlation between decomposition components.
    
    This measures how "orthogonal" the structure/texture split is.
    Values close to 0 indicate good separation; values near 1 indicate
    redundant or highly correlated components.
    
    Args:
        structure: Structure component of signal
        texture: Texture component of signal
        
    Returns:
        Coherence value in [0, 1]
    """
    norm_s = np.linalg.norm(structure)
    norm_t = np.linalg.norm(texture)
    
    if norm_s < 1e-10 or norm_t < 1e-10:
        return 0.0  # One component is effectively zero
    
    inner = np.abs(np.dot(structure, texture))
    return float(inner / (norm_s * norm_t))


class H3ARFTCascade(H3HierarchicalCascade):
    """
    H3-ARFT Hybrid Cascade Transform
    
    Replaces the standard RFT texture transform with the Operator-Based ARFT.
    The ARFT kernel is derived from the autocorrelation of the texture component.
    
    Mathematical Foundation:
        x = x_structure + x_texture
        C = {DCT(x_structure), ARFT(x_texture)}
        
    Where ARFT is the eigenbasis of the texture's autocorrelation operator.
    
    Note: This is a PROTOTYPE implementation. For production compression,
    use the native C++ pipeline (rftmw_native) with ANS entropy coding.
    """
    
    def __init__(self, kernel_size_ratio: float = 1/32):
        super().__init__(kernel_size_ratio)
        self.variant_name = "H3_ARFT_Cascade"
        self.last_kernel = None  # Store kernel for decoding
        self.last_decomposition = None  # Store for coherence calculation
        
    def encode(self, signal: np.ndarray, sparsity_target: float = 0.95) -> CascadeResult:
        start_time = time.perf_counter()
        
        # 1. Decompose Signal
        structure, texture = self._decompose(signal)
        self.last_decomposition = (structure, texture)
        
        # Compute actual coherence (not hardcoded!)
        coherence = compute_decomposition_coherence(structure, texture)
        
        # 2. Transform Structure (DCT)
        # Use standard DCT-II with ortho normalization
        struct_coeffs = dct(structure, norm='ortho')
        
        # 3. Transform Texture (ARFT)
        # Build adaptive kernel from texture autocorrelation
        N = len(texture)
        autocorr = np.correlate(texture, texture, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr[:N]
        
        # Build and store kernel
        self.last_kernel = build_operator_kernel(N, autocorr)
        
        # Apply ARFT
        texture_coeffs = arft_forward(texture, self.last_kernel)
        
        # 4. Combine Coefficients (70/30 split logic from H3)
        # We keep 70% of structure coeffs and 30% of texture coeffs
        # This is a simplified selection logic for the prototype
        
        # Sort by magnitude to pick top coefficients
        struct_indices = np.argsort(np.abs(struct_coeffs))[::-1]
        texture_indices = np.argsort(np.abs(texture_coeffs))[::-1]
        
        n_struct = int(0.7 * N)
        n_texture = int(0.3 * N)
        
        # Create sparse coefficient vectors
        final_struct = np.zeros_like(struct_coeffs)
        final_texture = np.zeros_like(texture_coeffs)
        
        final_struct[struct_indices[:n_struct]] = struct_coeffs[struct_indices[:n_struct]]
        final_texture[texture_indices[:n_texture]] = texture_coeffs[texture_indices[:n_texture]]
        
        # Combine for storage (interleaved or concatenated)
        combined_coeffs = np.concatenate([final_struct, final_texture])
        
        # 5. Calculate Metrics
        # BPP is estimated from non-zero count (actual BPP requires entropy coding)
        # NOTE: This is an ESTIMATE. For true BPP, use bench_h3_codec_real_bpp.py
        threshold = 0.01 * np.max(np.abs(combined_coeffs))  # 1% of max
        nz_count = np.count_nonzero(np.abs(combined_coeffs) > threshold)
        estimated_bpp = (nz_count * 16) / N  # 16 bits per non-zero coeff
        
        # Calculate PSNR
        # Reconstruct to measure error
        rec_struct = idct(final_struct, norm='ortho')
        rec_texture = self.last_kernel @ final_texture  # Inverse ARFT
        rec_signal = rec_struct + rec_texture
        
        mse = np.mean((signal - rec_signal) ** 2)
        max_val = np.max(np.abs(signal))
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else 100.0
        
        return CascadeResult(
            coefficients=combined_coeffs,
            bpp=estimated_bpp,
            coherence=coherence,  # Now actually computed!
            sparsity=1.0 - (nz_count / len(combined_coeffs)),
            variant=self.variant_name,
            time_ms=(time.perf_counter() - start_time) * 1000,
            psnr=psnr
        )

    def decode(self, coefficients: np.ndarray, original_length: int) -> np.ndarray:
        """
        Decode signal from cascade coefficients.
        
        Args:
            coefficients: Concatenated structure + texture coefficients
            original_length: Original signal length
            
        Returns:
            Reconstructed signal
        """
        N = original_length
        
        # Split coefficients
        # Note: encode() concatenates [final_struct, final_texture]
        # Both are length N
        struct_coeffs = coefficients[:N]
        texture_coeffs = coefficients[N:]
        
        # Inverse DCT for structure
        rec_struct = idct(struct_coeffs, norm='ortho')
        
        # Inverse ARFT for texture
        if self.last_kernel is None:
            # Fallback if kernel not available (e.g. cross-session decode)
            # In a real codec, the kernel (or its parameters) would be transmitted
            # For this prototype, we assume stateful decoding
            raise RuntimeError("ARFT kernel missing. Cannot decode without kernel state.")
            
        rec_texture = self.last_kernel @ texture_coeffs
        
        return rec_struct + rec_texture

