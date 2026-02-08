#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Quantonium-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# License: LICENSE-CLAIMS-NC.md (Research/Education Only)
"""Hybrid Φ-RFT / DCT decomposition utilities for Theorem 10 tests.

This module implements:
  - Φ-RFT forward/inverse transforms (unitary, FFT-based)
  - A hybrid DCT + Φ-RFT decomposition (structure + texture)
  - An adaptive front-end that chooses thresholds/ordering from signal features

This version integrates the canonical closed-form unitary transforms from
`algorithms.rft.core.closed_form_rft` and all 7 variants from
`algorithms.rft.variants.registry` to provide maximum flexibility for
breaking the 'ASCII bottleneck':

  variant="standard"       : Original β·frac(k/φ) phase (baseline, fully proven)
  variant="logphi"         : Log-frequency warped phase (low-freq emphasis)
  variant="mixed"          : Convex blend of standard + logphi
  variant="original"       : Matrix-based original Φ-RFT from registry
  variant="harmonic_phase" : Cubic time-base for nonlinear filtering
  variant="fibonacci_tilt" : Integer lattice alignment for crypto
  variant="chaotic_mix"    : Haar-like randomness for secure scrambling
  variant="geometric_lattice" : Phase-engineered for analog/optical
  variant="phi_chaotic_hybrid": Structure + disorder for resilient codecs
  variant="adaptive_phi"   : Meta selection for universal compression
  variant="log_periodic"   : Log-frequency phase warp for symbols
  variant="convex_mix"     : Hybrid log/standard adaptive textures
  variant="golden_ratio_exact": Full resonance lattice for validation

All variants remain unitary (error < 10⁻¹⁴ in exact arithmetic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import dct, idct
from numpy.fft import fft, ifft
from scipy.stats import kurtosis

# Import canonical closed-form RFT
try:
    from algorithms.rft.core.resonant_fourier_transform import (
        rft_forward_square as canonical_rft_forward,
        rft_inverse_square as canonical_rft_inverse,
        PHI as CANONICAL_PHI,
    )
    HAS_CANONICAL_RFT = True
except ImportError:
    HAS_CANONICAL_RFT = False

# Import all 7 variants from registry
try:
    from algorithms.rft.variants.registry import (
        VARIANTS,
        generate_original_phi_rft,
        generate_harmonic_phase,
        generate_fibonacci_tilt,
        generate_chaotic_mix,
        generate_geometric_lattice,
        generate_phi_chaotic_hybrid,
        generate_adaptive_phi,
        generate_log_periodic_phi_rft,
        generate_convex_mixed_phi_rft,
        generate_exact_golden_ratio_unitary,
    )
    HAS_VARIANTS = True
except ImportError:
    HAS_VARIANTS = False
    VARIANTS = {}

PHI: float = (1.0 + np.sqrt(5.0)) / 2.0


# ---------------------------------------------------------------------------
# Variant matrix cache (for matrix-based variants)
# ---------------------------------------------------------------------------
_VARIANT_MATRIX_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


def _get_variant_matrix(variant: str, n: int) -> Optional[np.ndarray]:
    """Get or generate the unitary matrix for a registry variant."""
    if not HAS_VARIANTS:
        return None
    
    cache_key = (variant, n)
    if cache_key in _VARIANT_MATRIX_CACHE:
        return _VARIANT_MATRIX_CACHE[cache_key]
    
    if variant not in VARIANTS:
        return None
    
    matrix = VARIANTS[variant].generator(n)
    _VARIANT_MATRIX_CACHE[cache_key] = matrix
    return matrix


def list_available_variants() -> Dict[str, str]:
    """List all available Φ-RFT variants with their use cases.
    
    Returns
    -------
    Dict[str, str]
        Mapping of variant name to description/use case.
    """
    variants = {}
    
    # Phase-based (always available)
    variants["standard"] = "Original β·frac(k/φ) phase - baseline, fully proven"
    variants["logphi"] = "Log-frequency warped - low-frequency emphasis"
    variants["mixed"] = "Convex blend of standard + logphi"
    variants["canonical"] = "Closed-form RFT from core module (O(n log n))"
    
    # Matrix-based (from registry)
    if HAS_VARIANTS:
        for name, info in VARIANTS.items():
            variants[name] = f"{info.name} - {info.use_case}"
    
    return variants


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _frac(values: np.ndarray) -> np.ndarray:
    """Return fractional part of values, preserving dtype."""
    return values - np.floor(values)


def _phi_phase(
    k: np.ndarray,
    n: int,
    *,
    beta: float = 0.83,
    kind: str = "standard",
    mix: float = 0.25,
) -> np.ndarray:
    """Generate Golden Ratio phase factors for different Φ-RFT variants.

    Parameters
    ----------
    k : np.ndarray
        Frequency indices [0, 1, ..., n-1] as float64.
    n : int
        Transform length.
    beta : float
        Global scaling of the φ-phase.
    kind : {"standard", "logphi", "mixed"}
        - "standard": original phase φ-based on frac(k/φ)
        - "logphi"  : log-frequency warped phase in [0,1]
        - "mixed"   : convex blend of the two, then reprojected to S^1
    mix : float
        Blend factor for "mixed" in [0,1]. mix=0 → standard, mix=1 → logphi.

    Returns
    -------
    np.ndarray
        Complex unit-modulus array of shape (n,) giving the φ-phase diagonal.
    """
    if kind not in ("standard", "logphi", "mixed"):
        raise ValueError(f"Unsupported φ-phase kind: {kind}")

    # Original φ-phase: irrational rotation on the unit circle
    frac_k = _frac(k / PHI)
    theta_std = 2.0 * np.pi * beta * frac_k

    if kind == "standard":
        return np.exp(1j * theta_std)

    # Log-frequency warped phase: compresses high-k spacing, more resolution
    # near low frequencies which dominate symbol statistics.
    # logk ∈ [0, 1], smooth and monotone.
    logk = np.log1p(k) / np.log1p(float(n))
    theta_log = 2.0 * np.pi * beta * logk

    if kind == "logphi":
        return np.exp(1j * theta_log)

    # "mixed": convex combination of the phase parameters, then exponentiated.
    # This preserves unit modulus while letting you dial between behaviors.
    mix_clamped = float(np.clip(mix, 0.0, 1.0))
    theta_mix = (1.0 - mix_clamped) * theta_std + mix_clamped * theta_log
    return np.exp(1j * theta_mix)


# ---------------------------------------------------------------------------
# Φ-RFT forward/inverse transforms (FFT-factorized, unitary)
# Supports both local phase-based variants AND registry matrix-based variants
# ---------------------------------------------------------------------------

# Registry variant names that use matrix multiplication
MATRIX_VARIANTS = {
    "original", "harmonic_phase", "fibonacci_tilt", "chaotic_mix",
    "geometric_lattice", "phi_chaotic_hybrid", "adaptive_phi",
    "log_periodic", "convex_mix", "golden_ratio_exact"
}

# Phase-based variants (FFT-factorized, O(n log n))
PHASE_VARIANTS = {"standard", "logphi", "mixed", "canonical"}


def rft_forward(
    x: ArrayLike,
    *,
    beta: float = 0.83,
    sigma: float = 1.25,
    kind: str = "standard",
    mix: float = 0.25,
    variant: Optional[str] = None,
) -> np.ndarray:
    """Φ-RFT forward transform for one-dimensional sequences.

    Supports two modes:
    1. Phase-based (O(n log n)): kind in {"standard", "logphi", "mixed", "canonical"}
    2. Matrix-based (O(n²)): variant in registry (7+ unitary matrices)

    Parameters
    ----------
    x : ArrayLike
        Input signal (real or complex), shape (N,).
    beta : float
        Golden phase scaling parameter (phase-based only).
    sigma : float
        Quadratic chirp scaling parameter (phase-based only).
    kind : {"standard", "logphi", "mixed", "canonical"}
        Phase-based φ-variant. "canonical" uses closed_form_rft.py.
    mix : float
        Blend factor for "mixed" variant.
    variant : str, optional
        If provided, use a matrix-based variant from the registry.
        Overrides `kind`. Options: "original", "harmonic_phase",
        "fibonacci_tilt", "chaotic_mix", "geometric_lattice",
        "phi_chaotic_hybrid", "adaptive_phi", "log_periodic",
        "convex_mix", "golden_ratio_exact".

    Returns
    -------
    np.ndarray
        Φ-RFT coefficients, shape (N,), complex128.
    """
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.shape[0]
    
    # --- Matrix-based variants (from registry) ---
    if variant is not None and variant in MATRIX_VARIANTS:
        matrix = _get_variant_matrix(variant, n)
        if matrix is not None:
            return matrix @ signal
        # Fallback to phase-based if matrix unavailable
        kind = "standard"
    
    # --- Canonical closed-form RFT ---
    if kind == "canonical" and HAS_CANONICAL_RFT:
        return canonical_rft_forward(signal, beta=beta, sigma=sigma)
    
    # --- Phase-based variants (O(n log n)) ---
    k = np.arange(n, dtype=np.float64)

    phase_golden = _phi_phase(k, n, beta=beta, kind=kind, mix=mix)
    phase_quadratic = np.exp(1j * np.pi * sigma * (k * k) / float(n))

    # Unitary: D_phi * C_sigma * F
    return phase_golden * (phase_quadratic * fft(signal, norm="ortho"))


def rft_inverse(
    y: ArrayLike,
    *,
    beta: float = 0.83,
    sigma: float = 1.25,
    kind: str = "standard",
    mix: float = 0.25,
    variant: Optional[str] = None,
) -> np.ndarray:
    """Inverse Φ-RFT transform matching :func:`rft_forward`.

    Supports two modes:
    1. Phase-based (O(n log n)): kind in {"standard", "logphi", "mixed", "canonical"}
    2. Matrix-based (O(n²)): variant in registry (7+ unitary matrices)

    Parameters
    ----------
    y : ArrayLike
        Φ-RFT coefficients, shape (N,).
    beta : float
        Golden phase scaling parameter (must match forward).
    sigma : float
        Quadratic chirp scaling parameter (must match forward).
    kind : {"standard", "logphi", "mixed", "canonical"}
        φ-phase variant (must match forward). "canonical" uses closed_form_rft.py.
    mix : float
        Blend factor for "mixed" variant.
    variant : str, optional
        If provided, use a matrix-based variant from the registry.
        Must match the variant used in forward transform.

    Returns
    -------
    np.ndarray
        Reconstructed signal, shape (N,), complex128.
    """
    coeffs = np.asarray(y, dtype=np.complex128)
    n = coeffs.shape[0]
    
    # --- Matrix-based variants (from registry) ---
    if variant is not None and variant in MATRIX_VARIANTS:
        matrix = _get_variant_matrix(variant, n)
        if matrix is not None:
            # Inverse of unitary matrix is conjugate transpose
            return np.conj(matrix.T) @ coeffs
        # Fallback to phase-based if matrix unavailable
        kind = "standard"
    
    # --- Canonical closed-form RFT ---
    if kind == "canonical" and HAS_CANONICAL_RFT:
        return canonical_rft_inverse(coeffs, beta=beta, sigma=sigma)
    
    # --- Phase-based variants (O(n log n)) ---
    k = np.arange(n, dtype=np.float64)

    phase_golden = _phi_phase(k, n, beta=beta, kind=kind, mix=mix)
    phase_quadratic = np.exp(1j * np.pi * sigma * (k * k) / float(n))

    # Inverse of (D_phi * C_sigma * F) is F^H * C_sigma^H * D_phi^H
    return ifft(
        np.conj(phase_quadratic) * np.conj(phase_golden) * coeffs,
        norm="ortho",
    )


@dataclass
class HybridResult:
    """Container for hybrid decomposition outputs."""

    structural: np.ndarray
    texture: np.ndarray
    residual: np.ndarray
    metadata: Dict[str, object]


def _energy(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector) ** 2)


def hybrid_decomposition(
    x: ArrayLike,
    *,
    max_iter: int = 5,
    threshold_dct: float = 0.05,
    threshold_rft: float = 0.05,
    strategy: str = "balanced",
    verbose: bool = False,
    # Φ-RFT controls (phase-based):
    rft_kind: str = "standard",
    rft_mix: float = 0.25,
    rft_beta: float = 0.83,
    rft_sigma: float = 1.25,
    # NEW: Matrix-based variant from registry
    rft_variant: Optional[str] = None,
) -> HybridResult:
    """Split ``x`` into DCT-sparse structure and Φ-RFT-sparse texture.

    Parameters
    ----------
    x : ArrayLike
        Input signal, shape (N,).
    max_iter : int
        Maximum number of alternating DCT / Φ-RFT refinement passes.
    threshold_dct : float
        Relative magnitude threshold for DCT coefficients (0..1).
    threshold_rft : float
        Relative magnitude threshold for Φ-RFT coefficients (0..1).
    strategy : {"balanced", "dct_first", "rft_first"}
        Order of applying basis updates per iteration.
    verbose : bool
        If True, prints energy breakdown per iteration.
    rft_kind : {"standard", "logphi", "mixed", "canonical"}
        Phase-based Φ-RFT variant for the texture component.
    rft_mix : float
        Blend factor when rft_kind="mixed".
    rft_beta : float
        Golden phase scaling, forwarded to rft_forward / rft_inverse.
    rft_sigma : float
        Quadratic chirp scaling, forwarded to rft_forward / rft_inverse.
    rft_variant : str, optional
        Matrix-based variant from registry. Overrides rft_kind.
        Options: "original", "harmonic_phase", "fibonacci_tilt",
        "chaotic_mix", "geometric_lattice", "phi_chaotic_hybrid",
        "adaptive_phi", "log_periodic", "convex_mix", "golden_ratio_exact".

    Returns
    -------
    HybridResult
        structural : DCT-sparse component
        texture    : Φ-RFT-sparse component
        residual   : residual after `max_iter` passes
        metadata   : diagnostic curves (energies, sparsities, features, weights)
    """
    signal = np.asarray(x, dtype=np.complex128)
    structural = np.zeros_like(signal)
    texture = np.zeros_like(signal)
    residual = signal.copy()

    metadata: Dict[str, object] = {
        "iterations": 0,
        "struct_energy": [],
        "texture_energy": [],
        "residual_energy": [],
        "struct_sparsity": [],
        "texture_sparsity": [],
        "rft_variant": rft_variant if rft_variant else rft_kind,
    }

    total_energy = _energy(signal)

    for idx in range(max_iter):
        metadata["iterations"] = idx + 1

        # Determine order for this iteration
        if strategy == "rft_first":
            steps = [("rft", threshold_rft), ("dct", threshold_dct)]
        elif strategy == "dct_first":
            steps = [("dct", threshold_dct), ("rft", threshold_rft)]
        else:
            # Balanced: Default to DCT first as it captures broad structure
            steps = [("dct", threshold_dct), ("rft", threshold_rft)]

        for step_type, thresh in steps:
            if step_type == "dct":
                # --- DCT Update ---
                if np.iscomplexobj(residual):
                    coeffs_real = dct(residual.real, norm="ortho")
                    coeffs_imag = dct(residual.imag, norm="ortho")
                    dct_coeffs = coeffs_real + 1j * coeffs_imag
                else:
                    dct_coeffs = dct(residual, norm="ortho")

                dct_thresh_val = (
                    thresh * np.max(np.abs(dct_coeffs)) if dct_coeffs.size else 0.0
                )
                mask_dct = np.abs(dct_coeffs) > dct_thresh_val
                dct_sparse = dct_coeffs * mask_dct

                if np.iscomplexobj(dct_sparse):
                    update_real = idct(dct_sparse.real, norm="ortho")
                    update_imag = idct(dct_sparse.imag, norm="ortho")
                    structural_update = update_real + 1j * update_imag
                else:
                    structural_update = idct(dct_sparse, norm="ortho")

                structural += structural_update
                residual -= structural_update

                metadata["struct_energy"].append(_energy(structural_update))
                metadata["struct_sparsity"].append(
                    1.0 - (np.sum(mask_dct) / mask_dct.size)
                )
                # Zero entry for the other component to keep lists aligned
                metadata["texture_energy"].append(0.0)
                metadata["texture_sparsity"].append(1.0)

            elif step_type == "rft":
                # --- Φ-RFT Update (supports all unitary variants) ---
                rft_coeffs = rft_forward(
                    residual,
                    beta=rft_beta,
                    sigma=rft_sigma,
                    kind=rft_kind,
                    mix=rft_mix,
                    variant=rft_variant,  # NEW: matrix-based variant
                )
                rft_thresh_val = (
                    thresh * np.max(np.abs(rft_coeffs)) if rft_coeffs.size else 0.0
                )

                # Enforce strict sparsity for RFT to avoid capturing broadband structure.
                # Only keep the top few coefficients (e.g., 5) if they exceed threshold.
                sorted_indices = np.argsort(np.abs(rft_coeffs))[::-1]
                mask_rft = np.zeros_like(rft_coeffs, dtype=bool)

                top_k = 5
                for i in range(min(top_k, rft_coeffs.size)):
                    idx_coeff = sorted_indices[i]
                    if np.abs(rft_coeffs[idx_coeff]) > rft_thresh_val:
                        mask_rft[idx_coeff] = True
                    else:
                        break

                rft_sparse = rft_coeffs * mask_rft

                texture_update = rft_inverse(
                    rft_sparse,
                    beta=rft_beta,
                    sigma=rft_sigma,
                    kind=rft_kind,
                    mix=rft_mix,
                    variant=rft_variant,  # NEW: matrix-based variant
                )
                texture += texture_update
                residual -= texture_update

                metadata["texture_energy"].append(_energy(texture_update))
                metadata["texture_sparsity"].append(
                    1.0 - (np.sum(mask_rft) / mask_rft.size)
                )
                metadata["struct_energy"].append(0.0)
                metadata["struct_sparsity"].append(1.0)

        residual_energy = _energy(residual)
        metadata["residual_energy"].append(residual_energy)

        if verbose:
            struct_ratio = (
                np.sum(metadata["struct_energy"]) / total_energy if total_energy else 0.0
            )
            texture_ratio = (
                np.sum(metadata["texture_energy"]) / total_energy if total_energy else 0.0
            )
            residual_ratio = residual_energy / total_energy if total_energy else 0.0
            print(
                f"Iteration {idx + 1}: "
                f"struct={struct_ratio:.2%}, "
                f"texture={texture_ratio:.2%}, "
                f"residual={residual_ratio:.2%}"
            )

        # Check convergence using the minimum of the two thresholds squared
        min_thresh = min(threshold_dct, threshold_rft)
        if total_energy > 0.0 and residual_energy / total_energy < min_thresh**2:
            break

    return HybridResult(structural, texture, residual, metadata)


# ---------------------------------------------------------------------------
# Feature analysis and adaptive front-end
# ---------------------------------------------------------------------------

def analyze_signal(x: ArrayLike) -> Dict[str, float]:
    """Extract simple descriptive statistics used for weight heuristics."""
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.size

    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode="full")[n - 1 :]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
    autocorr_decay = float(np.mean(np.abs(autocorr) > 0.5))

    if n > 2:
        diff_signal = np.diff(signal)
        second_diff = np.abs(np.diff(signal, n=2))
        std_diff = np.std(second_diff) if np.std(second_diff) > 0 else 1.0
        edge_density = float(np.mean(second_diff > std_diff))

        # Kurtosis of the first derivative helps distinguish steps (high kurtosis)
        # from waves (low kurtosis). Sine wave derivative kurtosis is ~ -1.5.
        if np.all(np.imag(signal) == 0):
            k_val = float(kurtosis(np.real(diff_signal)))
        else:
            k_val = float(kurtosis(np.abs(diff_signal)))
    else:
        edge_density = 0.0
        k_val = 0.0

    spectrum_abs = np.abs(fft(signal))
    spectrum_power = spectrum_abs**2
    spectrum_sum = np.sum(spectrum_power)
    if spectrum_sum <= 0:
        spectral_entropy_norm = 0.0
    else:
        p = spectrum_power / spectrum_sum
        valid = p > 1e-10
        entropy = -np.sum(p[valid] * np.log2(p[valid]))
        spectral_entropy_norm = float(entropy / np.log2(n)) if n > 1 else 0.0

    top_peaks = np.sort(spectrum_abs)[-5:]
    if top_peaks.size < 2 or np.any(top_peaks[1:] == 0):
        quasi_periodicity = 0.0
    else:
        ratios = top_peaks[:-1] / top_peaks[1:]
        # Check for ratios close to 1/PHI or 1/PHI^2
        is_phi = np.abs(ratios - (1.0 / PHI)) < 0.15
        is_phi2 = np.abs(ratios - (1.0 / PHI**2)) < 0.15
        quasi_periodicity = float(np.mean(is_phi | is_phi2))

    if n > 1:
        diff_signal = np.diff(signal)
        var_diff = np.mean(np.abs(diff_signal - np.mean(diff_signal)) ** 2)
        smoothness = float(1.0 / (1.0 + var_diff))
    else:
        smoothness = 1.0

    return {
        "autocorr_decay": autocorr_decay,
        "edge_density": edge_density,
        "spectral_entropy": spectral_entropy_norm,
        "quasi_periodicity": quasi_periodicity,
        "smoothness": smoothness,
        "kurtosis": k_val,
    }


def predict_weights(features: Dict[str, float]) -> Dict[str, float]:
    """Heuristic allocation of DCT vs Φ-RFT preference weights.

    We explicitly guard the ASCII bottleneck case: strong edges + high
    derivative kurtosis → treat as step-like (symbols), favor DCT.
    """
    # Strong edge density indicates symbolic/step-like data (ASCII, Code)
    # We prioritize this to avoid the "ASCII Bottleneck" where RFT fails.
    # Pure sine waves also have high edge density, but low kurtosis.
    is_step_like = features["edge_density"] > 0.45 and features["kurtosis"] > -1.2

    if is_step_like:
        # Prioritize DCT to clear steps, but keep RFT active for underlying waves
        return {"dct": 0.70, "rft": 0.30}

    if features["quasi_periodicity"] > 0.35:
        if features["spectral_entropy"] > 0.35:
            return {"dct": 0.50, "rft": 0.50}
        return {"dct": 0.20, "rft": 0.80}

    if features["edge_density"] > 0.3:
        return {"dct": 0.95, "rft": 0.05}
    if features["smoothness"] > 0.8:
        return {"dct": 0.85, "rft": 0.15}
    if features["spectral_entropy"] > 0.9:
        return {"dct": 0.50, "rft": 0.50}
    return {"dct": 0.65, "rft": 0.35}


def adaptive_hybrid_compress(
    x: ArrayLike,
    *,
    verbose: bool = False,
    max_iter: int = 5,
    # Phase-based knobs:
    rft_kind: str = "standard",
    rft_mix: float = 0.25,
    rft_beta: float = 0.83,
    rft_sigma: float = 1.25,
    # NEW: Matrix-based variant from registry
    rft_variant: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, object]]:
    """Run adaptive hybrid decomposition using any unitary Φ-RFT variant.

    This function integrates:
    - The canonical closed-form RFT (O(n log n))
    - All 7+ matrix-based variants from the registry (O(n²))
    - Phase-based variants for experimentation

    Parameters
    ----------
    x : ArrayLike
        Input signal, shape (N,).
    verbose : bool
        If True, prints intermediate diagnostics.
    max_iter : int
        Maximum hybrid iterations.
    rft_kind : {"standard", "logphi", "mixed", "canonical"}
        Phase-based Φ-RFT variant for the texture component.
        "canonical" uses the closed_form_rft.py implementation.
    rft_mix : float
        Blend factor for "mixed" variant (0..1).
    rft_beta : float
        Golden phase scaling parameter.
    rft_sigma : float
        Quadratic chirp scaling parameter.
    rft_variant : str, optional
        Matrix-based variant from registry. Overrides rft_kind.
        Options:
        - "original": Original Φ-RFT (quantum simulation)
        - "harmonic_phase": Cubic time-base (nonlinear filtering)
        - "fibonacci_tilt": Integer lattice (post-quantum crypto)
        - "chaotic_mix": Haar-like randomness (secure scrambling)
        - "geometric_lattice": Phase-engineered (analog/optical)
        - "phi_chaotic_hybrid": Structure+disorder (resilient codecs)
        - "adaptive_phi": Meta selection (universal compression)
        - "log_periodic": Log-frequency (symbol compression)
        - "convex_mix": Hybrid log/standard (adaptive textures)
        - "golden_ratio_exact": Full resonance (theorem validation)

    Returns
    -------
    structural : np.ndarray
        DCT-sparse component.
    texture : np.ndarray
        Φ-RFT-sparse component.
    weights : Dict[str, float]
        Estimated DCT/RFT weights used.
    metadata : Dict[str, object]
        Additional metadata (energies, sparsities, features, etc.).
    """
    features = analyze_signal(x)
    weights = predict_weights(features)

    if verbose:
        print("Signal features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        print("Predicted weights:")
        print(f"  DCT: {weights['dct']:.2f}")
        print(f"  RFT: {weights['rft']:.2f}")
        if rft_variant:
            print(f"  RFT Variant: {rft_variant}")
        else:
            print(f"  RFT Kind: {rft_kind}")

    # Translate weights into thresholds: more weight → lower threshold
    threshold_dct = 0.15 * (1.0 - weights["dct"])
    threshold_rft = 0.15 * (1.0 - weights["rft"])
    # Ensure minimum threshold
    threshold_dct = max(threshold_dct, 1e-3)
    threshold_rft = max(threshold_rft, 1e-3)

    # Determine strategy based on dominant weight
    if weights["dct"] > 0.6:
        strategy = "dct_first"
    else:
        strategy = "rft_first"

    result = hybrid_decomposition(
        x,
        max_iter=max_iter,
        threshold_dct=threshold_dct,
        threshold_rft=threshold_rft,
        strategy=strategy,
        verbose=verbose,
        rft_kind=rft_kind,
        rft_mix=rft_mix,
        rft_beta=rft_beta,
        rft_sigma=rft_sigma,
        rft_variant=rft_variant,  # NEW: pass variant through
    )
    result.metadata["features"] = features
    result.metadata["weights"] = weights

    return result.structural, result.texture, weights, result.metadata


def braided_hybrid_mca(
    x: ArrayLike,
    *,
    max_iter: int = 10,
    threshold: float = 0.05,
    dct_weight: float = 1.0,
    rft_weight: float = 1.0,
    verbose: bool = False,
) -> HybridResult:
    """Braided DCT + Φ-RFT decomposition via per-bin competition.

    This is an MCA-style variant where DCT and Φ-RFT both see the same residual
    each iteration, and *compete per frequency bin*. For each k, only one of
    (cS[k], cT[k]) is kept (winner-takes-all), so DCT can no longer steal bins
    that Φ-RFT explains better.

    Parameters
    ----------
    x : array-like
        Input signal.
    max_iter : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Relative threshold (fraction of max energy) for bin activation.
    dct_weight : float, optional
        Weight scaling for DCT energy when comparing per-bin scores.
    rft_weight : float, optional
        Weight scaling for RFT energy when comparing per-bin scores.
    verbose : bool, optional
        If True, prints progress per iteration.

    Returns
    -------
    HybridResult
        structural (DCT), texture (RFT), residual, metadata.
    """
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.size

    structural = np.zeros_like(signal)
    texture = np.zeros_like(signal)
    residual = signal.copy()

    metadata: Dict[str, object] = {
        "iterations": 0,
        "residual_energy": [],
        "mode": "braid_mca",
        "threshold": threshold,
        "dct_weight": dct_weight,
        "rft_weight": rft_weight,
    }

    total_energy = _energy(signal)

    for it in range(max_iter):
        metadata["iterations"] = it + 1

        # 1) Project residual onto both bases
        if np.iscomplexobj(residual):
            cS_real = dct(residual.real, norm="ortho")
            cS_imag = dct(residual.imag, norm="ortho")
            cS = cS_real + 1j * cS_imag
        else:
            cS = dct(residual, norm="ortho").astype(np.complex128)

        cT = rft_forward(residual)

        # 2) Compute per-bin energies (with optional weights)
        eS = dct_weight * (np.abs(cS) ** 2)
        eT = rft_weight * (np.abs(cT) ** 2)

        max_energy = max(np.max(eS), np.max(eT)) if n > 0 else 0.0
        if max_energy == 0.0:
            break

        tau2 = (threshold ** 2) * max_energy

        # 3) Winner-takes-all routing per bin
        mask_active = (eS > tau2) | (eT > tau2)

        # Decide winner where active
        choose_dct = (eS >= eT) & mask_active
        choose_rft = (eT > eS) & mask_active

        cS_braid = np.zeros_like(cS)
        cT_braid = np.zeros_like(cT)

        cS_braid[choose_dct] = cS[choose_dct]
        cT_braid[choose_rft] = cT[choose_rft]

        # 4) Reconstruct updates
        if np.iscomplexobj(cS_braid):
            updS_real = idct(cS_braid.real, norm="ortho")
            updS_imag = idct(cS_braid.imag, norm="ortho")
            delta_s = updS_real + 1j * updS_imag
        else:
            delta_s = idct(cS_braid, norm="ortho")

        delta_t = rft_inverse(cT_braid)

        structural += delta_s
        texture += delta_t
        residual -= (delta_s + delta_t)

        res_e = _energy(residual)
        metadata["residual_energy"].append(res_e)

        if verbose and total_energy > 0.0:
            print(
                f"[braid] iter {it + 1}: "
                f"residual = {res_e / total_energy:.2%}"
            )

        # Simple convergence check
        if total_energy > 0.0 and res_e / total_energy < 1e-4:
            break

    return HybridResult(structural, texture, residual, metadata)


def soft_braided_hybrid_mca(
    x: ArrayLike,
    *,
    max_iter: int = 10,
    threshold: float = 0.05,
    dct_weight: float = 1.0,
    rft_weight: float = 1.0,
    verbose: bool = False,
) -> HybridResult:
    """Soft-threshold braided DCT + Φ-RFT decomposition.

    Instead of winner-takes-all per bin, this uses soft routing:
    - cS_soft[k] = cS[k] * (eS[k] / (eS[k] + eT[k]))
    - cT_soft[k] = cT[k] * (eT[k] / (eS[k] + eT[k]))

    This preserves phase coherence while still allowing per-bin competition.

    Parameters
    ----------
    x : array-like
        Input signal.
    max_iter : int, optional
        Maximum number of iterations.
    threshold : float, optional
        Relative threshold for minimum bin energy before routing.
    dct_weight : float, optional
        Weight scaling for DCT energy in competition.
    rft_weight : float, optional
        Weight scaling for RFT energy in competition.
    verbose : bool, optional
        If True, prints progress per iteration.

    Returns
    -------
    HybridResult
        structural (DCT), texture (RFT), residual, metadata.
    """
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.size

    structural = np.zeros_like(signal)
    texture = np.zeros_like(signal)
    residual = signal.copy()

    metadata: Dict[str, object] = {
        "iterations": 0,
        "residual_energy": [],
        "mode": "soft_braid_mca",
        "threshold": threshold,
        "dct_weight": dct_weight,
        "rft_weight": rft_weight,
    }

    total_energy = _energy(signal)

    for it in range(max_iter):
        metadata["iterations"] = it + 1

        # 1) Project residual onto both bases
        if np.iscomplexobj(residual):
            cS_real = dct(residual.real, norm="ortho")
            cS_imag = dct(residual.imag, norm="ortho")
            cS = cS_real + 1j * cS_imag
        else:
            cS = dct(residual, norm="ortho").astype(np.complex128)

        cT = rft_forward(residual)

        # 2) Compute per-bin energies (with optional weights)
        eS = dct_weight * (np.abs(cS) ** 2)
        eT = rft_weight * (np.abs(cT) ** 2)

        max_energy = max(np.max(eS), np.max(eT)) if n > 0 else 0.0
        if max_energy == 0.0:
            break

        tau2 = (threshold ** 2) * max_energy

        # 3) Soft routing: Proportional allocation
        # Add epsilon to prevent division by zero
        eps = 1e-14
        eS_safe = eS + eps
        eT_safe = eT + eps
        e_sum = eS_safe + eT_safe

        # Soft weights
        w_dct = eS_safe / e_sum
        w_rft = eT_safe / e_sum

        # Apply threshold mask (only route bins above threshold)
        mask_active = (eS > tau2) | (eT > tau2)

        cS_soft = np.zeros_like(cS)
        cT_soft = np.zeros_like(cT)

        cS_soft[mask_active] = cS[mask_active] * w_dct[mask_active]
        cT_soft[mask_active] = cT[mask_active] * w_rft[mask_active]

        # 4) Reconstruct updates
        if np.iscomplexobj(cS_soft):
            updS_real = idct(cS_soft.real, norm="ortho")
            updS_imag = idct(cS_soft.imag, norm="ortho")
            delta_s = updS_real + 1j * updS_imag
        else:
            delta_s = idct(cS_soft, norm="ortho")

        delta_t = rft_inverse(cT_soft)

        structural += delta_s
        texture += delta_t
        residual -= (delta_s + delta_t)

        res_e = _energy(residual)
        metadata["residual_energy"].append(res_e)

        if verbose and total_energy > 0.0:
            print(
                f"[soft_braid] iter {it + 1}: "
                f"residual = {res_e / total_energy:.2%}"
            )

        # Convergence check
        if total_energy > 0.0 and res_e / total_energy < 1e-4:
            break

    return HybridResult(structural, texture, residual, metadata)


__all__ = [
    "PHI",
    "HybridResult",
    "adaptive_hybrid_compress",
    "analyze_signal",
    "braided_hybrid_mca",
    "hybrid_decomposition",
    "predict_weights",
    "rft_forward",
    "rft_inverse",
    "soft_braided_hybrid_mca",
]
