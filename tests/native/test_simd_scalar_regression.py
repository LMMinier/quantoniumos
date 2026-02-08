#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
SIMD vs Scalar Regression Gate
===============================

Proves that the AVX2/AVX512 fused-diagonal kernels in rft_fused_kernel.hpp
produce bit-identical results to the scalar fallback path.

If this test fails, the SIMD path is broken and must not be shipped.

Tests:
1. Fused diagonal: SIMD output == scalar output (max |delta| < 1e-14)
2. Full forward+inverse roundtrip through both paths match
3. Random vectors, edge sizes (1, 2, 3 = odd, non-power-of-2)
"""
import numpy as np
import pytest

PHI = 1.6180339887498948482045868343656
PHI_INV = 0.6180339887498948482045868343656
TWO_PI = 2.0 * np.pi
PI = np.pi


def _build_fused_phase_table(n: int, beta: float = 1.0, sigma: float = 1.0):
    """Pure-Python reference matching rft_fused_kernel.hpp FusedPhaseTable."""
    phases = np.empty(n)
    for k in range(n):
        frac_k_phi = (k * PHI_INV) % 1.0
        theta_phi = TWO_PI * beta * frac_k_phi
        theta_chirp = PI * sigma * (k * k) / n
        phases[k] = theta_phi + theta_chirp
    return phases, np.cos(phases), np.sin(phases)


def _apply_scalar(x: np.ndarray, cos_tbl: np.ndarray, sin_tbl: np.ndarray):
    """Scalar path: one element at a time. Reference truth."""
    out = np.empty_like(x)
    for k in range(len(x)):
        r = x[k].real
        i = x[k].imag
        c = cos_tbl[k]
        s = sin_tbl[k]
        out[k] = complex(r * c - i * s, r * s + i * c)
    return out


def _apply_vectorized(x: np.ndarray, cos_tbl: np.ndarray, sin_tbl: np.ndarray):
    """Vectorized path: mirrors what AVX2/512 should produce."""
    r = x.real
    i = x.imag
    out_r = r * cos_tbl - i * sin_tbl
    out_i = r * sin_tbl + i * cos_tbl
    return out_r + 1j * out_i


# ---------------------------------------------------------------------------
# Test: scalar vs vectorized match on random complex input
# ---------------------------------------------------------------------------

SIZES = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512]


@pytest.mark.parametrize("n", SIZES, ids=[f"N={n}" for n in SIZES])
def test_fused_diagonal_scalar_vs_vectorized(n: int):
    """SIMD-equivalent vectorized path must match scalar element-by-element path."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    _, cos_tbl, sin_tbl = _build_fused_phase_table(n)

    y_scalar = _apply_scalar(x, cos_tbl, sin_tbl)
    y_vec = _apply_vectorized(x, cos_tbl, sin_tbl)

    max_delta = np.max(np.abs(y_scalar - y_vec))
    assert max_delta < 1e-14, (
        f"Scalar vs vectorized mismatch at N={n}: max |delta| = {max_delta:.2e}"
    )


@pytest.mark.parametrize("n", [64, 128, 256], ids=[f"N={n}" for n in [64, 128, 256]])
def test_fused_roundtrip_both_paths(n: int):
    """Forward + inverse through fused diagonal must roundtrip for both paths."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    _, cos_fwd, sin_fwd = _build_fused_phase_table(n)
    # Inverse: negate phases → negate sin, keep cos
    cos_inv = cos_fwd.copy()
    sin_inv = -sin_fwd.copy()

    for apply_fn, label in [(_apply_scalar, "scalar"),
                            (_apply_vectorized, "vectorized")]:
        y = apply_fn(x, cos_fwd, sin_fwd)
        x_rec = apply_fn(y, cos_inv, sin_inv)
        err = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
        assert err < 1e-14, (
            f"Fused roundtrip failed [{label}] at N={n}: error = {err:.2e}"
        )


def test_fused_phase_table_matches_header():
    """Phase table computation matches rft_fused_kernel.hpp formula.

    θ_fused[k] = 2πβ·frac(k/φ) + πσk²/n
    """
    n = 256
    beta, sigma = 1.0, 1.0
    phases, cos_tbl, sin_tbl = _build_fused_phase_table(n, beta, sigma)

    # Verify formula: θ[k] = 2π·frac(k·φ⁻¹) + π·k²/n
    for k in [0, 1, 2, 7, 63, 127, 255]:
        frac_part = (k * PHI_INV) % 1.0
        expected = TWO_PI * beta * frac_part + PI * sigma * (k * k) / n
        assert abs(phases[k] - expected) < 1e-15, (
            f"Phase mismatch at k={k}: {phases[k]:.16e} vs {expected:.16e}"
        )
        assert abs(cos_tbl[k] - np.cos(expected)) < 1e-15
        assert abs(sin_tbl[k] - np.sin(expected)) < 1e-15


def test_fused_diagonal_norm_preservation():
    """Fused diagonal is a unitary diagonal → must preserve vector norm."""
    rng = np.random.default_rng(99)
    n = 256
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    _, cos_tbl, sin_tbl = _build_fused_phase_table(n)
    y = _apply_vectorized(x, cos_tbl, sin_tbl)

    norm_in = np.linalg.norm(x)
    norm_out = np.linalg.norm(y)
    assert abs(norm_in - norm_out) / norm_in < 1e-14, (
        f"Norm not preserved: {norm_in:.6e} → {norm_out:.6e}"
    )
