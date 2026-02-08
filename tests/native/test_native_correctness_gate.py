#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Native RFT Correctness Gate
===========================

This test MUST pass before any claims about the native C++/ASM pipeline.

Tests:
1. Native forward + inverse roundtrip error < 1e-10
2. Native matches Python implementation (same phase formula)
3. Norm preservation

If any test fails, the native module is NOT equivalent to the canonical RFT.
"""
from __future__ import annotations

import os

import numpy as np
import pytest


rft = pytest.importorskip("rftmw_native")

# Canonical Python reference implementation (same object used by theorem tests)
from algorithms.rft.core.transform_theorems import canonical_unitary_basis


def _sizes() -> list[int]:
    # The native canonical RFT is O(N^2). Keep CI fast by default.
    # Opt into the larger stress size by setting RFTMW_NATIVE_GATE_FULL=1.
    if os.environ.get("RFTMW_NATIVE_GATE_FULL", "0") == "1":
        return [64, 128, 256, 512, 1024]
    return [64, 128, 256, 512]


def _py_forward_canonical(x: np.ndarray) -> np.ndarray:
    U = canonical_unitary_basis(int(x.shape[0]))
    return U.conj().T @ x


def _py_inverse_canonical(X: np.ndarray) -> np.ndarray:
    U = canonical_unitary_basis(int(X.shape[0]))
    return U @ X


def test_native_roundtrip():
    """Native forward + inverse must roundtrip with error < 1e-10."""
    rng = np.random.default_rng(42)

    errors: list[tuple[int, float]] = []
    for n in _sizes():
        x = rng.normal(size=n).astype(np.float64, copy=False)
        x_norm = float(np.linalg.norm(x))

        X = rft.forward(x)
        x_rec = rft.inverse(X)

        err = float(np.linalg.norm(x - x_rec) / (x_norm + 1e-15))
        errors.append((n, err))
        print(f"  N={n:4d}: roundtrip error = {err:.2e}")

    max_err = max(e for _, e in errors)
    assert max_err <= 1e-10, f"Native roundtrip error {max_err:.2e} > 1e-10"


def test_native_matches_python():
    """Native canonical output must match canonical Python reference."""

    rng = np.random.default_rng(42)
    n = 128
    x = rng.normal(size=n).astype(np.float64, copy=False)

    X_py = _py_forward_canonical(x)
    X_native = rft.forward(x)

    rel = float(np.linalg.norm(X_py - X_native) / (np.linalg.norm(X_py) + 1e-15))
    print(f"  Python(canonical) vs Native forward rel error: {rel:.2e}")
    assert rel <= 1e-10


def test_norm_preservation():
    """Transform should preserve energy (Parseval's theorem)."""
    rng = np.random.default_rng(42)

    for n in [64, 256]:
        x = rng.normal(size=n).astype(np.float64, copy=False)
        x_norm = float(np.linalg.norm(x))

        X = rft.forward(x)
        X_norm = float(np.linalg.norm(X))

        rel_err = abs(x_norm - X_norm) / (x_norm + 1e-15)
        print(f"  N={n:4d}: ||x||={x_norm:.4f}, ||X||={X_norm:.4f}, rel_err={rel_err:.2e}")
        assert rel_err <= 1e-10
