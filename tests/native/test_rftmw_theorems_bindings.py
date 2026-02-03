# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos

from __future__ import annotations

import numpy as np
import pytest


a = pytest.importorskip("rftmw_native")

from algorithms.rft.core import transform_theorems as py_th


def test_phi_frequencies_match_python() -> None:
    N = 64
    cpp = a.theorems.phi_frequencies(N)
    py = py_th.phi_frequencies(N)
    assert cpp.shape == (N,)
    assert np.allclose(cpp, py, atol=1e-14, rtol=0.0)


def test_golden_roots_z_match_python() -> None:
    N = 64
    cpp = a.theorems.golden_roots_z(N)
    py = py_th.golden_roots_z(N)
    assert cpp.shape == (N,)
    assert np.allclose(cpp, py, atol=1e-13, rtol=0.0)


def test_k99_sanity() -> None:
    X = np.zeros(32, dtype=np.complex128)
    X[0] = 1.0 + 0.0j
    assert a.theorems.k99(X, 0.99) == 1

    X2 = np.ones(100, dtype=np.complex128)
    assert a.theorems.k99(X2, 0.99) == 99


def test_golden_companion_shift_shape_dtype() -> None:
    N = 16
    C = a.theorems.golden_companion_shift(N)
    assert C.shape == (N, N)
    assert C.dtype == np.complex128


def test_golden_drift_ensemble_shape_unit_modulus() -> None:
    N = 128
    M = 10
    X = a.theorems.golden_drift_ensemble(N, M, seed=0)
    assert X.shape == (M, N)
    assert X.dtype == np.complex128
    # By construction exp(i*theta), magnitude should be ~1.
    assert np.allclose(np.abs(X), 1.0, atol=1e-12, rtol=0.0)
