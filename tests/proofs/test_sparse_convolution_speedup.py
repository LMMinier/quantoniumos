# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import numpy as np


def _k_sparse_spectrum(N: int, K: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=K, replace=False)
    X = np.zeros(N, dtype=np.complex128)
    X[idx] = rng.normal(size=K) + 1j * rng.normal(size=K)
    return X


def _circular_convolution_naive(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Naive circular convolution y[n] = sum_m x[m] h[n-m mod N]."""

    x = np.asarray(x)
    h = np.asarray(h)
    N = x.shape[0]
    y = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        acc = 0.0 + 0.0j
        for m in range(N):
            acc += x[m] * h[(n - m) % N]
        y[n] = acc
    return y


def _ifft_unitary(X: np.ndarray) -> np.ndarray:
    return np.fft.ifft(X, norm="ortho")


def test_sparse_convolution_matches_spectral_multiply() -> None:
    """Convolution theorem sanity check.

    Even without claiming a sparse-FFT algorithm, if signals are represented in
    wave-space, convolution reduces to pointwise spectral multiplication.
    """

    # Keep N modest: naive O(N^2) is used as a correctness oracle.
    N = 128
    K = 16

    # Use the SAME support so the product spectrum stays K-sparse.
    X = _k_sparse_spectrum(N, K, seed=0)
    H = _k_sparse_spectrum(N, K, seed=0)

    x = _ifft_unitary(X)
    h = _ifft_unitary(H)

    y_naive = _circular_convolution_naive(x, h)

    Y = X * H
    # With unitary FFT conventions (norm='ortho'), the convolution theorem has
    # a √N scaling factor: conv_circ(x,h) = √N * IFFT_unitary(FFT_unitary(x)*FFT_unitary(h)).
    y_spec = np.sqrt(N) * _ifft_unitary(Y)

    assert np.allclose(y_naive, y_spec, atol=1e-10, rtol=0.0)


def test_sparse_representation_implies_lower_opcount() -> None:
    """Deterministic complexity comparison (counts, not wall-clock).

    If you *already have* a K-sparse wave-space representation, the multiply is
    O(K) (or O(N) if dense). A naive time-domain circular convolution is O(N^2).

    This test just encodes the inequality for a representative (N, K).
    """

    N = 1024
    K = 32

    # Naive circular conv multiplications ~ N^2.
    ops_time = N * N

    # Wave-space multiply on K active bins ~ K.
    ops_wave = K

    assert ops_wave < ops_time
