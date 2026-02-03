# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos

"""Transform-theory constructions for the canonical Resonant Fourier Transform (RFT).

This module centralizes the concrete operator/basis objects used by the repo-ready
"missing theorem" tests:

- Canonical unitary basis U = Φ(ΦᴴΦ)^{-1/2}
- Nearest-unitary optimality: U is the polar factor of Φ (matrix nearness)
- Golden companion shift operator Cφ built from roots z_k = exp(i2π frac((k+1)φ))
- Golden convolution algebra Hφ(h) = Σ h[m] Cφ^m
- Golden shift operator in time domain Tφ = U Λ Uᴴ (diagonalized by U)
- Structure metrics: Toeplitz/banded/shift×diag proximity
- Signal ensemble + loss for empirical optimality (Theorem E candidate)

Design goals:
- deterministic, testable, fast for moderate N (<=128)
- definitions do not depend on FFT or other transforms

See also: tests/proofs/test_rft_transform_theorems.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resonant_fourier_transform import PHI, rft_basis_matrix


def phi_frequencies(N: int) -> np.ndarray:
    """f_k = frac((k+1) φ) in cycles/sample."""

    k = np.arange(int(N), dtype=np.float64)
    return np.mod((k + 1.0) * PHI, 1.0)


def golden_roots_z(N: int) -> np.ndarray:
    """z_k = exp(i 2π f_k) where f_k = frac((k+1) φ)."""

    f = phi_frequencies(N)
    return np.exp(1j * 2.0 * np.pi * f)


def raw_phi_basis(N: int) -> np.ndarray:
    """Raw φ-grid exponential basis Φ (generally non-orthogonal at finite N)."""

    return rft_basis_matrix(int(N), int(N), use_gram_normalization=False)


def canonical_unitary_basis(N: int) -> np.ndarray:
    """Canonical unitary basis U = Φ(ΦᴴΦ)^{-1/2}."""

    return rft_basis_matrix(int(N), int(N), use_gram_normalization=True)


def companion_matrix_from_roots(roots: np.ndarray) -> np.ndarray:
    """Frobenius companion matrix for p(z)=∏(z-roots).

    If p(z)=z^N + a_{N-1} z^{N-1} + ... + a0, then the companion matrix C satisfies:
    det(zI - C) = p(z).
    """

    roots = np.asarray(roots, dtype=np.complex128)
    coeffs = np.poly(roots)  # length N+1, monic
    n = roots.size

    C = np.zeros((n, n), dtype=np.complex128)
    C[:-1, 1:] = np.eye(n - 1, dtype=np.complex128)
    # Last row = [-a0, -a1, ..., -a_{n-1}]
    C[-1, :] = -coeffs[-1:0:-1]
    return C


def golden_companion_shift(N: int) -> np.ndarray:
    """Cφ: companion shift operator built solely from the golden roots {z_k}."""

    return companion_matrix_from_roots(golden_roots_z(int(N)))


def vandermonde_evecs(roots: np.ndarray) -> np.ndarray:
    """V = [v_0 ... v_{N-1}] where v_k = (1, z_k, ..., z_k^{N-1})^T."""

    roots = np.asarray(roots, dtype=np.complex128)
    n = roots.size
    powers = np.arange(n, dtype=np.int64).reshape(-1, 1)
    return roots.reshape(1, -1) ** powers


def golden_filter_operator(C: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Hφ(h)=Σ h[m] C^m."""

    C = np.asarray(C, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)

    N = C.shape[0]
    H = np.zeros_like(C)
    for m in range(N):
        H += h[m] * np.linalg.matrix_power(C, m)
    return H


def golden_shift_operator_T(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct Tφ = U Λ Uᴴ with Λ=diag(exp(-i2π f_k)).

    Returns: (Tφ, Λ)
    """

    N = int(N)
    U = canonical_unitary_basis(N)
    f = phi_frequencies(N)
    Lambda = np.diag(np.exp(-1j * 2.0 * np.pi * f)).astype(np.complex128)
    T_phi = U @ Lambda @ U.conj().T
    return T_phi, Lambda


def fft_unitary_matrix(N: int) -> np.ndarray:
    """Unitary DFT matrix with exp(-i2π nk/N)/√N."""

    N = int(N)
    n = np.arange(N)
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(n, k) / N) / np.sqrt(N)


def offdiag_ratio(U: np.ndarray, A: np.ndarray) -> float:
    """||offdiag(Uᴴ A U)||_F / ||A||_F."""

    U = np.asarray(U, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)

    B = U.conj().T @ A @ U
    off = B - np.diag(np.diag(B))
    return float(np.linalg.norm(off, ord="fro") / np.linalg.norm(A, ord="fro"))


def toeplitz_projection(A: np.ndarray) -> np.ndarray:
    """Nearest Toeplitz matrix under Frobenius norm (average each diagonal)."""

    A = np.asarray(A, dtype=np.complex128)
    N = A.shape[0]
    T = np.zeros_like(A)
    for d in range(-(N - 1), N):
        vals = []
        for i in range(N):
            j = i - d
            if 0 <= j < N:
                vals.append(A[i, j])
        mean = sum(vals) / len(vals)
        for i in range(N):
            j = i - d
            if 0 <= j < N:
                T[i, j] = mean
    return T


def band_projection(A: np.ndarray, bandwidth: int) -> np.ndarray:
    """Keep only entries with |i-j|<=bandwidth (non-cyclic band)."""

    A = np.asarray(A, dtype=np.complex128)
    N = A.shape[0]
    B = np.zeros_like(A)
    for i in range(N):
        lo = max(0, i - bandwidth)
        hi = min(N, i + bandwidth + 1)
        B[i, lo:hi] = A[i, lo:hi]
    return B


def shift_matrix(N: int, m: int = 1) -> np.ndarray:
    S = np.zeros((int(N), int(N)), dtype=np.complex128)
    for i in range(int(N)):
        S[(i + int(m)) % int(N), i] = 1.0
    return S


def best_shift_times_diag_approx(A: np.ndarray, *, shift: int = 1) -> np.ndarray:
    """Fit A ≈ S^shift · diag(d) using the corresponding shifted diagonal entries."""

    A = np.asarray(A, dtype=np.complex128)
    N = A.shape[0]
    S = shift_matrix(N, shift)
    d = np.zeros(N, dtype=np.complex128)
    for j in range(N):
        d[j] = A[(j + shift) % N, j]
    return S @ np.diag(d)


@dataclass(frozen=True)
class StructureMetrics:
    toeplitz_residual: float
    band2_residual: float
    shift1_diag_residual: float


def structure_metrics(A: np.ndarray) -> StructureMetrics:
    """Compute three simple structure residuals for A."""

    A = np.asarray(A, dtype=np.complex128)
    nA = float(np.linalg.norm(A, ord="fro"))

    toe = toeplitz_projection(A)
    band2 = band_projection(A, bandwidth=2)
    sd = best_shift_times_diag_approx(A, shift=1)

    return StructureMetrics(
        toeplitz_residual=float(np.linalg.norm(A - toe, ord="fro") / nA),
        band2_residual=float(np.linalg.norm(A - band2, ord="fro") / nA),
        shift1_diag_residual=float(np.linalg.norm(A - sd, ord="fro") / nA),
    )


def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """QR-based approximate Haar unitary (deterministic given rng)."""

    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph


def k99(X: np.ndarray, *, frac_energy: float = 0.99) -> int:
    """Smallest K such that the largest-K energy mass is ≥ frac_energy."""

    p = np.abs(X) ** 2
    p = p / p.sum()
    idx = np.argsort(p)[::-1]
    c = np.cumsum(p[idx])
    return int(np.searchsorted(c, frac_energy) + 1)


def golden_drift_ensemble(N: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Golden quasi-periodic ensemble for Theorem 8.

    Generates M signals of length N from the ensemble:
        x[n] = exp(i 2π (f₀ n + a · frac(n φ)))

    where f₀ ~ Uniform[0,1] and a ~ Uniform[-1,1].

    This is the signal model for the Golden Spectral Concentration Inequality.
    """
    N = int(N)
    M = int(M)
    n = np.arange(N, dtype=np.float64)
    frac = np.mod(n * PHI, 1.0)

    out = np.empty((M, N), dtype=np.complex128)
    for i in range(M):
        f0 = rng.uniform(0.0, 1.0)
        a = rng.uniform(-1.0, 1.0)
        out[i] = np.exp(1j * 2.0 * np.pi * (f0 * n + a * frac))
    return out
