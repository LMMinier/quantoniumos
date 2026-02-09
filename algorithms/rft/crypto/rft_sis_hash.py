# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
RFT-SIS Cryptographic Hash — Research Prototype
================================================

Hybrid RFT-SIS hash: φ-structured + pseudo-random matrix.

Extracted from algorithms.rft.core.resonant_fourier_transform for clarity.
"""
from __future__ import annotations

import hashlib

import numpy as np

from algorithms.rft.core.resonant_fourier_transform import (
    PHI,
    rft_basis_function,
)


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

    def __init__(
        self,
        sis_n: int = 512,
        sis_m: int = 1024,
        sis_q: int = 3329,
        sis_beta: int = 100,
        domain_salt: bytes = b"RFT_SIS_DEFAULT_DOMAIN_2026",
    ):
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

        Security note:
        Any reduction to standard SIS assumes R is uniform and unpredictable.
        Here R is generated for research reproducibility and should not be
        treated as a cryptographic randomness source.
        """
        m, n, q = self.sis_m, self.sis_n, self.sis_q

        # Build A_φ: deterministic φ-structured matrix
        A_phi = np.zeros((m, n), dtype=np.int64)
        for i in range(m):
            for j in range(n):
                val = (i + 1) * (j + 1) * PHI
                A_phi[i, j] = int((val % 1.0) * q)

        # Build R: random matrix from salted PRNG
        seed_hash = hashlib.sha3_256(self.domain_salt).digest()
        seed_int = int.from_bytes(seed_hash[:4], "little")
        rng = np.random.default_rng(seed_int)
        R = rng.integers(0, q, size=(m, n), dtype=np.int64)

        # Combine: A = A_φ + R (mod q)
        A = (A_phi + R) % q

        return A.astype(np.int32)

    def _expand_input(self, data: bytes) -> np.ndarray:
        """Expand input bytes to sis_n floats via hash chain."""
        expanded = np.zeros(self.sis_n, dtype=np.float64)

        current_hash = data
        idx = 0

        while idx < self.sis_n:
            h = hashlib.sha3_256(
                current_hash if isinstance(current_hash, bytes) else current_hash.encode()
            )
            digest = h.digest()

            for i in range(0, min(32, self.sis_n - idx), 4):
                uint_val = int.from_bytes(digest[i : i + 4], "little")
                float_val = (uint_val / (2**32)) * 2 - 1
                expanded[idx] = float_val
                idx += 1
                if idx >= self.sis_n:
                    break

            current_hash = digest + (
                data if isinstance(data, bytes) else data.encode()
            )

        return expanded

    def _rft_transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply RFT and extract magnitude/phase features."""
        transformed = self._rft_matrix @ signal.astype(np.complex128)

        magnitude = np.abs(transformed)
        phase = np.angle(transformed)

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
