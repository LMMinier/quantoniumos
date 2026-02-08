# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos

import numpy as np

from algorithms.rft.core.absolute_novelty import (
    certified_abs_novelty_lower_bound_to_dft,
    heuristic_abs_novelty_upper_bound,
)
from algorithms.rft.core.transform_theorems import canonical_unitary_basis, fft_unitary_matrix


def test_absolute_transform_novelty_certificate_vs_dft_family():
    """Repo-enforced absolute novelty certificate for the transform itself.

    We certify a true lower bound for

        inf_{D1,D2,P} ||U_phi - D1 P F D2||_F / sqrt(N)

    by using magnitude mismatch, which is invariant under the allowed symmetries
    for the unitary DFT family.

    We also compute a heuristic best-alignment distance over a bounded
    permutation set as a sanity check.
    """

    # Observed certified lower bounds are ~0.15–0.19 for N=32/64.
    # Use a meaningful but non-brittle threshold.
    eps = 5e-2
    for N in (32, 64):
        U = canonical_unitary_basis(N)

        lb = certified_abs_novelty_lower_bound_to_dft(U)
        assert lb > eps, f"Certified novelty lower bound too small for N={N}: {lb}"

        # Sanity: best-found alignment distance (upper bound on the true infimum)
        F = fft_unitary_matrix(N)
        best = heuristic_abs_novelty_upper_bound(
            U,
            F,
            num_random_perms=8,
            phase_iters=25,
            seed=0x5EED,
        )

        # Any explicit aligned distance must respect the certified lower bound.
        assert best.distance + 1e-12 >= lb, (
            f"Unexpected: heuristic distance < certified lower bound for N={N}: "
            f"best={best.distance}, lb={lb}"
        )

        # Optional guardrail: best-found should also be nontrivial.
        assert best.distance > eps, f"Heuristic novelty distance too small for N={N}: {best.distance}"


def test_absolute_novelty_lower_bound_is_zero_for_dft_itself():
    """Control: the certified LB is 0 for the DFT (and anything with flat modulus)."""

    for N in (32, 64):
        F = fft_unitary_matrix(N)
        lb = certified_abs_novelty_lower_bound_to_dft(F)
        assert np.isclose(lb, 0.0, atol=1e-12), f"Expected LB=0 for DFT itself, got {lb}"
