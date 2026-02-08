# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import numpy as np

from algorithms.rft.core.true_wave_compute import WaveComputer


def test_soft_logic_support_properties() -> None:
    wc = WaveComputer(N=64)
    rng = np.random.default_rng(0)

    c1 = (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(np.complex128)
    c2 = (rng.normal(size=64) + 1j * rng.normal(size=64)).astype(np.complex128)

    m1 = wc.spectral_support_mask(c1, frac_of_max=0.25)
    m2 = wc.spectral_support_mask(c2, frac_of_max=0.25)

    c_and = wc.wave_and_soft(c1, c2, frac_of_max=0.25)
    m_and = np.abs(c_and) > 0

    # AND result must be supported only where both supports are active.
    assert np.all(m_and <= (m1 & m2))


def test_soft_conditional_interpolates() -> None:
    wc = WaveComputer(N=32)

    then_c = np.ones(32, dtype=np.complex128)
    else_c = np.zeros(32, dtype=np.complex128)

    cond = np.zeros(32, dtype=np.complex128)
    cond[0] = 10.0 + 0.0j  # strong DC

    out = wc.conditional_select_soft(cond, then_c, else_c)
    assert np.allclose(out, then_c, atol=1e-12, rtol=0.0)
