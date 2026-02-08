#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC

"""Generate canonical RFT kernel for Theorem 8 Option A (Q1.15 memh).

We need the *complex* canonical unitary basis used by the proof code:
  U = canonical_unitary_basis(N)

The transform coefficients are y = U^H x, i.e. y[k] = sum_n conj(U[n,k]) * x[n].
This script writes conj(U) as two memh files (real/imag) in signed Q1.15.

Outputs (default, from hardware/tb):
  - theorem8_kernel_uconj_real_N{N}_q15.memh
  - theorem8_kernel_uconj_imag_N{N}_q15.memh

These are consumed by tb_theorem8_energy_dump.sv via plusargs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algorithms.rft.core.transform_theorems import canonical_unitary_basis


def _float_to_q1_15(value: float) -> int:
    scale = (1 << 15) - 1
    scaled = int(np.round(float(value) * scale))
    if scaled > 32767:
        scaled = 32767
    if scaled < -32768:
        scaled = -32768
    return scaled


def write_kernel_memh(U_conj: np.ndarray, out_real: Path, out_imag: Path) -> None:
    out_real.parent.mkdir(parents=True, exist_ok=True)
    out_imag.parent.mkdir(parents=True, exist_ok=True)

    N = int(U_conj.shape[0])
    if U_conj.shape != (N, N):
        raise ValueError("U_conj must be square")

    with out_real.open("w", encoding="utf-8") as fr, out_imag.open("w", encoding="utf-8") as fi:
        fr.write("// conj(U)[n,k] real in Q1.15, flattened k-major (k*N + n)\n")
        fr.write(f"// N={N}\n")
        fi.write("// conj(U)[n,k] imag in Q1.15, flattened k-major (k*N + n)\n")
        fi.write(f"// N={N}\n")

        # Flatten in the order the SV TB uses: kernel[k*N + n]
        for k in range(N):
            for n in range(N):
                v = U_conj[n, k]
                wr = _float_to_q1_15(float(np.real(v)))
                wi = _float_to_q1_15(float(np.imag(v)))
                fr.write(f"{(wr & 0xFFFF):04X}\n")
                fi.write(f"{(wi & 0xFFFF):04X}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument(
        "--out-real",
        type=str,
        default="theorem8_kernel_uconj_real_N64_q15.memh",
    )
    ap.add_argument(
        "--out-imag",
        type=str,
        default="theorem8_kernel_uconj_imag_N64_q15.memh",
    )
    args = ap.parse_args()

    U = canonical_unitary_basis(int(args.N))
    U_conj = np.conjugate(U)

    out_real = Path(args.out_real)
    out_imag = Path(args.out_imag)
    write_kernel_memh(U_conj, out_real, out_imag)

    print(f"Wrote: {out_real}")
    print(f"Wrote: {out_imag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
