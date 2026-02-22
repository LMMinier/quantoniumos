# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Visualization Script: Ψ†F Heatmap
Optional plot showing non-trivial structure of Ψ†F matrix
"""
import numpy as np
import matplotlib.pyplot as plt
from algorithms.rft.core.resonant_fourier_transform import rft_forward_square as rft_forward

def main(n=128, beta=0.83, sigma=1.25, out="docs/figs/psiH_F.png"):
    E = np.eye(n, dtype=np.complex128)
    Psi = np.column_stack([rft_forward(E[:,j], beta=beta, sigma=sigma) for j in range(n)])
    F = np.fft.fft(np.eye(n), norm="ortho", axis=0)
    S = Psi.conj().T @ F
    plt.figure(figsize=(6,5), dpi=140)
    plt.imshow(np.abs(S), aspect='auto', interpolation='nearest')
    plt.title(r"$|\Psi^{H}F|$ heatmap")
    plt.xlabel("F columns")
    plt.ylabel("Ψ rows")
    plt.colorbar()
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    print("wrote", out)

if __name__ == "__main__":
    main()
