#!/usr/bin/env python3
"""
Generate two publication-quality figures for the RFT Diophantine paper:

  Figure 1: Cumulative energy curves (DFT vs RFT) for synthetic quasicrystal signal
  Figure 2: Heatmap of |U| vs |F| showing non-DFT magnitude structure

Usage:
    python figures/generate_paper_figures.py

Outputs:
    figures/fig1_cumulative_energy.pdf
    figures/fig1_cumulative_energy.png
    figures/fig2_basis_heatmap.pdf
    figures/fig2_basis_heatmap.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv, dft

# ── Reproducibility ──
np.random.seed(42)

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

PHI = (1 + np.sqrt(5)) / 2


def golden_freq_grid(N):
    """Return golden frequency grid f_k = frac((k+1)*phi)."""
    return np.array([(k + 1) * PHI % 1 for k in range(N)])


def build_Phi(N):
    """Build the raw phi-grid Vandermonde basis."""
    f = golden_freq_grid(N)
    n = np.arange(N)
    Phi = np.exp(2j * np.pi * np.outer(n, f)) / np.sqrt(N)
    return Phi


def build_U(Phi):
    """Canonical RFT basis via Löwdin orthogonalization (eigendecomposition path)."""
    G = Phi.conj().T @ Phi
    eigvals, Q = np.linalg.eigh(G)
    G_inv_sqrt = Q @ np.diag(1.0 / np.sqrt(eigvals)) @ Q.conj().T
    U = Phi @ G_inv_sqrt
    # Verify unitarity
    err = np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]))
    assert err < 1e-10, f"U not unitary: error = {err}"
    return U


def build_F(N):
    """Unitary DFT matrix."""
    return dft(N, scale="sqrtn")


def cumulative_energy(W, x):
    """Compute sorted cumulative energy curve for transform W applied to x."""
    c = W.conj().T @ x
    e = np.abs(c) ** 2
    e_sorted = np.sort(e)[::-1]
    cum = np.cumsum(e_sorted)
    return cum / np.sum(e)


def make_quasicrystal_signal(N, fib_indices=(1, 2, 3, 5, 8), rng=None):
    """Generate signal from the Golden-Harmonic Subspace Ensemble (§6.1).
    
    Uses complex exponentials: x = V @ c, where V has columns 
    v_m[n] = (1/sqrt(N)) exp(i 2π frac(m·φ) n) and c ~ CN(0, I).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = np.arange(N)
    K = len(fib_indices)
    V = np.zeros((N, K), dtype=complex)
    for j, a in enumerate(fib_indices):
        freq = (a * PHI) % 1
        V[:, j] = np.exp(2j * np.pi * freq * n) / np.sqrt(N)
    # Random complex coefficients
    c = rng.standard_normal(K) + 1j * rng.standard_normal(K)
    return V @ c


# ═══════════════════════════════════════
# FIGURE 1: Cumulative Energy Curves
# ═══════════════════════════════════════

def generate_figure1():
    N = 256
    n_trials = 500
    fib_indices = (1, 2, 3, 5, 8)

    Phi = build_Phi(N)
    U = build_U(Phi)
    F = build_F(N)

    rng = np.random.default_rng(42)

    # Accumulate cumulative curves over many trials
    rft_curves = np.zeros((n_trials, N))
    dft_curves = np.zeros((n_trials, N))

    for t in range(n_trials):
        x = make_quasicrystal_signal(N, fib_indices, rng)
        rft_curves[t] = cumulative_energy(U, x)
        dft_curves[t] = cumulative_energy(F, x)

    rft_mean = rft_curves.mean(axis=0)
    dft_mean = dft_curves.mean(axis=0)
    rft_std = rft_curves.std(axis=0)
    dft_std = dft_curves.std(axis=0)

    k = np.arange(1, N + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # Mean curves
    ax.plot(k, rft_mean, color="#2166ac", linewidth=2, label="RFT (canonical $U$)")
    ax.plot(k, dft_mean, color="#b2182b", linewidth=2, label="DFT ($F$)")

    # ±1 std bands
    ax.fill_between(k, rft_mean - rft_std, rft_mean + rft_std, alpha=0.15, color="#2166ac")
    ax.fill_between(k, dft_mean - dft_std, dft_mean + dft_std, alpha=0.15, color="#b2182b")

    # 99% threshold
    ax.axhline(0.99, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(N * 0.75, 0.985, "$\\tau = 0.99$", color="gray", fontsize=9)

    # Mark K_0.99 for each
    rft_k99 = np.searchsorted(rft_mean, 0.99) + 1
    dft_k99 = np.searchsorted(dft_mean, 0.99) + 1

    ax.annotate(
        f"$K_{{0.99}}^{{\\mathrm{{RFT}}}} \\approx {rft_k99}$",
        xy=(rft_k99, 0.99), xytext=(rft_k99 + 15, 0.92),
        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.2),
        fontsize=10, color="#2166ac",
    )
    ax.annotate(
        f"$K_{{0.99}}^{{\\mathrm{{DFT}}}} \\approx {dft_k99}$",
        xy=(dft_k99, 0.99), xytext=(dft_k99 + 10, 0.86),
        arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.2),
        fontsize=10, color="#b2182b",
    )

    ax.set_xlim(1, 60)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("Coefficients retained ($j$)")
    ax.set_ylabel("Cumulative energy captured")
    ax.set_title(
        "Spectral Concentration: RFT vs DFT\n"
        "Synthetic quasicrystal signal, $N = 256$, 5 Fibonacci-index tones, 500 trials"
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/fig1_cumulative_energy.pdf")
    fig.savefig("figures/fig1_cumulative_energy.png")
    plt.close(fig)
    print(f"Figure 1 saved. RFT K_0.99 ≈ {rft_k99}, DFT K_0.99 ≈ {dft_k99}")


# ═══════════════════════════════════════
# FIGURE 2: |U| vs |F| Heatmaps
# ═══════════════════════════════════════

def generate_figure2():
    N = 32  # Small enough to see structure clearly

    Phi = build_Phi(N)
    U = build_U(Phi)
    F = build_F(N)

    absU = np.abs(U)
    absF = np.abs(F)

    # Shared colorbar range
    vmin = 0
    vmax = max(absU.max(), absF.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1, 1]})

    # DFT
    im0 = axes[0].imshow(absF, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title("$|F|$ (DFT)", fontsize=12)
    axes[0].set_xlabel("Frequency index $k$")
    axes[0].set_ylabel("Time index $n$")

    # RFT
    im1 = axes[1].imshow(absU, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title("$|U|$ (Canonical RFT)", fontsize=12)
    axes[1].set_xlabel("Frequency index $k$")
    axes[1].set_ylabel("Time index $n$")

    # Difference
    diff = absU - absF
    dmax = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="equal")
    axes[2].set_title("$|U| - |F|$  (Difference)", fontsize=12)
    axes[2].set_xlabel("Frequency index $k$")
    axes[2].set_ylabel("Time index $n$")

    # Colorbars
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Magnitude")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="$\\Delta$ Magnitude")

    # Stats annotation
    mag_std = absU.std()
    novelty = np.linalg.norm(absU - absF, "fro") / np.sqrt(N)
    fig.text(
        0.5, -0.02,
        f"$N = {N}$    |    DFT magnitudes: constant $1/\\sqrt{{N}} = {1/np.sqrt(N):.4f}$"
        f"    |    RFT magnitude std: {mag_std:.4f}"
        f"    |    $\\mathcal{{N}}_{{\\mathrm{{abs}}}} = {novelty:.3f}$",
        ha="center", fontsize=9, color="gray",
    )

    fig.suptitle(
        "Basis Magnitude Structure: DFT vs Canonical RFT ($N = 32$)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("figures/fig2_basis_heatmap.pdf", bbox_inches="tight")
    fig.savefig("figures/fig2_basis_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved. Mag std = {mag_std:.4f}, Novelty = {novelty:.3f}")


if __name__ == "__main__":
    print("Generating Figure 1 (cumulative energy curves)...")
    generate_figure1()
    print()
    print("Generating Figure 2 (basis magnitude heatmaps)...")
    generate_figure2()
    print()
    print("Done. Figures saved to figures/")
