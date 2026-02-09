#!/usr/bin/env python3
"""
Generate all publication-quality figures for canonical_rft_paper.tex.
600 DPI, IEEE column-width conventions, Type-1 fonts, no rasterised text.

Figures:
  fig_basis_heatmap.pdf     — |Φ̃| magnitude heatmap vs DFT (N=64)
  fig_frequency_grid.pdf    — φ-grid vs uniform grid on unit circle
  fig_unitarity_scaling.pdf — ||Φ̃^H Φ̃ − I||_F  vs  N
  fig_energy_concentration.pdf — K_{0.99} bar chart (4 signal classes)
  fig_spectral_comparison.pdf  — magnitude spectra (golden chirp, RFT vs FFT)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── IEEE / DSP journal figure style ──────────────────────────────────────
DPI = 600
COL_W = 3.5          # IEEE single-column width (inches)
DBL_W = 7.16         # IEEE double-column width (inches)

rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "text.usetex": False,       # safe fallback; enable if full texlive available
    "pdf.fonttype": 42,         # TrueType → no bitmap fonts in PDF
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# ── colours (IEEE-friendly, colour-blind safe) ───────────────────────────
C_RFT  = "#0072B2"   # blue
C_FFT  = "#D55E00"   # vermillion
C_GRID = "#009E73"   # teal
C_GRAY = "#999999"

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers — canonical RFT (self-contained, no imports from repo)
# ═══════════════════════════════════════════════════════════════════════════
def rft_basis_canonical(N):
    """Return canonical unitary Φ̃ = Φ G^{-1/2}, shape (N, N)."""
    ks = np.arange(N)
    freqs = np.mod((ks + 1) * PHI, 1.0)           # f_k = {(k+1)φ}
    ns = np.arange(N).reshape(-1, 1)
    Phi = np.exp(1j * 2 * np.pi * freqs * ns) / np.sqrt(N)
    G = Phi.conj().T @ Phi
    eigvals, V = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 1e-15)
    G_inv_sqrt = V @ np.diag(eigvals ** -0.5) @ V.conj().T
    return Phi @ G_inv_sqrt

def dft_basis(N):
    """Return unitary DFT matrix, shape (N, N)."""
    ns = np.arange(N).reshape(-1, 1)
    ks = np.arange(N).reshape(1, -1)
    return np.exp(-1j * 2 * np.pi * ns * ks / N) / np.sqrt(N)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1 — Basis magnitude heatmap: |Φ̃| vs |F|
# ═══════════════════════════════════════════════════════════════════════════
def fig_basis_heatmap():
    N = 64
    Phi = rft_basis_canonical(N)
    F   = dft_basis(N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.4))

    im1 = ax1.imshow(np.abs(F), aspect="equal", cmap="viridis",
                     interpolation="nearest", vmin=0, vmax=np.abs(F).max())
    ax1.set_title("(a)  DFT  $|\\mathbf{F}|$")
    ax1.set_xlabel("Frequency index $k$")
    ax1.set_ylabel("Time index $n$")

    im2 = ax2.imshow(np.abs(Phi), aspect="equal", cmap="viridis",
                     interpolation="nearest", vmin=0, vmax=np.abs(Phi).max())
    ax2.set_title(r"(b)  Canonical $\varphi$-RFT  $|\widetilde{\Phi}|$")
    ax2.set_xlabel("Frequency index $k$")
    ax2.set_ylabel("Time index $n$")

    for ax in (ax1, ax2):
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_yticks([0, 16, 32, 48, 63])

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.82, pad=0.02, label="Magnitude")
    fig.savefig(os.path.join(OUT, "fig_basis_heatmap.pdf"))
    plt.close(fig)
    print("  ✓ fig_basis_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2 — Frequency grid: φ-grid vs uniform on unit circle
# ═══════════════════════════════════════════════════════════════════════════
def fig_frequency_grid():
    N = 32
    # DFT grid
    dft_freqs = np.arange(N) / N
    # φ-grid
    phi_freqs = np.mod((np.arange(N) + 1) * PHI, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.8),
                                    subplot_kw={"projection": "polar"})

    # DFT
    theta_dft = 2 * np.pi * dft_freqs
    ax1.scatter(theta_dft, np.ones(N), s=18, c=C_FFT, zorder=5, label="DFT")
    ax1.set_title("(a)  DFT grid ($k/N$)", pad=12)
    ax1.set_rticks([])
    ax1.set_ylim(0, 1.15)

    # φ-RFT
    theta_phi = 2 * np.pi * phi_freqs
    ax2.scatter(theta_phi, np.ones(N), s=18, c=C_RFT, zorder=5,
                label=r"$\varphi$-RFT")
    ax2.set_title(r"(b)  $\varphi$-grid ($\{(k\!+\!1)\varphi\}$)", pad=12)
    ax2.set_rticks([])
    ax2.set_ylim(0, 1.15)

    fig.savefig(os.path.join(OUT, "fig_frequency_grid.pdf"))
    plt.close(fig)
    print("  ✓ fig_frequency_grid.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3 — Unitarity error vs N (log-log)
# ═══════════════════════════════════════════════════════════════════════════
def fig_unitarity_scaling():
    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    errors = []
    for N in sizes:
        U = rft_basis_canonical(N)
        err = np.linalg.norm(U.conj().T @ U - np.eye(N), "fro")
        errors.append(err)

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.semilogy(sizes, errors, "o-", color=C_RFT, markersize=4, label=r"$\|\widetilde{\Phi}^H\widetilde{\Phi} - I\|_F$")
    ax.axhline(1e-12, ls="--", color=C_GRAY, lw=0.6, label=r"$10^{-12}$ threshold")
    ax.set_xlabel("Transform size $N$")
    ax.set_ylabel("Frobenius error")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Unitarity error vs. transform size")
    fig.savefig(os.path.join(OUT, "fig_unitarity_scaling.pdf"))
    plt.close(fig)
    print("  ✓ fig_unitarity_scaling.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4 — Energy concentration bar chart (K_{0.99})
# ═══════════════════════════════════════════════════════════════════════════
def k99(coeffs):
    """Number of sorted-magnitude coefficients for 99% energy."""
    power = np.abs(coeffs) ** 2
    total = power.sum()
    idx = np.argsort(power)[::-1]
    cumsum = np.cumsum(power[idx])
    return int(np.searchsorted(cumsum, 0.99 * total) + 1)

def fig_energy_concentration():
    N = 256
    rng = np.random.default_rng(42)
    U = rft_basis_canonical(N)
    F = dft_basis(N)

    # signals
    impulse = np.zeros(N); impulse[0] = 1.0
    sine = np.sin(2 * np.pi * 7 * np.arange(N) / N)
    noise = rng.standard_normal(N)
    golden_chirp = np.cos(2 * np.pi * PHI ** (np.arange(N) / N * 4))

    signals = {"Impulse": impulse, "Pure sine": sine,
               "White noise": noise, "Golden chirp": golden_chirp}

    labels = list(signals.keys())
    k_fft = [k99(F.conj().T @ s) for s in signals.values()]
    k_rft = [k99(U.conj().T @ s) for s in signals.values()]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    bars1 = ax.bar(x - w/2, k_fft, w, color=C_FFT, edgecolor="white",
                   linewidth=0.3, label="FFT")
    bars2 = ax.bar(x + w/2, k_rft, w, color=C_RFT, edgecolor="white",
                   linewidth=0.3, label=r"$\widetilde{\Phi}$-RFT")

    # value labels
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 3,
                    str(int(h)), ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("$K_{0.99}$ (coefficients for 99% energy)")
    ax.set_title(r"Energy concentration comparison ($N\!=\!256$)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 280)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(OUT, "fig_energy_concentration.pdf"))
    plt.close(fig)
    print("  ✓ fig_energy_concentration.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 5 — Spectral magnitude comparison (golden chirp)
# ═══════════════════════════════════════════════════════════════════════════
def fig_spectral_comparison():
    N = 256
    n = np.arange(N)
    x = np.cos(2 * np.pi * PHI ** (n / N * 4))

    U = rft_basis_canonical(N)
    F = dft_basis(N)

    fft_coeffs = F.conj().T @ x
    rft_coeffs = U.conj().T @ x

    fft_mag = np.abs(fft_coeffs)
    rft_mag = np.abs(rft_coeffs)

    # sort descending for cumulative energy plot
    fft_sorted = np.sort(fft_mag ** 2)[::-1]
    rft_sorted = np.sort(rft_mag ** 2)[::-1]
    fft_cum = np.cumsum(fft_sorted) / fft_sorted.sum()
    rft_cum = np.cumsum(rft_sorted) / rft_sorted.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.2))

    # (a) magnitude spectra
    ax1.stem(np.arange(N), fft_mag, linefmt=C_FFT, markerfmt=".", basefmt=" ",
             label="FFT")
    ax1.stem(np.arange(N), rft_mag, linefmt=C_RFT, markerfmt=".", basefmt=" ",
             label=r"$\widetilde{\Phi}$-RFT")
    ax1.set_xlabel("Coefficient index $k$")
    ax1.set_ylabel("$|\\hat{x}_k|$")
    ax1.set_title("(a)  Magnitude spectra — golden chirp")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_xlim(-2, N + 2)

    # (b) cumulative energy
    ax2.plot(np.arange(1, N + 1), fft_cum, color=C_FFT, label="FFT")
    ax2.plot(np.arange(1, N + 1), rft_cum, color=C_RFT,
             label=r"$\widetilde{\Phi}$-RFT")
    ax2.axhline(0.99, ls="--", color=C_GRAY, lw=0.6, label="99% threshold")
    ax2.set_xlabel("Number of coefficients $K$")
    ax2.set_ylabel("Cumulative energy fraction")
    ax2.set_title("(b)  Cumulative energy — golden chirp")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.set_xlim(0, 80)
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUT, "fig_spectral_comparison.pdf"))
    plt.close(fig)
    print("  ✓ fig_spectral_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating 600 DPI publication figures …")
    fig_basis_heatmap()
    fig_frequency_grid()
    fig_unitarity_scaling()
    fig_energy_concentration()
    fig_spectral_comparison()
    print(f"\nAll figures saved to {OUT}/")
