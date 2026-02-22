# Deterministic Irrational-Frequency Frame Normalization (φ-Grid RFT)

This note formalizes the **finite-$N$** correction used in QuantoniumOS for an irrational-frequency exponential RFT: **Gram-matrix normalization** and (optionally) **dual-frame coefficient extraction**.

The key point is simple and fully general:

- A raw irrational-frequency exponential basis is **not guaranteed orthogonal** at finite $N$.
- If the square basis matrix $\Phi$ is full rank, then the normalized basis
  $$
  \widetilde{\Phi} = \Phi\,(\Phi^H\Phi)^{-1/2}
  $$
  is **unitary**, hence defines a **Parseval (tight) frame** and supports exact inversion by correlation.

This is the mathematically defensible way to talk about a deterministic irrational-frequency “RFT” while keeping unitarity as a verified property.

## Setup

Let $\Phi \in \mathbb{C}^{N\times N}$ be a square basis matrix whose columns are basis vectors $\{\varphi_k\}_{k=0}^{N-1}$.

In the φ-grid construction used by the square-kernel implementation:

- time index $n \in \{0,1,\dots,N-1\}$
- frequencies $f_k = \operatorname{frac}((k+1)\,\phi)$ (folded to $[0,1)$ in cycles/sample)
- columns are sampled complex exponentials:
  $$
  \varphi_k[n] = \frac{1}{\sqrt{N}}\,e^{j2\pi f_k n}
  $$

Define the Gram matrix:
$$
G = \Phi^H\Phi.
$$

If $\Phi$ is full rank, then $G$ is Hermitian positive definite.

## Theorem 1 (Gram-normalized basis is unitary)

**Claim.** If $\Phi \in \mathbb{C}^{N\times N}$ is full rank and $G = \Phi^H\Phi$, then
$$
\widetilde{\Phi} = \Phi\,G^{-1/2}
$$
satisfies $\widetilde{\Phi}^H\widetilde{\Phi}=I$.

**Proof.** Since $G$ is Hermitian positive definite, $G^{-1/2}$ exists and is Hermitian. Then:
$$
\widetilde{\Phi}^H\widetilde{\Phi}
= (G^{-1/2})^H\,\Phi^H\Phi\,G^{-1/2}
= G^{-1/2}\,G\,G^{-1/2}
= I.
$$
Therefore $\widetilde{\Phi}$ has orthonormal columns; it is unitary. ∎

## Corollary 1 (Tight frame / Parseval property)

Because $\widetilde{\Phi}$ is unitary, it is a **Parseval (tight) frame** (indeed an orthonormal basis) for $\mathbb{C}^N$:

- **Energy preservation**: $\lVert x\rVert_2^2 = \lVert \widetilde{\Phi}^H x\rVert_2^2$
- **Perfect reconstruction**: $x = \widetilde{\Phi}\,(\widetilde{\Phi}^H x)$

This is the strongest, simplest unitarity statement available at finite $N$.

## Theorem 2 (Dual-frame coefficients for a non-orthogonal basis)

If one wants to keep the raw irrational-frequency basis $\Phi$ (without Gram-normalizing it), the exact synthesis coefficients are:

$$
X = (\Phi^H\Phi)^{-1}\Phi^H x = G^{-1}\Phi^H x.
$$

This is the (square, full-rank) special case of the dual-frame reconstruction formula.

## Asymptotic orthogonality as $N\to\infty$ (what is true, what is not)

For the unit-norm complex exponentials above,
$$
\langle \varphi_k,\varphi_\ell\rangle = \frac{1}{N}\sum_{n=0}^{N-1} e^{j2\pi (f_k-f_\ell)n}
= \frac{1}{N}\,\frac{1-e^{j2\pi N\Delta f}}{1-e^{j2\pi \Delta f}},\quad \Delta f=f_k-f_\ell.
$$
Hence, when $\Delta f \not\in \mathbb{Z}$,
$$
|\langle \varphi_k,\varphi_\ell\rangle| \le \frac{1}{N\,|\sin(\pi\Delta f)|}.
$$

For irrational spacing (e.g., $f_k=\operatorname{frac}((k+1)\phi)$), the frequency differences $\Delta f$ are equidistributed mod 1, so **typical** pairs have $|\sin(\pi\Delta f)|=\Theta(1)$ and therefore **typical correlations scale like $O(1/N)$**.

However, **worst-case** correlations are more subtle: because $\{m\phi\}$ can be arbitrarily close to 0 mod 1 for some integers $m$ (Diophantine approximation), there can exist indices with unusually small $|\Delta f|$ at large $N$. In other words:

- Average off-diagonal energy can decay with $N$ (empirically measurable).
- The maximum off-diagonal correlation need not monotonically vanish without additional constraints on the frequency set.

This repo therefore treats asymptotic “orthogonality” as an empirical/statistical statement (measured by mutual coherence and average cross-energy), while keeping **finite-$N$ unitarity** as the primary verified property via Gram normalization.

## Reproducibility hooks

- Implementation: `algorithms/rft/core/resonant_fourier_transform.py` (`use_gram_normalization=True` for square mode)
- Gram utilities: `algorithms/rft/core/gram_utils.py`
- Validation: `tests/validation/test_phi_frame_normalization.py`
- Benchmarks:
  - `benchmarks/rft_phi_frame_benchmark.py` (finite-$N$ reconstruction correctness)
  - `benchmarks/rft_phi_frame_asymptotics.py` (mutual coherence / average cross-energy vs FFT)
