# RFT Theory (Proof-First Layout)

This document provides a proof-first, textbook-style foundation for the Resonant Fourier Transform (RFT) used in this repository.

## 1. Canonical Definition

### Resonant Fourier Transform (RFT) — Canonical Definition

Domain: $x \in \mathbb{R}^N$ (or $\mathbb{C}^N$)  
Codomain: $X \in \mathbb{C}^N$

Let $\Phi \in \mathbb{C}^{N\times N}$ be a unitary basis matrix with elements

$$
\Phi_{k,t} = \exp\big(j 2\pi f_k \tfrac{t}{N} + j\,\phi_k\big)
$$

where:
- $f_k$ are deterministic resonant frequencies derived from a fixed irrational sequence (e.g., golden ratio modulation)
- $\phi_k$ are deterministic phase offsets

**Forward transform:**

$$
X = \Phi^H x
$$

**Inverse transform:**

$$
x = \Phi X
$$

Implementation note: in finite precision and finite $N$, exact unitarity can be enforced by Gram-matrix normalization of the raw exponential basis (see `algorithms/rft/core/resonant_fourier_transform.py`).

---

## 2. Lemmas

### Lemma 1: Linearity
For all scalars $a,b$ and signals $x_1,x_2$:

$$
\mathcal{R}\{a x_1 + b x_2\} = a\,\mathcal{R}\{x_1\} + b\,\mathcal{R}\{x_2\}
$$

**Justification:** $\mathcal{R}\{x\} = \Phi^H x$ is linear as a matrix-vector product.

### Lemma 2: Invertibility
If $\Phi$ is unitary, i.e. $\Phi^H\Phi = I$, then

$$
\mathcal{R}^{-1}\{X\} = \Phi X
$$

and $\mathcal{R}^{-1}(\mathcal{R}(x)) = x$ up to numerical precision.

---

## 3. Theorem — Unitarity Condition

**Theorem:** The RFT is unitary iff

$$
\Phi^H \Phi = I
$$

**Proof (standard):** A linear transform $X = \Phi^H x$ preserves inner products (and hence energy) iff $\Phi$ is unitary, i.e. $\langle x,y\rangle = \langle \Phi^H x, \Phi^H y\rangle$ for all $x,y$. This is equivalent to $\Phi^H\Phi=I$.

**Commentary:** Whether the raw irrational-frequency exponential basis is exactly unitary at finite $N$ depends on the frequency set. In practice, this repo provides a numerically unitary basis by orthonormalizing the raw basis when requested.

---

## 4. Candidate Extremal Property (Hypothesis)

**Hypothesis:** Among deterministic unitary bases, the RFT basis (for a chosen irrational-frequency rule) minimizes coefficient entropy for quasi-periodic signals whose spectral spacing follows irrational ratios:

$$
\Phi^* = \arg\min_{\Phi \in U(N)} H(\Phi^H x)
$$

where $H$ is Shannon entropy and $U(N)$ is the unitary group.

This is treated as an empirical hypothesis in this repository; it should be benchmarked against FFT/DCT/KLT baselines under a fixed evaluation protocol.

---

## 5. Complexity Characterization

| Transform | Complexity | Notes |
|----------|------------|------|
| DFT | $O(N^2)$ | Standard definition |
| FFT | $O(N\log N)$ | Cooley–Tukey family |
| DCT | $O(N\log N)$ | Fast cosine transforms |
| RFT (naïve) | $O(N^2)$ | Dense matrix-vector multiplication |
| RFT (potential) | $O(N\log N)$ | If exploitable structure exists (unproven) |

---

## Gram-Matrix Normalization and Frame-Correct Inversion for Irrational Frequency RFT

For an irrationally-spaced exponential construction, finite-$N$ orthogonality is not guaranteed. In that case, “inverse by correlation” ($\Phi^H$) is not generally correct.

Let $\Phi \in \mathbb{C}^{N\times N}$ be a full-rank (invertible) basis matrix whose columns are basis vectors. Define the Gram matrix:

$$
G = \Phi^H \Phi.
$$

If $\Phi$ is full rank then $G$ is Hermitian positive definite.

### Frame-correct coefficients (dual-frame solve)

The exact coefficients that reconstruct $x$ via $x = \Phi X$ are:

$$
X = (\Phi^H\Phi)^{-1}\Phi^H x = G^{-1}\Phi^H x.
$$

This reduces to $X = \Phi^H x$ when $\Phi$ is unitary.

### Gram-normalized (unitary) basis

Define:

$$
\widetilde{\Phi} = \Phi\,G^{-1/2}.
$$

Then:

$$
\widetilde{\Phi}^H\widetilde{\Phi} = G^{-1/2}\,(\Phi^H\Phi)\,G^{-1/2} = I,
$$

so $\widetilde{\Phi}$ is unitary and correlation becomes a correct inverse.

Implementation in this repository:
- `algorithms/rft/core/gram_utils.py` provides $G$ and $G^{-1/2}$
- `algorithms/rft/core/resonant_fourier_transform.py` supports `use_gram_normalization=True` and `frame_correct=True` for square-kernel operation

Discrete-time frequency periodicity note:
- For sampled signals, complex exponentials are periodic in frequency (e.g., cycles/sample is defined modulo 1). The square-kernel implementation therefore folds the irrational frequency rule into the fundamental band before forming $\Phi$ to avoid numerical near-degeneracy at finite $N$.

For a concise, publication-ready proof note (tight frame / Parseval property, plus asymptotic orthogonality discussion), see:
- `docs/theory/RFT_FRAME_NORMALIZATION.md`

References (orientation):
- Oppenheim & Schafer, *Discrete-Time Signal Processing* (discrete exponential orthogonality for the DFT grid)
- O. Christensen, *An Introduction to Frames and Riesz Bases* (frame operators, dual frames)
- Encyclopaedia Britannica, “Fourier analysis”

---

## References (for orientation)

- Encyclopaedia Britannica, “Fourier Analysis” (general transform framing)
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing* (standard DFT/transform properties)
- R. N. Bracewell, *The Fourier Transform and Its Applications*
