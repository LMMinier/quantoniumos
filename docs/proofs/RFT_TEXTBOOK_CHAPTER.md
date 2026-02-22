# Chapter: The Resonant Fourier Transform

## 8.1 Introduction

This chapter introduces the Resonant Fourier Transform (RFT) as a deterministic, Fourier-type transform defined by a unitary basis matrix. The RFT is positioned as a member of the broad class of linear transforms used for signal representation, analysis, and synthesis.

## 8.2 Mathematical Definition

Let $x \in \mathbb{R}^N$ (or $\mathbb{C}^N$). Let $\Phi \in \mathbb{C}^{N\times N}$ be a unitary basis matrix.

A canonical RFT construction starts from deterministic complex exponentials:

$$
\Phi_{k,t} = \exp\big(j 2\pi f_k \tfrac{t}{N} + j\,\phi_k\big)
$$

where $f_k$ follow a fixed irrational rule (e.g., golden ratio modulation) and $\phi_k$ are deterministic phase offsets.

The forward and inverse transforms are:

$$
X = \Phi^H x, \quad x = \Phi X.
$$

## 8.3 Properties

### Linearity
The transform is linear because it is defined by matrix multiplication.

### Invertibility
If $\Phi$ is unitary ($\Phi^H\Phi=I$), then the inverse exists and equals $\Phi$.

### Energy Preservation (Parseval-type relation)
For unitary transforms, energy is preserved:

$$
\|x\|_2^2 = \|X\|_2^2.
$$

## 8.4 Computational Aspects

A direct implementation via dense matrix multiplication costs $O(N^2)$. Faster implementations may be possible only if the basis admits additional structure (e.g., recursion or sparsity in a factorization), which is treated as an open research problem.

## 8.5 Applications

In this repository, RFT-related methods are applied in:
- deterministic signal representations
- transform-domain compression pipelines
- experimental cryptographic feature extraction

The mathematical kernel itself does not assume any application; applications live in separate modules.

## 8.6 Comparison with Fourier, DCT, and KLT

- **DFT/FFT:** uses integer-spaced harmonics; fast algorithms exist.
- **DCT:** real cosine basis; strong energy compaction for smooth signals.
- **KLT:** statistically optimal for a distribution but data-dependent.
- **RFT:** deterministic unitary basis with frequencies derived from an irrational rule; evaluated empirically under fixed benchmarks.

## References

- Encyclopaedia Britannica, “Fourier Analysis”
- A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*
- R. N. Bracewell, *The Fourier Transform and Its Applications*
