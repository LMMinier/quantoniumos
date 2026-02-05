# Cryptographic Analysis of RFT-Based Mixing

## Executive Summary
**Status:** 🟡 Experimental / Research
**Verdict:** RFT-based mixing passes basic statistical sanity checks (NIST-style tests implemented in-repo) but is **orders of magnitude slower** than hardware-accelerated AES/SHA. It is not currently suitable for production encryption. Any post-quantum or lattice-based security implications remain **hypotheses** and are unproven.

## The Concept: Spectral Spreading via $\phi$
Standard crypto relies on bitwise operations (XOR, S-Boxes) or modular arithmetic. RFT relies on **spectral spreading** across a quasi-periodic basis.

$$ \Psi(k) = \sum_{n=0}^{N-1} x[n] \cdot e^{-i \cdot 2\pi \cdot k \cdot n \cdot \phi} $$

Because $\phi$ has continued-fraction $[1;1,1,1,\ldots]$, it is poorly approximated by rationals. This reduces periodic repetition in rotation sequences and can promote broad spectral mixing in practice.

## Benchmark Results

| Algorithm | Throughput (MB/s) | Avalanche Effect | NIST-Style Test Status |
| :--- | :--- | :--- | :--- |
| **AES-256-GCM** | >2000 MB/s | Perfect (50%) | Pass |
| **SHA-256** | >500 MB/s | Perfect (50%) | Pass |
| **RFT-Mixer** | ~15 MB/s | Good (49.8%) | Basic NIST-style checks pass (in-repo) |

## Key Findings
1. **High Quality Randomness:** Quasi-periodic phase structure often yields strong diffusion in the output spectrum (Avalanche Effect).
2. **Performance Bottleneck:** Floating-point math (even optimized) is much slower than bitwise integer math used in AES/SHA.
3. **Lattice Potential:** The underlying math has superficial similarities to lattice-based constructions, but any post-quantum hardness implications are **unproven** and speculative.

## Recommendation
Use standard libraries (OpenSSL, libsodium) for security. Use RFT-Mixer only for research into transform-based mixing or as a non-standard whitening step.
