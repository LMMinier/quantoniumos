# Proof Map

This map links **canonical theorems** to their **implementations** and **verification tests**.

## Proof Standard (Non‑Negotiable)

“Proven” means a complete mathematical proof independent of implementation, benchmarks, or simulations.
Empirical results, crypto conjectures, and constructions without reductions do **not** belong in `docs/proofs/`.

## Quick Reference (2026-01-29)

| Theorem | Status | Validation |
|---------|--------|------------|
| Unitarity (Gram-normalized) | ✅ Proven | `tests/validation/test_mathematical_correctness.py` |
| Unitarity (Fast Φ-RFT) | ✅ Proven | `tests/validation/test_numerical_stability.py` |
| Resonance kernel non-quadratic | ✅ Proven | `scripts/reproduce_validation.py` |
| Twisted convolution | ✅ Proven | [THEOREMS_RFT_IRONCLAD.md] |
| Canonical ≠ DLCT | ⚠️ Open | Requires invariant for post-Gram matrix |

## Canonical Theorem Sources
- [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- [docs/proofs/RFT_THEOREMS.md](docs/proofs/RFT_THEOREMS.md)
- [docs/proofs/THEOREM_10_HYBRID.md](docs/proofs/THEOREM_10_HYBRID.md)
- [docs/proofs/VALIDATED_THEOREMS.md](docs/proofs/VALIDATED_THEOREMS.md) — Empirically validated
- [docs/INVARIANT_ANALYSIS.md](../INVARIANT_ANALYSIS.md) — Formal separation theorem

---

## Separation Theorem (Formal Impossibility Result)

### Resonance Kernel ∉ Quadratic-Phase Class
- **Document**: [docs/INVARIANT_ANALYSIS.md](../INVARIANT_ANALYSIS.md)
- **Definitions**: Quadratic-phase class $\mathcal{Q}_N$, equivalence relation $\sim$, extended class $\overline{\mathcal{Q}}_N$
- **Lemma 1**: $K \in \overline{\mathcal{Q}}_N \Rightarrow \Delta^2_k \arg(K_{n,k}) = \text{const}$
- **Theorem 1**: Resonance kernel $R_{n,k} = e^{-i2\pi n\phi^{-k}}$ violates this invariant
- **Validation**: Numerical verification shows $\Delta^2 g(k) \propto \phi^{-k}$ (exponential decay, not constant)
- **Scope**: Proven for generating kernel $R$; post-Gram $\widetilde{\Phi}$ requires separate analysis

---

## RFT Core Theorems

### Theorem 1 — Unitarity of RFT Variants
- **Proof**: [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- **Implementation**:
  - [algorithms/rft/core/resonant_fourier_transform.py](algorithms/rft/core/resonant_fourier_transform.py)
  - [algorithms/rft/variants/](algorithms/rft/variants/)
- **Tests**:
  - [tests/rft/test_variant_unitarity.py](tests/rft/test_variant_unitarity.py)
  - [tests/rft/test_canonical_rft.py](tests/rft/test_canonical_rft.py)

### Theorem 2 — Non-Equivalence to LCT/FrFT
- **Proof**: [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- **Implementation Evidence**:
  - [experiments/proofs/non_equivalence_proof.py](experiments/proofs/non_equivalence_proof.py)
  - [experiments/proofs/non_equivalence_theorem.py](experiments/proofs/non_equivalence_theorem.py)
- **Tests**:
  - [tests/rft/test_rft_vs_fft.py](tests/rft/test_rft_vs_fft.py)

### Theorem 3 — Sparsity Advantage on Resonant Signals
- **Proof**: [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- **Implementation Evidence**:
  - [experiments/proofs/sparsity_theorem.py](experiments/proofs/sparsity_theorem.py)
  - [benchmarks/rft_realworld_benchmark.py](benchmarks/rft_realworld_benchmark.py)
- **Tests**:
  - [tests/rft/test_rft_advantages.py](tests/rft/test_rft_advantages.py)

### Theorem 4 — Twisted Convolution Diagonalization
- **Proof**: [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- **Implementation**:
  - [algorithms/rft/core/resonant_fourier_transform.py](algorithms/rft/core/resonant_fourier_transform.py)
- **Tests**:
  - [tests/rft/test_dft_correlation.py](tests/rft/test_dft_correlation.py)

### Theorem 5 — Algebraic Properties of ⋆_Ψ
- **Proof**: [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- **Implementation**:
  - [algorithms/rft/core/resonant_fourier_transform.py](algorithms/rft/core/resonant_fourier_transform.py)
- **Tests**:
  - [tests/rft/test_psihf_entropy.py](tests/rft/test_psihf_entropy.py)

### Theorem 6 — Resonance Kernel Non-Equivalence
- **Proof**: [docs/proofs/THEOREMS_RFT_IRONCLAD.md](docs/proofs/THEOREMS_RFT_IRONCLAD.md)
- **Implementation Evidence**:
  - [experiments/proofs/non_equivalence_proof.py](experiments/proofs/non_equivalence_proof.py)

---

## Hybrid Theorem

### Theorem 10 — Coherence-Free Hybrid Boundary
- **Proof**: [docs/proofs/THEOREM_10_HYBRID.md](docs/proofs/THEOREM_10_HYBRID.md)
- **Implementation**:
  - [algorithms/rft/hybrids/cascade_hybrids.py](algorithms/rft/hybrids/cascade_hybrids.py)
  - [algorithms/rft/hybrids/h3_arft_cascade.py](algorithms/rft/hybrids/h3_arft_cascade.py)
- **Tests**:
  - [tests/benchmarks/test_coherence.py](tests/benchmarks/test_coherence.py)
  - [tests/rft/test_boundary_effects.py](tests/rft/test_boundary_effects.py)

---

## Quantum Correctness (Textbook Definitions)

### Gate Definitions and Unitarity
- **Source**: Nielsen & Chuang (textbook equivalence)
- **Implementation**:
  - [algorithms/rft/quantum_inspired/quantum_gates.py](algorithms/rft/quantum_inspired/quantum_gates.py)
- **Tests**:
  - [tests/proofs/test_entanglement_protocols.py](tests/proofs/test_entanglement_protocols.py)

### Grover’s Algorithm Correctness
- **Source**: Grover (1996)
- **Implementation**:
  - [algorithms/rft/quantum_inspired/quantum_search.py](algorithms/rft/quantum_inspired/quantum_search.py)

---

## Crypto Claims (Standard Primitives)

### AES S-Box / HKDF / HMAC Correctness
- **Sources**: NIST FIPS 197, RFC 5869
- **Implementation**:
  - [algorithms/rft/crypto/enhanced_cipher.py](algorithms/rft/crypto/enhanced_cipher.py)
- **Tests**:
  - [tests/crypto/test_avalanche.py](tests/crypto/test_avalanche.py)
  - [tests/crypto/test_property_encryption.py](tests/crypto/test_property_encryption.py)
  - [tests/crypto/test_rft_sis_hash.py](tests/crypto/test_rft_sis_hash.py)

---

## Research Constructions (Not Proofs)

Cryptographic constructions and conjectures are **not** proofs unless backed by reductions to standard assumptions.
Keep those items in research scope and non‑claims documentation:

- [docs/NOVEL_ALGORITHMS.md](../../docs/NOVEL_ALGORITHMS.md)

---

## Maintenance

- Add new theorems to **docs/proofs/** and update this map.
- Keep the proof-to-test link **1:1** whenever possible.
