# QuantoniumOS Release Notes

## v2.0.3 — Theorem 8 Diophantine Upgrade

### Theorem 8: CONSTRUCTIVE + COMPUTATIONAL → CONSTRUCTIVE + DIOPHANTINE

The golden spectral concentration inequality (Theorem 8) has been upgraded from constructive + computational to **constructive + Diophantine** by grounding the DFT spectral leakage in classical number theory.

**What changed:**
1. **New Diophantine proof module** — `algorithms/rft/theory/theorem8_diophantine.py` implements 6 lemmas (8.4a–f) proving that DFT spectral leakage on golden-quasi-periodic signals is a **number-theoretic theorem**, not merely a computational observation.
2. **Classical foundations** — Hurwitz (1891), Steinhaus-Sós (1957), Weyl (1916), Erdős-Turán (1948), Roth (1955).
3. **Key insight** — The DFT **must** leak energy because golden frequencies never align with DFT bins — this is Hurwitz's theorem, with optimal constant 1/√5 for the golden ratio.
4. **Test suite** — 46 new tests in `tests/proofs/test_theorem8_diophantine.py` (9 test classes), integrated into `FormalProofEngine` with DIOPHANTINE status.
5. **Documentation updated** — THEOREMS_RFT_IRONCLAD.md, MATHEMATICAL_CLAIMS_INVENTORY.md, TECHNICAL_DETAILS.md, DEFINITION_FIREWALL.md, STATE.md, RELEASE_NOTES.md.

**Diophantine lemma chain (8.4a–f):**
- 8.4a: Three-Distance Theorem (Steinhaus-Sós 1957)
- 8.4b: Hurwitz Irrationality Bound (1891) — |φ − p/q| ≥ 1/(√5·q²)
- 8.4c: Quantitative Weyl Discrepancy (Erdős-Turán 1948)
- 8.4d: Per-Harmonic DFT Leakage (Dirichlet kernel + Hurwitz)
- 8.4e: RFT Zero-Misalignment Principle (constructive)
- 8.4f: Diophantine Gap Theorem (punchline)

**Proof status summary:** All 11 theorems (1–11) now have formal proofs. Theorem 8 is the strongest: CONSTRUCTIVE + DIOPHANTINE. Only Conjecture 12 (variational minimality) remains empirical.

---

## v2.0.2 — Theorem 8 Formal Proof Upgrade

### Theorem 8: PARTIALLY PROVEN → CONSTRUCTIVE + COMPUTATIONAL

The golden spectral concentration inequality (Theorem 8) has been upgraded from empirical/partially-proven to a full constructive + computational proof with zero empirical claims remaining.

**What changed:**
1. **New formal proof module** — `algorithms/rft/theory/theorem8_formal_proof.py` implements 5 lemmas (8.3a–e) proving concentration via Vandermonde algebra rather than the originally conjectured Landau-Widom eigenvalue decay.
2. **Key discovery** — The ensemble covariance has *exact* rank K = O(log N) (not just exponential decay). The N−K eigenvalues are machine-zero (~10⁻¹⁷).
3. **Proven results** — K₀.₉₉(RFT) = K = O(log N), K₀.₉₉(DFT) ∝ N^0.75, gap ΔK₀.₉₉ ∝ N^1.04.
4. **Test suite** — 33 new tests in `tests/proofs/test_theorem8_formal_proof.py`, integrated into `FormalProofEngine` (38 total proof engine tests).
5. **Documentation updated** — THEOREMS_RFT_IRONCLAD.md, MATHEMATICAL_CLAIMS_INVENTORY.md, TECHNICAL_DETAILS.md, DEFINITION_FIREWALL.md, STATE.md.

**Proof status summary:** All 11 theorems (1–11) now have formal proofs. Only Conjecture 12 (variational minimality) remains empirical.

---

## v2.0.1 — February 7, 2026

### Audit Remediation (10 fixes)

Full repository audit (2,264 tests: 1844 pass, 6 fail, 415 skip, 1 error) identified
critical issues. All resolved in this patch:

1. **scipy.stats.erfc fix** — `erfc` imported from `scipy.special` (was `scipy.stats`).
2. **Claims lint .venv exclusion** — `test_claims_lint.py` no longer scans virtual-env files.
3. **Theorem 8 contradiction** — `MATHEMATICAL_CLAIMS_INVENTORY.md` aligned with `THEOREMS_RFT_IRONCLAD.md`; status changed to PARTIALLY PROVEN (later upgraded to CONSTRUCTIVE + COMPUTATIONAL in v2.0.2).
4. **SIMD made functional** — `rftmw_core.hpp` and `rft_fused_kernel.hpp` AVX2 paths now perform real SIMD operations.
5. **Feistel roundtrip fixed** — `rft_phi_permute` rewritten with canonical φ formula. Encrypt/decrypt corrected. `rft_sis.c` phases updated.
6. **phi_phase_fft_optimized deprecated** — `DeprecationWarning` added at module and function level.
7. **README.md split** — From 1181 to 145 lines; technical detail in `docs/TECHNICAL_DETAILS.md`.
8. **Mobile screens wired** — All 8 disconnected screens registered in AppNavigator.
9. **Stale docs updated** — `STATE.md` and `RELEASE_NOTES.md` updated; Structural Distinctness marked proven (Theorem 6).
10. **Dependency management restructured** — Added reproducible core lockfile (`requirements-lock-core.txt`) derived from CI/dev venv via `pip freeze`. Split requirements into core/dev/ml-extra. `requirements.txt` retains version ranges; lock captures exact installed versions only.

### Canonical φ-Formula Alignment

| Component | Old | New (Canonical) |
|-----------|-----|-----------------|
| RFT-SIS (`rft_sis.c`) | `frac(k/φ)` | `frac((k+1)·φ)` |
| Feistel permute | `(i*1618/1000) % len` | `floor(frac((i+1)·φ) * len) % len` |

---

## v2.0.0 — December 17, 2025

## Major Updates: Canonical RFT Definition

This release formalizes the **Canonical Resonant Fourier Transform (RFT)** definition, resolving inconsistencies in previous documentation and implementations.

### Key Changes

*   **Canonical RFT Redefinition:** The Canonical RFT is now explicitly defined as the **Gram-normalized irrational-frequency exponential basis** ($\widetilde{\Phi} = \Phi (\Phi^H \Phi)^{-1/2}$). This ensures exact unitarity at finite $N$ while preserving the golden-ratio resonance structure.
*   **Legacy Definition:** The previous "Eigenbasis of Resonance Operator" definition has been moved to a legacy/alternative status.
*   **Documentation Updates:**
    *   `docs/proofs/PHI_RFT_MATHEMATICAL_PROOFS.md` and `.tex` updated to reflect the Gram-normalized construction.
    *   `papers/RFTPU_TECHNICAL_SPECIFICATION_V2.tex` and `.md` updated.
    *   `CANONICAL.md` updated to point to the correct implementation.
    *   `algorithms/rft/README_RFT.md` updated.
*   **Implementation:** The core implementation in `algorithms/rft/core/resonant_fourier_transform.py` is now the authoritative reference for the Canonical RFT.

### Verification

*   Validated via `tests/validation/test_phi_frame_normalization.py`.
*   Confirmed unitarity and frame properties.

## Quantum Simulation Verification (v2.0.0-verified)

*   **Fidelity:** Verified 1.000000 fidelity for Superposition (Hadamard) and Entanglement (Bell State).
*   **Scaling:** Confirmed $O(N)$ scaling for Symbolic Compression vs $O(2^N)$ for Classical Simulators.
*   **Benchmarks:** `docs/validation/QUANTUM_VERIFICATION_REPORT_v2.0.0.txt`

## Patent Notice

**USPTO Application #19/169,399** covers the methods and systems described in this release.
