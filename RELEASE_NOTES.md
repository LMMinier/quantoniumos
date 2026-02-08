# QuantoniumOS Release Notes

## v2.0.1 — February 7, 2026

### Audit Remediation (10 fixes)

Full repository audit (2,264 tests: 1844 pass, 6 fail, 415 skip, 1 error) identified
critical issues. All resolved in this patch:

1. **scipy.stats.erfc fix** — `erfc` imported from `scipy.special` (was `scipy.stats`).
2. **Claims lint .venv exclusion** — `test_claims_lint.py` no longer scans virtual-env files.
3. **Theorem 8 contradiction** — `MATHEMATICAL_CLAIMS_INVENTORY.md` aligned with `THEOREMS_RFT_IRONCLAD.md`; status changed to PARTIALLY PROVEN.
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
