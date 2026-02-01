# Paper Claims Update Required

## Status: PENDING REVIEW

This document tracks required updates to published papers following the RFT definition correction (December 2025).

---

## Summary of Changes

### What Changed

The definition of "RFT" (Resonant Fourier Transform) has been corrected:

| Aspect | OLD (Deprecated) | NEW (Canonical) |
|--------|------------------|-----------------|
| **Formula** | Ψ = D_φ C_σ F | Eigenbasis of K = T(R·d) |
| **Sparsity** | None vs FFT | +15-20 dB on target signals |
| **Name** | "Φ-RFT" | "RFT" (Resonant Fourier Transform) |
| **Old name** | N/A | "φ-phase FFT" (deprecated) |

---

## Papers Requiring Updates

### 1. `paper.tex` / `paper.pdf`
- **Section 1:** Replace "Φ-RFT" first mention with operator definition
- **Equation 1-3:** Move Ψ = D_φ C_σ F to "Related Constructions" section
- **Claims:** Add caveat that φ-phase FFT has no sparsity advantage

### 2. `coherence_free_hybrid_transforms.tex`
- **Abstract:** Update to reference operator-based RFT
- **Section 2:** Add note that compression gains come from cascade architecture, not φ-phase FFT itself

### 3. `dev_manual.tex`
- **Chapter 1:** Update RFT definition
- **API Reference:** Point to `resonant_fourier_transform.py` as canonical

### 4. `zenodo_rftpu_publication.tex`
- **RFTPU Definition:** Clarify that RFTPU executes operator-based RFT kernels
- **Hardware claims:** Add that φ-phase FFT accelerator is a simpler (but less powerful) variant

---

## Recommended Language

### First Mention of RFT

> In earlier drafts we used "RFT" for a φ-phase FFT variant (Ψ = D_φ C_σ F). That construction is unitary but does not change coefficient magnitudes relative to FFT, providing no sparsity advantage.
>
> In the current work, **Resonant Fourier Transform (RFT)** refers strictly to the eigenbasis of a resonance operator K built from structured autocorrelation. The golden ratio φ, Fibonacci factors, and similar structures enter as parameters of K, not as the definition of RFT itself.

### Sparsity Claims

> RFT (operator-defined) shows improved sparsity vs FFT/DCT on resonance-structured signal families. On golden quasi-periodic signals, RFT achieves +15-20 dB PSNR at 10% coefficient retention compared to FFT/DCT baselines. See `tests/benchmarks/honest_rft_benchmark.py` for methodology.

### Deprecation Notice

> The original "Φ-RFT" formula (Ψ = D_φ C_σ F) is now classified as "φ-phase FFT" and is preserved for backwards compatibility. It is unitary and O(N log N), but has the property |(Ψx)_k| = |(Fx)_k| for all x, meaning it offers no compression advantage over standard FFT.

---

## Validation Suite

Run the following to verify all claims:

```bash
# Formal proofs
python3 algorithms/rft/theory/formal_framework.py

# Honest benchmark
python3 tests/benchmarks/honest_rft_benchmark.py

# Multi-scale validation
python3 tests/benchmarks/rft_multiscale_benchmark.py

# Real-world signals
python3 tests/benchmarks/rft_realworld_benchmark.py
```

---

## Timeline

| Task | Status |
|------|--------|
| README.md updated | ✅ Completed |
| algorithms/rft/README_RFT.md created | ✅ Completed |
| Core files renamed | ✅ Completed |
| Paper updates | 🔲 Pending |
| Zenodo re-upload | 🔲 Pending |

---

*Last Updated: December 2025*
