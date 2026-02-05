# RFT–SIS Hybrid Geometric Hash (v3.1)

**Status:** Research prototype. Empirical validation only.  
**License:** LICENSE-CLAIMS-NC.md (research/academic use only).  
**Patent:** Practices claims of USPTO App. #19/169,399 (RFT). Commercial use requires a license.

## What this is
A hash construction that combines:
- A unitary Resonance Fourier Transform (RFT) front-end (golden-ratio phase + Gram normalization)
- Lattice SIS mixing (n=512, m=1024, q=3329) with coordinate expansion via SHA3-256
- Reproducible matrix sampling via NumPy RNG (not a CSPRNG)

## What we’ve measured (empirical)
- Avalanche: **~50% ± 3%** down to 1e-15 perturbations  
- Collisions: **0 / 10,000** test points  
- Bit distribution: **~0.501 mean** ones  
- RFT unitarity: **||Ψ†Ψ − I||_F ≲ 3e-14** (N≤512)  
- Throughput (pure Python): **~800 hashes/s** on commodity CPU

**These are measurements, not proofs.** No security claims are made beyond standard SIS assumptions.

## Install & Run
```bash
pip install -e .
pytest algorithms/crypto/rft_sis/rft_sis_v31_validation_suite.py -q
