# Known Issues

Tracking sheet for known bugs, test gaps, and environment caveats.
Updated with each tagged release.

---

## Resolved in v2.0.1

### `test_energy_spread_threshold[sine]` — incorrect assertion (pre-existing)

**Symptom**: `tests/validation/test_mixing_quality.py::TestSpectralFlatness::test_energy_spread_threshold[sine]`
 failed on every run, including clean `main` HEAD.

**Root cause**: The test assumed *every* signal type should spread energy across ≥ 10 % of
RFT coefficients after transform.  A pure sine is narrowband—a correct unitary
transform *concentrates* its energy in very few bins (8 / 256 captured 90 % of
the energy), which is the expected behaviour.

**Fix** (commit in v2.0.1): Replaced the blanket "spread" assertion with
signal-appropriate checks:

| Signal   | Assertion                          | Rationale                          |
|----------|------------------------------------|------------------------------------|
| impulse  | `num_90 > N * 0.10` (spread)      | Broadband → energy should spread   |
| sine     | `num_90 < N * 0.25` (concentrate) | Narrowband → energy should compact |
| noise    | `num_90 > N * 0.30` (spread)      | Broadband → energy should spread   |

**Verified**: All three parametrised cases pass after the fix.

---

## Open Issues

### Legacy "Φ-RFT" naming in some files

**Status:** Partially resolved (major docs fixed; some secondary files still use "Φ-RFT")

**Issue:** The old φ-phase RFT formula (`φ_phase(n,k,N) = (n·φ^k)/N + (k·φ·n²)/N²`)
was historically called "the RFT" throughout the codebase and documentation.
This formula is **not** the canonical RFT — it is non-unitary (κ ≈ 400 at N=256)
and was superseded by the Gram-normalized `frac((k+1)·φ)` basis.

**What "the RFT" means:** The canonical RFT is defined as:
```
Φ_{n,k} = (1/√N) exp(j 2π frac((k+1)·φ) · n)
Φ̃ = Φ (Φᴴ Φ)^{-1/2}
```
See `README.md` and `algorithms/rft/core/resonant_fourier_transform.py`.

**Fixed in:**
- `docs/NOVEL_ALGORITHMS.md` — Definition 1.1 now clearly labeled as legacy
- `docs/TECHNICAL_DETAILS.md` — Variant table renamed, "Φ-RFT" → "RFT" in key sections
- `docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md` — "Φ-RFT" → "RFT"
- `docs/algorithms/rft/RFT_DEVELOPER_MANUAL.md` — Legacy forms labeled
- `algorithms/rft/core/resonant_fourier_transform.py` — Docstring reordered

**Remaining:** Some secondary docs (`docs/guides/`, `docs/reports/`, hardware files,
variant source files) still use "Φ-RFT" as shorthand. These uses are acceptable
when they clearly refer to the canonical (Gram-normalized) definition, but should
not be confused with the legacy φ-phase formula.

### RFT codec does not compete with industry image codecs

**Status:** Confirmed — this is a fundamental limitation, not a bug.

**Evidence:** Real rate-distortion benchmark on Kodak PhotoCD images
(`benchmarks/codec_rd_curve_real.py`, results in `results/rd_curves/`).

| Codec | BPP | PSNR (dB) | SSIM |
|-------|-----|-----------|------|
| RFT-Binary (prune=0.95) | 3.53 | 19.0 | 0.36 |
| JPEG (q=90) | 3.29 | 37.6 | 0.96 |
| WebP (q=95) | 2.91 | 37.5 | 0.97 |
| AVIF (q=90) | 2.87 | 37.7 | 0.97 |

At every matched bitrate, industry codecs deliver **15–19 dB higher PSNR**.
RFT's "lossless" mode costs **87.5 BPP** (11× raw 8-bit RGB) because it
serializes complex-valued (magnitude + phase) coefficients.

This is expected: the RFT φ-grid basis is not designed for natural-image
energy compaction. Prior "BPP" claims in `ascii_wall_final_hypotheses.py`
and `hybrid_mca_fixes.py` used sparsity-counting, not real compressed sizes.

**Action:** Compression claims should not be made for this codec without
first matching or exceeding JPEG on standard benchmarks.

---

## Environment Caveats

* **Python 3.12**: The `.venv` used for development targets Python 3.12.
  CI runs on Python 3.11 (see `.github/workflows/ci.yml`).
  Packages may resolve differently across Python versions.

* **Optional ML dependencies**: `pip install -e '.[ai]'` pulls PyTorch,
  Hugging Face, and diffusers.  These are *not* required by the core library
  or test suite.
