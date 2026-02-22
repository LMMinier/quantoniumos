# VERIFY.md — v2.0.1 Reproduction Checklist

Exact commands to verify the v2.0.1 release from a clean checkout.

## Prerequisites

```
Python >= 3.11
pip install -e .
```

## 1. Core test suite (should show 0 failures)

```bash
pytest tests/ -x -q --tb=short 2>&1 | tail -3
# Expected: 376 passed, 13 skipped
```

## 2. Validation subset (the 33 tests that cover fixes 1-9)

```bash
pytest tests/validation/ tests/proofs/test_rft_transform_theorems.py -v --tb=short
# Expected: all PASSED
```

## 3. Pre-existing test that was fixed (sine energy concentration)

```bash
pytest tests/validation/test_mixing_quality.py::TestSpectralFlatness::test_energy_spread_threshold -v
# Expected: 3 passed (impulse, sine, noise)
```

## 4. Canonical RFT import

```bash
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('OK')"
# Expected: OK
```

## 5. RFT unitarity (roundtrip error < 1e-14)

```bash
python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np
rft = CanonicalTrueRFT(256)
x = np.random.randn(256) + 1j*np.random.randn(256)
err = np.linalg.norm(x - rft.inverse_transform(rft.forward_transform(x))) / np.linalg.norm(x)
print(f'Unitarity error: {err:.2e}')
assert err < 1e-14
"
# Expected: Unitarity error: ~1e-16
```

## 6. RFT ≠ FFT (non-equivalence)

```bash
python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np
np.random.seed(42); N=256
x = np.random.randn(N)+1j*np.random.randn(N)
y_rft = CanonicalTrueRFT(N).forward_transform(x)
y_fft = np.fft.fft(x, norm='ortho')
corr = abs(np.vdot(y_rft,y_fft))/(np.linalg.norm(y_rft)*np.linalg.norm(y_fft))
print(f'RFT-FFT correlation: {corr:.4f}')
assert corr < 0.5
"
# Expected: correlation ~0.07
```

## Git state at release

| Item | Value |
|------|-------|
| Tag | `v2.0.1` |
| Commit 1 (audit fixes) | `7643cf5` |
| Commit 2 (deps + docs) | `46040ee` |
| Parent | `be71c47` |
| Tests | 376 passed, 0 failed, 13 skipped |
| Lock packages | 77 (from `pip freeze`, Python 3.12) |

## CI Notes

GitHub Actions CI (`.github/workflows/ci.yml`) runs on push to `main`/`develop`
when paths under `src/`, `algorithms/`, `tests/`, `requirements*.txt`, or
`pyproject.toml` change. CI installs a minimal set (numpy, scipy, sympy,
matplotlib, pytest, brotli, zstandard) — **not** from requirements files. It
runs:

- RFT unitarity + non-equivalence inline checks
- ANS codec roundtrip
- H3 hypothesis reproducibility
- Crypto avalanche
- `tests/transforms/test_rft_correctness.py` (43 tests)
- `tests/codec_tests/`
- `tests/validation/test_transform_absolute_novelty.py`

The fixed `test_energy_spread_threshold` is in `tests/validation/test_mixing_quality.py`,
which is **not** in the CI subset — so the pre-existing failure never broke CI.
The fix is still correct and validated locally.
