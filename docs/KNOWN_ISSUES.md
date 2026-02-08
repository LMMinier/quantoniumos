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

_None at present._

---

## Environment Caveats

* **Python 3.12**: The `.venv` used for development and lock-file generation
  targets Python 3.12.  CI runs on Python 3.11 (see `.github/workflows/ci.yml`).
  The lock file (`requirements-lock-core.txt`) is informational for 3.12;
  packages may resolve differently on 3.11.

* **Optional ML dependencies**: `requirements-ml-extra.txt` pulls PyTorch,
  Hugging Face, and diffusers.  These are *not* required by the core library
  or test suite.
