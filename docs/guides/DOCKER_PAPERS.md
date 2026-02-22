# QuantoniumOS Paper Validation via Docker

Reproduce all claims from the QuantoniumOS research papers in an isolated environment.

## Quick Start

```bash
# Build the container
docker build -f Dockerfile.papers -t quantoniumos-papers .

# Run full validation suite
docker run --rm quantoniumos-papers
```

## Papers Validated

| Paper | DOI | Claims |
|-------|-----|--------|
| TechRxiv | `10.36227/techrxiv.175384307.75693850/v1` | Theorems 1-10 |
| Zenodo | `10.5281/zenodo.17712905` | All source code |
| USPTO | Application 19/169,399 | Patent claims 1-4 |

## Validation Tests

The container runs 5 validation suites:

1. **Irrevocable Truths** - All 7 Φ-RFT variants are unitary (error < 10⁻¹⁴)
2. **Scaling Laws** - Theorem 3: 61.8%+ sparsity for golden signals
3. **Variant Claims** - Theorems 4-7: Each variant solves its niche
4. **Unit Tests** - 24+ pytest tests for core algorithms
5. **ASCII Bottleneck** - Theorem 10: Hybrid DCT+RFT codec

## Run Specific Tests

```bash
# Just the unitarity proof
docker run --rm quantoniumos-papers python scripts/irrevocable_truths.py

# Just scaling laws
docker run --rm quantoniumos-papers python scripts/verify_scaling_laws.py

# Just unit tests
docker run --rm quantoniumos-papers pytest tests/rft/ -v

# Interactive shell
docker run -it --rm quantoniumos-papers /bin/bash
```

## Expected Output

```
========================================
QuantoniumOS Paper Validation Suite
========================================

[1/5] Irrevocable Truths (7 Variants)
========================================
Variant 'standard': unitary error = 2.22e-16 ✓
Variant 'logphi': unitary error = 3.33e-16 ✓
...
ALL 7 VARIANTS PROVEN UNITARY

[2/5] Scaling Laws (Theorem 3)
========================================
N=512: sparsity = 98.63% ✓ (threshold: 61.8%)

[3/5] Variant Claims (Theorems 4-7)
========================================
standard: baseline ✓
logphi: low-frequency emphasis ✓
...

[4/5] Unit Tests
========================================
tests/rft/test_closed_form.py::test_unitarity PASSED
tests/rft/test_variants.py::test_all_variants PASSED
...
24 passed, 1 failed

[5/5] ASCII Bottleneck (Theorem 10)
========================================
Hybrid codec: 37% improvement over single-basis ✓

========================================
Validation Complete
========================================
```

## Requirements

- Docker 20.10+
- ~500MB disk space
- ~2 minutes to build
- ~30 seconds to run validation

## Troubleshooting

### Build fails
```bash
# Clean build
docker build --no-cache -f Dockerfile.papers -t quantoniumos-papers .
```

### Permission denied
```bash
# Run as root (not recommended for production)
docker run --rm --user root quantoniumos-papers
```

### View logs
```bash
docker run --rm quantoniumos-papers 2>&1 | tee validation.log
```

## License

AGPL-3.0-or-later with patent provisions. See LICENSE.md for details.
