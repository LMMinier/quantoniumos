# Repo Strip Plan + Vision Check (QuantoniumOS)

**Date:** 2025-12-15  
**Purpose:** Give a practical, repo-native answer to:
1) What is actually *core* vs *optional* vs *generated*,
2) What you can strip without losing the scientific/core contribution,
3) What “the vision” is (in a defensible way),
4) How close the repo is to each possible “finish line”.

This is written to support *decisive pruning* (smaller, clearer, more publishable), not to preserve everything.

---

## 0) Ground truth: what’s already in your repo

Your repo already contains a strong self-audit boundary:
- [CANONICAL.md](../../CANONICAL.md) explicitly marks what is claim-bearing/canonical vs exploratory.
- [NOVELTY_AUDIT_REPORT.md](../../NOVELTY_AUDIT_REPORT.md) is unusually honest about what is and isn’t novel.
- [docs/NON_CLAIMS.md](../NON_CLAIMS.md) + [docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md](../LIMITATIONS_AND_REVIEWER_CONCERNS.md) correctly reduce overclaim risk.

This means you *can* ship something scientifically credible—if you strip aggressively and keep the story consistent.

---

## 1) The repo is currently several repos glued together

A realistic “vision” requires choosing one primary deliverable. Right now you have at least three:

1) **Core science / transform framework** (math + code + tests + benchmark harness)
2) **System demos** (desktop UI, mobile, UI assets)
3) **Hardware feasibility** (RTL / testbenches / 3D viewer)

All three can coexist, but if you want outside readers/reviewers to understand it, the repo needs a default center.

---

## 2) Inventory by size (what’s actually consuming space)

From `du` (dev container):

- `.venv/` ~ **8.9G** (generated local environment)
- `quantonium-mobile/` ~ **392M** (mostly `node_modules/`)
- `hardware/` ~ **252M** (TBs + viewer)
- `data/` ~ **123M** (physionet/mitbih downloads etc.)
- `figures/` ~ **32M**
- `papers/` ~ **15M**

Core code is small by comparison:
- `algorithms/rft/` ~ **2.4M**
- `tools/` ~ **1.2M**
- `src/` ~ **1.1M**
- `tests/` ~ **3.2M**
- `docs/` ~ **1.9M**

**Implication:** You can strip 90%+ of repo size without touching core science.

---

## 3) Always-strip (generated / cache / local-only)

These should not be committed and should not be part of any “release artifact”:

- `.venv/`
- `.pytest_cache/`
- `.hypothesis/` (Hypothesis test artifacts)
- `**/__pycache__/`
- `quantonium-mobile/node_modules/`
- any build artifacts under `hardware/**/build` or `src/**/build` if present

**Status check:** `.hypothesis/constants/*` currently shows up as untracked files in git status. That is a pure artifact; it should be ignored.

---

## 4) Keep vs Optional vs Strip (directory-level)

This is the practical pruning map.

### Keep (core science deliverable)

These are the minimum to preserve a scientifically defensible “transform/framework” artifact:

- `algorithms/`
  - `algorithms/rft/core/` (canonical kernels + the finite-N correction utilities)
  - `algorithms/rft/theory/` + `algorithms/rft/theorems/` (formal math support)
  - `algorithms/rft/compression/` (only if compression claims are central)
  - `algorithms/rft/kernels/` (only if native/variant kernels are central)
- `tests/` (keeps the repo honest)
- `benchmarks/` (canonical evaluation entrypoints)
- `tools/benchmarking/` (reproducible harness + CSV outputs)
- `docs/` (but prune to the story you’re actually telling)
- Root governance docs: `BENCHMARK_PROTOCOL.md`, `CANONICAL.md`, `SECURITY.md`, `LICENSE*`, `PATENT_NOTICE.md`, `CITATION.cff`

### Optional (keep only if you *actively* ship them)

- `src/` (native acceleration) — keep if you want speed story or native engine credibility
- `papers/` — keep if you actively compile/publish from this repo
- `figures/` — keep if papers/docs refer to them directly
- `data/` — keep *scripts/configs*, but large downloaded datasets should be fetched, not stored
- `experiments/` — keep if you want reproducibility of exploratory claims; otherwise archive
- `results/` — keep only if the repo is meant to include “frozen” benchmark artifacts

### Strip / archive out of the main repo (unless they are the product)

- `quantonium-mobile/` (huge; largely demo/UI; belongs in a separate repo)
- `quantonium_os_src/` (desktop demo; separate repo if it’s not the main research artifact)
- `ui/` (shared assets; can move with UI repos)
- `hardware/` (hardware feasibility is real work, but it’s a different audience and lifecycle)

---

## 5) File-level keep list (the ones that define the scientific core)

If you want the smallest possible “science-core” release, the following files/areas are the critical spine:

- Canonical definition + API surface:
  - `algorithms/rft/README_RFT.md`
  - `algorithms/rft/core/resonant_fourier_transform.py`
  - `algorithms/rft/core/canonical_true_rft.py` (if you keep the operator/eigenbasis story)
  - `algorithms/rft/core/phi_phase_fft.py` (only as deprecated baseline / negative result)
  - `algorithms/rft/core/gram_utils.py` (finite-N correction utilities)
  - `algorithms/rft/core/rft_phi_legacy.py` (only if legacy behavior matters)

- Validation + correctness:
  - `tests/rft/`
  - `tests/validation/`
  - `pytest.ini`, `pyproject.toml`, `requirements*.txt`

- Reproducible benchmarks:
  - `benchmarks/` (the canonical “Classes” harness)
  - `tools/benchmarking/rft_vs_fft_benchmark.py` (explicitly labels `rft_impl`)
  - `docs/research/benchmarks/VERIFIED_BENCHMARKS.md`

---

## 6) CI/workflows: what matters

- `.github/workflows/shannon_tests.yml`
  - **Useful if** entropy/coherence/codec tests are part of your scientific promise.
  - **Optional if** you are narrowing to just “transform correctness + minimal benchmarks”.

- `.github/workflows/spdx-inject.yml`
  - This is a manual “rollout” workflow. **Keep if** you intend to maintain SPDX hygiene across many files.
  - Otherwise it can live in `tools/` only; the workflow itself is optional.

---

## 7) Vision (what the repo should be) + how close you are

### Vision A: “Science-core transform framework” (recommended)

**Definition:** a reproducible research artifact: canonical transform(s), correct math, tests, and honest benchmarks.

**You’re close because:**
- Tests are green (full pytest run passes in this workspace).
- Benchmarks produce CSV artifacts under `results/patent_benchmarks/`.
- You already wrote the non-claims/limitations and a novelty audit.

**Main gap:** narrative consistency. You need to choose *one* canonical story for “RFT” and treat everything else as baseline/deprecated/experimental.

**Closeness:** ~80% (engineering-wise). The remaining ~20% is mostly editorial + pruning + ablation design.

### Vision B: “End-to-end product demo (desktop/mobile)”

**Definition:** a platform repo that happens to include a research transform.

**You’re close because:** the demos exist.  
**Main gap:** productization and scope explosion (this becomes a software product, not a paper).

**Closeness:** ~40% (because the bar is much higher than tests passing).

### Vision C: “Hardware feasibility as primary deliverable”

**Definition:** RFTPU/accelerator is the product; software exists to validate hardware math.

**You’re close because:** hardware tree is large and nontrivial.

**Main gap:** external reproducibility and audience fit (hardware needs its own docs/tests, and reviewers differ).

**Closeness:** ~50–60% (depending on whether you target “design study” vs “tapeout-ready”).

---

## 8) Concrete strip plan (actionable)

If the goal is Vision A (science-core), do this:

1) **Hard separate “science” vs “demo/hardware”**
   - Keep science in this repo.
   - Move `quantonium-mobile/`, `quantonium_os_src/`, `ui/`, `hardware/` to separate repos (or a separate branch/tag).

2) **Keep only one canonical transform definition**
   - Pick the canonical definition you’re willing to defend.
   - Mark everything else as baseline/legacy/deprecated.

3) **Make results reproducible, not stored**
   - Either keep `results/patent_benchmarks/*.csv` as a frozen appendix *or*
   - Treat results as generated artifacts and store them as CI artifacts/releases.

4) **Add an ablation benchmark suite (the fastest way to find “real wins”)**
   - Compare φ-grid vs random nonuniform vs jittered uniform vs FFT on identical metrics and datasets.

---

## 9) What to do next (if you want a “clean publishable repo”)

- Create a “science-core” README section that points to:
  - canonical transform definition
  - tests to reproduce correctness
  - benchmark commands that generate the CSVs
  - an explicit “what wins / what doesn’t” table

- Then prune until the repo’s default story matches that README.

---

## 10) Your provided path list: recommended disposition

This appendix maps the specific paths you listed into **Keep / Optional / Strip**.

### Keep

- `.github/workflows/` (keep at least minimal CI)
  - `.github/workflows/shannon_tests.yml` (keep if entropy/codec is part of your promise)
  - `.github/workflows/spdx-inject.yml` (keep if SPDX hygiene is ongoing)
- `algorithms/` (core science)
  - `algorithms/rft/core/` (canonical kernel(s) + correction utilities)
  - `algorithms/rft/theory/`, `algorithms/rft/theorems/`
  - `algorithms/rft/README_RFT.md`, `algorithms/rft/RFT_RESEARCH_SUMMARY.md`
- `benchmarks/` (canonical evaluation entrypoints)
- `tests/` (unit + validation suites)
- `tools/` (benchmark harnesses, validation tooling)
- Root governance + licensing docs (e.g. `CANONICAL.md`, `BENCHMARK_PROTOCOL.md`, `LICENSE*`, `PATENT_NOTICE.md`, `SECURITY.md`, `CITATION.cff`)
- `docs/` (but prune to the single story you intend to defend)

### Optional

- `src/` (native acceleration)
- `experiments/` (only if you ship exploratory runs as part of the story)
- `papers/` and `docs/research/THE_PHI_RFT_FRAMEWORK_PAPER.tex` (only if actively maintained/compiled)
- `figures/` (only if required by papers/docs)
- `data/` (keep fetch scripts/config; avoid bundling large downloaded datasets)
- `results/patent_benchmarks/` (either treat as frozen appendix *or* keep as generated-only)
- `demos/`, `examples/` (nice for onboarding; not required for science-core)

### Strip (or move to separate repos)

- `quantonium-mobile/` (mobile app + `node_modules` lifecycle; separate repo)
- `quantonium_os_src/` (desktop demo; separate repo)
- `ui/` (UI assets; move with UI repos)
- `hardware/` (hardware feasibility is a different audience/release cadence)

### Always-strip (generated)

- `.hypothesis/`
- `.pytest_cache/`
- `.venv/`
- `**/__pycache__/`
