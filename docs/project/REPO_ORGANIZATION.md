# QuantoniumOS – Repository Organization

Purpose: define a single, stable map for the repo without moving directories.  
This file explains what each top-level component is for, and where to go next.

---

## 1. Top-Level Layout (Conceptual Map)

| Category | Paths | What Lives Here |
| --- | --- | --- |
| Core compute (RFT + crypto + native) | `algorithms/`, `quantoniumos/`, `src/` | Canonical RFT kernels, variants, crypto, native engines, package shim |
| Applications (OS, DAW, mobile, UI) | `quantonium_os_src/`, `quantonium-mobile/`, `ui/` | Desktop "QuantoniumOS" apps, mobile app, shared UI assets |
| Research & benchmarks | `benchmarks/`, `experiments/`, `examples/`, `data/`, `figures/` | Formal benchmark suite, experiments, demos, result JSONs, plots |
| Hardware | `hardware/` | RTL/TL-Verilog/FPGA, testbenches, waveforms, synthesis artifacts |
| Docs & papers | `docs/`, `papers/`, root `*.md`, `CITATION.cff` | Architecture docs, validation reports, LaTeX, paper assets, citation info |
| Tooling & infra | `tools/`, `scripts/`, `tests/`, `.github/`, `release/` | CLI tools, automation scripts, pytest suites, CI workflows, release artifacts |
| Licensing & IP | `LICENSE.md`, `LICENSE-CLAIMS-NC.md`, `PATENT_*.md`, `CLAIMS_PRACTICING_FILES.txt`, `SECURITY.md` | Legal split, patent notice, claims mapping, security policy |

---

## 1.1 Scientific Partition Overlay (Journal/Patent-Friendly)

Purpose: provide an explicit scientific/evaluation/demo split **without moving directories**.

These top-level folders are *indexes + policy boundaries* that point to the authoritative code/docs already living under `algorithms/`, `benchmarks/`, `tests/`, `hardware/`, etc.

| Section | Path | Marking | Canonical sources (already in repo) |
| --- | --- | --- | --- |
| Core Science (PRIMARY ASSET) | `rft-core/` | CANONICAL SCIENTIFIC REFERENCE IMPLEMENTATION | `algorithms/rft/core/`, `algorithms/rft/kernels/`, `algorithms/rft/theory/`, `tests/rft/`, `tests/validation/` |
| Validation & Benchmarks (SUPPORT) | `rft-validation/` | REPRODUCIBLE EMPIRICAL EVALUATION | `benchmarks/`, `algorithms/rft/benchmarks/`, `tests/medical/`, `tests/validation/`, `data/`, `figures/` |
| Experimental Variants (NON-CANONICAL) | `rft-experiments/` | EXPLORATORY / NON-CANONICAL RESEARCH | `algorithms/rft/experiments/`, `experiments/` |
| Demonstrations (UI/OS/Mobile) | `demos/` | TECHNOLOGY DEMONSTRATIONS — NOT SCIENTIFIC CLAIMS | `quantonium_os_src/`, `quantonium-mobile/`, `ui/`, `examples/`, `hardware/rftpu-3d-viewer/` |

Hardware has its own feasibility label; see `hardware/README.md`.

**Onboarding path:**

1. `README.md` → project summary, RFT update, quick start.
2. `GETTING_STARTED.md` → first run + examples.
3. `SETUP_GUIDE.md` → installation, native builds, troubleshooting.
4. `DOCUMENTATION_INDEX.md` → doc tree + task-based navigation.

---

## 2. Core Compute (RFT, Crypto, Native Engines)

### 2.1 Algorithms

Root: `algorithms/`

Focus: all math and algorithmic logic for RFT and related systems.

Key subtrees:

| Path | Purpose | Typical Entry |
| --- | --- | --- |
| `algorithms/rft/` | Canonical RFT definition, theory, variants, hybrids, applications | Indexed by `algorithms/rft/README_RFT.md` and `algorithms/rft/RFT_RESEARCH_SUMMARY.md` |
| `algorithms/rft/kernels/` | Canonical RFT kernels (e.g. `resonant_fourier_transform.py`), ARFT kernel, φ-phase legacy kernels | Import from algorithms, or via `quantoniumos` package re-exports |
| `algorithms/rft/variants/` | Operator variants + hybrid manifests | Routed through variant manifest into benchmarks and tests |
| `algorithms/rft/fast/` | Fast/approximate RFT explorations (low-rank, structured eigensolvers) | Used from benchmarks/experiments; not the default API |
| `algorithms/rft/theory/` | Formal math proofs + theorem analysis | Referenced by RFT research docs |
| `algorithms/rft/applications/` | Focused domain benchmarks (biomedical, compression, etc.) | Called by `benchmarks/` and `tests/` suites |

See `algorithms/rft/README_RFT.md` and `algorithms/rft/RFT_RESEARCH_SUMMARY.md` for the authoritative definition and current status of RFT.

### 2.2 Python Package Shim

Root: `quantoniumos/`

Purpose: stable, user-facing Python imports.

- Re-exports canonical pieces like `CanonicalTrueRFT` and `EnhancedRFTCryptoV2`.
- Keeps upstream code layout flexible while preserving `from quantoniumos import ...` as the public API.

Use this for anything "public": treat the raw `algorithms.*` imports as internal.

### 2.3 Native Engines

Root: `src/` (includes `rftmw_native` and other native code).

Role:

- C/ASM kernels and C++ AVX2/AVX-512 engine (`rftmw_native`) used for 3–10× speedups.
- Built via the commands in `README.md` and `SETUP_GUIDE.md` (cmake + make, then shared library copy).

---

## 3. Applications (Desktop OS, Mobile, UI)

### 3.1 Desktop "QuantoniumOS" Environment

Root: `quantonium_os_src/`

Structure:

- `frontend/quantonium_desktop.py`: main desktop UI (Q logo launcher).
- `apps/`: individual PyQt applications:
  - `symbolic_simulator`, `quantum_inspired_crypto`, `q_notes`, `q_vault`, `rft_validator`, `rft_visualizer`, `system_monitor`.
- `engine/`: reserved for future engine integration.
- `resources/icons/`: UI icons.

Use `scripts/quantonium_boot.py` to launch the full desktop; or run `quantonium_os_src/frontend/quantonium_desktop.py` directly.

### 3.2 Mobile App

Root: `quantonium-mobile/`

Role:

- React Native / Expo style mobile client for QuantoniumOS concepts.
- Keep all mobile-only logic isolated here to avoid polluting core compute or desktop OS trees.

### 3.3 Shared UI Assets

Root: `ui/`

Role:

- Central bucket for icons, styles, and UI components that aren't tied to a specific app.
- Use this as the shared visual layer between `quantonium_os_src/` and `quantonium-mobile/` where possible.

---

## 4. Research, Experiments, Benchmarks

### 4.1 Benchmarks

Root: `benchmarks/`

Purpose: formal, multi-class benchmark suite.

- `run_all_benchmarks.py`: master entrypoint for Classes A–E (quantum simulation, transform/DSP, compression, crypto, audio/DAW).
- Class-specific modules (e.g., `class_b_transform_dsp.py`) call into `algorithms/rft/*` and friends.

Use this folder for any "honest comparison" vs baseline algorithms or libraries.

### 4.2 Experiments

Root: `experiments/`

Role:

- Free-form research runs: entropy studies, scaling laws, SOTA comparison scripts, exploratory notebooks.
- Outputs should land under `data/` and `figures/`, not in `algorithms/`.

### 4.3 Examples

Root: `examples/`

Role:

- Minimal "how to use this" scripts.
- Demo integration of core libraries (`algorithms/`, `quantoniumos` package) into simple pipelines.
- Should stay small and tutorial-style.

### 4.4 Data and Figures

Roots: `data/`, `figures/`

- `data/`: JSON + configs + result dumps (e.g., `quantum_compression_results.json`, codec outputs, scaling configs).
- `figures/`: PNG/SVG/GIF, LaTeX plot data, diagrams for papers and docs.

Rule: no executable code lives here; only artifacts.

---

## 5. Hardware

Root: `hardware/`

Role:

- RTL, TL-Verilog blueprints, FPGA tops, testbenches, synthesis reports, and visualization scripts for the RFTPU/RPU.
- Waveform dumps, OpenLane/OpenROAD scripts, WebFPGA artifacts.

Usage:

- Treat this tree as the canonical hardware definition that corresponds to the software RFT variants referenced in `README_RFT.md` and your chip paper.
- Any new tile layouts, NoC changes, or macro placements belong here, not in `algorithms/`.

---

## 6. Documentation, Papers, and Validation Reports

### 6.1 Docs Tree

Root: `docs/`

Contains:

- Architecture docs: `docs/ARCHITECTURE*.md` (end-to-end stack from ASM → C → C++ → Python).
- Validation reports: `docs/FINAL_VALIDATION_REPORT.md` and friends (Phase 4 summary).
- Manuals and guides (engineer/dev manuals, system maps).

Index: `DOCUMENTATION_INDEX.md` – always start here for anything prose-heavy.

### 6.2 Papers

Root: `papers/`

Role:

- LaTeX sources, .bib, and figure references for the Zenodo/TechRxiv papers.
- Only paper-specific glue should live here; core equations and implementations stay in `algorithms/` and `docs/theory`.

### 6.3 Root-Level Reports

Important root docs:

- `RFT_VALIDATION_SNAPSHOT.md` – latest RFT validation snapshot (unitarity, robustness, honest claims).
- `CLAIMS_AUDIT_REPORT.md` – maps what is proven, overstated, or experimental.
- `PATENT_COMPLIANCE_REPORT.md` – maps patent claims to code paths and tests.
- `SYSTEM_STATUS_SUMMARY.md` – high-level health and cleanup opportunities.
- `REPRODUCING_RESULTS.md` – how to reproduce the benchmarks and validation results.
- `CITATION.cff` – citation metadata (Zenodo DOIs).

---

## 7. Tooling, Scripts, Tests, CI, Releases

### 7.1 Tools

Root: `tools/`

Role:

- CLI utilities for benchmarking, compression, crypto, model management, dev helpers.
- Anything that "wraps" the core libs but isn't itself a library, app, or test.

### 7.2 Scripts

Root: `scripts/`

Organized subdirectories:

- `scripts/launchers/` – App launchers (desktop OS entry points)
- `scripts/benchmarks/` – Benchmark runner scripts
- `scripts/figures/` – Figure/visualization generators

Key scripts:

- `run_full_suite.sh` (or equivalent) – full validation pipeline.
- Figure generators and paper helpers.
- Launchers (e.g., `quantonium_boot.py` for the desktop OS).

### 7.3 Tests

Root: `tests/`

Role:

- All pytest suites (unit, integration, slow).
- Test wiring is validated by `verify_test_wiring.py` and `test_imports.py`.
- Markers: `integration`, `slow` as configured in `pytest.ini`.

### 7.4 CI / Packaging / Release

- `.github/workflows/` – CI pipelines (lint, tests, possibly docs/build).
- `pyproject.toml` – build metadata, dependencies, package includes/excludes.
- `Dockerfile`, `Dockerfile.papers` – containerized validation for core stack and paper results.
- `quantoniumos-bootstrap.sh`, `verify_setup.sh`, `validate_system.py`, `verify_medical_tests.py`, `run_demo.sh` – bootstrap/health check entrypoints.
- `release/` – curated release artifacts and release notes (benchmarks zip, release README).

---

## 8. Licensing and Patent Compliance

**Licensing split:**

- `LICENSE.md`: default AGPL-3.0-or-later for all files not listed as patent-practicing.
- `LICENSE-CLAIMS-NC.md`: non-commercial research license for patent-practicing files.
- `CLAIMS_PRACTICING_FILES.txt`: authoritative list of files covered by the patent license.
- `PATENT_NOTICE.md`: high-level patent notice and how to contact for commercial licensing.
- `PATENT_COMPLIANCE_REPORT.md`, `CLAIMS_AUDIT_REPORT.md`: link claims ↔ implementations ↔ tests.
- `SECURITY.md`: security policy and cryptography disclaimers.

**Rule of thumb:**

- If you touch RFT core, kernels, crypto, hybrids, or hardware, always check `CLAIMS_PRACTICING_FILES.txt` and the compliance report first.
- New files implementing claim-critical logic should either be added to that list or explicitly noted as outside patent claims.

---

## 9. Navigation Cheatsheet

Common tasks → where to go:

| Task | Destination |
| --- | --- |
| Understand what RFT actually is | `algorithms/rft/README_RFT.md`, `algorithms/rft/RFT_RESEARCH_SUMMARY.md`, `docs/ARCHITECTURE.md` |
| Use the transform in my own code | Import from `quantoniumos` package; see `README.md` quick start |
| Run all benchmarks | `benchmarks/run_all_benchmarks.py`, `REPRODUCING_RESULTS.md` |
| Inspect hardware design | `hardware/` plus chip paper in `papers/`, figures in `figures/` |
| Desktop OS or mobile app | `quantonium_os_src/` (desktop), `quantonium-mobile/` (mobile), `ui/` (shared assets) |
| Legal or security review | `PATENT_NOTICE.md`, `PATENT_COMPLIANCE_REPORT.md`, `CLAIMS_AUDIT_REPORT.md`, `CLAIMS_PRACTICING_FILES.txt`, `SECURITY.md` |

---

**This file is the source of truth for repo structure.** If you add major new components, extend this map instead of inventing new ad-hoc categories.
