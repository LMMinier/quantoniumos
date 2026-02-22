#!/usr/bin/env bash
# QuantoniumOS Repository Cleanup Script
# Run from the repo root: bash cleanup_repo.sh
#
# This script implements the full cleanup plan:
#   Phase 1  – Untrack gitignored files
#   Phase 2  – Delete docs/archive/
#   Phase 3  – Remove quantonium-mobile/ (extract to separate repo first!)
#   Phase 4  – Delete redundant requirements files
#   Phase 5  – Deduplicate license files
#   Phase 6  – Deduplicate hardware test vectors
#   Phase 7  – Consolidate scripts/ and tools/
#   Phase 8  – Consolidate docs/ subdirectories
#   Phase 9  – Merge demos/ and examples/
#   Phase 10 – Remove empty/placeholder directories
#   Phase 12 – Tidy loose ends
#
# NOTE: Phase 11 (fix broken references) was already applied via file edits
# NOTE: Phase 4 pyproject.toml updates were already applied via file edits

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== QuantoniumOS Repo Cleanup ==="
echo ""

# ─────────────────────────────────────────────────────────────
# Phase 1: Untrack files that are in .gitignore
# ─────────────────────────────────────────────────────────────
echo "Phase 1: Untracking gitignored files..."

# __pycache__ directories
git rm -r --cached --ignore-unmatch -- '*/__pycache__' '__pycache__' 2>/dev/null || true

# .egg-info
git rm -r --cached --ignore-unmatch -- 'quantoniumos.egg-info' 2>/dev/null || true

# sbom.json (auto-generated)
git rm --cached --ignore-unmatch -- sbom.json 2>/dev/null || true

# results/ (generated test outputs)
git rm -r --cached --ignore-unmatch -- results/ 2>/dev/null || true

# Hardware build artifacts (.vcd, compiled binaries, generated text)
git rm --cached --ignore-unmatch -- \
  hardware/full_coverage.vcd \
  hardware/quantoniumos_sim.vcd \
  hardware/fpga_sim \
  hardware/fpga_top_sim \
  hardware/full_coverage_sim \
  hardware/full_sim \
  hardware/fpga_sim_test \
  hardware/rftpu_sim \
  hardware/fpga_top_sim_output.txt \
  hardware/simulation_output.txt \
  hardware/synth_report.txt \
  2>/dev/null || true

# quantonium-mobile/node_modules (if tracked)
git rm -r --cached --ignore-unmatch -- 'quantonium-mobile/node_modules' 2>/dev/null || true

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 2: Delete docs/archive/ (contradicts current claims)
# ─────────────────────────────────────────────────────────────
echo "Phase 2: Removing docs/archive/..."
git rm -rf -- docs/archive/ 2>/dev/null || true
echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 3: Remove quantonium-mobile/
# ─────────────────────────────────────────────────────────────
echo "Phase 3: Removing quantonium-mobile/ (extract to a separate repo first!)..."
echo "  IMPORTANT: Before running this, push quantonium-mobile/ to its own repo."
echo "  Proceeding with removal from this repo..."
git rm -rf -- quantonium-mobile/ 2>/dev/null || true
echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 4: Delete redundant requirements files
# ─────────────────────────────────────────────────────────────
echo "Phase 4: Removing redundant requirements files (pyproject.toml is now single source)..."
git rm -f --ignore-unmatch -- \
  requirements.txt \
  requirements-core.txt \
  requirements-dev.txt \
  requirements-lock-core.txt \
  requirements-ml-extra.txt \
  requirements.in \
  experiments/competitors/requirements-bench.txt \
  2>/dev/null || true
echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 5: Deduplicate license files
# ─────────────────────────────────────────────────────────────
echo "Phase 5: Deduplicating license files..."

# Remove near-duplicate LICENSE.agpl
git rm -f --ignore-unmatch -- LICENSE.agpl 2>/dev/null || true

# Remove duplicate notice in docs/licensing/
git rm -f --ignore-unmatch -- docs/licensing/notice.md 2>/dev/null || true

# Merge LICENSE_SPLIT + LICENSING_OVERVIEW + LICENSING_DIAGNOSIS → LICENSING.md
# We keep LICENSING_OVERVIEW.md content as the base (most comprehensive)
if [ -f docs/licensing/LICENSING_OVERVIEW.md ]; then
  cp docs/licensing/LICENSING_OVERVIEW.md docs/licensing/LICENSING.md
  git add docs/licensing/LICENSING.md
fi
git rm -f --ignore-unmatch -- \
  docs/licensing/LICENSE_SPLIT.md \
  docs/licensing/LICENSING_OVERVIEW.md \
  docs/licensing/LICENSING_DIAGNOSIS.md \
  2>/dev/null || true

# Move DOCKER_PAPERS.md from licensing/ to guides/ (misplaced)
if [ -f docs/licensing/DOCKER_PAPERS.md ]; then
  git mv docs/licensing/DOCKER_PAPERS.md docs/guides/DOCKER_PAPERS.md 2>/dev/null || true
fi

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 6: Deduplicate hardware test vectors
# ─────────────────────────────────────────────────────────────
echo "Phase 6: Removing duplicate root hardware_test_vectors/..."
git rm -rf -- hardware_test_vectors/ 2>/dev/null || true
echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 7: Consolidate scripts/ and tools/
# ─────────────────────────────────────────────────────────────
echo "Phase 7: Consolidating scripts/ and tools/..."

# Create target subdirectories in scripts/
mkdir -p scripts/benchmarks scripts/figures scripts/paper scripts/validation \
         scripts/infra scripts/ai scripts/crypto scripts/compression scripts/optimization

# --- Move top-level scripts into categorized subdirectories ---

# Benchmarks
for f in scripts/benchmark_*.py scripts/test_real_data.py scripts/test_rft_quick.py scripts/test_desktop.py; do
  [ -f "$f" ] && git mv "$f" scripts/benchmarks/ 2>/dev/null || true
done

# Figures (merge figure_generation/ and figures/ into scripts/figures/)
for f in scripts/generate_all_theorem_figures.py scripts/generate_figures_only.sh \
         scripts/generate_medical_figures.py scripts/generate_paper_figures.sh \
         scripts/generate_pdf_figures_for_latex.py scripts/generate_rft_gifs.py \
         scripts/generate_rft_gifs_simple.py scripts/generate_rft_hardware_gifs.py \
         scripts/generate_zenodo_figures.py scripts/regenerate_ieee_figures.py \
         scripts/plot_psihf.py scripts/visualize_rft_analysis.py; do
  [ -f "$f" ] && git mv "$f" scripts/figures/ 2>/dev/null || true
done
# Merge scripts/figure_generation/* into scripts/figures/
if [ -d scripts/figure_generation ]; then
  for f in scripts/figure_generation/*; do
    [ -f "$f" ] && git mv "$f" scripts/figures/ 2>/dev/null || true
  done
  rmdir scripts/figure_generation 2>/dev/null || true
fi
# Merge old scripts/figures/ content (if separate files remain)

# Paper building
for f in scripts/build_ieee_paper.py scripts/build_ieee_paper_v2.py \
         scripts/compile_paper.sh scripts/populate_ieee_docx.py scripts/md_to_pdf.py; do
  [ -f "$f" ] && git mv "$f" scripts/paper/ 2>/dev/null || true
done

# Validation
for f in scripts/run_all_validations.py scripts/run_paper_validation_suite.py \
         scripts/run_proofs.py scripts/run_quick_paper_tests.py scripts/run_verify_now.py \
         scripts/validate_all.sh scripts/validate_paper_claims.py \
         scripts/verify_ascii_bottleneck.py scripts/verify_braided_comprehensive.py \
         scripts/verify_hw_sw_alignment.py scripts/verify_hybrid_mca_recovery.py \
         scripts/verify_paper_claims.py scripts/verify_performance_and_crypto.py \
         scripts/verify_rate_distortion.py scripts/verify_scaling_laws.py \
         scripts/verify_soft_vs_hard_braiding.py scripts/verify_variant_claims.py; do
  [ -f "$f" ] && git mv "$f" scripts/validation/ 2>/dev/null || true
done

# Infrastructure
for f in scripts/generate_sbom.py scripts/add_spdx_headers.py scripts/build.py \
         scripts/fast_start.sh scripts/quantonium_boot.py scripts/reproduce_warning.py; do
  [ -f "$f" ] && git mv "$f" scripts/infra/ 2>/dev/null || true
done

# AI/LLM
for f in scripts/quick_chatbox_with_ai.py scripts/load_ai_with_weights.py \
         scripts/run_local_chat_offline.sh; do
  [ -f "$f" ] && git mv "$f" scripts/ai/ 2>/dev/null || true
done

# Crypto
for f in scripts/estimate_sis_security.py scripts/archive_crypto_tests.py \
         scripts/run_nist_sts.py scripts/run_shannon_tests.py; do
  [ -f "$f" ] && git mv "$f" scripts/crypto/ 2>/dev/null || true
done

# Catch remaining uncategorized scripts
for f in scripts/archive_*.py scripts/analyze_*.py; do
  [ -f "$f" ] && git mv "$f" scripts/validation/ 2>/dev/null || true
done
for f in scripts/run_ascii_test.sh scripts/run_full_suite.sh scripts/run_gen.py scripts/run_research_suite.sh; do
  [ -f "$f" ] && git mv "$f" scripts/validation/ 2>/dev/null || true
done
[ -f scripts/irrevocable_truths.py ] && git mv scripts/irrevocable_truths.py scripts/validation/ 2>/dev/null || true
git rm -f --ignore-unmatch -- scripts/__init__.py 2>/dev/null || true

# Move scripts/launchers/ into scripts/infra/launchers/ (keep as subdir)
if [ -d scripts/launchers ]; then
  mkdir -p scripts/infra/launchers
  for f in scripts/launchers/*; do
    [ -f "$f" ] && git mv "$f" scripts/infra/launchers/ 2>/dev/null || true
  done
  rmdir scripts/launchers 2>/dev/null || true
fi

# --- Merge tools/ into scripts/ ---

# tools/benchmarking/ → scripts/benchmarks/
if [ -d tools/benchmarking ]; then
  for f in tools/benchmarking/*.py; do
    [ -f "$f" ] && git mv "$f" scripts/benchmarks/ 2>/dev/null || true
  done
fi

# tools/validation/ → scripts/validation/
if [ -d tools/validation ]; then
  for f in tools/validation/*.py; do
    [ -f "$f" ] && git mv "$f" scripts/validation/ 2>/dev/null || true
  done
fi

# tools/compression/ → scripts/compression/
if [ -d tools/compression ]; then
  for f in tools/compression/*.py; do
    [ -f "$f" ] && git mv "$f" scripts/compression/ 2>/dev/null || true
  done
fi

# tools/crypto/ + tools/sts/ + tools/leak_check/ → scripts/crypto/
for d in tools/crypto tools/sts tools/leak_check; do
  if [ -d "$d" ]; then
    for f in "$d"/*.py; do
      [ -f "$f" ] && git mv "$f" scripts/crypto/ 2>/dev/null || true
    done
  fi
done

# tools/optimization/ → scripts/optimization/
if [ -d tools/optimization ]; then
  for f in tools/optimization/*.py; do
    [ -f "$f" ] && git mv "$f" scripts/optimization/ 2>/dev/null || true
  done
fi

# tools/licenses/ + tools/spdx_inject.py → scripts/infra/
if [ -d tools/licenses ]; then
  for f in tools/licenses/*.py; do
    [ -f "$f" ] && git mv "$f" scripts/infra/ 2>/dev/null || true
  done
fi
[ -f tools/spdx_inject.py ] && git mv tools/spdx_inject.py scripts/infra/ 2>/dev/null || true

# tools/model_management/ → scripts/ai/
if [ -d tools/model_management ]; then
  for f in tools/model_management/*.py; do
    [ -f "$f" ] && git mv "$f" scripts/ai/ 2>/dev/null || true
  done
fi

# tools/rft_quick_reference.py → scripts/
[ -f tools/rft_quick_reference.py ] && git mv tools/rft_quick_reference.py scripts/ 2>/dev/null || true

# Remove the snippet file
git rm -f --ignore-unmatch -- tools/rft8x8_closed_form_kernel.tlv.snippet 2>/dev/null || true

# Remove tools/__init__.py and empty dirs
git rm -f --ignore-unmatch -- tools/__init__.py 2>/dev/null || true

# Clean up empty tools/ subdirectories
find tools/ -type d -empty -delete 2>/dev/null || true
rmdir tools 2>/dev/null || true

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 8: Consolidate docs/ subdirectories
# ─────────────────────────────────────────────────────────────
echo "Phase 8: Consolidating docs/..."

# Merge docs/manuals/ → docs/guides/
if [ -d docs/manuals ]; then
  for f in docs/manuals/*; do
    [ -f "$f" ] && git mv "$f" docs/guides/ 2>/dev/null || true
  done
  rmdir docs/manuals 2>/dev/null || true
fi

# Merge docs/theory/ → docs/proofs/
if [ -d docs/theory ]; then
  for f in docs/theory/*; do
    [ -f "$f" ] && git mv "$f" docs/proofs/ 2>/dev/null || true
  done
  rmdir docs/theory 2>/dev/null || true
fi

# Dissolve single-file directories
[ -f docs/api/README.md ] && git mv docs/api/README.md docs/reference/API.md 2>/dev/null || true
rmdir docs/api 2>/dev/null || true

[ -f docs/medical/README.md ] && git mv docs/medical/README.md docs/scientific_domains/MEDICAL.md 2>/dev/null || true
rmdir docs/medical 2>/dev/null || true

[ -f docs/safety/AI_SAFETY_CERTIFICATION.md ] && git mv docs/safety/AI_SAFETY_CERTIFICATION.md docs/reference/AI_SAFETY_CERTIFICATION.md 2>/dev/null || true
rmdir docs/safety 2>/dev/null || true

[ -f docs/user/README.md ] && git mv docs/user/README.md docs/guides/USER_GUIDE.md 2>/dev/null || true
rmdir docs/user 2>/dev/null || true

# Remove duplicate THEOREMS_RFT_IRONCLAD.md stub in proofs/
git rm -f --ignore-unmatch -- docs/proofs/THEOREMS_RFT_IRONCLAD.md 2>/dev/null || true

# Remove old-formula RFT_THEOREMS.md in validation/ (keep proofs/ version)
git rm -f --ignore-unmatch -- docs/validation/RFT_THEOREMS.md 2>/dev/null || true

# Remove empty docs/licensing/ if it's now empty
rmdir docs/licensing 2>/dev/null || true

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 9: Merge demos/ and examples/
# ─────────────────────────────────────────────────────────────
echo "Phase 9: Merging demos/ and examples/..."

# Move examples/ content into demos/
if [ -d examples ] && [ -f examples/routing_integration_demo.py ]; then
  git mv examples/routing_integration_demo.py demos/ 2>/dev/null || true
  rmdir examples 2>/dev/null || true
fi

# Rename demos/ → examples/ (more conventional)
if [ -d demos ]; then
  git mv demos examples 2>/dev/null || true
fi

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 10: Remove empty/placeholder directories
# ─────────────────────────────────────────────────────────────
echo "Phase 10: Removing empty/placeholder dirs..."

# experiments/crypto/ (empty)
rmdir experiments/crypto 2>/dev/null || true

# algorithms/rft/benchmarks/ (only __init__.py)
git rm -rf --ignore-unmatch -- algorithms/rft/benchmarks/ 2>/dev/null || true

# release/ (only a README referencing nonexistent files)
git rm -rf --ignore-unmatch -- release/ 2>/dev/null || true

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Phase 12: Tidy loose ends
# ─────────────────────────────────────────────────────────────
echo "Phase 12: Tidying loose ends..."

# Move AGENT.md into .github/
mkdir -p .github
git mv AGENT.md .github/AGENT.md 2>/dev/null || true

# Rename 'quantonium' script to 'quantonium-cli' to avoid name confusion
if [ -f quantonium ] && [ ! -d quantonium ]; then
  git mv quantonium quantonium-cli 2>/dev/null || true
fi

echo "  Done."

# ─────────────────────────────────────────────────────────────
# Update README.md table: remove mobile row, fix examples path
# ─────────────────────────────────────────────────────────────
echo "Phase 12b: Note — README.md mobile row and examples path need manual update."
echo "  (Already handled via file edits if applicable.)"

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Changes staged but NOT committed. Review with:"
echo "  git status"
echo "  git diff --cached --stat"
echo ""
echo "When satisfied, commit with:"
echo "  git commit -m 'chore: major repo cleanup — remove confusion, duplicates, and clutter"
echo ""
echo "    - Untrack gitignored files (__pycache__, .egg-info, sbom.json, results/, hw artifacts)"
echo "    - Delete docs/archive/ (contradicted current claims)"
echo "    - Remove quantonium-mobile/ (extract to separate repo)"
echo "    - Remove 6 redundant requirements files (pyproject.toml is single source)"
echo "    - Deduplicate license files (LICENSE.agpl, 3 licensing docs → 1)"
echo "    - Remove duplicate root hardware_test_vectors/"
echo "    - Consolidate scripts/ into categorized subdirs + merge tools/ into scripts/"
echo "    - Merge docs/manuals/ → guides/, docs/theory/ → proofs/, dissolve single-file dirs"
echo "    - Merge demos/ + examples/ → examples/"
echo "    - Remove empty/placeholder dirs (experiments/crypto, algorithms/rft/benchmarks, release)"
echo "    - Fix broken references (LICENSE.md → LICENSE, DOCS_INDEX → INDEX, phi_phase_fft_optimized)"
echo "    - Move AGENT.md → .github/, rename quantonium → quantonium-cli'"
echo ""
echo "Remaining manual steps:"
echo "  1. Push quantonium-mobile/ to its own repo before merging this"
echo "  2. Update any CI workflows that reference moved files"
echo "  3. Run: pytest tests/ -x -q   to verify nothing is broken"
echo "  4. Run: pip install -e '.[dev]' to verify pyproject.toml works"
