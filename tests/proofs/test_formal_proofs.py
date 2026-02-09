# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
test_formal_proofs.py — Machine-verified formal proof test suite
================================================================

Runs the FormalProofEngine, verifies every theorem's logical chain,
and checks that empirical certificates are within tolerance.

Each test maps to exactly one theorem in the proof registry.
Status semantics:
  - PASS   = proof chain valid AND all numerical witnesses within tolerance
  - FAIL   = either a logical step is unjustified or a witness violates bounds
  - XFAIL  = conjecture (expected to be empirical only)
"""

import json
import pytest
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from algorithms.rft.theory.proof_engine import (
    FormalProofEngine, ProofStatus, InferenceRule, TheoremRecord,
)


@pytest.fixture(scope="module")
def engine():
    """Run the full proof engine once for all tests."""
    eng = FormalProofEngine(sizes=[8, 16, 32, 64])
    eng.prove_all()
    return eng


# ────────────────────────────────────────────────────────────────────────────────
# Structural tests — verify the proof engine itself
# ────────────────────────────────────────────────────────────────────────────────

class TestProofEngineStructure:
    def test_axiom_count(self, engine):
        assert len(engine.axioms) >= 7, "Expected at least 7 axioms"

    def test_theorem_count(self, engine):
        assert len(engine.theorems) >= 7, "Expected at least 7 theorems"

    def test_all_theorems_have_proofs(self, engine):
        for tid, rec in engine.theorems.items():
            assert rec.proof is not None, f"{tid} has no proof"
            assert len(rec.proof.steps) > 0, f"{tid} has empty proof"

    def test_no_circular_dependencies(self, engine):
        """Verify DAG structure: no theorem depends on itself transitively."""
        visited = set()
        rec_stack = set()

        def dfs(tid):
            if tid in rec_stack:
                pytest.fail(f"Circular dependency detected involving {tid}")
            if tid in visited:
                return
            rec_stack.add(tid)
            if tid in engine.theorems:
                for dep in engine.theorems[tid].proof.dependencies:
                    dfs(dep)
            rec_stack.discard(tid)
            visited.add(tid)

        for tid in engine.theorems:
            dfs(tid)

    def test_dependencies_exist(self, engine):
        """Every dependency reference must point to an existing theorem or axiom."""
        valid_ids = set(engine.theorems.keys()) | set(engine.axioms.keys())
        for tid, rec in engine.theorems.items():
            for dep in rec.proof.dependencies:
                assert dep in valid_ids, (
                    f"{tid} depends on '{dep}' which doesn't exist"
                )

    def test_every_step_has_justification(self, engine):
        for tid, rec in engine.theorems.items():
            for step in rec.proof.steps:
                assert step.rule is not None, (
                    f"{tid} step {step.step_id} has no inference rule"
                )
                assert isinstance(step.rule, InferenceRule), (
                    f"{tid} step {step.step_id}: rule is not an InferenceRule"
                )


# ────────────────────────────────────────────────────────────────────────────────
# Individual theorem verification
# ────────────────────────────────────────────────────────────────────────────────

class TestTheorem1PhiFullRank:
    def test_verified(self, engine):
        rec = engine.theorems["THM1"]
        assert rec.proof.verified, "Theorem 1 proof not verified"

    def test_status_constructive(self, engine):
        assert engine.theorems["THM1"].proof.status == ProofStatus.CONSTRUCTIVE

    def test_sigma_min_positive(self, engine):
        rec = engine.theorems["THM1"]
        for key, val in rec.numerical_bounds.items():
            if "sigma_min" in key:
                assert val > 0, f"{key} = {val} is not positive"

    def test_frequency_gaps_positive(self, engine):
        rec = engine.theorems["THM1"]
        for key, val in rec.numerical_bounds.items():
            if "min_gap" in key:
                assert val > 0, f"{key} = {val}: frequencies not distinct"

    def test_depends_only_on_axioms(self, engine):
        assert engine.theorems["THM1"].proof.dependencies == []


class TestTheorem2CanonicalUnitary:
    def test_verified(self, engine):
        rec = engine.theorems["THM2"]
        assert rec.proof.verified, "Theorem 2 proof not verified"

    def test_status_constructive(self, engine):
        assert engine.theorems["THM2"].proof.status == ProofStatus.CONSTRUCTIVE

    def test_unitarity_error_below_tolerance(self, engine):
        rec = engine.theorems["THM2"]
        for key, val in rec.numerical_bounds.items():
            if "unitarity_err" in key:
                assert val < 1e-10, f"{key} = {val} exceeds tolerance"

    def test_depends_on_theorem_1(self, engine):
        assert "THM1" in engine.theorems["THM2"].proof.dependencies


class TestTheorem6PhiNeqDFT:
    def test_verified(self, engine):
        rec = engine.theorems["THM6"]
        assert rec.proof.verified, "Theorem 6 proof not verified"

    def test_magnitude_nonuniform(self, engine):
        rec = engine.theorems["THM6"]
        for key, val in rec.numerical_bounds.items():
            if "mag_std" in key:
                assert val > 1e-6, f"{key} too small — columns look DFT-like"

    def test_dft_distance_positive(self, engine):
        rec = engine.theorems["THM6"]
        for key, val in rec.numerical_bounds.items():
            if "dft_dist" in key:
                assert val > 1e-6, f"{key} too small — too close to DFT"


class TestTheorem9MaassenUffink:
    def test_verified(self, engine):
        rec = engine.theorems["THM9"]
        assert rec.proof.verified, "Theorem 9 proof not verified"

    def test_status_deductive(self, engine):
        assert engine.theorems["THM9"].proof.status == ProofStatus.DEDUCTIVE

    def test_all_entropy_sums_exceed_bound(self, engine):
        rec = engine.theorems["THM9"]
        for step in rec.proof.steps:
            if step.numerical_witness and "H_sum" in step.numerical_witness:
                w = step.numerical_witness
                assert w["H_sum"] >= w.get("bound", 0) - 1e-10, (
                    f"N={w.get('N')}, signal={w.get('signal')}: "
                    f"H_sum={w['H_sum']:.4f} < bound={w.get('bound', 0):.4f}"
                )

    def test_coherence_bounded(self, engine):
        rec = engine.theorems["THM9"]
        for key, val in rec.numerical_bounds.items():
            if key.startswith("mu_"):
                # mu should be between 1/sqrt(N) and 1
                assert 0 < val <= 1.0, f"{key} = {val} out of range"


class TestTheorem10PolarUniqueness:
    def test_verified(self, engine):
        rec = engine.theorems["THM10"]
        assert rec.proof.verified, "Theorem 10 proof not verified"

    def test_polar_match(self, engine):
        rec = engine.theorems["THM10"]
        for key, val in rec.numerical_bounds.items():
            if "polar_err" in key:
                assert val < 1e-10, f"{key} = {val}: doesn't match scipy polar"

    def test_positive_definite_factor(self, engine):
        rec = engine.theorems["THM10"]
        for key, val in rec.numerical_bounds.items():
            if "min_eig" in key:
                assert val > 0, f"{key} = {val}: U†Φ not positive definite"

    def test_no_random_beats_polar(self, engine):
        rec = engine.theorems["THM10"]
        for key, val in rec.numerical_bounds.items():
            if "beat_count" in key:
                assert val == 0, f"{key} = {val}: some random unitary was closer"


class TestTheorem11NoExactDiag:
    def test_verified(self, engine):
        rec = engine.theorems["THM11"]
        assert rec.proof.verified, "Theorem 11 proof not verified"

    def test_companion_non_normal(self, engine):
        rec = engine.theorems["THM11"]
        for key, val in rec.numerical_bounds.items():
            if "normality_err" in key:
                assert val > 1e-6, f"{key} = {val}: C_φ appears normal"

    def test_offdiag_nonzero(self, engine):
        rec = engine.theorems["THM11"]
        for key, val in rec.numerical_bounds.items():
            if "offdiag_ratio" in key:
                assert val > 0, f"{key} = {val}: exact diagonalization achieved"


class TestTheoremANearestUnitary:
    def test_verified(self, engine):
        rec = engine.theorems["THMA"]
        assert rec.proof.verified, "Theorem A proof not verified"

    def test_positive_margin(self, engine):
        rec = engine.theorems["THMA"]
        for key, val in rec.numerical_bounds.items():
            assert val > 0, f"{key} = {val}: random unitary was closer than U_φ"


class TestConjecture12Variational:
    """Conjecture — expected to be empirical only, not formally proven."""

    def test_status_is_empirical(self, engine):
        rec = engine.theorems["CONJ12"]
        assert rec.proof.status == ProofStatus.EMPIRICAL

    def test_empirical_support(self, engine):
        rec = engine.theorems["CONJ12"]
        # Should be supported even if not formally proven
        for key, val in rec.numerical_bounds.items():
            if "beat_count" in key:
                assert val == 0, (
                    f"{key} = {val}: conjecture refuted at some size"
                )


# ────────────────────────────────────────────────────────────────────────────────
# Cross-cutting verification
# ────────────────────────────────────────────────────────────────────────────────

class TestCrossCuttingVerification:
    def test_all_constructive_proofs_verified(self, engine):
        for tid, rec in engine.theorems.items():
            if rec.proof.status == ProofStatus.CONSTRUCTIVE:
                assert rec.proof.verified, f"{tid} is constructive but not verified"

    def test_all_deductive_proofs_verified(self, engine):
        for tid, rec in engine.theorems.items():
            if rec.proof.status == ProofStatus.DEDUCTIVE:
                assert rec.proof.verified, f"{tid} is deductive but not verified"

    def test_report_generation(self, engine):
        report = engine.generate_report()
        assert "VERIFIED" in report
        assert "AXIOM SYSTEM" in report
        assert len(report) > 500

    def test_json_export(self, engine):
        """Verify JSON export is valid and round-trips."""
        j = engine.export_json()
        data = json.loads(j)
        assert len(data) == len(engine.theorems)
        for tid in engine.theorems:
            assert tid in data
            assert "verified" in data[tid]
            assert "steps" in data[tid]

    def test_proof_chain_integrity(self, engine):
        """Every non-axiom step must reference at least one prior item."""
        for tid, rec in engine.theorems.items():
            for step in rec.proof.steps:
                if step.rule != InferenceRule.AXIOM:
                    assert len(step.references) > 0 or step.rule == InferenceRule.NUMERICAL_CERTIFICATE, (
                        f"{tid} step {step.step_id}: non-axiom step with no references"
                    )
