# AGENT.md — GitHub Actions Workflow Sprawl Control (QuantoniumOS)

## Mission
Reduce GitHub Actions run volume and self-checkout traffic WITHOUT breaking CI, security posture, or research validation.
No speculative edits. No "big bang" refactors. Legacy-safe only.

## Hard Rules (DO NOT VIOLATE)
1) **No deletions** of workflows on the first pass.
2) **No major dependency bumps** as part of workflow changes.
3) **No new network telemetry** or tracking.
4) Any change must include:
   - what workflow(s) changed
   - why
   - expected reduction in runs
   - proof: before/after run counts or at least a reproducible calculation

## Scope
Only touch:
- `.github/workflows/**.yml`
- `.github/workflows/**.yaml`
- optionally `.github/dependabot.yml` (only if asked)
- docs explaining CI (optional)
DO NOT touch application code unless needed to keep CI green.

---

## Part A — Measure: exact self-checkout ceiling (NO GUESSING)

### A1) Inventory checkout usage per workflow
For each file in `.github/workflows/`:
- Count how many jobs use `actions/checkout@v4` (or any checkout action).
- Output table:

| Workflow File | Workflow Name | Jobs | Jobs w/ checkout | Checkouts/Run | Triggers | Notes |
|---|---|---:|---:|---:|---|---|

**Definition:** `Checkouts/Run = count(jobs with checkout)`  
If a job checks out multiple times (rare), count each.

### A2) Combine with actual run counts
Use this exact run-count input (provided by user):
- CI: 52
- CLI Verification: 1
- Cross-Implementation Validation: 72
- Dependabot Updates: 6
- Green Wall CI Pipeline: 114
- Minimal Working Pipeline: 10
- Nightly Slow Validations: 24
- Package Build and Test: 53
- QuantoniumOS Encryption Tests: 21
- QuantoniumOS Incremental Pipeline: 10
- SPDX Header Rollout: 1
- Security & Supply Chain: 53
- Security Scan: 76
- Shannon Entropy Tests: 46

Compute:
`internal_checkouts = Σ(runs(workflow) × checkouts_per_run(workflow))`

Output table:

| Workflow Name | Runs | Checkouts/Run | Internal Checkouts |
|---|---:|---:|---:|

Also output:
- total runs
- total internal checkouts
- max internal checkouts (this total is the ceiling for "self-inflicted clones")

### A3) Interpretation
State clearly:
- "GitHub Actions can explain at most **X** clones/checkouts in this window."
- If GitHub Traffic clones >> X, conclude "external scanners/users dominate".

---

## Part B — Reduce Runs Safely (3 PRs max)

### Strategy Hierarchy (apply in this order)
1) **concurrency** (stop duplicates)
2) **path filters** (don't run when irrelevant files change)
3) **trigger tightening** (push→PR-only, schedule-only, workflow_dispatch)
4) **merge redundant workflows** (only after measurements confirm overlap)

### B1) Add concurrency everywhere (safe, minimal risk)
Add at workflow top-level:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

If a workflow must never cancel (rare), document why and skip.

### B2) Add strict path filters

For each workflow, define what it *actually* cares about.

Examples:

* Python tests:
  * `src/**`, `algorithms/**`, `tests/**`, `pyproject.toml`, `requirements*.txt`
* Mobile/node:
  * `quantonium-mobile/**`
* Hardware:
  * `hardware/**`, `rtl/**`, `*.sv`
* Docs-only changes should NOT trigger heavy CI.

Implement in triggers:

```yaml
on:
  push:
    paths:
      - "src/**"
      - "tests/**"
      - "algorithms/**"
      - "pyproject.toml"
      - "requirements*.txt"
  pull_request:
    paths:
      - "src/**"
      - "tests/**"
      - "algorithms/**"
      - "pyproject.toml"
      - "requirements*.txt"
```

### B3) Convert heavy pipelines to schedule-only

Candidates typically:

* "Nightly Slow Validations"
* "Cross-Implementation Validation"
* "Green Wall CI Pipeline" (if it duplicates CI)

Make heavy ones:

```yaml
on:
  schedule:
    - cron: "0 6 * * *"   # daily 6am UTC (or weekly)
  workflow_dispatch: {}
```

### B4) Remove redundancy by consolidation (only after B1–B3)

Likely redundancies:

* "Security Scan" vs "Security & Supply Chain"
* "CI" vs "Green Wall CI Pipeline" vs "Minimal Working Pipeline" vs "Incremental Pipeline"

Rule:

* Only ONE "fast CI" on push/PR.
* Only ONE "security scan" on PR + weekly cron.
* Everything else: scheduled or manual.

---

## Part C — Required Deliverables

### C1) Before/After Run Reduction Estimate

Provide:

* current expected runs per week
* proposed expected runs per week
* estimated reduction %

### C2) PR Plan (3 PRs)

* PR1: concurrency + path filters (no trigger deletions)
* PR2: move heavy workflows to schedule/manual
* PR3: merge duplicate security workflows (if needed)

### C3) Verification Checklist

For each PR:

* confirm YAML validates
* confirm at least one successful run of the main CI workflow
* confirm security scan still runs on PR and on schedule

---

## Forbidden Moves

* Do NOT add new workflows.
* Do NOT add "workflow_run" chains.
* Do NOT add any action that clones the repo more than once per job.
* Do NOT use `pull_request_target` unless explicitly required and reviewed.

---

## Definition of Done

1. Computed internal checkout ceiling with tables (Part A).
2. Reduced total workflow runs materially (target: >50% reduction) without breaking CI.
3. Repo still has:
   * one fast CI
   * one security pipeline
   * slow validations scheduled/manual
