#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Claims Lint Test - Catches forbidden marketing language.
"""

import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    re.compile(r"\b1000\s*-?\s*qubit(s)?\b", re.IGNORECASE),
    re.compile(r"\b1000\s*\+\s*qubit(s)?\b", re.IGNORECASE),
    re.compile(r"\b10\s*million\s*qubit(s)?\b", re.IGNORECASE),
    re.compile(r"\b10M\s*qubit(s)?\b", re.IGNORECASE),
    re.compile(r"\bmillion\s*qubit(s)?\b", re.IGNORECASE),
]

INCLUDE_EXTENSIONS = {".md", ".tex", ".txt", ".py"}
EXCLUDE_DIRS = {".git", "node_modules", "dist", "build", "__pycache__"}


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                continue
        if path.is_file() and path.suffix in INCLUDE_EXTENSIONS:
            yield path


def test_no_overbroad_qubit_claims():
    root = Path(__file__).resolve().parents[2]
    violations = []

    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                violations.append(f"{path}:{line_no}: {match.group(0)}")

    assert not violations, "Overbroad qubit-simulation claims detected:\n" + "\n".join(violations)
