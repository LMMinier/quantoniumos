#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Scan repository for SPDX license identifiers in code files and flag incompatible ones.

We only scan code-like extensions to avoid docs/data noise. If no SPDX tags are found,
we do not fail the build (informational). If any incompatible license tags are detected,
exit non-zero.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

CODE_EXTS = {".py", ".c", ".h", ".cpp", ".hpp", ".cc", ".js", ".ts", ".sh", ".bat"}
ALLOW = {
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "MPL-2.0",
    "ISC",
    "PSF-2.0",
    "LGPL-3.0-or-later",
    "LGPL-2.1-or-later",
    "AGPL-3.0-or-later",
    "GPL-3.0-or-later",  # compatible with AGPL-3.0
}
DENY = {
    "GPL-2.0-only",
    "LGPL-2.1-only",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
}
SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9\.-]+)")


def scan_repo(root: Path) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    found_allow: list[tuple[Path, str]] = []
    found_deny: list[tuple[Path, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in CODE_EXTS:
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        m = SPDX_RE.search(text)
        if m:
            lic = m.group(1)
            if lic in DENY:
                found_deny.append((p, lic))
            elif lic in ALLOW:
                found_allow.append((p, lic))
    return found_allow, found_deny


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    allow, deny = scan_repo(root)
    print(f"SPDX-allowed tags found: {len(allow)}")
    for p, lic in allow[:10]:
        print(f"  {lic} {p}")
    if deny:
        print("Incompatible SPDX tags detected:")
        for p, lic in deny:
            print(f"  {lic} {p}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
