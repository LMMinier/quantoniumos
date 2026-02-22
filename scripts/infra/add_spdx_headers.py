#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Add SPDX license headers to all Python source files.

This script scans the repository and adds AGPL-3.0-only SPDX headers
to files that don't already have them, respecting existing headers.

Usage:
    python scripts/add_spdx_headers.py [--dry-run]
"""

import argparse
import os
from pathlib import Path
from typing import List, Set

# SPDX header templates
SPDX_PYTHON_HEADER = """# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""

SPDX_CLAIMS_HEADER = """# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""

# Files/patterns to skip
SKIP_PATTERNS = {
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'node_modules',
    '.pytest_cache',
    '.mypy_cache',
    'build',
    'dist',
    '*.egg-info',
}

# Directories that should use claims license
CLAIMS_DIRS = {
    'algorithms/rft/core',
    'algorithms/rft/crypto',
    'algorithms/rft/quantum',
    'algorithms/rft/compression',
}


def load_claims_files(repo_root: Path) -> Set[str]:
    """Load list of claim-practicing files."""
    claims_file = repo_root / 'CLAIMS_PRACTICING_FILES.txt'
    if not claims_file.exists():
        return set()
    
    claims = set()
    with open(claims_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                claims.add(line)
    return claims


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    parts = path.parts
    for pattern in SKIP_PATTERNS:
        if pattern in parts:
            return True
    return False


def has_spdx_header(content: str) -> bool:
    """Check if file already has SPDX header."""
    return 'SPDX-License-Identifier' in content[:500]


def has_shebang(content: str) -> bool:
    """Check if file starts with shebang."""
    return content.startswith('#!')


def is_claims_file(filepath: Path, claims_files: Set[str], repo_root: Path) -> bool:
    """Check if file is a claims-practicing file."""
    rel_path = str(filepath.relative_to(repo_root))
    
    # Check explicit list
    if rel_path in claims_files:
        return True
    
    # Check if in claims directory
    for claims_dir in CLAIMS_DIRS:
        if rel_path.startswith(claims_dir):
            return True
    
    return False


def add_header(filepath: Path, claims_files: Set[str], repo_root: Path, dry_run: bool) -> bool:
    """Add SPDX header to a file. Returns True if modified."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError):
        return False
    
    if has_spdx_header(content):
        return False  # Already has header
    
    # Determine which header to use
    if is_claims_file(filepath, claims_files, repo_root):
        header = SPDX_CLAIMS_HEADER
    else:
        header = SPDX_PYTHON_HEADER
    
    # Handle shebang
    if has_shebang(content):
        lines = content.split('\n', 1)
        shebang = lines[0] + '\n'
        rest = lines[1] if len(lines) > 1 else ''
        new_content = shebang + header + rest
    else:
        new_content = header + content
    
    if dry_run:
        print(f"[DRY-RUN] Would add header to: {filepath}")
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"[ADDED] {filepath}")
    
    return True


def find_python_files(repo_root: Path) -> List[Path]:
    """Find all Python files in repository."""
    files = []
    for path in repo_root.rglob('*.py'):
        if not should_skip(path):
            files.append(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Add SPDX headers to Python files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.resolve()
    claims_files = load_claims_files(repo_root)
    
    print(f"Repository root: {repo_root}")
    print(f"Loaded {len(claims_files)} claim-practicing files")
    print(f"Dry run: {args.dry_run}")
    print()
    
    python_files = find_python_files(repo_root)
    print(f"Found {len(python_files)} Python files")
    print()
    
    modified = 0
    skipped = 0
    
    for filepath in python_files:
        if add_header(filepath, claims_files, repo_root, args.dry_run):
            modified += 1
        else:
            skipped += 1
    
    print()
    print(f"Summary:")
    print(f"  Modified: {modified}")
    print(f"  Skipped (already has header): {skipped}")
    print(f"  Total: {len(python_files)}")


if __name__ == '__main__':
    main()
