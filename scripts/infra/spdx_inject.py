#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
SPDX Header Injector (AGPL vs Claims-NC)

- Reads CLAIMS_PRACTICING_FILES.txt at repo root
- Adds top-of-file SPDX headers where safe (files with comment syntax)
- Skips files already containing an SPDX header
- For JSON (no comments), writes a companion "<file>.license" file per REUSE spec

Usage:
  python spdx_inject.py --dry-run
  python spdx_inject.py
"""
import argparse, sys, os, re
from pathlib import Path

# Comment styles per extension
LINE_HASH = "# {tag}: {id}\n"
LINE_SLASH = "// {tag}: {id}\n"
BLOCK_C = "/* {tag}: {id} */\n"
# Map of extensions to header style in priority order
COMMENT_STYLES = {
    # hash comment
    ".py": LINE_HASH, ".sh": LINE_HASH, ".bash": LINE_HASH, ".zsh": LINE_HASH,
    ".md": LINE_HASH, ".txt": LINE_HASH, ".yaml": LINE_HASH, ".yml": LINE_HASH,
    ".toml": LINE_HASH, ".ts": LINE_HASH, ".tsx": LINE_HASH, ".js": LINE_HASH, ".jsx": LINE_HASH,
    ".sv": LINE_SLASH, ".svh": LINE_SLASH, ".v": LINE_SLASH, ".vh": LINE_SLASH,
    ".asm": LINE_SLASH, ".s": LINE_SLASH,
    ".c": BLOCK_C, ".h": BLOCK_C, ".cpp": BLOCK_C, ".hpp": BLOCK_C, ".cc": BLOCK_C, ".cxx": BLOCK_C,
}

SPDX_TAG = "SPDX-License-Identifier"
SPDX_RE = re.compile(r"^\s*SPDX-License-Identifier\s*:", re.IGNORECASE | re.MULTILINE)

# File size cap (bytes) to avoid huge/binary-ish blobs
SIZE_CAP = 2_000_000

def load_claims_list(repo: Path):
    claims_file = repo / "CLAIMS_PRACTICING_FILES.txt"
    claims = set()
    if claims_file.exists():
        for line in claims_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            claims.add(line)
    return claims

def choose_id(path: Path, claims_set: set):
    # Paths in claims list are repo-relative with forward slashes
    rel = path.as_posix()
    if rel in claims_set:
        return "LicenseRef-QuantoniumOS-Claims-NC"
    return "AGPL-3.0-or-later"

def get_style_for(path: Path):
    ext = path.suffix.lower()
    return COMMENT_STYLES.get(ext, None)

def has_spdx(text: str) -> bool:
    return bool(SPDX_RE.search(text))

def inject_header(path: Path, header_line: str, dry_run: bool = False):
    raw = path.read_bytes()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="ignore")
    if has_spdx(txt):
        return False  # already has header
    # Preserve shebang if present
    lines = txt.splitlines(keepends=True)
    if lines and lines[0].startswith("#!") and header_line.startswith("#"):
        new_txt = lines[0] + header_line + "".join(lines[1:])
    else:
        new_txt = header_line + txt
    if not dry_run:
        path.write_text(new_txt, encoding="utf-8")
    return True

def write_license_sidecar(path: Path, spdx_id: str, dry_run: bool = False):
    lic = path.with_suffix(path.suffix + ".license")
    content = f"{SPDX_TAG}: {spdx_id}\n"
    if not dry_run:
        lic.write_text(content, encoding="utf-8")
    return lic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="scan and report without modifying files")
    ap.add_argument("--repo", default=".", help="path to repo root")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    claims = load_claims_list(repo)

    changed, sidecars, skipped, present = [], [], [], []

    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        # skip large files
        try:
            if path.stat().st_size > SIZE_CAP:
                continue
        except OSError:
            continue

        rel = path.relative_to(repo)
        # skip obvious non-source (git dir, images, archives, etc.)
        rel_s = rel.as_posix()
        if any(x in rel_s.split("/") for x in [".git", ".venv", "node_modules", "__pycache__", "build", "dist", "quantoniumos.egg-info", ".pytest_cache", ".mypy_cache"]):
            continue
        if rel_s.startswith(".git/") or "/.git/" in rel_s:
            continue
        if any(rel_s.lower().endswith(ext) for ext in (".png",".jpg",".jpeg",".gif",".mp4",".zip",".tar",".gz",".bz2",".xz",".7z",".pdf",".pkl",".bin",".pt",".onnx",".so",".dylib",".dll")):
            continue

        try:
            raw = path.read_bytes()
        except Exception:
            skipped.append(rel_s + " [unreadable]")
            continue

        # Try to decode as text
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                txt = raw.decode("latin-1", errors="ignore")
            except Exception:
                skipped.append(rel_s + " [binary?]")
                continue

        if has_spdx(txt):
            present.append(rel_s)
            continue

        spdx_id = choose_id(rel, claims)

        # JSON has no comments -> use sidecar
        if path.suffix.lower() == ".json":
            sidecar = write_license_sidecar(path, spdx_id, dry_run=args.dry_run)
            sidecars.append(sidecar.relative_to(repo).as_posix())
            continue

        style = get_style_for(path)
        if style is None:
            # fallback: try hash comment if first line looks textual and not XML
            if txt.lstrip().startswith("<"):
                skipped.append(rel_s + " [xml-like]")
                continue
            style = "# {tag}: {id}\n"
        header_line = style.format(tag=SPDX_TAG, id=spdx_id)
        try:
            if inject_header(path, header_line, dry_run=args.dry_run):
                changed.append(rel_s)
        except Exception as e:
            skipped.append(rel_s + f" [inject error: {e}]")

    # Print a concise report
    print("== SPDX Injection Report ==")
    print(f"Repo: {repo}")
    print(f"Claims entries: {len(claims)}")
    print(f"Added headers to: {len(changed)} files")
    print(f"Created sidecar licenses for JSON: {len(sidecars)} files")
    print(f"Already had SPDX: {len(present)} files")
    print(f"Skipped: {len(skipped)} files")
    if args.dry_run:
        print("\n-- DRY RUN ONLY -- no files were modified.")
    else:
        print("\nChanges written to disk.")
    # Emit machine-readable summary
    summary = {
        "added": changed,
        "sidecars": sidecars,
        "present": present,
        "skipped": skipped,
    }
    try:
        import json
        Path("spdx_inject_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Summary JSON: spdx_inject_summary.json")
    except Exception:
        pass

if __name__ == "__main__":
    main()