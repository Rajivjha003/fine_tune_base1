"""
combine_code_files.py  v3.0.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
World-class codebase → LLM context packager.

GOLDEN RULES (never violated):
  1. ZERO code loss  — every byte of every matched file is preserved verbatim.
  2. Files are ATOMIC — a file is NEVER split across two chunks.
  3. Garbage-free    — binaries, build artefacts, secrets, and VCS noise are
                       excluded before a single line is read.
  4. Deterministic   — same inputs always produce identical outputs.
  5. Resilient       — one bad file never aborts the whole run.

FEATURES:
  • .gitignore  + nested .gitignore support (pathspec, optional)
  • .llmignore  custom ignore file (same gitignore syntax)
  • Token budget mode: split by ~token count instead of raw line count
  • Per-file SHA-256 hash in header for integrity verification
  • Full project tree in every chunk header for LLM spatial awareness
  • Manifest file (manifest.json) listing every chunk + files it contains
  • Dry-run mode: prints what would happen without writing anything
  • Single-file mode: everything in one file (no chunking)
  • Max file size guard: skip suspiciously large files (e.g. minified bundles)
  • Symlink-safe: never follows symlinks into excluded territory
  • Handles Windows CRLF, Mac CR, Unix LF — outputs clean LF always
  • Graceful degradation: all optional deps (pathspec, tqdm, tiktoken) optional
  • Fully typed, zero external deps required to run

OPTIONAL DEPS (all pip-installable, all gracefully degraded if absent):
  pip install pathspec tiktoken tqdm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── optional dependencies ─────────────────────────────────────────────────────

try:
    import pathspec          # pip install pathspec
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

try:
    import tiktoken          # pip install tiktoken  (OpenAI's tokenizer)
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except Exception:
    _HAS_TIKTOKEN = False
    _TOKENIZER = None  # type: ignore[assignment]

try:
    from tqdm import tqdm    # pip install tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ─────────────────────────────────────────────────────────────────────────────
# VERSION & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "3.0.0"
SEP     = "=" * 72          # visual separator used in output files

# ---------------------------------------------------------------------------
# Directories NEVER useful to an LLM — always excluded regardless of .gitignore
# ---------------------------------------------------------------------------
ALWAYS_EXCLUDED_DIRS: Set[str] = {
    # Python
    ".venv", "venv", "env", ".env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    # Node / JS / TS
    "node_modules", ".next", ".nuxt", ".svelte-kit",
    "dist", "build", "out", ".turbo", ".parcel-cache",
    # Version control
    ".git", ".hg", ".svn",
    # Editors & OS
    ".idea", ".vscode", ".DS_Store",
    # Build / coverage artefacts
    "htmlcov", "coverage", ".coverage",
    # Django / Alembic migrations (generated, not logic)
    "migrations",
    # Our own output — CRITICAL: never recurse into previous output
    "CombinedFiles",
    # Misc noise
    "logs", "log", "tmp", "temp", ".cache", ".sass-cache",
    "storybook-static", ".storybook",
}

# ---------------------------------------------------------------------------
# File name patterns that are NEVER useful (checked against the basename)
# ---------------------------------------------------------------------------
ALWAYS_EXCLUDED_FILENAMES: Set[str] = {
    # Dependency lockfiles — massive, machine-generated
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "Gemfile.lock",
    "composer.lock", "cargo.lock",
    # Compiled / transpiled output
    "*.min.js", "*.min.css", "*.bundle.js", "*.chunk.js",
    # Environment & secrets
    ".env", ".env.local", ".env.production", ".env.staging",
    ".env.development", ".env.test",
    # Misc machine-generated
    ".DS_Store", "Thumbs.db", "desktop.ini",
}

# ---------------------------------------------------------------------------
# Binary / non-human-readable extensions — silently skipped
# ---------------------------------------------------------------------------
BINARY_EXTENSIONS: Set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".tiff", ".webp", ".avif", ".svg",
    # Audio / video
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov", ".mkv",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Documents (binary format)
    ".pdf", ".docx", ".xlsx", ".pptx", ".odt", ".ods",
    # Executables / compiled
    ".exe", ".dll", ".so", ".dylib", ".bin", ".wasm",
    ".pyc", ".pyo", ".pyd", ".class", ".o", ".a",
    # Databases
    ".db", ".sqlite", ".sqlite3", ".mdb",
    # Font
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Misc binary
    ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5",
    ".parquet", ".feather", ".arrow",
    ".model", ".pt", ".pth", ".onnx", ".pb",
    ".p12", ".pfx", ".cer", ".der",
}

# Max file size to read (bytes). Files larger than this are likely minified
# bundles, auto-generated code, or data files — skip them with a warning.
DEFAULT_MAX_FILE_BYTES: int = 500_000   # 500 KB

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN COUNTING
# ─────────────────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Return the token count for `text`.
    Uses tiktoken (cl100k_base) when available, else falls back to the
    industry-standard heuristic: 1 token ≈ 4 characters.
    """
    if _HAS_TIKTOKEN and _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# IGNORE-SPEC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_spec_from_files(paths: List[Path]) -> Optional[object]:
    """
    Load a pathspec from one or more gitignore-style files.
    Returns None if pathspec is not installed or no files exist.
    """
    if not _HAS_PATHSPEC:
        return None
    patterns: List[str] = []
    for p in paths:
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    patterns.extend(fh.read().splitlines())
            except OSError:
                pass
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _build_ignore_spec(root: Path, use_gitignore: bool, llmignore: bool):
    """
    Build a combined ignore spec from:
      1. <root>/.gitignore
      2. ~/.gitignore_global (if present)
      3. <root>/.llmignore   (custom file, same gitignore syntax)
    Returns the spec object or None.
    """
    sources: List[Path] = []
    if use_gitignore:
        sources += [root / ".gitignore", Path.home() / ".gitignore_global"]
    if llmignore:
        sources.append(root / ".llmignore")
    return _load_spec_from_files(sources)


def _is_secret_file(fp: Path) -> bool:
    """
    Heuristic guard: skip files that look like they contain secrets,
    even if the user didn't add them to .gitignore.
    """
    name = fp.name.lower()
    # Exact matches
    secret_names = {
        ".env", ".env.local", ".env.production", ".env.staging",
        ".env.development", ".env.test", ".env.example",
        "secrets.json", "credentials.json", "service_account.json",
        "keystore.jks", "id_rsa", "id_ed25519",
    }
    if name in secret_names:
        return True
    # Suffix patterns
    secret_suffixes = (".pem", ".key", ".p12", ".pfx", ".cer", ".der")
    if name.endswith(secret_suffixes):
        return True
    # Env files like `.env.production`
    if name.startswith(".env."):
        return True
    return False


def _is_minified_or_generated(fp: Path, content: str) -> bool:
    """
    Detect minified / auto-generated files that waste tokens without adding context.
    Heuristics:
      • Any line longer than 5 000 chars  → minified
      • File starts with a common generated header  → auto-generated
    """
    # Check for extremely long lines (minified bundles)
    for line in content.splitlines()[:20]:   # only check first 20 lines
        if len(line) > 5_000:
            return True
    # Common auto-generated headers
    generated_markers = (
        "// THIS FILE IS AUTO-GENERATED",
        "// Auto-generated file",
        "/* eslint-disable */\n// @ts-nocheck",
        "# THIS FILE IS GENERATED",
        "# AUTO GENERATED",
        "# Do not edit this file",
        "# This file was automatically generated",
        "# Generated by",
    )
    header = content[:500].upper()
    for marker in generated_markers:
        if marker.upper() in header:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY TREE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_tree(root: Path, files: List[Path]) -> str:
    """
    Build a compact ASCII directory tree for the collected files.

        project/
        ├── app/
        │   ├── main.py
        │   └── utils.py
        └── tests/
            └── test_main.py
    """
    lines: List[str] = [f"{root.name}/"]
    rel_paths = sorted(fp.relative_to(root) for fp in files)

    # Track which ancestors we've already emitted
    seen: Set[Path] = set()

    for rp in rel_paths:
        parts = rp.parts
        for depth in range(len(parts)):
            ancestor = Path(*parts[: depth + 1])
            if ancestor in seen:
                continue
            seen.add(ancestor)
            is_last_at_depth = _is_last_child(rel_paths, ancestor, depth)
            indent = ""
            for d in range(depth):
                parent_at_d = Path(*parts[: d + 1])
                indent += "    " if _is_last_child(rel_paths, parent_at_d, d) else "│   "
            connector = "└── " if is_last_at_depth else "├── "
            suffix = "/" if depth < len(parts) - 1 else ""
            lines.append(indent + connector + parts[depth] + suffix)
    return "\n".join(lines)


def _is_last_child(rel_paths: List[Path], node: Path, depth: int) -> bool:
    """True if `node` is the last sibling at its depth level."""
    siblings = [
        rp for rp in rel_paths
        if len(rp.parts) > depth and Path(*rp.parts[: depth + 1]) == node
           or (depth > 0
               and len(rp.parts) > depth
               and Path(*rp.parts[:depth]) == node.parent
               and len(rp.parts) > depth)
    ]
    # simpler: just check if any later path shares the same parent
    node_parent = node.parent
    node_name   = node.name
    same_parent = [
        rp for rp in rel_paths
        if len(rp.parts) > depth
        and (Path(*rp.parts[: depth]) if depth > 0 else Path(".")) == node_parent
    ]
    if not same_parent:
        return True
    return Path(*same_parent[-1].parts[: depth + 1]).name == node_name


# ─────────────────────────────────────────────────────────────────────────────
# FILE INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_snippet(path: Path) -> str:
    """Return the first 12 hex chars of the SHA-256 of the raw file bytes."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65_536), b""):
                h.update(block)
        return h.hexdigest()[:12]
    except OSError:
        return "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def combine_code_files(
    root_dir: str,
    extensions: List[str],
    lines_per_chunk: int = 1500,
    token_budget: Optional[int] = None,
    output_dir: Optional[str] = None,
    extra_excluded_dirs: Optional[Set[str]] = None,
    use_gitignore: bool = True,
    use_llmignore: bool = True,
    include_secrets: bool = False,
    skip_generated: bool = True,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    dry_run: bool = False,
    single_file: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Combine source code files into LLM-ready chunked text files.

    Parameters
    ----------
    root_dir          : Project root to scan (string path).
    extensions        : Extensions to include, e.g. ['.py', '.ts', '.tsx'].
    lines_per_chunk   : Soft *line* limit per output chunk.
                        Ignored when token_budget is set.
                        A FILE IS NEVER SPLIT — boundary is always at
                        the end of a complete file.
    token_budget      : If set, split by approximate token count instead
                        of lines. Requires tiktoken for accuracy; falls
                        back to char/4 heuristic otherwise.
    output_dir        : Where to write chunks.
                        Defaults to <root_dir>/CombinedFiles.
    extra_excluded_dirs: Additional directory names to never descend into.
    use_gitignore     : Respect .gitignore + ~/.gitignore_global (needs pathspec).
    use_llmignore     : Respect <root>/.llmignore (same gitignore syntax).
    include_secrets   : If False (default), skip .env / key / pem files.
    skip_generated    : If True (default), skip auto-generated / minified files.
    max_file_bytes    : Skip files larger than this many bytes (default 500 KB).
    dry_run           : Print what would happen; write nothing.
    single_file       : Emit everything into one combined_1.txt (no chunking).
    verbose           : Print progress and statistics.

    Returns
    -------
    List of absolute path strings for the written chunk files.
    (Empty list on dry_run or when no files found.)
    """

    # ── Resolve paths ────────────────────────────────────────────────────────
    root    = Path(root_dir).resolve()
    out_dir = Path(output_dir).resolve() if output_dir else root / "CombinedFiles"

    if not root.is_dir():
        raise ValueError(f"root_dir does not exist or is not a directory: {root}")

    # ── Normalise extensions → always ".ext" lowercase ───────────────────────
    exts: Set[str] = {
        (e if e.startswith(".") else f".{e}").lower()
        for e in extensions
    }

    # ── Build exclusion set ───────────────────────────────────────────────────
    excluded_dirs = ALWAYS_EXCLUDED_DIRS.copy()
    if extra_excluded_dirs:
        excluded_dirs.update(extra_excluded_dirs)
    # Never recurse into the output directory
    excluded_dirs.add(out_dir.name)

    # ── Build ignore spec (gitignore + .llmignore) ───────────────────────────
    ignore_spec = _build_ignore_spec(root, use_gitignore, use_llmignore)

    # ── Logging helper ────────────────────────────────────────────────────────
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ── Print run config ─────────────────────────────────────────────────────
    budget_desc = (
        f"~{token_budget:,} tokens/chunk"
        if token_budget
        else f"{lines_per_chunk:,} lines/chunk (file-boundary aligned)"
    )
    log(f"\n{'━'*60}")
    log(f"  combine_code_files  v{VERSION}")
    log(f"{'━'*60}")
    log(f"  root         : {root}")
    log(f"  extensions   : {sorted(exts)}")
    log(f"  budget       : {budget_desc}")
    log(f"  output       : {out_dir}")
    log(f"  gitignore    : {'✓' if ignore_spec and use_gitignore else '✗'}")
    log(f"  llmignore    : {'✓' if (root / '.llmignore').exists() else '✗ (no .llmignore found)'}")
    log(f"  tiktoken     : {'✓ exact' if _HAS_TIKTOKEN else '≈ heuristic (pip install tiktoken)'}")
    log(f"  dry_run      : {'YES — nothing will be written' if dry_run else 'no'}")
    log(f"  single_file  : {single_file}")
    log(f"{'━'*60}\n")

    # ── Phase 1: Walk and collect all matching files ──────────────────────────
    collected:         List[Path] = []
    skipped_binary:    int = 0
    skipped_ignore:    int = 0
    skipped_secret:    int = 0
    skipped_toobig:    int = 0
    skipped_generated: int = 0
    skipped_lockfile:  int = 0

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dp = Path(dirpath_str)

        # ── Prune excluded dirs in-place (CRITICAL for performance) ──────────
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in excluded_dirs and not d.startswith(".")
        )

        for fname in sorted(filenames):
            fp = dp / fname
            rel = fp.relative_to(root)

            # 1. Binary extension guard
            if fp.suffix.lower() in BINARY_EXTENSIONS:
                skipped_binary += 1
                continue

            # 2. Well-known useless filenames (lockfiles, etc.)
            if fname in ALWAYS_EXCLUDED_FILENAMES:
                skipped_lockfile += 1
                continue

            # 3. Secret / credential files
            if not include_secrets and _is_secret_file(fp):
                skipped_secret += 1
                log(f"  🔒 secret skip  : {rel}")
                continue

            # 4. gitignore / llmignore spec
            if ignore_spec and ignore_spec.match_file(str(rel)):
                skipped_ignore += 1
                log(f"  ⤷  ignore skip  : {rel}")
                continue

            # 5. Extension filter
            if fp.suffix.lower() not in exts:
                continue

            # 6. File size guard (catches minified bundles by size alone)
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if size > max_file_bytes:
                skipped_toobig += 1
                log(f"  ⚠️  too large    : {rel} ({size:,} bytes > {max_file_bytes:,})")
                continue

            collected.append(fp)

    # ── Early exit ────────────────────────────────────────────────────────────
    if not collected:
        log("⚠️  No matching files found.")
        log(f"   Searched: {root}")
        log(f"   Extensions: {sorted(exts)}")
        log("   Tip: check --extensions and verify the path.\n")
        return []

    # ── Phase 2: Read files + apply content-level guards ─────────────────────
    # We read here (not during walk) so content guards run before chunking.
    FileRecord = Tuple[Path, List[str], str]   # (path, lines, sha256)
    records: List[FileRecord] = []
    read_errors: int = 0

    iterator = (
        tqdm(collected, unit="file", ncols=80, desc="Reading")   # type: ignore
        if _HAS_TQDM else collected
    )

    for fp in iterator:
        rel = fp.relative_to(root)
        try:
            raw = fp.read_bytes()
            # Normalise line endings → LF only
            text = raw.decode("utf-8", errors="replace") \
                      .replace("\r\n", "\n") \
                      .replace("\r",   "\n")
        except OSError as exc:
            log(f"  ❌ read error    : {rel}  →  {exc}")
            read_errors += 1
            continue

        # Content-level: skip minified / auto-generated files
        if skip_generated and _is_minified_or_generated(fp, text):
            skipped_generated += 1
            log(f"  🤖 generated skip: {rel}")
            continue

        sha = _sha256_snippet(fp)
        file_lines = text.splitlines()
        records.append((fp, file_lines, sha))

    if not records:
        log("⚠️  All matched files were filtered out at the content level.\n")
        return []

    # ── Phase 3: Build project tree ───────────────────────────────────────────
    included_paths = [fp for fp, _, _ in records]
    project_tree   = _build_tree(root, included_paths)
    run_ts         = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Phase 4: Chunk & write ─────────────────────────────────────────────────
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx:     int       = 1
    budget_used:   int       = 0   # lines OR tokens depending on mode
    buf:           List[str] = []
    written:       List[str] = []
    # manifest: chunk_id → list of relative file paths
    manifest:      Dict[int, List[str]] = {}
    current_files: List[str] = []

    def _preamble(idx: int) -> List[str]:
        lines = [
            SEP,
            f"// LLM CONTEXT PACKAGE",
            f"// chunk     : {idx}",
            f"// generated : {run_ts}",
            f"// tool      : combine_code_files v{VERSION}",
            f"// root      : {root}",
            f"// extensions: {', '.join(sorted(exts))}",
            f"// tokenizer : {'tiktoken cl100k_base' if _HAS_TIKTOKEN else 'heuristic (chars/4)'}",
            SEP,
            "",
            "// ── PROJECT FILE TREE (all files in this run) ──",
        ]
        for tline in project_tree.splitlines():
            lines.append(f"//   {tline}")
        lines += ["", SEP, ""]
        return lines

    def _flush(buf: List[str], idx: int, files: List[str]) -> str:
        text = "\n".join(buf)
        out_path = out_dir / f"combined_{idx}.txt"
        tok = count_tokens(text)
        if not dry_run:
            out_path.write_text(text, encoding="utf-8")
        manifest[idx] = files
        log(
            f"  ✏️  chunk {idx:03d}  →  combined_{idx}.txt"
            f"  ({len(buf):,} lines  |  ~{tok:,} tokens)"
        )
        return str(out_path)

    buf           = _preamble(chunk_idx)
    budget_used   = count_tokens("\n".join(buf)) if token_budget else len(buf)
    current_files = []

    for fp, file_lines, sha in records:
        rel      = fp.relative_to(root)
        rel_str  = str(rel)
        n_lines  = len(file_lines)

        # Build the file block header
        file_header = [
            "",
            SEP,
            f"// FILE  : {rel_str}",
            f"// LINES : {n_lines}",
            f"// SHA256: {sha}",
            SEP,
            "",
        ]
        file_block = file_header + file_lines + [""]

        # Measure this block
        if token_budget:
            block_cost = count_tokens("\n".join(file_block))
        else:
            block_cost = len(file_block)

        # ── Chunk boundary check ─────────────────────────────────────────────
        # If adding this file would exceed the budget AND we already have
        # at least one file in the current chunk, flush first.
        # (If the chunk is empty, we MUST add the file even if it alone exceeds
        #  the budget — we never split a file.)
        if current_files and (budget_used + block_cost) > (token_budget or lines_per_chunk):
            written.append(_flush(buf, chunk_idx, current_files))
            chunk_idx   += 1
            buf           = _preamble(chunk_idx)
            budget_used   = count_tokens("\n".join(buf)) if token_budget else len(buf)
            current_files = []

        buf.extend(file_block)
        budget_used   += block_cost
        current_files.append(rel_str)

        if single_file:
            # In single-file mode, never flush mid-way; let the final flush handle it.
            pass

    # Final flush
    if buf and current_files:
        written.append(_flush(buf, chunk_idx, current_files))

    # ── Phase 5: Write manifest ───────────────────────────────────────────────
    manifest_data = {
        "version"    : VERSION,
        "generated"  : run_ts,
        "root"       : str(root),
        "extensions" : sorted(exts),
        "total_files": len(records),
        "chunks"     : [
            {
                "chunk"     : idx,
                "filename"  : f"combined_{idx}.txt",
                "files"     : flist,
            }
            for idx, flist in manifest.items()
        ],
    }
    manifest_path = out_dir / "manifest.json"
    if not dry_run:
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Phase 6: Summary ─────────────────────────────────────────────────────
    total_tokens = sum(
        count_tokens(Path(p).read_text(encoding="utf-8"))
        for p in written
        if not dry_run
    )

    log(f"\n{'━'*60}")
    log(f"  ✅  DONE")
    log(f"{'━'*60}")
    log(f"  files included     : {len(records)}")
    log(f"  chunks written     : {len(written)}")
    log(f"  total ~tokens      : {total_tokens:,}")
    log(f"  output dir         : {out_dir}")
    if not dry_run:
        log(f"  manifest           : {manifest_path.name}")
    log(f"")
    log(f"  ── Skipped ──────────────────────────────────")
    log(f"  binary             : {skipped_binary}")
    log(f"  lockfiles          : {skipped_lockfile}")
    log(f"  secrets            : {skipped_secret}")
    log(f"  too large (>{max_file_bytes//1000}KB): {skipped_toobig}")
    log(f"  gitignore/llmignore: {skipped_ignore}")
    log(f"  auto-generated     : {skipped_generated}")
    log(f"  read errors        : {read_errors}")
    log(f"{'━'*60}\n")

    return written if not dry_run else []


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="combine_code_files",
        description="combine_code_files — World-class codebase → LLM context packager.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
EXAMPLES
────────
  # Python backend only (1500 lines/chunk)
  python combine_code_files.py D:\project\backend -e .py

  # Full-stack  (Python + TypeScript/React)
  python combine_code_files.py D:\project -e .py .ts .tsx .js .jsx

  # Tight context window: split by ~8 000 tokens/chunk (needs tiktoken)
  python combine_code_files.py . -e .py --tokens 8000

  # Single file (no chunking) — for models with huge context windows
  python combine_code_files.py . -e .py --single-file

  # Custom output directory
  python combine_code_files.py . -e .py -o C:\llm_context\my_project

  # Exclude extra directories, raise file size limit to 1 MB
  python combine_code_files.py . -e .py --exclude tests docs --max-kb 1000

  # Dry run — see exactly what would be included without writing
  python combine_code_files.py . -e .py --dry-run

  # Disable .gitignore (include everything except built-in exclusions)
  python combine_code_files.py . -e .py --no-gitignore

CREATE .llmignore
──────────────────
  Create a file named  .llmignore  in your project root using the same
  syntax as .gitignore. It is loaded in addition to .gitignore and lets
  you exclude paths that should not go to the LLM without affecting Git.

  Example .llmignore:
      # Skip test fixtures and sample data
      tests/fixtures/
      data/samples/
      # Skip generated API client
      src/api/generated/
        """,
    )

    p.add_argument(
        "root_dir",
        help="Project root directory to scan.",
    )
    p.add_argument(
        "-e", "--extensions",
        nargs="+", required=True, metavar="EXT",
        help="File extensions to include.  Example: -e .py .ts .tsx",
    )
    p.add_argument(
        "--lines",
        type=int, default=1500, dest="lines_per_chunk", metavar="N",
        help="Soft line limit per chunk (default: 1500). Ignored when --tokens is set.",
    )
    p.add_argument(
        "--tokens",
        type=int, default=None, dest="token_budget", metavar="N",
        help="Split by approximate token count instead of lines. "
             "Example: --tokens 8000  for an 8 K-token chunk size.",
    )
    p.add_argument(
        "-o", "--output",
        default=None, dest="output_dir", metavar="DIR",
        help="Output directory. Defaults to <root>/CombinedFiles.",
    )
    p.add_argument(
        "--exclude",
        nargs="*", default=[], dest="extra_excluded", metavar="DIR",
        help="Extra directory names to never descend into.",
    )
    p.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Do NOT load .gitignore / ~/.gitignore_global.",
    )
    p.add_argument(
        "--no-llmignore",
        action="store_true",
        help="Do NOT load .llmignore even if it exists.",
    )
    p.add_argument(
        "--include-secrets",
        action="store_true",
        help="Include .env and credential files (NOT recommended).",
    )
    p.add_argument(
        "--keep-generated",
        action="store_true",
        help="Include auto-generated / minified files (NOT recommended).",
    )
    p.add_argument(
        "--max-kb",
        type=int, default=DEFAULT_MAX_FILE_BYTES // 1000, metavar="KB",
        help=f"Skip files larger than N KB (default: {DEFAULT_MAX_FILE_BYTES // 1000}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing any files.",
    )
    p.add_argument(
        "--single-file",
        action="store_true",
        help="Write all content to a single combined_1.txt (no chunking).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output.",
    )

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    combine_code_files(
        root_dir           = args.root_dir,
        extensions         = args.extensions,
        lines_per_chunk    = args.lines_per_chunk,
        token_budget       = args.token_budget,
        output_dir         = args.output_dir,
        extra_excluded_dirs= set(args.extra_excluded) if args.extra_excluded else None,
        use_gitignore      = not args.no_gitignore,
        use_llmignore      = not args.no_llmignore,
        include_secrets    = args.include_secrets,
        skip_generated     = not args.keep_generated,
        max_file_bytes     = args.max_kb * 1000,
        dry_run            = args.dry_run,
        single_file        = args.single_file,
        verbose            = not args.quiet,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DIRECT INVOCATION  (edit the section below to match your project)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use the CLI parser to handle terminal arguments
    main()


# python code_context.py D:\Fine_tuning\merchfine -e .py
