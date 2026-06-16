"""E8: replace Q-format magic literals with primitives constants.

Migration record + reusable remediation (handoff packet E8). First run
2026-06-11 (492 replacements across 52 files, result committed in
748654f). REUSABLE: if tests/test_q15_hygiene.py fails because new magic
literals crept into the engine hot paths, re-run this to remediate. The
rewrite is value-preserving by construction; bit-exact gate the golden
traces afterward.

Tokenize-based (NUMBER tokens only - comments, docstrings, and strings
are never touched). Usage (from repo root):

    pixi run python tools/migrate_q15_literals.py --dry  [paths...]  # report
    pixi run python tools/migrate_q15_literals.py        [paths...]  # apply

Rules:
- 32767 -> Q15_ONE, 16384 -> Q15_HALF, 8192 -> Q15_QUARTER,
  24576 -> Q15_3_4, 65536 -> Q16_ONE
- 32768 -> Q15_SCALE, except unary "-32768" -> Q15_NEG_ONE
- Constant-definition lines (NAME = <literal>) are left alone
  (they are redefinition sites, handled manually).
- VETO list excludes instances that are not Q-format semantics
  (buffer sizes, FFT lengths, etc.) - reviewed by hand from --dry.
- Adds the needed `from ...primitives import (...)` line per file.
"""

import io
import re
import sys
import token
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # tools/ -> repo root
SRC = REPO / "src"
PKG = SRC / "symcrash_python_ap"

MAP = {
    "32767": "Q15_ONE",
    "16384": "Q15_HALF",
    "8192": "Q15_QUARTER",
    "24576": "Q15_3_4",
    "65536": "Q16_ONE",
    "32768": "Q15_SCALE",  # unary-minus case becomes Q15_NEG_ONE
}

DEFAULT_TARGETS = [
    PKG,
    SRC / "symcrash_workbench",
    SRC / "symcrash_preset_editor",
]
EXCLUDE = {PKG / "primitives.py", PKG / "fixed.py"}

# (relpath-as-posix, lineno) pairs reviewed by hand: literal is NOT Q-format
VETO: set = {
    # sine_core: 65536 = LUT table size; 16384 = quarter-cycle phase offset
    # (90 deg for cosine lookup), not Q15_HALF
    ("src/symcrash_python_ap/wavelet/sine_core.py", 102),
    ("src/symcrash_python_ap/wavelet/sine_core.py", 103),
    ("src/symcrash_python_ap/wavelet/sine_core.py", 123),
    ("src/symcrash_python_ap/wavelet/sine_core.py", 124),
    ("src/symcrash_python_ap/wavelet/sine_core.py", 161),
    ("src/symcrash_python_ap/wavelet/sine_core.py", 168),
    ("src/symcrash_python_ap/wavelet/sine_core.py", 337),
    # MIDI pitch-bend center (2^13), not Q15_QUARTER
    ("src/symcrash_python_ap/cockpit_engine.py", 237),
    # cents scale factor, not Q15_QUARTER
    ("src/symcrash_python_ap/voice/time_ownership.py", 514),
}

# (relpath, lineno) -> {literal: name} per-line semantic overrides
OVERRIDES = {
    # LFO phase comparisons: phase is Q16 (0..65535); 32768 = half cycle
    ("src/symcrash_python_ap/envelope_suite.py", 737): {"32768": "Q16_HALF"},
    ("src/symcrash_python_ap/envelope_suite.py", 740): {"32768": "Q16_HALF"},
    ("src/symcrash_python_ap/envelope_suite.py", 745): {"32768": "Q16_HALF"},
    ("src/symcrash_python_ap/envelope_suite.py", 753): {"32768": "Q16_HALF"},
}

CONST_DEF_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z_0-9]*\s*=\s*-?\d+\s*(#.*)?$")
# token strings before a '-' that make it unary
UNARY_BEFORE = {"(", "[", "{", ",", "=", ":", "+", "-", "*", "/", "//", "%",
                "**", "<<", ">>", "<", ">", "<=", ">=", "==", "!=", "&", "|",
                "^", "~", "and", "or", "not", "in", "if", "else", "return",
                "yield", "while", "assert", "lambda"}


def relative_import_prefix(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(PKG)
        depth = len(rel.parts) - 1  # 0 = top-level module in the package
        return "." * (depth + 1) + "primitives"
    except ValueError:
        return "symcrash_python_ap.primitives"


def existing_imported_names(text: str) -> set:
    names = set()
    for m in re.finditer(
            r"from\s+[\w.]*(?:primitives|fixed)\s+import\s+(\(([^)]*)\)|([^\n]*))",
            text):
        body = m.group(2) or m.group(3) or ""
        for part in body.replace("\n", " ").split(","):
            name = part.strip().split(" as ")[0].strip()
            if name:
                names.add(name)
    return names


def find_import_insert_line(lines) -> int:
    """Line index AFTER the module docstring and leading imports (ast-based:
    a >120-line docstring broke the old text heuristic by swallowing the
    inserted import)."""
    import ast
    tree = ast.parse("".join(lines))
    last = 0
    for node in tree.body:
        is_docstring = (isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str))
        if is_docstring or isinstance(node, (ast.Import, ast.ImportFrom)):
            last = node.end_lineno  # 1-based -> insert index after it
            continue
        break
    return last


def process(path: Path, dry: bool) -> list:
    raw = path.read_bytes()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    rel = path.resolve().relative_to(REPO).as_posix()

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenizeError as e:
        print(f"!! tokenize failed {rel}: {e}")
        return []

    sig = [t for t in tokens if t.type not in
           (token.NL, token.NEWLINE, token.COMMENT, token.INDENT,
            token.DEDENT)]
    pos_of = {id(t): i for i, t in enumerate(sig)}

    repls = []  # (row, col_start, col_end, newtext, old, label)
    for t in tokens:
        if t.type != token.NUMBER or t.string not in MAP:
            continue
        row, col = t.start
        if (rel, row) in VETO:
            continue
        line = lines[row - 1]
        if "q15-literal-ok" in line:
            continue  # reviewed non-Q15 number (same marker the lint honors)
        if CONST_DEF_RE.match(line.rstrip("\r\n")):
            continue  # redefinition site, handled manually
        new = OVERRIDES.get((rel, row), {}).get(t.string) or MAP[t.string]
        start_col = col
        if new == "Q15_SCALE" and t.string == "32768" and id(t) in pos_of:
            i = pos_of[id(t)]
            prev = sig[i - 1] if i > 0 else None
            before = sig[i - 2] if i > 1 else None
            if (prev is not None and prev.string == "-"
                    and prev.end == t.start
                    and (before is None or before.string in UNARY_BEFORE
                         or before.type in (token.NEWLINE, token.INDENT))):
                new = "Q15_NEG_ONE"
                start_col = prev.start[1]
        repls.append((row, start_col, t.end[1], new, t.string, line.rstrip()))

    if not repls:
        return []

    if dry:
        return [(rel, r[0], r[4], r[3], r[5]) for r in repls]

    # apply right-to-left per line
    by_line = {}
    for row, c0, c1, new, _old, _ctx in repls:
        by_line.setdefault(row, []).append((c0, c1, new))
    for row, items in by_line.items():
        line = lines[row - 1]
        for c0, c1, new in sorted(items, reverse=True):
            line = line[:c0] + new + line[c1:]
        lines[row - 1] = line

    needed = sorted({r[3] for r in repls})
    have = existing_imported_names(text)
    missing = [n for n in needed if n not in have]
    if missing:
        prefix = relative_import_prefix(path)
        imp = f"from {prefix} import {', '.join(missing)}\n"
        at = find_import_insert_line(lines)
        lines.insert(at, imp)

    out = "".join(lines)
    if crlf and "\r\n" not in out:
        out = out.replace("\n", "\r\n")
    path.write_bytes(out.encode("utf-8"))
    return [(rel, r[0], r[4], r[3], r[5]) for r in repls]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if a != "--dry"]
    dry = "--dry" in sys.argv
    targets = [Path(a).resolve() for a in args] if args else DEFAULT_TARGETS
    files = []
    for t in targets:
        files += [t] if t.is_file() else sorted(t.rglob("*.py"))
    files = [f for f in files if f.resolve() not in
             {e.resolve() for e in EXCLUDE}]

    total = 0
    per_file = {}
    for f in files:
        rows = process(f, dry)
        if rows:
            per_file[rows[0][0]] = rows
            total += len(rows)

    mode = "DRY" if dry else "APPLIED"
    for rel, rows in per_file.items():
        print(f"\n{rel}  ({len(rows)})")
        for _rel, lineno, old, new, ctx in rows:
            print(f"  {lineno:5d}: {old} -> {new:12s} | {ctx.strip()[:90]}")
    print(f"\n[{mode}] {total} replacements in {len(per_file)} files")


if __name__ == "__main__":
    main()
