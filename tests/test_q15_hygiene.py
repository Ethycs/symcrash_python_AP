"""
Q-format constant hygiene lint (E8, 2026-06-11).

Engine hot paths must use the canonical constants from
symcrash_python_ap.primitives (Q15_ONE, Q15_SCALE, Q15_HALF, ...)
instead of magic literals. The 32767 (clamp bound) vs 32768 (scale
divisor) distinction is a real fixed-point bug class; named constants
make a transposition visible in review.

Allowed exceptions:
- comments and strings (tokenize: only NUMBER tokens are checked)
- constant-definition lines (NAME = <literal>)
- lines carrying a `q15-literal-ok` marker comment (reviewed cases
  where the number is NOT Q-format semantics, e.g. LUT sizes or
  quarter-cycle phase offsets in sine_core.py)
"""

import io
import re
import token
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "symcrash_python_ap"

SCAN_DIRS = ["vm", "wavelet", "voice", "material", "safety", "symbol"]
FORBIDDEN = {"32767", "32768", "16384", "8192", "24576", "65536"}
CONST_DEF_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z_0-9]*\s*=\s*-?\d+\s*(#.*)?$")
MARKER = "q15-literal-ok"


def test_no_q_format_magic_literals_in_hot_paths():
    offenders = []
    for d in SCAN_DIRS:
        for path in sorted((SRC / d).rglob("*.py")):
            text = path.read_bytes().decode("utf-8")
            lines = text.splitlines()
            for t in tokenize.generate_tokens(io.StringIO(text).readline):
                if t.type != token.NUMBER or t.string not in FORBIDDEN:
                    continue
                line = lines[t.start[0] - 1]
                if MARKER in line or CONST_DEF_RE.match(line):
                    continue
                offenders.append(
                    f"{path.relative_to(SRC.parent)}:{t.start[0]}: "
                    f"{t.string} | {line.strip()[:80]}")
    assert not offenders, (
        "Q-format magic literals in engine hot paths - use the constants "
        "from symcrash_python_ap.primitives (or mark a reviewed non-Q15 "
        f"number with `# {MARKER}`):\n" + "\n".join(offenders))
