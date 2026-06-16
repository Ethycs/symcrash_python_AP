"""Merge fixed.py into primitives.py; rewrite fixed.py as a re-export shim.

Historical one-shot migration (handoff packet E8, step 1). Executed
2026-06-11; result committed in 748654f. NOT idempotent and not meant to
re-run (the merge has already happened) - retained as the executable
record of how primitives.py absorbed the WSE-STD-001 arithmetic layer.
Run from repo root (paths are CWD-relative).

Deterministic text splice:
- primitives.py gains the WSE-STD-001 arithmetic layer (everything in
  fixed.py from the SATURATION section down), plus the STD-001 constant
  names aliased onto the existing primitives constants.
- fixed.py becomes an explicit re-export shim (no behavior change for
  its 33 importers).
"""
from pathlib import Path

PRIM = Path("src/symcrash_python_ap/primitives.py")
FIXED = Path("src/symcrash_python_ap/fixed.py")

prim = PRIM.read_bytes().decode("utf-8")
fixed = FIXED.read_bytes().decode("utf-8")

# ---- extract the arithmetic layer from fixed.py (SATURATION section -> EOF)
marker = "# =============================================================================\n# SATURATION"
idx = fixed.index(marker)
arith = fixed[idx:].rstrip() + "\n"

# ---- constants bridge: STD-001 names aliased onto primitives' constants
bridge = '''
# =============================================================================
# FIXED-POINT ARITHMETIC LAYER (WSE-STD-001, merged from fixed.py 2026-06-11)
# =============================================================================
# fixed.py is now a re-export shim; this module is the single canonical
# source for Q-format constants and bit-exact arithmetic primitives.

# Q1.15 format constants (WSE-STD-001 Section 3)
Q15_SHIFT = 15
ONE_Q15 = Q15_ONE             # Canonical STD-001 name (0x7FFF)
Q15_MINUS_ONE = Q15_NEG_ONE   # STD-001 name for -1.0 (-32768)
Q15_3_4 = 24576               # 0.75 in Q15
ZERO = 0

# Integer bounds (STD-001 Section 2)
MIN_S16 = -32768
MAX_S16 = 32767
INT16_MIN = MIN_S16           # Alias for compatibility
INT16_MAX = MAX_S16           # Alias for compatibility
INT32_MAX = 2147483647
INT32_MIN = -2147483648


'''

# ---- splice into primitives.py just before the EXPORTS section
exports_marker = "# =============================================================================\n# EXPORTS"
assert exports_marker in prim
prim = prim.replace(exports_marker, bridge.lstrip("\n") + arith + "\n\n" + exports_marker)

# ---- imports: numpy (for LUT builders) + Tuple (used by REGIME_SYMBOL_OPS)
prim = prim.replace(
    "from enum import IntEnum, Enum\nfrom typing import Dict, List, NamedTuple",
    "from enum import IntEnum, Enum\nfrom typing import Dict, List, NamedTuple, Tuple\n\nimport numpy as np",
)

# ---- extend __all__
ARITH_NAMES = [
    'Q15_SHIFT', 'ONE_Q15', 'Q15_MINUS_ONE', 'Q15_3_4', 'ZERO',
    'MIN_S16', 'MAX_S16', 'INT16_MIN', 'INT16_MAX', 'INT32_MAX', 'INT32_MIN',
    'SAT16', 'SAT32', 'sat16', 'sat32',
    'RSHIFT_RNE', 'shift_round', 'shift_round_sat16',
    'MUL_Q15', 'MUL_Q15_TO_Q31', 'mac_q15', 'mul_q15', 'mul_q15_sat',
    'ADD_Q15', 'MIX_Q15', 'ABS16',
    'slew_q15', 'slew_q15_asym',
    'tanh_q15', 'tanh_q15_interp', 'sqrt_q15', 'soft_clip_q15',
    'vec_mul_q15', 'vec_add_sat', 'vec_scale_q15', 'vec_dot_q15',
    'PITCH_SAMPLE_RATE', 'PITCH_PHASE_BITS',
    'midi_to_phase_inc_fixed', 'midi_to_freq_mhz', 'midi_to_freq_hz',
]
all_addition = (
    "    # Fixed-point arithmetic layer (WSE-STD-001, merged from fixed.py)\n"
    + "".join(f"    '{n}',\n" for n in ARITH_NAMES)
)
prim = prim.replace(
    "    'REGIME_SYMBOL_OPS', 'get_operators_for_state', 'compute_timbral_mode_from_intent',\n]",
    "    'REGIME_SYMBOL_OPS', 'get_operators_for_state', 'compute_timbral_mode_from_intent',\n"
    + all_addition + "]",
)

PRIM.write_bytes(prim.encode("utf-8"))

# ---- fixed.py shim
shim_names = ARITH_NAMES + ['Q15_ONE', 'Q15_HALF', 'Q15_SCALE']
shim = '''"""
Fixed-Point Arithmetic Layer — RE-EXPORT SHIM
==============================================

The WSE-STD-001 arithmetic layer was merged into primitives.py
(2026-06-11, E8 constant-hygiene consolidation). primitives.py is the
single canonical source for Q-format constants and bit-exact
arithmetic. This shim preserves the historical import surface:

    from symcrash_python_ap.fixed import RSHIFT_RNE, SAT16, Q15_ONE

New code should import from symcrash_python_ap.primitives directly.
"""

from .primitives import (  # noqa: F401
'''
for n in sorted(shim_names):
    shim += f"    {n},\n"
shim += ")\n\n__all__ = [\n"
for n in sorted(shim_names):
    shim += f"    '{n}',\n"
shim += "]\n"

FIXED.write_bytes(shim.encode("utf-8"))
print("merged: primitives.py", len(prim.splitlines()), "lines; fixed.py shim",
      len(shim.splitlines()), "lines")
