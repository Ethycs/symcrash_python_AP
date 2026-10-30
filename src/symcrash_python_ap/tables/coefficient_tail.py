"""Python coefficient-tail profile; all runtime parameters are integers.

Energy is the sum of squared Q15 samples (Q30), without an implicit shift.
The legacy SymbolTail caps and the CCS caps use different normalization.
CCS retains its established RMS headroom: its default K=4 view has 64 leaves
with four temporal slots, independent of the 32-node operator mask.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from ..primitives import Q15_ONE, Q15_HALF, Q15_SCALE

E_LEAF_MAX = 0x40000000
E_FINE_MAX = 0x30000000
E_GLOBAL_MAX = 0x7FFFFFFF

CCS_MAX_RMS_Q15 = 23170
CCS_E_LEAF_MAX = 4 * Q15_SCALE**2
CCS_E_FINE_MAX = 4 * CCS_E_LEAF_MAX
CCS_E_GLOBAL_MAX = 256 * CCS_MAX_RMS_Q15**2
MAX_ENERGY_BUDGET = 512 * Q15_SCALE**2

# Squared amplitude profiles from operator_library.build_normalize_scale.
# Common integer denominators cancel when distributing the energy budget.
CURVE_ENERGY_WEIGHTS = {
    "flat": (1, 1, 1, 1, 1, 1),
    "pink": (3600, 900, 400, 225, 144, 100),
    "bright": (25, 36, 49, 64, 81, 100),
}
LEAF_BANDS = ((0, 2), (2, 6), (6, 14), (14, 26), (26, 42), (42, 64))
FINE_LEAF_START = 60


@dataclass(frozen=True)
class CoefficientTailPolicy:
    normalize_curve: str = "pink"
    fineclamp_cap_ratio_q15: int = Q15_HALF
    slew_max_delta: int | None = None
    leaf_energy_max: int = CCS_E_LEAF_MAX
    fine_energy_max: int = CCS_E_FINE_MAX
    global_energy_max: int = CCS_E_GLOBAL_MAX
    # Disjoint leaf pairs: corresponding temporal slots must agree.
    sheaf_pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if self.normalize_curve not in CURVE_ENERGY_WEIGHTS:
            raise ValueError("normalize_curve must be flat, pink, or bright")
        for name, upper in (("fineclamp_cap_ratio_q15", Q15_ONE),
                            ("leaf_energy_max", MAX_ENERGY_BUDGET),
                            ("fine_energy_max", MAX_ENERGY_BUDGET),
                            ("global_energy_max", MAX_ENERGY_BUDGET)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= upper:
                raise ValueError(f"{name} must be an integer in 0..{upper}")
        if self.slew_max_delta is not None and (
            type(self.slew_max_delta) is not int or not 0 <= self.slew_max_delta <= 0xFFFF
        ):
            raise ValueError("slew_max_delta must be None or an integer in 0..65535")
        seen = set()
        if type(self.sheaf_pairs) is not tuple:
            raise ValueError("sheaf_pairs must be an immutable tuple")
        for pair in self.sheaf_pairs:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("each sheaf pair must be a tuple of two leaves")
            for leaf in pair:
                if type(leaf) is not int or not 0 <= leaf < 64 or leaf in seen:
                    raise ValueError("sheaf pairs must contain distinct leaves in 0..63")
                seen.add(leaf)


DEFAULT_TAIL_POLICY = CoefficientTailPolicy()


@lru_cache(maxsize=128)
def compile_tail_policy(normalize_curve: str, fineclamp_cap_ratio: float,
                        slew_max_delta: int | None) -> CoefficientTailPolicy:
    """Quantize authoring parameters once; the tail itself never uses floats."""
    if not math.isfinite(fineclamp_cap_ratio) or not 0 <= fineclamp_cap_ratio <= 1:
        raise ValueError("fineclamp_cap_ratio must be finite and in 0..1")
    return CoefficientTailPolicy(
        normalize_curve=normalize_curve,
        fineclamp_cap_ratio_q15=min(Q15_ONE, round(fineclamp_cap_ratio * Q15_SCALE)),
        slew_max_delta=slew_max_delta,
    )
