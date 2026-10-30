"""Bounded diagonal pressure gains from the authored operator defaults.

PressureRamp (15): +10% in bands 2/3, -3% elsewhere.
FoldPressure (20): +30% in band 2, +15% in neighbors, -6% elsewhere.
These are gain deltas in signed Q15, allowing gains above unity. They are
linear operators before the mandatory saturating coefficient write.
"""
from ..operator_library import BAND_SLICES

PRESSURE_BAND_DELTAS_Q15 = (-983, -983, 3277, 3277, -983, -983)
FOLD_BAND_DELTAS_Q15 = (-1966, 4915, 9830, 4915, -1966, -1966)
PRESSURE_NODE_DELTAS_Q15 = tuple(
    delta for (start, end), delta in zip(BAND_SLICES, PRESSURE_BAND_DELTAS_Q15)
    for _ in range(start // 8, end // 8)
)
FOLD_NODE_DELTAS_Q15 = tuple(
    delta for (start, end), delta in zip(BAND_SLICES, FOLD_BAND_DELTAS_Q15)
    for _ in range(start // 8, end // 8)
)
