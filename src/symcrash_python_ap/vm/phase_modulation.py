"""WSE-STD-210 8.3: Q15 modulation and depth map to full-turn Q32 phase."""
import numpy as np
from ..primitives import Q15_NEG_ONE, Q15_ONE


def phase_modulation_q32(sample_q15: int, depth_q15: int) -> int:
    # Q15 * Q15 is Q30 turns. Q32 turns needs two more fractional bits.
    # Keep the signed product wide; wrapping belongs to the phase accumulator.
    return int(sample_q15) * int(depth_q15) * 4


def modulation_lanes_q15(samples, lanes: int) -> np.ndarray:
    """L1 sampling: first K samples; a missing suffix is silence, never wrap."""
    if isinstance(samples, (list, tuple)) and any(isinstance(x, (bool, np.bool_)) for x in samples):
        raise ValueError('FM modulator samples must be integer Q15 values, not booleans')
    values = np.asarray(samples)
    if values.ndim != 1 or (values.size and values.dtype.kind not in 'iu'):
        raise ValueError('FM modulator must be a one-dimensional integer Q15 buffer')
    if values.size and (np.any(values < Q15_NEG_ONE) or np.any(values > Q15_ONE)):
        raise ValueError('FM modulator samples must be signed Q15 values')
    result = np.zeros(lanes, dtype=np.int16)
    n = min(lanes, len(values))
    result[:n] = values[:n]
    return result
