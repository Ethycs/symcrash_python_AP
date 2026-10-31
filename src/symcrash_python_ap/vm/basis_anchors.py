"""Fixed Mode B pitch-anchor selection and bounded coefficient interpolation.

The phase accumulator always uses the accepted Q32 pitch. These anchors
approximate only the within-block basis response (WSE-STD-210 section 5).
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit
from numba.extending import register_jitable

from ..pitch_rom import MIDI_CENT_PHASE_Q32
from ..primitives import Q15_NEG_ONE, Q15_ONE
from .carrier_spectrum import harmonic_limit


ANCHORS = MIDI_CENT_PHASE_Q32[::100]
WEIGHT_BITS = 30
WEIGHT_ONE = 1 << WEIGHT_BITS


def anchor_span(cent):
    """Four fixed neighboring anchors, including one-sided endpoint stencils."""
    return min(max(cent//100-1, 0), len(ANCHORS)-4)


@lru_cache(maxsize=1)
def anchor_capacities():
    # An anchor may be the uppermost point in a four-point stencil. Retain
    # every harmonic that any legal target of that stencil can request.
    result = [0]*len(ANCHORS)
    for cent, pitch in enumerate(MIDI_CENT_PHASE_Q32):
        first = anchor_span(cent)
        capacity = harmonic_limit(pitch)
        for i in range(first, first+4):
            result[i] = max(result[i], capacity)
    return dict(zip(ANCHORS, result))


def capacity_for_anchor(pitch):
    return anchor_capacities().get(pitch, harmonic_limit(pitch))


def frequency_directory(keys):
    """Reuse equal-frequency harmonics; offsets are int16 words in basis data."""
    by_frequency = {}
    offset = 0
    for pitch in keys:
        count = capacity_for_anchor(pitch)
        for h in range(1,count+1):
            by_frequency.setdefault(pitch*h, (offset+(h-1)*256, offset+(count+h-1)*256))
        offset += count*512
    return tuple((frequency,*by_frequency[frequency]) for frequency in sorted(by_frequency))


def interpolation_grid(directory):
    """Coalesce sub-millihertz duplicates only for selecting stable stencils.

    Exact entries remain separately addressable. Multiplying a rounded pitch
    can differ by a few Q32 units from its octave anchor; these nearly equal
    nodes must not form a numerically singular interpolation interval.
    """
    result = []
    for i, (frequency, *_) in enumerate(directory):
        if not result or frequency-directory[result[-1]][0] >= 64:
            result.append(i)
    return tuple(result)


def frequency_stencil(frequency, grid):
    """Bracket the target; keep outer nodes at least one inner span away."""
    left = min(max(bisect_right(grid,frequency)-1,0),len(grid)-2)
    if left == 0:
        return (0,1,2,3)
    if left == len(grid)-2:
        return tuple(range(len(grid)-4,len(grid)))
    width = grid[left+1]-grid[left]
    outer_left = max(0,bisect_right(grid,grid[left]-width)-1)
    outer_right = min(len(grid)-1,bisect_left(grid,grid[left+1]+width))
    return (outer_left,left,left+1,outer_right)


@register_jitable
def _divide_even(numerator, denominator):
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    quotient,remainder = divmod(numerator,denominator)
    return quotient + (2*remainder > denominator or
                       (2*remainder == denominator and quotient%2 != 0))


def weights_for_frequency(frequency, nodes):
    """Fixed Q30 ratios and products; each division/product rounds to even."""
    weights=[]
    for i,node in enumerate(nodes):
        weight=WEIGHT_ONE
        for j,other in enumerate(nodes):
            if i != j:
                ratio=_divide_even((frequency-other)*WEIGHT_ONE,node-other)
                weight=_divide_even(weight*ratio,WEIGHT_ONE)
        weights.append(weight)
    largest=0
    for i in range(1,4):
        if abs(weights[i]) > abs(weights[largest]):
            largest=i
    weights[largest] += WEIGHT_ONE-sum(weights)
    return weights


compiled_weights_for_frequency = njit(cache=True)(weights_for_frequency)


@njit(cache=True)
def compiled_descriptor_arrays(pitch, count, directory, grid_indices, grid):
    """Bit-exact fixed-width port of PreparedBasisAtlas's descriptor loop."""
    offsets = np.empty((count,4,2),dtype=np.int32)
    weights = np.zeros((count,4),dtype=np.int32)
    frequencies = directory[:,0]
    for h in range(count):
        frequency = pitch*(h+1)
        exact = np.searchsorted(frequencies,frequency)
        if exact < len(frequencies) and frequencies[exact] == frequency:
            for j in range(4):
                offsets[h,j,0] = directory[exact,1]
                offsets[h,j,1] = directory[exact,2]
            weights[h,0] = 1<<30
            continue
        left = min(max(np.searchsorted(grid,frequency,side='right')-1,0),len(grid)-2)
        if left == 0:
            stencil = (0,1,2,3)
        elif left == len(grid)-2:
            stencil = (len(grid)-4,len(grid)-3,len(grid)-2,len(grid)-1)
        else:
            width = np.int64(grid[left+1])-np.int64(grid[left])
            outer_left = max(0,np.searchsorted(grid,np.int64(grid[left])-width,side='right')-1)
            outer_right = min(len(grid)-1,np.searchsorted(grid,np.int64(grid[left+1])+width))
            stencil = (outer_left,left,left+1,outer_right)
        nodes = (np.int64(grid[stencil[0]]),np.int64(grid[stencil[1]]),
                 np.int64(grid[stencil[2]]),np.int64(grid[stencil[3]]))
        values = compiled_weights_for_frequency(frequency,nodes)
        for j in range(4):
            weights[h,j] = values[j]
            offsets[h,j,0] = directory[grid_indices[stencil[j]],1]
            offsets[h,j,1] = directory[grid_indices[stencil[j]],2]
    return offsets,weights


def pitch_index(phase_inc):
    index = bisect_left(MIDI_CENT_PHASE_Q32, phase_inc)
    if index == len(MIDI_CENT_PHASE_Q32) or MIDI_CENT_PHASE_Q32[index] != phase_inc:
        raise ValueError('Pitch increment is absent from the prepared basis domain')
    return index


def interpolate_arrays(sources, weights):
    """Diagnostic/reference materialization; the audio kernel streams scalars."""
    total = np.zeros(sources[0].shape, dtype=np.int64)
    for source, weight in zip(sources, weights):
        total += source.astype(np.int64)*int(weight)
    quotient, remainder = total >> WEIGHT_BITS, total & (WEIGHT_ONE-1)
    rounded = quotient + ((remainder > WEIGHT_ONE//2) |
        ((remainder == WEIGHT_ONE//2) & ((quotient & 1) != 0)))
    result = np.clip(rounded, Q15_NEG_ONE, Q15_ONE).astype(np.int16)
    return np.frombuffer(result.tobytes(), dtype=np.int16).reshape(result.shape)


@dataclass(frozen=True)
class InterpolatedBasis:
    """Small immutable descriptor referencing mapped anchors; no copied bank."""
    DC_frame: np.ndarray
    coefficients: np.ndarray
    source_offsets: np.ndarray
    interpolation_weights: np.ndarray
    H_max: int
    anchor_note: int
    anchor_freq_hz: float

    @property
    def Ucos_h(self):
        return self._materialize(0)

    @property
    def Usin_h(self):
        return self._materialize(1)

    def _materialize(self, component):
        result=np.empty((self.H_max,64,4),dtype=np.int16)
        for h in range(self.H_max):
            sources=[self.coefficients[int(offset):int(offset)+256].reshape(64,4)
                     for offset in self.source_offsets[h,:,component]]
            result[h]=interpolate_arrays(sources,self.interpolation_weights[h])
        return np.frombuffer(result.tobytes(),dtype=np.int16).reshape(result.shape)
