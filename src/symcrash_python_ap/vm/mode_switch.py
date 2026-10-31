"""Fixed four-serviced-block coefficient crossfade for WSE-STD-210 section 9."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..primitives import RSHIFT_RNE, Q16_ONE, Q15_SCALE, Q15_NEG_ONE, Q15_ONE


FADE_BLOCKS = 4
MIX_ONE = Q16_ONE


@dataclass
class ModeSwitchState:
    initialized: bool = False
    mix_b_q16: int = 0
    start_mix_b_q16: int = 0
    remaining_blocks: int = 0
    b_harmonic_gains: Optional[tuple] = None
    b_gain_bound: int = Q15_SCALE
    b_fm_source: int = 0

    def step(self, mode: int, previous_mode: int, *, starting: bool) -> bool:
        """Latch a target and advance its fade; return whether phase must hand off."""
        if mode not in (0, 1):
            self.initialized = False
            self.remaining_blocks = 0
            return False
        target = mode * MIX_ONE
        if starting or not self.initialized or previous_mode not in (0, 1):
            self.initialized = True
            self.mix_b_q16 = self.start_mix_b_q16 = target
            self.remaining_blocks = 0
            return False
        switched = previous_mode != mode
        if switched:
            self.start_mix_b_q16 = self.mix_b_q16
            self.remaining_blocks = FADE_BLOCKS
        if self.remaining_blocks:
            self.remaining_blocks -= 1
            elapsed = FADE_BLOCKS - self.remaining_blocks
            self.mix_b_q16 = RSHIFT_RNE(
                self.start_mix_b_q16 * (FADE_BLOCKS-elapsed) + target * elapsed, 2)
        return switched

    def validate(self, mode=None):
        """Reject malformed replay state before adopting it."""
        from .spectral_modulation import validate_harmonic_gains
        if type(self.initialized) is not bool:
            raise ValueError('Invalid mode-switch initialization flag')
        for value, high in ((self.mix_b_q16, MIX_ONE), (self.start_mix_b_q16, MIX_ONE),
                            (self.remaining_blocks, FADE_BLOCKS-1), (self.b_fm_source, 2)):
            if type(value) is not int or not 0 <= value <= high:
                raise ValueError('Invalid mode-switch state')
        validate_harmonic_gains(self.b_harmonic_gains, self.b_gain_bound)
        if not self.initialized and self.remaining_blocks:
            raise ValueError('Uninitialized mode switch cannot have a pending fade')
        if mode is not None and self.initialized:
            if type(mode) is not int or mode not in (0, 1):
                raise ValueError('Initialized mode switch requires Mode A or B')
            elapsed = FADE_BLOCKS - self.remaining_blocks
            expected = RSHIFT_RNE(self.start_mix_b_q16*self.remaining_blocks
                                 + mode*MIX_ONE*elapsed, 2)
            if self.mix_b_q16 != expected:
                raise ValueError('Mode-switch mix disagrees with target and fade progress')


def blend_coefficients(a, b, mix_b_q16):
    """One wide weighted sum, signed RNE, exact endpoints and no Q15 gain loss."""
    weighted = a.astype(np.int64)*(MIX_ONE-mix_b_q16) + b.astype(np.int64)*mix_b_q16
    whole = weighted >> 16
    remainder = weighted & (MIX_ONE-1)
    rounded = whole + ((remainder > MIX_ONE//2) |
                       ((remainder == MIX_ONE//2) & ((whole & 1) != 0)))
    return np.clip(rounded, Q15_NEG_ONE, Q15_ONE).astype(np.int16)
