"""
Noise Exciter (Packet E4 - Acoustic Attack Transients)
======================================================

Deterministic, seeded broadband noise source for acoustic excitation
(bow scrape, breath, mallet contact, pick). Q15 integer output,
block-based, mixed into the voice output pre-gain in vm_step_block.

Design (WSE-STD conventions):
- Counter-based PRNG: splitmix64 of (seed_base + absolute_sample_index).
  No global RNG state; the value of sample n depends only on the preset
  seed (meta.seed_u32), voice_id, and n - so repeated renders and
  snapshot/restore replays are bit-identical. seed_base is derived via
  prng.hash64_mix (WSE-STD-040 Section 4.3).
- Q15 conversion follows WSE-STD-040 Section 9: (u32 >> 17) - 0x4000,
  yielding white noise in [-16384, 16383].
- Colors:
    WHITE (0): raw counter noise (flat spectrum)
    PINK  (1): fixed-point Paul Kellett 3-pole approximation (~-3 dB/oct)
    BROWN (2): leaky integrator (~-6 dB/oct)
  Pink/brown filters carry integer state across blocks (NoiseExciterState)
  so there are no block-rate restart artifacts; the state is part of the
  VMState snapshot for deterministic replay.
- Envelope: linear attack then linear decay (block-rate breakpoints,
  per-sample interpolated inside the block - no zipper steps). One-shot,
  anchored at note-on (block_index 0).

All arithmetic is integer (Python ints / numpy int64), bit-exact across
the JIT and reference VM paths: the exciter runs outside the per-sample
hot-path kernels, so one implementation serves both.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .tables import BLOCK_SAMPLES
from ..primitives import Q15_ONE
from ..prng import hash64_mix

# Noise colors (matches instrument_ir.NoiseColor order)
NOISE_WHITE = 0
NOISE_PINK = 1
NOISE_BROWN = 2

# splitmix64 constants (Steele/Lea/Flood; standard counter-based mixer)
_SM64_GAMMA = 0x9E3779B97F4A7C15
_SM64_M1 = 0xBF58476D1CE4E5B9
_SM64_M2 = 0x94D049BB133111EB
_U64_MASK = 0xFFFFFFFFFFFFFFFF

# Domain tag mixed into the seed so the exciter stream never collides with
# other consumers of meta.seed_u32 ("NoisExct" in ASCII).
_NOISE_DOMAIN_TAG = 0x4E6F6973_45786374

# Paul Kellett "economy" pink filter coefficients, Q15 fixed point:
#   b0 = 0.99765*b0 + w*0.0990460
#   b1 = 0.96300*b1 + w*0.2965164
#   b2 = 0.57000*b2 + w*1.0526913
#   pink = b0 + b1 + b2 + w*0.1848
_PINK_A0, _PINK_G0 = 32691, 3246    # 0.99765, 0.0990460
_PINK_A1, _PINK_G1 = 31556, 9716    # 0.96300, 0.2965164
_PINK_A2, _PINK_G2 = 18678, 34495   # 0.57000, 1.0526913
_PINK_GW = 6056                     # 0.1848

# Brown leaky integrator: acc += w; acc -= acc >> LEAK; out = acc >> OUT
_BROWN_LEAK_SHIFT = 7   # leak time constant ~128 samples (~2.7 ms @ 48k)
_BROWN_OUT_SHIFT = 4


@dataclass
class NoiseExciterState:
    """Per-voice colored-noise filter state (deterministic, snapshotable)."""
    pink_b0: int = 0
    pink_b1: int = 0
    pink_b2: int = 0
    brown_acc: int = 0

    def reset(self) -> None:
        """Reset filter state (note-on)."""
        self.pink_b0 = 0
        self.pink_b1 = 0
        self.pink_b2 = 0
        self.brown_acc = 0

    def to_dict(self) -> dict:
        """Serialize for VMState snapshot."""
        return {
            'pink_b0': self.pink_b0,
            'pink_b1': self.pink_b1,
            'pink_b2': self.pink_b2,
            'brown_acc': self.brown_acc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NoiseExciterState':
        """Deserialize from VMState snapshot."""
        return cls(
            pink_b0=data.get('pink_b0', 0),
            pink_b1=data.get('pink_b1', 0),
            pink_b2=data.get('pink_b2', 0),
            brown_acc=data.get('brown_acc', 0),
        )


def noise_seed_base(seed_u32: int, voice_id: int) -> int:
    """
    Derive the 64-bit counter base from preset seed and voice id.

    Per WSE-STD-040 Section 4.3 (HASH64 seeding): deterministic and
    platform-independent, no floating point.
    """
    return hash64_mix(seed_u32 & 0xFFFFFFFF,
                      (_NOISE_DOMAIN_TAG ^ (voice_id & 0xFFFFFFFF)) & _U64_MASK)


def white_block_q15(seed_base_u64: int, block_index: int) -> np.ndarray:
    """
    Generate one block of counter-based white noise, Q15 in [-16384, 16383].

    Value of sample n = splitmix64(seed_base + n*GAMMA) depends only on
    (seed_base, n): blocks can be generated in any order or skipped without
    affecting later blocks.
    """
    n0 = block_index * BLOCK_SAMPLES
    ctr = np.arange(n0, n0 + BLOCK_SAMPLES, dtype=np.uint64)
    z = np.uint64(seed_base_u64 & _U64_MASK) + ctr * np.uint64(_SM64_GAMMA)
    z = z ^ (z >> np.uint64(30))
    z = z * np.uint64(_SM64_M1)
    z = z ^ (z >> np.uint64(27))
    z = z * np.uint64(_SM64_M2)
    z = z ^ (z >> np.uint64(31))
    u32 = (z >> np.uint64(32)).astype(np.int64)
    # WSE-STD-040 Section 9: q15 = (r >> 17) - 0x4000
    return ((u32 >> 17) - 0x4000).astype(np.int32)


def _color_filter_block(ns: NoiseExciterState, color: int,
                        white: np.ndarray) -> np.ndarray:
    """
    Apply the color filter to one white block. Integer math only;
    pink/brown carry state in `ns` across blocks (phase-continuous).
    """
    if color == NOISE_PINK:
        b0, b1, b2 = ns.pink_b0, ns.pink_b1, ns.pink_b2
        out = np.empty(BLOCK_SAMPLES, dtype=np.int32)
        for i in range(BLOCK_SAMPLES):
            w = int(white[i])
            b0 = (b0 * _PINK_A0 + w * _PINK_G0) >> 15
            b1 = (b1 * _PINK_A1 + w * _PINK_G1) >> 15
            b2 = (b2 * _PINK_A2 + w * _PINK_G2) >> 15
            pink = b0 + b1 + b2 + ((w * _PINK_GW) >> 15)
            # Normalize (sum of poles ~2.3x input RMS) and clamp to Q15
            pink >>= 1
            out[i] = max(-Q15_ONE, min(Q15_ONE, pink))
        ns.pink_b0, ns.pink_b1, ns.pink_b2 = b0, b1, b2
        return out

    if color == NOISE_BROWN:
        acc = ns.brown_acc
        out = np.empty(BLOCK_SAMPLES, dtype=np.int32)
        for i in range(BLOCK_SAMPLES):
            acc += int(white[i])
            acc -= acc >> _BROWN_LEAK_SHIFT
            v = acc >> _BROWN_OUT_SHIFT
            out[i] = max(-Q15_ONE, min(Q15_ONE, v))
        ns.brown_acc = acc
        return out

    # WHITE (default)
    return white


def noise_env_q15(block_index: int, attack_blocks: int,
                  decay_blocks: int) -> int:
    """
    One-shot exciter envelope value at a block boundary (Q15).

    Linear attack over attack_blocks, then linear decay to 0 over
    decay_blocks. Anchored at note-on (block_index 0).
    """
    if block_index < 0:
        return 0
    attack = max(0, attack_blocks)
    if attack > 0 and block_index < attack:
        return (block_index * Q15_ONE) // attack
    if decay_blocks <= 0:
        return 0
    d = block_index - attack
    if d >= decay_blocks:
        return 0
    return ((decay_blocks - d) * Q15_ONE) // decay_blocks


def render_noise_block(
    ns: NoiseExciterState,
    seed_base_u64: int,
    block_index: int,
    color: int,
    level_q15: int,
    attack_blocks: int,
    decay_blocks: int,
) -> Optional[np.ndarray]:
    """
    Render one enveloped, leveled noise block.

    Returns an int32 array of BLOCK_SAMPLES Q15 samples ready to sum into
    the voice output, or None when the envelope is zero for the entire
    block (the exciter contributes nothing - callers skip the mix, which
    preserves bit-exactness and costs nothing after the transient).
    """
    if level_q15 <= 0:
        return None

    env_a = noise_env_q15(block_index, attack_blocks, decay_blocks)
    env_b = noise_env_q15(block_index + 1, attack_blocks, decay_blocks)
    if env_a == 0 and env_b == 0:
        return None

    white = white_block_q15(seed_base_u64, block_index)
    colored = _color_filter_block(ns, color, white).astype(np.int64)

    # Per-sample linear envelope interpolation across the block
    # (BLOCK_SAMPLES = 64 = 1 << 6)
    idx = np.arange(BLOCK_SAMPLES, dtype=np.int64)
    env = env_a + (((env_b - env_a) * idx) >> 6)

    scaled = ((colored * env) >> 15) * level_q15 >> 15
    return scaled.astype(np.int32)


__all__ = [
    'NOISE_WHITE',
    'NOISE_PINK',
    'NOISE_BROWN',
    'NoiseExciterState',
    'noise_seed_base',
    'white_block_q15',
    'noise_env_q15',
    'render_noise_block',
]
