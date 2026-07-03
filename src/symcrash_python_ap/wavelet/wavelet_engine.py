"""
Wavelet Engine
==============

LEGACY - pre-VM path (quarantined 2026-07-02). Superseded by
vm/executor.vm_step_block, the single realization kernel (see
PIPELINE-ARCHITECTURE-GAP.md "Unified Execution Layer"). Kept for the
legacy integration tests and FPGA reference material only; not part of
the preset story. Do not add new consumers.

Renders VoiceFrames to audio via inverse wavelet packet synthesis.
Uses CDF 5/3 lifting with 5 tree types.

ARCHITECTURAL NOTE:
All coefficient changes are slewed or micro-interpolated HERE, not in Safety.
Safety handles block-boundary OLA only; musical transitions are Wavelet Engine's job.

The Regularity Operator R provides:
- Event-local micro-bridge: C¹ cosine ramp at coefficient change points
- Slew limiting on audio output for any residual discontinuities
"""

from typing import List, Dict, Tuple, Optional
import math
import numpy as np

from ..fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)
from ..primitives import Q15_SCALE
from ..frames import VoiceFrame
from .wavelet_packet_trees import PacketTreeSet, BLOCK_SIZE, LEAF_COUNT, NODE_SIZE


# =============================================================================
# REGULARITY OPERATOR R (WSE-STD-070)
# =============================================================================
#
# The Regularity Operator projects audio onto a bounded-variation set.
# Clicks are impulses; impulses are unbounded derivatives.
#
# R consists of two components:
# 1. Event-local micro-bridge: C¹ cosine ramp at known change points
# 2. Slope projection fallback: symmetric slew limiter for any remaining spikes
#
# This lives in Wavelet Engine because it handles MUSICAL transitions
# (note changes, coefficient morphs, tree blends), NOT block boundaries.

# Constants
R_BRIDGE_LEN = 16         # Samples for micro-bridge (event-local)
R_SLEW_MAX = 0x2000       # Maximum sample-to-sample change for audio
R_CURVATURE_MAX = 0x1000  # Maximum second difference (curvature)

# Microfade constants (per WSE-STD-070 Section 4)
MICROFADE_LEN = 32        # Samples for allocation microfade (fade-in from 0)

# Precomputed leaf envelope for coefficient expansion
# sin((coeff_idx + 0.5) / NODE_SIZE * pi) for NODE_SIZE=4
# This avoids 256 math.sin calls per block per voice
_LEAF_ENVELOPE_Q15 = [
    int(round(math.sin((i + 0.5) / 4 * math.pi) * Q15_ONE))
    for i in range(4)  # NODE_SIZE = 4
]


def _build_cosine_bridge_lut(n: int) -> List[int]:
    """
    Build C¹ cosine bridge LUT for micro-bridge.

    The bridge function w(t) satisfies:
    - w(0) = 0, w(1) = 1 (amplitude continuity)
    - w'(0) = 0, w'(1) = 0 (derivative continuity at endpoints)

    Uses smoothstep: w(t) = 3t² - 2t³
    """
    lut = []
    for i in range(n + 1):
        t = i / n
        # Smoothstep: 3t² - 2t³ (C¹ continuous)
        w = 3 * t * t - 2 * t * t * t
        lut.append(int(round(w * Q15_ONE)))
    return lut


# Precomputed micro-bridge LUT
_BRIDGE_LUT = _build_cosine_bridge_lut(R_BRIDGE_LEN)


def micro_bridge(
    old_samples: List[int],
    new_samples: List[int],
    bridge_len: int = R_BRIDGE_LEN,
) -> List[int]:
    """
    Apply C¹-continuous micro-bridge between old and new generator states.

    This is the event-local regularity fix: when something changes at a known
    sample index, we blend old->new using a smoothstep that matches both value
    and derivative at the endpoints.

    Args:
        old_samples: Samples from the "old" state (at least bridge_len)
        new_samples: Samples from the "new" state (at least bridge_len)
        bridge_len: Number of samples for the transition

    Returns:
        Blended samples (length = bridge_len)
    """
    if len(old_samples) < bridge_len or len(new_samples) < bridge_len:
        # Not enough context, fall back to simple blend
        bridge_len = min(len(old_samples), len(new_samples), bridge_len)
        if bridge_len == 0:
            return []

    result = []
    for i in range(bridge_len):
        # w goes from 0 to Q15_ONE over bridge_len
        if bridge_len == R_BRIDGE_LEN:
            w = _BRIDGE_LUT[i]
        else:
            t = i / (bridge_len - 1) if bridge_len > 1 else 1.0
            w = int((3 * t * t - 2 * t * t * t) * Q15_ONE)

        inv_w = Q15_ONE - w

        # Blend: out = old * (1-w) + new * w
        out = (old_samples[i] * inv_w + new_samples[i] * w) >> 15
        result.append(sat16(out))

    return result


def slope_project(
    samples: List[int],
    prev_sample: int = 0,
    max_slew: int = R_SLEW_MAX,
) -> Tuple[List[int], int]:
    """
    Project audio onto bounded-slope set (slew limiter).

    GUARANTEE: Output always satisfies |x[n] - x[n-1]| <= max_slew.

    Args:
        samples: Audio samples
        prev_sample: Last sample from previous block
        max_slew: Maximum |x[n] - x[n-1]|

    Returns:
        (limited_samples, last_sample)
    """
    if not samples:
        return [], prev_sample

    out = []
    prev = prev_sample
    for s in samples:
        delta = s - prev
        if delta > max_slew:
            limited = prev + max_slew
        elif delta < -max_slew:
            limited = prev - max_slew
        else:
            limited = s
        out.append(limited)
        prev = limited

    return out, out[-1] if out else prev_sample


class RegularityOperator:
    """
    Universal Regularity Operator R for the Wavelet Engine.

    Enforces bounded variation (no impulses) through:
    1. Event-local micro-bridge at known change points (coefficient morphs)
    2. Slope projection fallback for unknown/residual discontinuities
    3. Microfade on voice allocation (per WSE-STD-070 Section 4)

    Voice Lifecycle (WSE-STD-070):
    - voice_allocate(): New owner of voice slot. Arms microfade, sets prev_sample.
    - voice_on(): Retrigger/gate rise. NEVER resets state.
    - voice_free(): Voice becomes inactive. Keeps last sample state.

    Usage in render path:
        R = RegularityOperator()

        # On new voice allocation:
        R.allocate(last_known_sample)  # Arms microfade

        # When coefficients change (note, tree, morph):
        audio = R.apply_transition(old_audio, new_audio, change_idx)

        # Always apply slope projection as final safety:
        audio = R.finalize(audio)  # Also applies microfade if armed
    """

    def __init__(
        self,
        bridge_len: int = R_BRIDGE_LEN,
        max_slew: int = R_SLEW_MAX,
        microfade_len: int = MICROFADE_LEN,
    ):
        self.bridge_len = bridge_len
        self.max_slew = max_slew
        self.microfade_len = microfade_len
        self.prev_sample = 0

        # Microfade state: armed on voice_allocate, decremented each sample
        self._microfade_counter = 0

    def allocate(self, last_sample: int = 0) -> None:
        """
        Signal new voice allocation (new logical owner of this voice slot).

        Per WSE-STD-070 Section 4:
        - Sets prev_sample to last known output (or 0)
        - Arms microfade to prevent clicks from stale/unknown state

        Call this on voice_allocate, NOT on voice_on (retrigger).

        Args:
            last_sample: Last known output sample from this voice slot,
                         or 0 if unknown. The microfade ensures smooth
                         transition regardless.
        """
        self.prev_sample = last_sample
        self._microfade_counter = self.microfade_len

    def is_fading(self) -> bool:
        """Check if microfade is currently active."""
        return self._microfade_counter > 0

    def apply_transition(
        self,
        old_samples: List[int],
        new_samples: List[int],
        blend_start: int = 0,
    ) -> List[int]:
        """
        Apply micro-bridge at a coefficient transition point.

        Args:
            old_samples: Audio from previous coefficient state
            new_samples: Audio from new coefficient state
            blend_start: Sample index where transition begins

        Returns:
            Blended audio with C¹-continuous transition
        """
        if not old_samples or not new_samples:
            return new_samples if new_samples else old_samples

        # Build output: old before transition, bridge, new after
        result = list(old_samples[:blend_start])

        # Get context for bridge
        old_context = old_samples[blend_start:blend_start + self.bridge_len]
        new_context = new_samples[blend_start:blend_start + self.bridge_len]

        if old_context and new_context:
            bridge = micro_bridge(old_context, new_context, self.bridge_len)
            result.extend(bridge)

            # Continue with new samples after bridge
            after_bridge = blend_start + self.bridge_len
            if after_bridge < len(new_samples):
                result.extend(new_samples[after_bridge:])
        else:
            # Not enough context, just use new samples
            result.extend(new_samples[blend_start:])

        return result

    def finalize(self, samples: List[int]) -> List[int]:
        """
        Final slope projection to guarantee bounded slew.

        Also applies microfade if armed (from voice_allocate).

        Call this at the end of render to ensure no clicks remain.

        IMPORTANT: prev_sample is updated to the FINAL output sample (post-microfade)
        to ensure continuity at block boundaries.
        """
        # Apply microfade FIRST if armed (before slope projection)
        # This way slope projection sees the faded signal and can limit it properly
        if self._microfade_counter > 0:
            samples = self._apply_microfade(samples)

        # Then apply slope projection
        result, self.prev_sample = slope_project(
            samples, self.prev_sample, self.max_slew
        )

        return result

    def _apply_microfade(self, samples: List[int]) -> List[int]:
        """
        Apply microfade ramp to samples.

        The microfade is a smoothstep from 0 to 1 over microfade_len samples,
        applied to prevent clicks on voice allocation.

        Fade starts from prev_sample (the last known output) and ramps
        the new signal in smoothly.
        """
        if not samples:
            return samples

        result = []
        for s in samples:
            if self._microfade_counter > 0:
                # Compute fade position: starts at 0, ends at 1
                # remaining = microfade_counter, so position = (len - remaining) / len
                pos = (self.microfade_len - self._microfade_counter) / self.microfade_len

                # Smoothstep for C¹ continuity
                w = 3 * pos * pos - 2 * pos * pos * pos

                # Crossfade from prev_sample to new signal
                # At pos=0: output = prev_sample
                # At pos=1: output = s
                faded = int(self.prev_sample * (1 - w) + s * w)
                result.append(sat16(faded))

                self._microfade_counter -= 1
            else:
                result.append(s)

        return result

    def reset(self):
        """
        Reset internal state.

        WARNING: This should ONLY be called on voice_allocate, never on voice_on.
        Use allocate() instead for proper microfade handling.
        """
        self.prev_sample = 0
        self._microfade_counter = 0


class WaveletEngine:
    """
    Renders VoiceFrames to audio blocks via wavelet packet synthesis.
    
    Uses CDF 5/3 lifting with 5 selectable tree structures:
    - Tree 0: Balanced (uniform resolution)
    - Tree 1: Harmonic (wider lows, detailed highs)
    - Tree 2: Texture (high-band resolution)
    - Tree 3: Transient (mid-band emphasis)
    - Tree 4: Drone (low-band detail)
    """
    
    def __init__(self, block_size: int = 64):
        self.B = block_size
        self.trees = PacketTreeSet()

        # Boundary/continuity state per voice
        self.boundary_state: Dict[int, Dict] = {}

        # Tree transition state per voice (for crossfade ramping)
        self.voice_tree_state: Dict[int, Dict] = {}

        # Previous coefficients per voice (for coefficient smoothing)
        self.prev_coeffs: Dict[int, List[int]] = {}

        # Crossfade duration (blocks)
        self.xfade_blocks = 32

        # Regularity Operator R per voice (handles musical transitions)
        self.regularity: Dict[int, RegularityOperator] = {}

        # Phase state per voice for continuous synthesis (256 floats per voice)
        # Maps voice_id -> List[float] of phase accumulators for each coefficient
        self.voice_phase_state: Dict[int, List[float]] = {}

        # Coefficient smoothing rate (Q15 slew per block)
        # Higher = faster tracking, Lower = smoother but laggier
        self.coeff_slew_rate = 0x2000  # ~25% per block

        # Mix gain state (for smooth voice count transitions)
        self.mix_gain_q15 = Q15_ONE  # Start at unity gain

    def get_regularity(self, voice_id: int) -> RegularityOperator:
        """Get or create RegularityOperator for a voice."""
        if voice_id not in self.regularity:
            self.regularity[voice_id] = RegularityOperator()
        return self.regularity[voice_id]

    def get_phase_state(self, voice_id: int) -> List[float]:
        """
        Get or create phase state array for a voice.

        Phase state is critical for pitched audio synthesis - it ensures
        continuity of sinusoidal components across block boundaries.

        Returns:
            256-element array of phase accumulators (one per coefficient)
        """
        if voice_id not in self.voice_phase_state:
            # Initialize with position-based phase spread (same as synthesizer default)
            phase_state = []
            for leaf_idx in range(LEAF_COUNT):
                for coeff_idx in range(NODE_SIZE):
                    # Spread coeffs evenly across 2*pi and leaves across pi
                    phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
                    phase_state.append(phase)
            self.voice_phase_state[voice_id] = phase_state
        return self.voice_phase_state[voice_id]

    def reset_phase_state(self, voice_id: int) -> None:
        """Reset phase state for a voice (on new note allocation)."""
        if voice_id in self.voice_phase_state:
            del self.voice_phase_state[voice_id]

    def get_last_sample(self, voice_id: int) -> int:
        """
        Get last known output sample for a voice.

        Used by scheduler to pass to voice_allocate when reassigning a voice slot.
        """
        if voice_id in self.regularity:
            return self.regularity[voice_id].prev_sample
        return 0

    def voice_allocate(self, voice_id: int, last_sample: Optional[int] = None) -> None:
        """
        Allocate voice slot to new logical owner.

        Per WSE-STD-070 Section 4 (Voice Lifecycle):
        - Called when a voice slot is assigned to a NEW note/sound
        - Sets prev_sample to last known output
        - Arms 32-sample microfade (smoothstep 0→1)
        - Clears any tree transition state

        This is DIFFERENT from voice_on, which is a retrigger/gate-rise and
        MUST NOT reset state.

        Args:
            voice_id: Voice slot index
            last_sample: Last output sample from this slot, or None to query.
                        Pass explicit 0 only if voice was silent.
        """
        # Get or create RegularityOperator
        R = self.get_regularity(voice_id)

        # Determine last sample if not provided
        if last_sample is None:
            last_sample = R.prev_sample

        # Arm microfade via RegularityOperator.allocate()
        R.allocate(last_sample)

        # Clear tree transition state for fresh start
        if voice_id in self.voice_tree_state:
            del self.voice_tree_state[voice_id]

        # Clear boundary state
        if voice_id in self.boundary_state:
            del self.boundary_state[voice_id]

        # Clear previous coefficients (new voice starts fresh)
        if voice_id in self.prev_coeffs:
            del self.prev_coeffs[voice_id]

        # Reset phase state for new note (ensures clean start)
        self.reset_phase_state(voice_id)

    def voice_on(self, voice_id: int) -> None:
        """
        Signal voice retrigger / gate rise.

        Per WSE-STD-070: voice_on is a RETRIGGER, not a new allocation.
        - MUST NOT reset prev_sample or arm microfade
        - MUST NOT clear tree/boundary state
        - Only ensures RegularityOperator exists

        Use voice_allocate() when a voice slot gets a new owner.
        Use voice_on() for retriggering the same sound.
        """
        # Ensure RegularityOperator exists (but do NOT reset anything)
        self.get_regularity(voice_id)

    def voice_free(self, voice_id: int) -> None:
        """
        Mark voice as inactive / available for reallocation.

        Per WSE-STD-070:
        - Voice slot becomes available for scheduler
        - Keeps prev_sample state (next allocate will use it for microfade)
        - Does NOT clear RegularityOperator

        The next voice_allocate() will arm microfade from the retained prev_sample.
        """
        # Intentionally minimal: we keep R.prev_sample for next allocate
        # Could add a flag or notify scheduler here if needed
        pass

    def voice_off(self, voice_id: int) -> None:
        """
        Signal voice gate fall (legacy compatibility).

        For backward compatibility with existing code. Equivalent to voice_free().
        """
        self.voice_free(voice_id)

    def reset_wet_state(self, voice_id: int) -> None:
        """
        Reset all wet/IDWPT state for a voice.

        Call this on voice start/stop and on explicit "force silence" to ensure
        no stale state causes drones or DC leakage.

        Clears:
        - prev_coeffs (no coefficient memory)
        - phase_state (fresh phase accumulator)
        - boundary_state (no boundary memory)
        - voice_tree_state (fresh tree blend state)
        - RegularityOperator history (no slope memory)

        This is the nuclear option for ensuring the wet path starts clean.
        """
        # Clear previous coefficients
        if voice_id in self.prev_coeffs:
            del self.prev_coeffs[voice_id]

        # Reset phase state
        self.reset_phase_state(voice_id)

        # Clear boundary state
        if voice_id in self.boundary_state:
            del self.boundary_state[voice_id]

        # Clear tree transition state
        if voice_id in self.voice_tree_state:
            del self.voice_tree_state[voice_id]

        # Reset RegularityOperator (clear slope history)
        if voice_id in self.regularity:
            R = self.regularity[voice_id]
            R.prev_sample = 0
            R._microfade_counter = 0

    def render_voice(self, vf: VoiceFrame, apply_regularity: bool = True) -> List[int]:
        """
        Render single VoiceFrame to audio block.

        Args:
            vf: VoiceFrame with coefficients, tree selection, gain
            apply_regularity: Apply Regularity Operator R for click-free output

        ARCHITECTURE NOTE (v2 Carrier+Lift):
        With the carrier owning pitch/phase, the IDWPT is now a "lift" (effect),
        NOT an oscillator. We disable coefficient slewing and micro-bridge as
        they were oscillator-stability hacks that now cause drones/stale state.

        Anti-click mechanisms (retained for effect stability):
        1. Intra-block alpha ramp: smooth tree blend transitions (still useful)
        2. Slope projection: final safety net for any remaining spikes
        3. DC blocking: handled in CarrierLiftPipeline

        DISABLED (were oscillator hacks, now harmful):
        - Coefficient slewing: kept wet alive when it should die
        - Micro-bridge: had ordering bugs and smeared transients
        """
        voice_id = vf.voice_id

        # Use Option A tree control: tree_a_id, tree_b_id, tree_blend_q15
        # Fall back to legacy tree_id if tree_a_id is 0 and tree_id is set
        tree_a_id = vf.tree_a_id if vf.tree_a_id > 0 else vf.tree_id
        tree_b_id = vf.tree_b_id if vf.tree_b_id > 0 else tree_a_id
        blend_q15 = vf.tree_blend_q15

        # Get previous blend state for this voice (for intra-block ramping)
        if voice_id not in self.voice_tree_state:
            self.voice_tree_state[voice_id] = {
                'prev_tree_a': tree_a_id,
                'prev_tree_b': tree_b_id,
                'prev_blend': 0,
            }
        state = self.voice_tree_state[voice_id]
        prev_blend = state['prev_blend']

        # === COEFFICIENT HANDLING (v2: NO slewing, direct use) ===
        # With carrier leading pitch, we use coefficients directly.
        # Slewing was an oscillator-stability hack that causes drones.
        coeffs_to_use = vf.coeff_q15

        # Store current coefficients for tracking (but NOT for slewing)
        self.prev_coeffs[voice_id] = list(vf.coeff_q15)

        # Get phase state for this voice
        phase_state = self.get_phase_state(voice_id)

        # === SYNTHESIS (v2: NO micro-bridge, direct synthesis) ===
        # Micro-bridge was for oscillator phase continuity.
        # With carrier+lift, we just synthesize directly.
        x_q15 = self._synthesize_with_blend(
            voice_id, tree_a_id, tree_b_id, coeffs_to_use,
            prev_blend, blend_q15, phase_state
        )

        # Update state for next block
        state['prev_tree_a'] = tree_a_id
        state['prev_tree_b'] = tree_b_id
        state['prev_blend'] = blend_q15

        # Apply voice gain
        x_q15 = [mul_q15_sat(s, vf.gain_q15) for s in x_q15]

        # Ensure correct length
        if len(x_q15) < self.B:
            x_q15.extend([0] * (self.B - len(x_q15)))
        elif len(x_q15) > self.B:
            x_q15 = x_q15[:self.B]

        # Apply Regularity Operator R (handles musical transitions)
        # This ensures bounded slew rate across coefficient changes
        if apply_regularity:
            R = self.get_regularity(voice_id)
            x_q15 = R.finalize(x_q15)

        return x_q15

    def _synthesize_ramped_crossfade(
        self,
        tree_a_id: int,
        tree_b_id: int,
        coeff_q15: List[int],
        alpha_start: int,
        alpha_end: int,
        phase_state: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Synthesize crossfade with intra-block alpha ramping.

        Ramps alpha linearly from alpha_start to alpha_end over the block.
        This prevents clicks when blend factor changes between blocks.

        Args:
            tree_a_id: Primary tree ID
            tree_b_id: Secondary tree ID
            coeff_q15: Coefficients
            alpha_start: Blend at start of block (0 = all tree_a)
            alpha_end: Blend at end of block (Q15_ONE = all tree_b)
            phase_state: Phase accumulator array for continuous synthesis

        Returns:
            Audio samples with smooth crossfade ramp
        """
        # Synthesize both trees (phase_state applied to primary tree)
        x_a = self.trees.synthesize(tree_a_id, coeff_q15, phase_state=phase_state)
        x_b = self.trees.synthesize(tree_b_id, coeff_q15, phase_state=None)

        # Ramp alpha linearly across the block
        result = []
        for i in range(self.B):
            # Linear interpolation of alpha across block
            t = i / (self.B - 1) if self.B > 1 else 0
            alpha = int(alpha_start + t * (alpha_end - alpha_start))
            alpha_inv = Q15_ONE - alpha

            # Blend samples
            mixed = mul_q15_sat(x_a[i], alpha_inv) + mul_q15_sat(x_b[i], alpha)
            result.append(sat16(mixed))

        return result

    def apply_gain_pan(self, x: List[int], gain: int, pan: int) -> Tuple[List[int], List[int]]:
        """Apply gain and pan to mono signal, return stereo L/R."""
        # Pan law: constant power approximation
        # pan: -32768 = full left, 0 = center, +32767 = full right
        
        # Linear pan for simplicity (constant power would use sqrt)
        pan_norm = pan >> 1  # Scale pan to half range
        left_gain = sat16(Q15_HALF - pan_norm)
        right_gain = sat16(Q15_HALF + pan_norm)
        
        # Apply overall gain
        left_gain = mul_q15_sat(left_gain, gain)
        right_gain = mul_q15_sat(right_gain, gain)
        
        L = [mul_q15_sat(s, left_gain) for s in x]
        R = [mul_q15_sat(s, right_gain) for s in x]
        
        return L, R
    
    def mix_block(self, voice_blocks: List[Tuple[VoiceFrame, List[int]]]) -> Tuple[List[int], List[int]]:
        """
        Mix multiple voice blocks to stereo output.

        Uses slewed mix gain to prevent discontinuities when voice count changes.
        The gain is computed as Q15 * (1 / sqrt(n_voices)) and smoothly ramped
        within the block from the previous gain to the new gain.
        """
        # Accumulate in int32
        accL = [0] * self.B
        accR = [0] * self.B

        for vf, x in voice_blocks:
            # Note: gain already applied in render_voice, only apply pan here
            L, R = self.apply_pan(x, vf.pan_q15)

            for i in range(self.B):
                accL[i] += L[i]
                accR[i] += R[i]

        # Compute target mix gain: 1/sqrt(n_voices) to maintain constant power
        # This is smoother than shift-based division
        n_voices = max(1, len(voice_blocks))
        if n_voices == 1:
            target_gain = Q15_ONE
        else:
            # 1/sqrt(n) in Q15: 32767 / sqrt(n)
            target_gain = int(Q15_ONE / (n_voices ** 0.5))

        # Slew the mix gain to prevent clicks when voice count changes
        prev_gain = self.mix_gain_q15
        gain_delta = target_gain - prev_gain

        # Apply ramped gain across the block
        outL = []
        outR = []
        for i in range(self.B):
            # Linear ramp from prev_gain to target_gain
            t = i / (self.B - 1) if self.B > 1 else 1.0
            current_gain = prev_gain + int(gain_delta * t)

            # Apply gain and saturate
            outL.append(sat16((accL[i] * current_gain) >> 15))
            outR.append(sat16((accR[i] * current_gain) >> 15))

        # Update state for next block
        self.mix_gain_q15 = target_gain

        return outL, outR
    
    def apply_pan(self, x: List[int], pan: int) -> Tuple[List[int], List[int]]:
        """Apply pan to mono signal, return stereo L/R. No gain change at center."""
        # Pan law: constant power approximation
        # pan: -32768 = full left, 0 = center, +32767 = full right
        
        # At center (pan=0): both channels get full signal
        # At extremes: one channel gets full, other gets zero
        
        if pan <= -Q15_HALF:
            # Hard left
            return list(x), [0] * len(x)
        elif pan >= Q15_HALF:
            # Hard right
            return [0] * len(x), list(x)
        else:
            # Linear crossfade around center
            # pan=0: L=1, R=1 (mono compatible)
            # pan=-16384: L=1, R=0
            # pan=+16384: L=0, R=1
            
            # Scale pan to 0-1 range for right channel
            right_frac = (pan + Q15_HALF) / Q15_SCALE  # 0 to 1
            left_frac = 1.0 - right_frac * 0.5   # 1 to 0.5 to 1
            right_frac = 0.5 + right_frac * 0.5  # 0.5 to 1
            
            # Actually simpler: both channels full at center
            # For now, just pass through at center
            if abs(pan) < 1000:
                return list(x), list(x)
            
            left_gain = int(Q15_ONE * (1.0 - max(0, pan) / Q15_SCALE))
            right_gain = int(Q15_ONE * (1.0 + min(0, pan) / Q15_SCALE))

            L = [mul_q15_sat(s, left_gain) for s in x]
            R = [mul_q15_sat(s, right_gain) for s in x]

            return L, R

    def _compute_coeff_change(self, prev_coeffs: List[int], new_coeffs: List[int]) -> int:
        """
        Compute magnitude of coefficient change between blocks.

        Returns the maximum absolute difference across all coefficients.
        This determines whether micro-bridge is needed.

        Args:
            prev_coeffs: Previous block's coefficients
            new_coeffs: Current block's coefficients

        Returns:
            Maximum coefficient change (Q15 scale)
        """
        if len(prev_coeffs) != len(new_coeffs):
            # Different lengths means major change
            return Q15_ONE

        max_diff = 0
        for p, n in zip(prev_coeffs, new_coeffs):
            diff = abs(n - p)
            if diff > max_diff:
                max_diff = diff

        return max_diff

    def _slew_coefficients(self, prev_coeffs: List[int], new_coeffs: List[int]) -> List[int]:
        """
        Slew-limit coefficients toward new values.

        Prevents large coefficient jumps that cause clicks.
        Each coefficient moves at most coeff_slew_rate per block.

        Args:
            prev_coeffs: Previous block's coefficients
            new_coeffs: Target coefficients

        Returns:
            Slew-limited coefficients (may not reach target in one block)
        """
        result = []

        for i, (prev, new) in enumerate(zip(prev_coeffs, new_coeffs)):
            delta = new - prev
            if delta > self.coeff_slew_rate:
                # Positive change, limit to slew rate
                result.append(prev + self.coeff_slew_rate)
            elif delta < -self.coeff_slew_rate:
                # Negative change, limit to slew rate
                result.append(prev - self.coeff_slew_rate)
            else:
                # Within slew rate, use target directly
                result.append(new)

        # Handle length mismatch (new coeffs longer)
        if len(new_coeffs) > len(prev_coeffs):
            # New coefficients fade in from 0
            for i in range(len(prev_coeffs), len(new_coeffs)):
                target = new_coeffs[i]
                if target > self.coeff_slew_rate:
                    result.append(self.coeff_slew_rate)
                elif target < -self.coeff_slew_rate:
                    result.append(-self.coeff_slew_rate)
                else:
                    result.append(target)

        return result

    def _synthesize_with_blend(
        self,
        voice_id: int,
        tree_a_id: int,
        tree_b_id: int,
        coeff_q15: List[int],
        prev_blend: int,
        blend_q15: int,
        phase_state: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Synthesize audio with tree blending and intra-block ramping.

        If blend changed from previous block, uses intra-block alpha ramp
        to prevent discontinuities.

        Args:
            voice_id: Voice identifier (for phase state lookup)
            tree_a_id: Primary tree ID
            tree_b_id: Secondary tree ID
            coeff_q15: Coefficients to use
            prev_blend: Previous block's blend value
            blend_q15: Target blend value
            phase_state: Phase accumulator array for continuous synthesis

        Returns:
            Synthesized audio samples
        """
        # Check if we need blending at all
        if tree_a_id == tree_b_id or blend_q15 == 0:
            # Single tree, no blend needed
            return self.trees.synthesize(tree_a_id, coeff_q15, phase_state=phase_state)

        # Check if blend changed - if so, use ramped crossfade
        if prev_blend != blend_q15:
            # Blend changed, ramp within block to prevent click
            return self._synthesize_ramped_crossfade(
                tree_a_id, tree_b_id, coeff_q15,
                prev_blend, blend_q15, phase_state
            )
        else:
            # Blend stable, use constant blend
            # Note: phase_state is only passed to tree_a (primary)
            x_a = self.trees.synthesize(tree_a_id, coeff_q15, phase_state=phase_state)
            x_b = self.trees.synthesize(tree_b_id, coeff_q15, phase_state=None)

            # Blend: out = a * (1 - blend) + b * blend
            blend_inv = Q15_ONE - blend_q15
            result = []
            for a, b in zip(x_a, x_b):
                mixed = mul_q15_sat(a, blend_inv) + mul_q15_sat(b, blend_q15)
                result.append(sat16(mixed))

            return result


# =============================================================================
# IDWPT LIFT MODE (Carrier + Lift Architecture v2)
# =============================================================================

class IDWPTLift:
    """
    IDWPT as a "Lift" for the Carrier + Lift Architecture.

    In this mode, the IDWPT doesn't determine pitch - that's the carrier's job.
    Instead, it acts as a timbral shaping effect that "lifts" the spectral
    content defined by 64-leaf gains.

    Key differences from standard WaveletEngine:
    - Input: 64 leaf gains (not 256 coefficients)
    - No pitch responsibility (carrier handles that)
    - Optimized for audio-domain blending with carrier
    - 5-block latency modeled explicitly

    Usage:
        lift = IDWPTLift()
        lift.voice_allocate(voice_id)

        # Each block:
        x_wpt = lift.render_leaves(voice_id, leaf_gains_q15)
        # Then blend with carrier using equal-power crossfade
    """

    def __init__(self, block_size: int = BLOCK_SIZE):
        self.B = block_size
        self.trees = PacketTreeSet()

        # Per-voice state
        self.voice_state: Dict[int, Dict] = {}

        # Phase state per voice (256 floats for continuous synthesis)
        self.voice_phase_state: Dict[int, List[float]] = {}

        # Regularity operators per voice
        self.regularity: Dict[int, RegularityOperator] = {}

        # Latency tracking (5 blocks for D=6)
        self.LATENCY_BLOCKS = 5

        # Previous leaf gains per voice (for smoothing)
        self.prev_leaves: Dict[int, List[int]] = {}

        # Slew rate for leaf changes
        self.leaf_slew_rate = 0x2000  # ~25% per block

    def voice_allocate(self, voice_id: int) -> None:
        """Allocate voice slot for lift synthesis."""
        self.voice_state[voice_id] = {
            'blocks_rendered': 0,
            'tree_id': 1,  # Default to harmonic tree
        }

        # Initialize phase state
        if voice_id not in self.voice_phase_state:
            phase_state = []
            for leaf_idx in range(LEAF_COUNT):
                for coeff_idx in range(NODE_SIZE):
                    # Spread coeffs evenly across 2*pi and leaves across pi
                    phase = coeff_idx * (math.pi * 2 / NODE_SIZE) + leaf_idx * (math.pi / LEAF_COUNT)
                    phase_state.append(phase)
            self.voice_phase_state[voice_id] = phase_state

        # Initialize regularity operator
        if voice_id not in self.regularity:
            self.regularity[voice_id] = RegularityOperator()
            self.regularity[voice_id].allocate(0)

    def voice_free(self, voice_id: int) -> None:
        """Free voice slot."""
        self.voice_state.pop(voice_id, None)
        # Keep phase state and regularity for potential reuse

    def set_tree(self, voice_id: int, tree_id: int) -> None:
        """Set tree (tuning) for a voice. Should be slow-morphed externally."""
        if voice_id in self.voice_state:
            self.voice_state[voice_id]['tree_id'] = tree_id

    def render_leaves(
        self,
        voice_id: int,
        leaf_gains_q15: List[int],
        tree_id: Optional[int] = None,
    ) -> List[int]:
        """
        Render audio from 64 leaf gains.

        This is the core lift operation: takes spectral content defined by
        leaf gains and synthesizes audio via IDWPT.

        Args:
            voice_id: Voice identifier
            leaf_gains_q15: 64 Q15 gains, one per master leaf
            tree_id: Optional tree override (uses voice's tree if None)

        Returns:
            64 Q15 audio samples
        """
        # Ensure voice is allocated
        if voice_id not in self.voice_state:
            self.voice_allocate(voice_id)

        state = self.voice_state[voice_id]
        state['blocks_rendered'] += 1

        # Get tree to use
        tid = tree_id if tree_id is not None else state['tree_id']

        # === LEAF SMOOTHING ===
        # Slew-limit leaf changes to prevent clicks
        if voice_id in self.prev_leaves:
            leaf_gains_q15 = self._slew_leaves(
                self.prev_leaves[voice_id], leaf_gains_q15
            )
        self.prev_leaves[voice_id] = list(leaf_gains_q15)

        # === EXPAND 64 LEAVES TO 256 COEFFICIENTS ===
        # Each leaf expands to NODE_SIZE (8) coefficients
        # For now, use simple repetition with envelope shaping
        coeffs_256 = self._expand_leaves_to_coeffs(leaf_gains_q15)

        # Get phase state for this voice
        phase_state = self.voice_phase_state.get(voice_id)

        # === IDWPT SYNTHESIS ===
        x_q15 = self.trees.synthesize(tid, coeffs_256, phase_state=phase_state)

        # === APPLY REGULARITY ===
        R = self.regularity.get(voice_id)
        if R:
            x_q15 = R.finalize(x_q15)

        # Ensure correct length
        if len(x_q15) < self.B:
            x_q15.extend([0] * (self.B - len(x_q15)))
        elif len(x_q15) > self.B:
            x_q15 = x_q15[:self.B]

        return x_q15

    def _slew_leaves(
        self,
        prev_leaves: List[int],
        new_leaves: List[int],
    ) -> List[int]:
        """
        Slew-limit leaf changes to prevent clicks.

        Args:
            prev_leaves: Previous leaf gains
            new_leaves: Target leaf gains

        Returns:
            Slew-limited leaf gains
        """
        result = []
        for prev, new in zip(prev_leaves, new_leaves):
            delta = new - prev
            if delta > self.leaf_slew_rate:
                result.append(prev + self.leaf_slew_rate)
            elif delta < -self.leaf_slew_rate:
                result.append(prev - self.leaf_slew_rate)
            else:
                result.append(new)
        return result

    def _expand_leaves_to_coeffs(self, leaf_gains_q15: List[int]) -> List[int]:
        """
        Expand 64 leaf gains to 256 coefficients.

        Each leaf expands to NODE_SIZE (4) coefficients with envelope shaping.
        This creates smooth spectral content that IDWPT can process.

        With NODE_SIZE=4: 64 leaves × 4 coeffs = 256 total

        Args:
            leaf_gains_q15: 64 Q15 leaf gains

        Returns:
            256 Q15 coefficients
        """
        coeffs = []

        # Ensure we have exactly LEAF_COUNT leaves (or less)
        leaves = leaf_gains_q15[:LEAF_COUNT] if len(leaf_gains_q15) >= LEAF_COUNT else leaf_gains_q15

        for leaf_idx, gain in enumerate(leaves):
            # Create NODE_SIZE coefficients per leaf using precomputed envelope
            # _LEAF_ENVELOPE_Q15 contains sin((i + 0.5) / 4 * pi) in Q15 format
            for coeff_idx in range(NODE_SIZE):
                # Use precomputed Q15 envelope, then Q15 multiply with gain
                envelope_q15 = _LEAF_ENVELOPE_Q15[coeff_idx]
                # gain is already Q15, envelope_q15 is Q15
                # Multiply: (gain * envelope) >> 15
                coeff = mul_q15_sat(gain, envelope_q15)
                coeffs.append(coeff)

        # Pad to 256 if needed (shouldn't happen with 64 leaves × 4 coeffs)
        while len(coeffs) < 256:
            coeffs.append(0)

        return coeffs[:256]

    def is_wet_ready(self, voice_id: int) -> bool:
        """
        Check if wet signal is ready (past latency period).

        The IDWPT has ~5 blocks of latency. During this period,
        the carrier should be used alone with beta=0.

        Args:
            voice_id: Voice identifier

        Returns:
            True if at least LATENCY_BLOCKS have been rendered
        """
        if voice_id not in self.voice_state:
            return False
        return self.voice_state[voice_id]['blocks_rendered'] >= self.LATENCY_BLOCKS

    def get_blocks_rendered(self, voice_id: int) -> int:
        """Get number of blocks rendered for a voice."""
        if voice_id not in self.voice_state:
            return 0
        return self.voice_state[voice_id]['blocks_rendered']
