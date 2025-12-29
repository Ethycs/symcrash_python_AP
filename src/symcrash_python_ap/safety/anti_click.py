"""
AntiClickOperator - Single Matrix-Style Wrapper for Click-Free Audio
=====================================================================

This is the ONLY place that handles click prevention in the render pipeline.
Synths can be nonlinear/messy and produce one or two candidate blocks.
AntiClickOperator is responsible for:

1. Latency alignment (5-block FIFO for wet/carrier sync)
2. Carrier/wet mixing (equal-power crossfade with beta ramp)
3. Optional OLA seam smoothing

Key design: "Carrier leads, wavelet packet comes in late"
- Carrier provides immediate pitch response (zero latency)
- Wet/IDWPT path has ~5 blocks of latency
- Beta ramps from 0->target once wet is ready

Usage:
    anticlick = AntiClickOperator(AntiClickConfig())
    state = anticlick.init_state()

    anticlick.note_on(state)
    anticlick.set_wet_target(state, 1.0)  # Full wet

    for block in blocks:
        carrier = render_carrier(...)
        wet = render_wet(...)  # May be None during warmup
        output = anticlick.step(state, carrier, wet)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple, Deque, List, Dict
from functools import lru_cache

import numpy as np

# Use sine_core for deterministic, bit-exact trig (no math.sin/cos in hot paths)
from ..wavelet.sine_core import sine_q15, cosine_q15, PHASE_MASK

# Q15 constants for fixed-point compatibility
Q15_ONE = 32767
Q15_SCALE = 32768

# Phase constant for 90 degrees (pi/2)
PHASE_90_DEG = 0x40000000


# =============================================================================
# UTILITIES
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


# Pre-computed weight cache for common beta values (avoids trig calls)
_WEIGHT_CACHE: Dict[int, Tuple[float, float]] = {}


def equal_power_weights(beta: float) -> Tuple[float, float]:
    """
    Equal-power crossfade weights for beta in [0,1].

    Returns (dry_weight, wet_weight) such that:
    - beta=0: (1, 0) = pure dry
    - beta=1: (0, 1) = pure wet
    - beta=0.5: (0.707, 0.707) = equal power mix

    Uses sine_core LUT for deterministic, bit-exact results.
    Results are cached to avoid repeated LUT lookups.
    """
    beta = clamp(beta, 0.0, 1.0)

    # Quantize beta to 12-bit resolution for caching (4096 distinct values)
    # This provides ~0.02% precision which is inaudible
    beta_key = int(beta * 4096)

    if beta_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[beta_key]

    # Convert beta [0,1] to phase [0, pi/2] for sine_core
    # beta=0 -> phase=0, beta=1 -> phase=0x40000000 (90 degrees)
    phase = int(beta * PHASE_90_DEG) & PHASE_MASK

    # wet = sin(pi/2 * beta), dry = cos(pi/2 * beta)
    wet_q15 = sine_q15(phase)
    dry_q15 = cosine_q15(phase)

    # Convert Q15 to float for mixing (normalize by 32767)
    wet = wet_q15 / Q15_ONE
    dry = dry_q15 / Q15_ONE

    # Cache the result
    _WEIGHT_CACHE[beta_key] = (dry, wet)

    return dry, wet


def hann_window(K: int) -> np.ndarray:
    """
    Generate Hann window of length K.

    Used for OLA in/out windows that form a partition of unity:
    w_out + w_in = 1 (approximately exact with complementary Hann halves)

    Uses sine_core LUT for deterministic, bit-exact results.
    Formula: w[n] = 0.5 - 0.5 * cos(2π * n / (K-1))
    """
    if K <= 1:
        return np.ones((K,), dtype=np.float32)

    window = np.empty((K,), dtype=np.float32)
    for n in range(K):
        # Map n to phase: 2π * n / (K-1) corresponds to one full cycle
        # Phase 0 to 2^32 = one full cycle, so multiply by 2^32
        # phase = (n / (K-1)) * 2^32
        phase = int((n / (K - 1)) * (1 << 32)) & PHASE_MASK
        cos_q15 = cosine_q15(phase)
        # Convert Q15 to float and apply formula: 0.5 - 0.5 * cos(...)
        window[n] = 0.5 - 0.5 * (cos_q15 / Q15_ONE)

    return window


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AntiClickConfig:
    """Configuration for AntiClickOperator."""

    block_size: int = 64
    sample_rate: int = 48_000

    # Wet latency in blocks (D=6 IDWPT requires ~5 blocks)
    wet_latency_blocks: int = 5

    # How long to bloom wet in (seconds). Typical: 0.03..0.2
    bloom_time_s: float = 0.05

    # OLA seam smoothing on first K samples of each output block.
    # Set K=0 to disable OLA entirely.
    ola_K: int = 16

    # If True: equal-power crossfade; if False: linear crossfade
    equal_power: bool = True


# =============================================================================
# PER-VOICE STATE
# =============================================================================

@dataclass
class AntiClickState:
    """Per-voice state for AntiClickOperator."""

    # Crossfade position [0=dry, 1=wet]
    beta: float = 0.0
    beta_target: float = 0.0
    beta_step: float = 0.0  # Per-block increment toward target

    # OLA tail buffer (last K samples of previous output)
    tail: Optional[np.ndarray] = None

    # Carrier delay FIFO for alignment with wet latency
    carrier_fifo: Optional[Deque[np.ndarray]] = None

    # Whether wet has become valid for this voice yet
    wet_ready: bool = False

    # FIFO warmup counter: counts blocks until FIFO contains real carrier data
    # When fifo_warmup > 0, FIFO still contains zeros from initialization
    fifo_warmup: int = 0


# =============================================================================
# ANTI-CLICK OPERATOR
# =============================================================================

class AntiClickOperator:
    """
    Single "matrix operator" for click-free audio rendering.

    Implements structured operations:
    - Latency alignment (carrier FIFO)
    - 2-way mixing (equal-power or linear crossfade)
    - Optional OLA seam smoothing

    This replaces all scattered anti-click logic:
    - DC blockers in unified_renderer.py and carrier.py
    - Crossfade utilities in unified_renderer.py
    - OLA logic in safety_engine.py
    - BlendState in carrier_lift.py
    - RegularityOperator in wavelet_engine.py
    """

    def __init__(self, cfg: AntiClickConfig):
        """
        Initialize AntiClickOperator.

        Args:
            cfg: AntiClickConfig with block_size, latency, bloom time, etc.
        """
        self.cfg = cfg
        self._init_windows()

    def _init_windows(self) -> None:
        """Precompute OLA windows."""
        K = self.cfg.ola_K
        if K > 0:
            w = hann_window(K).astype(np.float32)
            # Complementary halves: w_in + w_out = 1
            self.w_in = w
            self.w_out = 1.0 - w
        else:
            self.w_in = None
            self.w_out = None

    def init_state(self) -> AntiClickState:
        """
        Create initial state for a voice.

        Returns:
            AntiClickState with initialized FIFO and tail buffer
        """
        st = AntiClickState()

        # Initialize OLA tail buffer
        if self.cfg.ola_K > 0:
            st.tail = np.zeros((self.cfg.ola_K,), dtype=np.float32)

        # Initialize carrier FIFO for latency alignment
        st.carrier_fifo = deque(
            [np.zeros((self.cfg.block_size,), dtype=np.float32)
             for _ in range(self.cfg.wet_latency_blocks)],
            maxlen=self.cfg.wet_latency_blocks
        )

        # FIFO warmup: need to wait for FIFO to fill with real carrier data
        # During warmup, FIFO returns zeros which don't match the carrier phase
        st.fifo_warmup = self.cfg.wet_latency_blocks

        # Compute beta_step per block from bloom_time_s
        block_dt = self.cfg.block_size / self.cfg.sample_rate
        # ceil(a/b) = (a + b - 1) // b for positive values
        # bloom_blocks = ceil(bloom_time_s / block_dt)
        bloom_blocks = max(1, int((self.cfg.bloom_time_s + block_dt - 1e-9) / block_dt))
        st.beta_step = 1.0 / bloom_blocks

        return st

    def note_on(self, st: AntiClickState) -> None:
        """
        Signal note-on / voice start.

        Resets state:
        - Carrier leads immediately
        - Wet fades in once ready (after latency period)
        """
        st.beta = 0.0
        st.beta_target = 0.0
        st.wet_ready = False

        # Reset FIFO warmup counter
        st.fifo_warmup = self.cfg.wet_latency_blocks

        # Clear OLA tail
        if st.tail is not None:
            st.tail.fill(0.0)

        # Clear carrier FIFO
        if st.carrier_fifo is not None:
            st.carrier_fifo.clear()
            for _ in range(self.cfg.wet_latency_blocks):
                st.carrier_fifo.append(
                    np.zeros((self.cfg.block_size,), dtype=np.float32)
                )

    def note_off(self, st: AntiClickState) -> None:
        """
        Signal note-off / voice release.

        Can be used to fade wet back to 0 for release tail.
        """
        # Optionally set beta_target to 0 for clean release
        # For now, we let the voice naturally decay
        pass

    def set_wet_target(self, st: AntiClickState, target: float) -> None:
        """
        Set wet/dry blend target.

        Args:
            st: Voice state
            target: Target beta [0=dry, 1=wet]
        """
        st.beta_target = clamp(target, 0.0, 1.0)

    def _mix2(self, A: np.ndarray, B: np.ndarray, beta: float) -> np.ndarray:
        """
        Mix two blocks with crossfade.

        Args:
            A: Dry signal (carrier)
            B: Wet signal (IDWPT)
            beta: Mix amount [0=A, 1=B]

        Returns:
            Mixed signal (clipped to [-1, 1])
        """
        beta = clamp(beta, 0.0, 1.0)

        if self.cfg.equal_power:
            a_weight, b_weight = equal_power_weights(beta)
        else:
            # Linear crossfade
            a_weight = 1.0 - beta
            b_weight = beta

        mixed = a_weight * A + b_weight * B
        # Clip to prevent overflow - equal power crossfade can exceed 1.0
        # when both signals are in-phase and at full scale
        return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def _ola(self, st: AntiClickState, x: np.ndarray) -> np.ndarray:
        """
        Apply OLA seam smoothing.

        NOTE: OLA was designed for tree-based synthesis where blocks might have
        discontinuities. For effect_mode (FWPT->mask->IDWPT), the carrier and
        wet signals are already phase-coherent and continuous, so OLA actually
        CREATES discontinuities rather than removing them.

        This method is now a NO-OP - it just updates the tail buffer for
        potential future use but does not modify the signal.

        Args:
            st: Voice state (contains tail buffer)
            x: Current block

        Returns:
            Input block unchanged
        """
        K = self.cfg.ola_K
        if K <= 0 or st.tail is None:
            return x

        # Update tail buffer for tracking, but DO NOT apply OLA blending
        # OLA blending causes discontinuities in effect_mode where signals
        # are already continuous.
        st.tail = x[-K:].copy()

        return x

    def step(
        self,
        st: AntiClickState,
        carrier_now: np.ndarray,
        wet_now: Optional[np.ndarray],
        Q: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process one block through the anti-click operator.

        This is the main render method. Call once per block per voice.

        During FIFO warmup (first N blocks where N = wet_latency_blocks):
        - Output carrier_now directly (no FIFO delay, no OLA)
        - This avoids discontinuities from zero-filled FIFO

        After warmup:
        - Use carrier_delayed from FIFO for consistent latency
        - Apply OLA for seam smoothing
        - Mix with wet if beta > 0

        Args:
            st: Per-voice state
            carrier_now: Current carrier block (zero-latency, immediate response)
            wet_now: Wet/IDWPT block if available, else None during warmup
            Q: Optional modulation vector (future use for dynamic beta_step, etc.)

        Returns:
            Click-safe output block (same shape as carrier_now)
        """
        N = self.cfg.block_size
        assert carrier_now.shape == (N,), f"carrier_now shape {carrier_now.shape} != ({N},)"
        if wet_now is not None:
            assert wet_now.shape == (N,), f"wet_now shape {wet_now.shape} != ({N},)"

        # Clip inputs to [-1, 1] to prevent overflow in mixing/OLA
        # This is the single point of input conditioning
        carrier_now = np.clip(carrier_now, -1.0, 1.0).astype(np.float32)
        if wet_now is not None:
            wet_now = np.clip(wet_now, -1.0, 1.0).astype(np.float32)

        # Push carrier into FIFO for alignment with wet latency
        assert st.carrier_fifo is not None
        st.carrier_fifo.append(carrier_now)

        # Track FIFO warmup (for wet path alignment)
        if st.fifo_warmup > 0:
            st.fifo_warmup -= 1

        # Track wet readiness
        if wet_now is not None:
            st.wet_ready = True

        # Advance beta toward target (slew limiting)
        if st.beta < st.beta_target:
            st.beta = min(st.beta_target, st.beta + st.beta_step)
        elif st.beta > st.beta_target:
            st.beta = max(st.beta_target, st.beta - st.beta_step)

        # Determine mixing mode:
        # - Pure dry (beta=0): use carrier_now directly
        # - Wet mixing (beta>0): mix carrier_now with wet_now
        #
        # Note: The FIFO was designed for tree-based synthesis with inherent latency.
        # For effect_mode (FWPT->mask->IDWPT), there is no latency - wet_now is
        # derived directly from carrier_now. So we mix carrier_now with wet_now
        # to maintain phase coherence.
        #
        # The FIFO is still maintained for potential future use with latent paths.
        use_wet = st.beta > 0.0 and wet_now is not None and st.wet_ready

        if use_wet:
            # Mix carrier_now with wet_now (both derived from same phase)
            # Effect mode has no latency, so no FIFO delay needed
            y = self._mix2(carrier_now, wet_now, st.beta)
            # OLA smooths any small differences between carrier and wet paths
            y = self._ola(st, y)
        else:
            # Pure dry mode - use carrier_now directly
            y = carrier_now.astype(np.float32)
            # Update tail in case we switch to wet later
            if st.tail is not None:
                K = self.cfg.ola_K
                st.tail = y[-K:].copy()

        return y

    def step_q15(
        self,
        st: AntiClickState,
        carrier_now: List[int],
        wet_now: Optional[List[int]],
        Q: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Process one block with Q15 fixed-point I/O.

        Convenience wrapper for integration with existing Q15-based pipeline.

        Args:
            st: Per-voice state
            carrier_now: Carrier samples as Q15 integers
            wet_now: Wet samples as Q15 integers, or None
            Q: Optional modulation vector

        Returns:
            Output samples as Q15 integers
        """
        # Convert Q15 to float
        carrier_f = np.array(carrier_now, dtype=np.float32) / Q15_SCALE
        wet_f = None
        if wet_now is not None:
            wet_f = np.array(wet_now, dtype=np.float32) / Q15_SCALE

        # Process
        y_f = self.step(st, carrier_f, wet_f, Q)

        # Convert back to Q15
        y_q15 = (y_f * Q15_SCALE).astype(np.int32)
        y_q15 = np.clip(y_q15, -Q15_SCALE, Q15_ONE)

        return y_q15.tolist()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AntiClickConfig',
    'AntiClickState',
    'AntiClickOperator',
    'equal_power_weights',
    'hann_window',
    'clamp',
]
