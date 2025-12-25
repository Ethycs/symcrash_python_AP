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
from typing import Optional, Tuple, Deque, List
import math

import numpy as np

# Q15 constants for fixed-point compatibility
Q15_ONE = 32767
Q15_SCALE = 32768


# =============================================================================
# UTILITIES
# =============================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def equal_power_weights(beta: float) -> Tuple[float, float]:
    """
    Equal-power crossfade weights for beta in [0,1].

    Returns (dry_weight, wet_weight) such that:
    - beta=0: (1, 0) = pure dry
    - beta=1: (0, 1) = pure wet
    - beta=0.5: (0.707, 0.707) = equal power mix

    Uses: dry = cos(pi/2 * beta), wet = sin(pi/2 * beta)
    """
    beta = clamp(beta, 0.0, 1.0)
    dry = math.cos(0.5 * math.pi * beta)
    wet = math.sin(0.5 * math.pi * beta)
    return dry, wet


def hann_window(K: int) -> np.ndarray:
    """
    Generate Hann window of length K.

    Used for OLA in/out windows that form a partition of unity:
    w_out + w_in = 1 (approximately exact with complementary Hann halves)
    """
    if K <= 1:
        return np.ones((K,), dtype=np.float32)
    n = np.arange(K, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(2.0 * math.pi * n / (K - 1))


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

        # Compute beta_step per block from bloom_time_s
        block_dt = self.cfg.block_size / self.cfg.sample_rate
        bloom_blocks = max(1, int(math.ceil(self.cfg.bloom_time_s / block_dt)))
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
            Mixed signal
        """
        beta = clamp(beta, 0.0, 1.0)

        if self.cfg.equal_power:
            a_weight, b_weight = equal_power_weights(beta)
        else:
            # Linear crossfade
            a_weight = 1.0 - beta
            b_weight = beta

        return (a_weight * A + b_weight * B).astype(np.float32)

    def _ola(self, st: AntiClickState, x: np.ndarray) -> np.ndarray:
        """
        Apply OLA seam smoothing.

        Blends current block start with previous block tail:
            y[0:K] = w_out * tail + w_in * x[0:K]
            tail <- x[N-K:N]

        Args:
            st: Voice state (contains tail buffer)
            x: Current block

        Returns:
            Seam-smoothed block
        """
        K = self.cfg.ola_K
        if K <= 0 or st.tail is None:
            return x

        y = x.copy()

        # Blend start of block with tail from previous block
        y[:K] = self.w_out * st.tail + self.w_in * y[:K]

        # Save tail for next block
        st.tail = y[-K:].copy()

        return y

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

        # Push carrier into FIFO for alignment with wet latency
        assert st.carrier_fifo is not None
        st.carrier_fifo.append(carrier_now.astype(np.float32))
        carrier_delayed = st.carrier_fifo[0].copy()  # 5 blocks ago

        # Determine whether wet is ready
        if wet_now is not None:
            st.wet_ready = True

        # Before wet is ready: output immediate carrier (no delay) for responsiveness
        # No OLA for pure carrier - it's already phase-continuous
        if not st.wet_ready:
            y = carrier_now.astype(np.float32)
            # Update tail for OLA continuity when wet becomes ready
            if st.tail is not None:
                K = self.cfg.ola_K
                st.tail = y[-K:].copy()
            return y

        # Wet is available - check if we should blend or stay dry
        # If beta_target is 0, user wants pure dry output
        if st.beta_target == 0.0:
            # Pure dry mode (beta=0 explicitly set)
            y = carrier_now.astype(np.float32)
            if st.tail is not None:
                K = self.cfg.ola_K
                st.tail = y[-K:].copy()
            return y

        # Advance beta toward target (slew limiting)
        if st.beta < st.beta_target:
            st.beta = min(st.beta_target, st.beta + st.beta_step)
        elif st.beta > st.beta_target:
            st.beta = max(st.beta_target, st.beta - st.beta_step)

        # Mix aligned carrier_delayed with wet_now
        y = self._mix2(carrier_delayed, wet_now, st.beta)

        # Apply OLA seam smoothing only when blending (beta > 0)
        # This smooths the seam between carrier and wet blocks
        if st.beta > 0.0:
            y = self._ola(st, y)
        else:
            # Update tail for future OLA continuity
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
