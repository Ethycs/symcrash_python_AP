"""
Safety Engine
=============

Makes audio blocks "legal" and block-invisible:
- Metrics measurement (RMS, peak, HF, DC)
- Regulation (RG smoothed tracking)
- Q projection (gain/filter adjustment)
- De-click/OLA (block boundary smoothing)
- Limiter (peak protection)
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass, field
import math

from fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15, slew_q15_asym,
    sqrt_q15, soft_clip_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)
from frames import ParamFrame, Metrics


# =============================================================================
# DE-CLICK / OLA CONSTANTS
# =============================================================================

FADE_LEN = 8          # Samples for fade in/out (power of 2)
OLA_OVERLAP = 8       # Samples of overlap-add per block edge

# Voice phases
VOICE_INACTIVE = 0
VOICE_STARTING = 1
VOICE_RUNNING = 2
VOICE_STOPPING = 3


# =============================================================================
# HANN WINDOW LUT (precomputed for FPGA compatibility)
# =============================================================================

def _build_hann_lut(n: int) -> List[int]:
    """Build Hann half-window LUT: 0.5 * (1 - cos(pi * i / n))"""
    lut = []
    for i in range(n + 1):
        # Raised cosine: 0 at i=0, 1 at i=n
        val = 0.5 * (1.0 - math.cos(math.pi * i / n))
        lut.append(int(round(val * Q15_ONE)))
    return lut

# Precomputed Hann window for FADE_LEN samples
_HANN_FADE_LUT = _build_hann_lut(FADE_LEN)
_HANN_OLA_LUT = _build_hann_lut(OLA_OVERLAP)


def hann_fade_in(n: int, fade_len: int = FADE_LEN) -> int:
    """Returns Q1.15 gain for sample n of fade-in (0..fade_len)."""
    if n <= 0:
        return 0
    if n >= fade_len:
        return Q15_ONE
    if fade_len == FADE_LEN:
        return _HANN_FADE_LUT[n]
    # Fallback for non-standard lengths
    val = 0.5 * (1.0 - math.cos(math.pi * n / fade_len))
    return int(round(val * Q15_ONE))


def hann_fade_out(n: int, fade_len: int = FADE_LEN) -> int:
    """Returns Q1.15 gain for sample n of fade-out (0..fade_len)."""
    return hann_fade_in(fade_len - n, fade_len)


# =============================================================================
# OLA STATE
# =============================================================================

@dataclass
class OLAState:
    """Per-voice OLA state for de-clicking."""
    prev_tail: List[int] = field(default_factory=list)  # Last OLA_OVERLAP samples
    voice_phase: int = VOICE_INACTIVE  # 0=inactive, 1=starting, 2=running, 3=stopping
    fade_pos: int = 0  # Position within fade (0..FADE_LEN)


@dataclass
class RGState:
    """Regulation state (smoothed metrics)."""
    E_hat: int = 0       # Smoothed RMS energy
    HF_hat: int = 0      # Smoothed HF energy
    peak_hat: int = 0    # Smoothed peak
    dc_hat: int = 0      # Smoothed DC offset


@dataclass
class LimiterState:
    """Limiter state."""
    gain: int = Q15_ONE  # Current gain (Q1.15)
    hold_count: int = 0  # Hold counter


class SafetyEngine:
    """
    Safety processing: metrics, regulation, de-click/OLA, limiting.

    Processing chain:
        Input -> Metrics -> RG -> Q -> De-click/OLA -> Limiter -> DC removal -> Output
    """

    # Target levels by K_mode
    E_TARGETS = [0x4000, 0x5000, 0x6000, 0x7000]  # Q1.15 RMS targets
    HF_BUDGETS = [0x2000, 0x3000, 0x4000, 0x5000]  # HF energy budgets

    # Limiter thresholds
    LIMIT_THRESHOLD = 0x7000  # ~0.875
    LIMIT_CEILING = 0x7F00    # ~0.996

    # Attack/release (samples per block)
    ATTACK_RATE = 2048   # Fast attack
    RELEASE_RATE = 64    # Slow release

    def __init__(self, block_size: int = 64):
        self.B = block_size

        # Regulation state (smoothed)
        self.rg = RGState()

        # Limiter state
        self.lim_L = LimiterState()
        self.lim_R = LimiterState()

        # Q state (slewed gains)
        self.q_gain = Q15_ONE
        self.q_hf_cut = Q15_ONE

        # Smoothing window
        self.W = 16  # Blocks for RG averaging
        self.alpha = Q15_ONE // self.W

        # OLA state per voice
        self.ola_state: Dict[int, OLAState] = {}

        # Master OLA state (for mixed stereo output)
        self.master_ola_L: List[int] = []
        self.master_ola_R: List[int] = []

    # =========================================================================
    # OLA STATE MANAGEMENT
    # =========================================================================

    def get_ola_state(self, voice_id: int) -> OLAState:
        """Get or create OLA state for a voice."""
        if voice_id not in self.ola_state:
            self.ola_state[voice_id] = OLAState()
        return self.ola_state[voice_id]

    def voice_on(self, voice_id: int):
        """Signal voice start - triggers fade-in."""
        state = self.get_ola_state(voice_id)
        state.voice_phase = VOICE_STARTING
        state.fade_pos = 0
        state.prev_tail = []

    def voice_off(self, voice_id: int):
        """Signal voice stop - triggers fade-out."""
        state = self.get_ola_state(voice_id)
        if state.voice_phase == VOICE_RUNNING:
            state.voice_phase = VOICE_STOPPING
            state.fade_pos = 0

    def is_voice_active(self, voice_id: int) -> bool:
        """Check if voice is active (not inactive)."""
        if voice_id not in self.ola_state:
            return False
        return self.ola_state[voice_id].voice_phase != VOICE_INACTIVE

    # =========================================================================
    # VOICE START/STOP DE-CLICK
    # =========================================================================

    def apply_voice_start(self, x: List[int], voice_id: int) -> List[int]:
        """Apply fade-in for voice start."""
        state = self.get_ola_state(voice_id)

        if state.voice_phase != VOICE_STARTING:
            return x

        out = list(x)
        samples_to_fade = min(FADE_LEN - state.fade_pos, len(x))

        for i in range(samples_to_fade):
            gain = hann_fade_in(state.fade_pos + i, FADE_LEN)
            out[i] = mul_q15_sat(x[i], gain)

        state.fade_pos += len(x)
        if state.fade_pos >= FADE_LEN:
            state.voice_phase = VOICE_RUNNING
            state.fade_pos = 0

        return out

    def apply_voice_stop(self, x: List[int], voice_id: int) -> List[int]:
        """Apply fade-out for voice stop."""
        state = self.get_ola_state(voice_id)

        if state.voice_phase != VOICE_STOPPING:
            return x

        out = list(x)
        samples_to_fade = min(FADE_LEN - state.fade_pos, len(x))

        for i in range(samples_to_fade):
            gain = hann_fade_out(state.fade_pos + i, FADE_LEN)
            out[i] = mul_q15_sat(x[i], gain)

        # Zero remaining samples after fade completes
        if state.fade_pos + len(x) >= FADE_LEN:
            for i in range(samples_to_fade, len(x)):
                out[i] = 0

        state.fade_pos += len(x)
        if state.fade_pos >= FADE_LEN:
            state.voice_phase = VOICE_INACTIVE
            state.fade_pos = 0
            state.prev_tail = []

        return out

    # =========================================================================
    # BLOCK BOUNDARY OLA
    # =========================================================================

    def apply_ola(self, x: List[int], voice_id: int) -> List[int]:
        """Overlap-add for block boundary smoothing."""
        state = self.get_ola_state(voice_id)

        # Skip if voice is inactive or just starting
        if state.voice_phase == VOICE_INACTIVE:
            return x

        if not state.prev_tail or len(state.prev_tail) < OLA_OVERLAP:
            # First block or insufficient tail - just save tail
            state.prev_tail = list(x[-OLA_OVERLAP:])
            return x

        out = list(x)

        # Crossfade at block start: blend previous tail with current start
        for i in range(OLA_OVERLAP):
            fade_out_gain = hann_fade_out(i, OLA_OVERLAP)
            fade_in_gain = hann_fade_in(i, OLA_OVERLAP)

            prev_contrib = mul_q15_sat(state.prev_tail[i], fade_out_gain)
            curr_contrib = mul_q15_sat(x[i], fade_in_gain)
            out[i] = sat16(prev_contrib + curr_contrib)

        # Save tail for next block
        state.prev_tail = list(x[-OLA_OVERLAP:])

        return out

    def apply_declick(self, x: List[int], voice_id: int) -> List[int]:
        """Full de-click chain for a voice block."""
        x = self.apply_voice_start(x, voice_id)
        x = self.apply_voice_stop(x, voice_id)
        x = self.apply_ola(x, voice_id)
        return x

    # =========================================================================
    # MASTER OLA (for mixed stereo output)
    # =========================================================================

    def apply_master_ola(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Apply OLA to mixed stereo output (block boundary smoothing)."""
        if not self.master_ola_L or len(self.master_ola_L) < OLA_OVERLAP:
            # First block - just save tail
            self.master_ola_L = list(L[-OLA_OVERLAP:])
            self.master_ola_R = list(R[-OLA_OVERLAP:])
            return L, R

        outL = list(L)
        outR = list(R)

        # Crossfade at block start
        for i in range(OLA_OVERLAP):
            fade_out_gain = hann_fade_out(i, OLA_OVERLAP)
            fade_in_gain = hann_fade_in(i, OLA_OVERLAP)

            # Left channel
            prev_L = mul_q15_sat(self.master_ola_L[i], fade_out_gain)
            curr_L = mul_q15_sat(L[i], fade_in_gain)
            outL[i] = sat16(prev_L + curr_L)

            # Right channel
            prev_R = mul_q15_sat(self.master_ola_R[i], fade_out_gain)
            curr_R = mul_q15_sat(R[i], fade_in_gain)
            outR[i] = sat16(prev_R + curr_R)

        # Save tail for next block
        self.master_ola_L = list(L[-OLA_OVERLAP:])
        self.master_ola_R = list(R[-OLA_OVERLAP:])

        return outL, outR

    # =========================================================================
    # METRICS
    # =========================================================================

    def measure_metrics(self, L: List[int], R: List[int]) -> Metrics:
        """Measure block metrics."""
        # Peak
        peak = 0
        for i in range(self.B):
            peak = max(peak, abs(L[i]), abs(R[i]))

        # RMS energy (sum of squares, scaled)
        energy = 0
        for i in range(self.B):
            energy += (L[i] * L[i] + R[i] * R[i]) >> 16
        rms = sqrt_q15(min(Q15_ONE, energy // (2 * self.B)))

        # HF energy (simple: sum of sample differences)
        hf = 0
        for i in range(1, self.B):
            dL = L[i] - L[i-1]
            dR = R[i] - R[i-1]
            hf += (dL * dL + dR * dR) >> 18
        hf_energy = min(Q15_ONE, hf // self.B)

        # DC offset (average)
        dc = sum(L) + sum(R)
        dc_offset = sat16(dc // (2 * self.B))

        # Crest factor (peak / rms)
        if rms > 0:
            crest = (peak << 15) // rms
            crest = min(Q15_ONE, crest)
        else:
            crest = Q15_ONE

        return Metrics(
            rms_q15=rms,
            peak_q15=peak,
            hf_energy_q15=hf_energy,
            dc_offset_q15=dc_offset,
            crest_factor_q15=crest
        )

    def update_rg(self, m: Metrics, K_mode: int):
        """Update regulation state with smoothed metrics."""
        # Exponential smoothing: hat = alpha * new + (1-alpha) * hat
        alpha = self.alpha
        alpha_inv = Q15_ONE - alpha

        self.rg.E_hat = shift_round(
            alpha * m.rms_q15 + alpha_inv * self.rg.E_hat,
            Q15_SHIFT
        )
        self.rg.HF_hat = shift_round(
            alpha * m.hf_energy_q15 + alpha_inv * self.rg.HF_hat,
            Q15_SHIFT
        )
        self.rg.peak_hat = shift_round(
            alpha * m.peak_q15 + alpha_inv * self.rg.peak_hat,
            Q15_SHIFT
        )
        self.rg.dc_hat = shift_round(
            alpha * m.dc_offset_q15 + alpha_inv * self.rg.dc_hat,
            Q15_SHIFT
        )

    # =========================================================================
    # Q PROJECTION
    # =========================================================================

    def compute_q_targets(self, K_mode: int) -> Tuple[int, int]:
        """Compute Q gain and HF cut targets based on RG state."""
        E_target = self.E_TARGETS[min(K_mode, 3)]
        HF_budget = self.HF_BUDGETS[min(K_mode, 3)]

        # Gain target: scale to bring RMS toward target
        if self.rg.E_hat > 0:
            gain_target = (E_target << 15) // self.rg.E_hat
            gain_target = min(Q15_ONE, gain_target)
        else:
            gain_target = Q15_ONE

        # HF cut target: reduce if HF over budget
        if self.rg.HF_hat > HF_budget:
            hf_cut_target = (HF_budget << 15) // self.rg.HF_hat
            hf_cut_target = max(Q15_HALF, min(Q15_ONE, hf_cut_target))
        else:
            hf_cut_target = Q15_ONE

        return gain_target, hf_cut_target

    def apply_q(self, L: List[int], R: List[int], K_mode: int) -> Tuple[List[int], List[int]]:
        """Apply Q projection (gain + HF control)."""
        gain_target, hf_cut_target = self.compute_q_targets(K_mode)

        # Slew gains
        self.q_gain = slew_q15(self.q_gain, gain_target, 256)
        self.q_hf_cut = slew_q15(self.q_hf_cut, hf_cut_target, 128)

        outL = []
        outR = []

        # Simple HF reduction: blend with previous sample
        blend = Q15_ONE - self.q_hf_cut

        for i in range(self.B):
            # Apply gain
            yL = mul_q15_sat(L[i], self.q_gain)
            yR = mul_q15_sat(R[i], self.q_gain)

            # HF reduction (simple lowpass blend)
            if i > 0 and blend > 0:
                yL = sat16(mul_q15_sat(yL, self.q_hf_cut) +
                          mul_q15_sat(outL[i-1], blend))
                yR = sat16(mul_q15_sat(yR, self.q_hf_cut) +
                          mul_q15_sat(outR[i-1], blend))

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    # =========================================================================
    # LIMITER
    # =========================================================================

    def apply_limiter(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Apply brick-wall limiter with attack/release."""
        outL = []
        outR = []

        for i in range(self.B):
            # Find peak
            peak = max(abs(L[i]), abs(R[i]))

            # Compute target gain
            if peak > self.LIMIT_THRESHOLD:
                # Gain to bring peak to ceiling
                target_gain = (self.LIMIT_CEILING << 15) // max(1, peak)
                target_gain = min(Q15_ONE, target_gain)
            else:
                target_gain = Q15_ONE

            # Asymmetric slew (fast attack, slow release)
            self.lim_L.gain = slew_q15_asym(
                self.lim_L.gain, target_gain,
                self.ATTACK_RATE, self.RELEASE_RATE
            )

            # Apply gain
            yL = mul_q15_sat(L[i], self.lim_L.gain)
            yR = mul_q15_sat(R[i], self.lim_L.gain)

            # Final hard clip (should rarely trigger)
            yL = sat16(yL)
            yR = sat16(yR)

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    def remove_dc(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Remove DC offset."""
        if abs(self.rg.dc_hat) < 100:
            return L, R

        dc = self.rg.dc_hat
        outL = [sat16(x - dc) for x in L]
        outR = [sat16(x - dc) for x in R]

        return outL, outR

    # =========================================================================
    # MAIN PROCESSING CHAIN
    # =========================================================================

    def step_block(self, L: List[int], R: List[int],
                   params: ParamFrame,
                   apply_master_ola: bool = True) -> Tuple[List[int], List[int]]:
        """
        Full safety processing chain.

        Chain: Metrics -> RG -> Q -> Master OLA -> Limiter -> DC removal

        For per-voice de-click, call apply_declick() on each voice block
        BEFORE mixing to stereo.
        """
        # 1) Measure metrics
        m = self.measure_metrics(L, R)

        # 2) Update RG (smoothed tracking)
        self.update_rg(m, params.K_mode)

        # 3) Apply Q (gain + HF control)
        Lq, Rq = self.apply_q(L, R, params.K_mode)

        # 4) Apply master OLA (block boundary smoothing on mixed output)
        if apply_master_ola:
            Lq, Rq = self.apply_master_ola(Lq, Rq)

        # 5) Apply limiter
        Lo, Ro = self.apply_limiter(Lq, Rq)

        # 6) Remove DC
        Lo, Ro = self.remove_dc(Lo, Ro)

        return Lo, Ro

    def step_block_with_voices(self, voice_blocks: List[Tuple[int, List[int], List[int]]],
                                params: ParamFrame) -> Tuple[List[int], List[int]]:
        """
        Process with per-voice de-clicking before mixing.

        voice_blocks: List of (voice_id, L, R) tuples

        This applies per-voice de-click, then mixes, then runs the rest of the chain.
        """
        # Apply per-voice de-click
        declicked_blocks = []
        for voice_id, vL, vR in voice_blocks:
            dL = self.apply_declick(vL, voice_id)
            dR = self.apply_declick(vR, voice_id)
            declicked_blocks.append((dL, dR))

        # Mix to stereo
        mix_L = [0] * self.B
        mix_R = [0] * self.B

        for dL, dR in declicked_blocks:
            for i in range(self.B):
                mix_L[i] = sat16(mix_L[i] + dL[i])
                mix_R[i] = sat16(mix_R[i] + dR[i])

        # Run through main chain (skip master OLA since we did per-voice)
        return self.step_block(mix_L, mix_R, params, apply_master_ola=False)

    def get_metrics(self) -> Metrics:
        """Get current smoothed metrics."""
        return Metrics(
            rms_q15=self.rg.E_hat,
            peak_q15=self.rg.peak_hat,
            hf_energy_q15=self.rg.HF_hat,
            dc_offset_q15=self.rg.dc_hat,
            crest_factor_q15=0
        )
