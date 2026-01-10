"""
Material Engine
===============

Applies selected "material" programs to audio:
- Tape Bus: hysteresis saturation
- Catastrophe: fold/cusp/swallowtail bifurcation maps
- Rank: median/trimmed mean nonlinear filtering
- Tropical: hard cut/max operations
- Clean: transparent pass-through with gentle glue
"""

from typing import List, Tuple
from dataclasses import dataclass

from ..fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15, slew_q15_asym,
    tanh_q15, soft_clip_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF, mac_q15
)
from ..frames import ParamFrame, MaterialParams


@dataclass
class TapeState:
    """State for tape hysteresis model."""
    m: int = 0  # Magnetization state (Q1.15)


@dataclass
class CatastropheState:
    """State for catastrophe bifurcation."""
    s: int = 0  # Slow state variable (Q1.15)
    regime: int = 0  # Current regime ID


@dataclass
class RankState:
    """State for rank statistics filter."""
    buffer: List[int] = None
    idx: int = 0

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = [0] * 17  # Max window size


@dataclass
class DelayState:
    """State for delay line effect (MONO)."""
    buffer: List[int] = None
    write_idx: int = 0
    # FPGA-friendly: 1 second mono = 48000 samples = 96KB
    # Fits in 128KB BRAM budget for mono operation
    max_samples: int = 48000  # 1 second at 48kHz

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = [0] * self.max_samples


@dataclass
class ChorusState:
    """State for chorus/flanger effect (MONO)."""
    buffer: List[int] = None
    write_idx: int = 0
    lfo_phase: int = 0  # Q32 phase accumulator
    # FPGA-friendly: 20ms mono = 960 samples = 1.9KB
    max_samples: int = 960  # 20ms at 48kHz (typical chorus range: 5-25ms)

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = [0] * self.max_samples


class MaterialEngine:
    """
    Applies material programs to stereo audio blocks.
    All processing is bit-accurate Q1.15.
    """
    
    def __init__(self, block_size: int = 64):
        self.B = block_size
        
        # Per-channel state
        self.tape_L = TapeState()
        self.tape_R = TapeState()
        
        self.cat_L = CatastropheState()
        self.cat_R = CatastropheState()
        
        self.rank_L = RankState()
        self.rank_R = RankState()

        # Delay state (material_id=6)
        self.delay_L = DelayState()
        self.delay_R = DelayState()

        # Chorus state (material_id=7)
        self.chorus_L = ChorusState()
        self.chorus_R = ChorusState()

        # Slewed parameters
        self.params = MaterialParams()
        
        # One-pole lowpass state for tone control
        self.lp_zL = 0
        self.lp_zR = 0
    
    def step_block(self, L: List[int], R: List[int], params: ParamFrame,
                   material_id: int, cat_id: int = 0) -> Tuple[List[int], List[int]]:
        """
        Apply selected material program to stereo block.
        
        material_id:
            0 = clean (transparent)
            1 = tape bus
            2 = rank grit
            3 = tropical cut
            4 = catastrophe snap (fold)
            5 = catastrophe A4 (cusp)
        """
        # Slew parameters
        self.params.slew_from(params)
        
        if material_id == 1:
            return self._tape_bus(L, R)
        elif material_id == 2:
            return self._rank_grit(L, R)
        elif material_id == 3:
            return self._tropical_cut(L, R)
        elif material_id == 4:
            return self._cat_fold(L, R)
        elif material_id == 5:
            return self._cat_cusp(L, R)
        elif material_id == 6:
            return self._delay_line(L, R)
        elif material_id == 7:
            return self._chorus(L, R)
        else:
            return self._clean_glue(L, R)
    
    # =========================================================================
    # TAPE BUS
    # =========================================================================
    
    def _tape_sample(self, x: int, state: TapeState) -> int:
        """
        Single-sample tape hysteresis.
        
        Model: m' = m + λ(tanh(drive * x + bias) - m) * (1 - ρ*m²)
        Output: tone_filter(m)
        """
        drive = self.params.drive_q15
        bias = self.params.bias_q15
        
        # λ (feedback rate) and ρ (saturation) depend on drive
        # Higher drive -> faster response, more saturation
        lam = min(Q15_ONE, drive + Q15_HALF)  # 0.5 to 1.5 range
        rho = drive >> 2  # Gentle saturation factor
        
        # Input with drive and bias
        driven = mul_q15_sat(x, drive)
        driven = sat16(driven + bias)
        
        # Tanh nonlinearity
        target = tanh_q15(driven)
        
        # Hysteresis: m' = m + λ(target - m) * (1 - ρ*m²)
        m_sq = mul_q15_sat(state.m, state.m)
        damp = Q15_ONE - mul_q15_sat(rho, m_sq)
        damp = max(0, damp)
        
        delta = mul_q15_sat(target - state.m, lam)
        delta = mul_q15_sat(delta, damp)
        
        state.m = sat16(state.m + (delta >> 1))
        
        return state.m
    
    def _tape_bus(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Tape bus processing."""
        outL = []
        outR = []
        
        for i in range(self.B):
            yL = self._tape_sample(L[i], self.tape_L)
            yR = self._tape_sample(R[i], self.tape_R)
            
            # Tone control (simple one-pole lowpass)
            yL = self._tone_filter(yL, 'L')
            yR = self._tone_filter(yR, 'R')
            
            outL.append(yL)
            outR.append(yR)
        
        return outL, outR
    
    def _tone_filter(self, x: int, ch: str) -> int:
        """Simple tone control via one-pole lowpass."""
        tone = self.params.tone_q15
        
        # tone = 0 -> full lowpass, tone = 32767 -> bypass
        # alpha = tone (mix coefficient)
        alpha = tone
        alpha_inv = Q15_ONE - alpha
        
        if ch == 'L':
            # y = alpha * x + (1-alpha) * z
            lp = mul_q15_sat(alpha_inv, self.lp_zL) + mul_q15_sat(alpha, x)
            self.lp_zL = sat16(lp)
            return self.lp_zL
        else:
            lp = mul_q15_sat(alpha_inv, self.lp_zR) + mul_q15_sat(alpha, x)
            self.lp_zR = sat16(lp)
            return self.lp_zR
    
    # =========================================================================
    # RANK STATISTICS
    # =========================================================================
    
    def _rank_sample(self, x: int, state: RankState, window: int = 9) -> int:
        """
        Rank statistics filter (median or trimmed mean).
        """
        # Add to circular buffer
        state.buffer[state.idx] = x
        state.idx = (state.idx + 1) % window
        
        # Get sorted window
        window_vals = state.buffer[:window]
        sorted_vals = sorted(window_vals)
        
        # Trimmed mean (drop top/bottom 2)
        trim = 2
        if window > 2 * trim:
            trimmed = sorted_vals[trim:-trim]
            result = sum(trimmed) // len(trimmed)
        else:
            # Median fallback
            result = sorted_vals[window // 2]
        
        return sat16(result)
    
    def _rank_grit(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Rank statistics processing with grit."""
        outL = []
        outR = []
        
        # Window size depends on drive (more drive -> smaller window -> grittier)
        drive = self.params.drive_q15
        window = 17 if drive < Q15_HALF else 9
        
        for i in range(self.B):
            yL = self._rank_sample(L[i], self.rank_L, window)
            yR = self._rank_sample(R[i], self.rank_R, window)
            
            # Mix with original based on drive
            mix = drive
            yL = sat16(mul_q15_sat(yL, mix) + mul_q15_sat(L[i], Q15_ONE - mix))
            yR = sat16(mul_q15_sat(yR, mix) + mul_q15_sat(R[i], Q15_ONE - mix))
            
            outL.append(yL)
            outR.append(yR)
        
        return outL, outR
    
    # =========================================================================
    # TROPICAL CUT
    # =========================================================================
    
    def _tropical_cut(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """
        Tropical algebra: min/max operations for hard cuts.
        """
        outL = []
        outR = []
        
        cut = self.params.cut_q15
        
        for i in range(self.B):
            # Hard threshold cut
            if abs(L[i]) < cut:
                yL = 0
            else:
                yL = L[i]
            
            if abs(R[i]) < cut:
                yR = 0
            else:
                yR = R[i]
            
            outL.append(yL)
            outR.append(yR)
        
        return outL, outR
    
    # =========================================================================
    # CATASTROPHE MAPS
    # =========================================================================
    
    def _cat_fold_sample(self, x: int, state: CatastropheState) -> int:
        """
        Fold catastrophe (A2): V(x) = x³ + ax
        
        Dynamics: x' = -∂V/∂x = -3x² - a
        We use x as fast variable, pressure as 'a' parameter.
        """
        pressure = self.params.pressure_q15
        
        # Scale input to state
        fast = x >> 2  # Reduce input range
        
        # Fold dynamics: s' = s + rate * (-s³ - pressure*s + input)
        s = state.s
        s_sq = mul_q15_sat(s, s)
        s_cu = mul_q15_sat(s_sq, s)
        
        # Potential gradient
        grad = -s_cu - mul_q15_sat(pressure, s)
        
        # Update with input driving
        rate = Q15_HALF  # Integration rate
        delta = mul_q15_sat(rate, sat16(grad + fast))
        
        state.s = sat16(s + (delta >> 2))
        
        # Output is state with soft clip
        return soft_clip_q15(state.s)
    
    def _cat_fold(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Fold catastrophe processing."""
        outL = []
        outR = []
        
        for i in range(self.B):
            yL = self._cat_fold_sample(L[i], self.cat_L)
            yR = self._cat_fold_sample(R[i], self.cat_R)
            
            outL.append(yL)
            outR.append(yR)
        
        return outL, outR
    
    def _cat_cusp_sample(self, x: int, state: CatastropheState) -> int:
        """
        Cusp catastrophe (A3): V(x) = x⁴ + ax² + bx
        
        Two control parameters: pressure (a) and bias (b).
        """
        pressure = self.params.pressure_q15
        bias = self.params.bias_q15
        
        fast = x >> 2
        s = state.s
        
        # Cusp dynamics
        s_sq = mul_q15_sat(s, s)
        s_cu = mul_q15_sat(s_sq, s)
        
        # Gradient: -4s³ - 2as - b
        grad = -mul_q15_sat(s_cu, 4 * Q15_ONE >> 15)
        grad = sat16(grad - mul_q15_sat(pressure, s << 1))
        grad = sat16(grad - bias)
        
        # Drive
        rate = Q15_HALF
        delta = mul_q15_sat(rate, sat16(grad + fast))
        
        state.s = sat16(s + (delta >> 2))
        
        return soft_clip_q15(state.s)
    
    def _cat_cusp(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """Cusp catastrophe processing."""
        outL = []
        outR = []
        
        for i in range(self.B):
            yL = self._cat_cusp_sample(L[i], self.cat_L)
            yR = self._cat_cusp_sample(R[i], self.cat_R)
            
            outL.append(yL)
            outR.append(yR)
        
        return outL, outR
    
    # =========================================================================
    # CLEAN GLUE
    # =========================================================================
    
    def _clean_glue(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """
        Transparent processing with gentle glue compression.
        """
        outL = []
        outR = []

        for i in range(self.B):
            # Very gentle soft clip only if near max
            yL = soft_clip_q15(L[i], threshold=30000)
            yR = soft_clip_q15(R[i], threshold=30000)

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    # =========================================================================
    # DELAY LINE (material_id=6)
    # =========================================================================

    def _delay_sample(self, x: int, state: DelayState, delay_samples: int,
                      feedback_q15: int, mix_q15: int) -> int:
        """
        Single-sample delay line with feedback.

        Args:
            x: Input sample (Q15)
            state: DelayState with circular buffer
            delay_samples: Delay time in samples
            feedback_q15: Feedback amount (Q15, 0-32767)
            mix_q15: Wet/dry mix (Q15: 0=dry, 32767=wet)

        Returns:
            Output sample (Q15)
        """
        # Clamp delay to valid range
        delay_samples = max(1, min(delay_samples, state.max_samples - 1))

        # Read from delay line (with wrap)
        read_idx = (state.write_idx - delay_samples) % state.max_samples
        delayed = state.buffer[read_idx]

        # Write input + feedback to delay line
        fb_signal = mul_q15_sat(delayed, feedback_q15)
        write_val = sat16(x + fb_signal)
        state.buffer[state.write_idx] = write_val

        # Advance write pointer
        state.write_idx = (state.write_idx + 1) % state.max_samples

        # Mix dry/wet
        dry = mul_q15_sat(x, Q15_ONE - mix_q15)
        wet = mul_q15_sat(delayed, mix_q15)

        return sat16(dry + wet)

    def _delay_line(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """
        Delay line processing.

        Uses drive_q15 as delay time (0-32767 maps to 0-500ms)
        Uses tone_q15 as feedback amount
        Uses pressure_q15 as wet/dry mix
        """
        outL = []
        outR = []

        # Map parameters to delay settings
        # drive: 0-32767 -> 0-48000 samples (0-1000ms at 48kHz)
        # FPGA-friendly: mono fits in BRAM (48000 samples = 96KB)
        delay_samples = (self.params.drive_q15 * 48000) >> 15

        # tone: feedback (0-32767)
        # Limit feedback to prevent runaway
        feedback_q15 = min(self.params.tone_q15, 30000)

        # pressure: wet/dry mix
        mix_q15 = self.params.pressure_q15

        for i in range(self.B):
            yL = self._delay_sample(L[i], self.delay_L, delay_samples, feedback_q15, mix_q15)
            yR = self._delay_sample(R[i], self.delay_R, delay_samples, feedback_q15, mix_q15)

            # Soft clip to prevent feedback buildup
            yL = soft_clip_q15(yL)
            yR = soft_clip_q15(yR)

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    # =========================================================================
    # CHORUS/FLANGER (material_id=7)
    # =========================================================================

    def _chorus_sample(self, x: int, state: ChorusState, base_delay: int,
                       depth: int, lfo_inc: int, mix_q15: int) -> int:
        """
        Single-sample chorus with LFO-modulated delay.

        Args:
            x: Input sample (Q15)
            state: ChorusState with circular buffer and LFO
            base_delay: Base delay time in samples
            depth: LFO modulation depth in samples
            lfo_inc: LFO phase increment per sample (Q32)
            mix_q15: Wet/dry mix (Q15)

        Returns:
            Output sample (Q15)
        """
        import math

        # Advance LFO phase
        state.lfo_phase = (state.lfo_phase + lfo_inc) & 0xFFFFFFFF

        # Convert LFO phase to sine (-1 to +1 as delay offset)
        # phase_normalized = state.lfo_phase / 2^32 -> [0, 1)
        # sin_val = sin(2*pi*phase_normalized) -> [-1, +1]
        phase_float = state.lfo_phase / (1 << 32)
        lfo_val = math.sin(2 * math.pi * phase_float)  # -1 to +1

        # Modulated delay time
        delay_samples = base_delay + int(lfo_val * depth)
        delay_samples = max(1, min(delay_samples, state.max_samples - 1))

        # Read from delay line
        read_idx = (state.write_idx - delay_samples) % state.max_samples
        delayed = state.buffer[read_idx]

        # Write input to delay line (no feedback for chorus)
        state.buffer[state.write_idx] = x

        # Advance write pointer
        state.write_idx = (state.write_idx + 1) % state.max_samples

        # Mix dry/wet
        dry = mul_q15_sat(x, Q15_ONE - mix_q15)
        wet = mul_q15_sat(delayed, mix_q15)

        return sat16(dry + wet)

    def _chorus(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """
        Chorus/flanger processing.

        Uses drive_q15 as LFO rate (0-32767 maps to 0.1-5 Hz)
        Uses tone_q15 as modulation depth
        Uses pressure_q15 as wet/dry mix
        Uses bias_q15 as stereo offset (L/R phase difference)
        """
        import math

        outL = []
        outR = []

        # Base delay for chorus effect (10-30ms typical)
        base_delay = 720  # 15ms at 48kHz

        # Map parameters
        # drive: LFO rate (0.1 Hz to 5 Hz)
        # LFO freq = 0.1 + (drive/32767) * 4.9 Hz
        # phase_inc per sample = freq / sample_rate * 2^32
        rate_hz = 0.1 + (self.params.drive_q15 / 32767) * 4.9
        lfo_inc = int(rate_hz / 48000 * (1 << 32))

        # tone: modulation depth (0-480 samples = 0-10ms)
        depth = (self.params.tone_q15 * 480) >> 15

        # pressure: wet/dry mix
        mix_q15 = self.params.pressure_q15

        # Stereo offset: fixed phase difference between L and R
        # bias_q15 = 16384 (0.5) gives 90 degree offset = pi/2
        # Map to radians: stereo_phase_offset = bias_q15 / 32767 * pi
        stereo_phase_offset = (self.params.bias_q15 / 32767) * math.pi

        for i in range(self.B):
            # Process L channel
            yL = self._chorus_sample(L[i], self.chorus_L, base_delay, depth, lfo_inc, mix_q15)

            # Process R channel with fixed phase offset
            # Temporarily apply offset to R LFO
            r_phase_with_offset = int(self.chorus_R.lfo_phase + stereo_phase_offset / (2 * math.pi) * (1 << 32)) & 0xFFFFFFFF
            old_r_phase = self.chorus_R.lfo_phase
            self.chorus_R.lfo_phase = r_phase_with_offset
            yR = self._chorus_sample(R[i], self.chorus_R, base_delay, depth, lfo_inc, mix_q15)
            # Restore R phase to keep L and R in sync (offset is applied each sample)
            self.chorus_R.lfo_phase = (old_r_phase + lfo_inc) & 0xFFFFFFFF

            outL.append(yL)
            outR.append(yR)

        return outL, outR
