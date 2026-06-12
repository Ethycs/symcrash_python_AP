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

from typing import List, Tuple, Generic, TypeVar, Callable
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
class SVFState:
    """
    State for TPT (Topology-Preserving Transform) SVF filter.

    The TPT SVF is derived from the analog SVF using the trapezoidal rule,
    giving excellent frequency response and stability at all frequencies.

    State variables:
    - ic1eq: First integrator state (Q15)
    - ic2eq: Second integrator state (Q15)

    Reference: Vadim Zavalishin, "The Art of VA Filter Design"
    """
    ic1eq: int = 0  # First integrator state (Q15)
    ic2eq: int = 0  # Second integrator state (Q15)

    def reset(self):
        """Reset filter state."""
        self.ic1eq = 0
        self.ic2eq = 0


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


@dataclass
class ReverbState:
    """
    State for Schroeder reverb (MONO).

    FPGA NOTE: This reverb requires significant memory for delay lines.
    Total buffer: ~100ms worth = 4800 samples per channel = ~10KB per channel.
    For stereo: ~20KB total. Use external SRAM/SDRAM, not BRAM.

    Structure: 4 parallel comb filters -> 2 series allpass filters
    """
    # Comb filter delay lines (parallel)
    # Delay times chosen for dense, non-metallic reverb
    comb1: List[int] = None  # 29.7ms = 1426 samples
    comb2: List[int] = None  # 37.1ms = 1781 samples
    comb3: List[int] = None  # 41.1ms = 1973 samples
    comb4: List[int] = None  # 43.7ms = 2098 samples
    comb_idx: List[int] = None

    # Allpass filter delay lines (series)
    ap1: List[int] = None    # 5.0ms = 240 samples
    ap2: List[int] = None    # 1.7ms = 82 samples
    ap_idx: List[int] = None

    # Comb filter delays (samples at 48kHz)
    COMB_DELAYS = (1426, 1781, 1973, 2098)
    # Allpass delays
    AP_DELAYS = (240, 82)

    def __post_init__(self):
        if self.comb1 is None:
            self.comb1 = [0] * self.COMB_DELAYS[0]
            self.comb2 = [0] * self.COMB_DELAYS[1]
            self.comb3 = [0] * self.COMB_DELAYS[2]
            self.comb4 = [0] * self.COMB_DELAYS[3]
            self.comb_idx = [0, 0, 0, 0]

            self.ap1 = [0] * self.AP_DELAYS[0]
            self.ap2 = [0] * self.AP_DELAYS[1]
            self.ap_idx = [0, 0]


@dataclass
class KarplusStrongState:
    """
    State for Karplus-Strong plucked string synthesis (MONO).

    The classic Karplus-Strong algorithm:
    1. Fill delay line with noise burst (exciter)
    2. Each sample: output = delay_line[read], write = avg(old, delay_line[read])
    3. The averaging filter determines decay rate and timbre

    FPGA NOTE: ~2ms delay for 500Hz fundamental = 96 samples = 192 bytes
    Full range 20Hz-5kHz needs ~2400 samples max = ~5KB per channel
    Fits comfortably in BRAM.

    The delay length sets the pitch: delay_samples = sample_rate / frequency
    """
    buffer: List[int] = None
    write_idx: int = 0
    delay_samples: int = 109  # Default ~440Hz at 48kHz
    prev_sample: int = 0      # For averaging filter
    max_samples: int = 2400   # ~20Hz at 48kHz

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = [0] * self.max_samples

    def reset(self, delay_samples: int = None):
        """Reset state, optionally setting new delay length."""
        if delay_samples is not None:
            self.delay_samples = max(1, min(delay_samples, self.max_samples - 1))
        self.buffer = [0] * self.max_samples
        self.write_idx = 0
        self.prev_sample = 0


# =============================================================================
# STEREO STATE WRAPPER
# =============================================================================

T = TypeVar('T')


@dataclass
class StereoState(Generic[T]):
    """
    Wrapper for stereo state pairs.

    Consolidates left/right channel state into a single container.
    Reduces boilerplate in MaterialEngine initialization.
    """
    left: T
    right: T

    @classmethod
    def create(cls, factory: Callable[[], T]) -> 'StereoState[T]':
        """Create stereo state from a factory function."""
        return cls(left=factory(), right=factory())


# =============================================================================
# MATERIAL STRATEGY FUNCTIONS (dispatch targets)
# =============================================================================
# These are the method signatures for material processing strategies.
# Type alias for clarity: (engine, L, R, **kwargs) -> (L_out, R_out)
MaterialStrategy = Callable[['MaterialEngine', List[int], List[int]], Tuple[List[int], List[int]]]


class MaterialEngine:
    """
    Applies material programs to stereo audio blocks.
    All processing is bit-accurate Q1.15.
    """
    
    def __init__(self, block_size: int = 64):
        self.B = block_size

        # Stereo state pairs using StereoState wrapper
        self.tape = StereoState.create(TapeState)
        self.cat = StereoState.create(CatastropheState)
        self.rank = StereoState.create(RankState)
        self.delay = StereoState.create(DelayState)       # material_id=6
        self.chorus = StereoState.create(ChorusState)     # material_id=7
        self.reverb = StereoState.create(ReverbState)     # material_id=8 (FPGA: use external SRAM)
        self.svf = StereoState.create(SVFState)           # material_id=9
        self.ks = StereoState.create(KarplusStrongState)  # material_id=10

        # Backward compatibility aliases (for existing code that uses _L/_R)
        self.tape_L, self.tape_R = self.tape.left, self.tape.right
        self.cat_L, self.cat_R = self.cat.left, self.cat.right
        self.rank_L, self.rank_R = self.rank.left, self.rank.right
        self.delay_L, self.delay_R = self.delay.left, self.delay.right
        self.chorus_L, self.chorus_R = self.chorus.left, self.chorus.right
        self.reverb_L, self.reverb_R = self.reverb.left, self.reverb.right
        self.svf_L, self.svf_R = self.svf.left, self.svf.right
        self.ks_L, self.ks_R = self.ks.left, self.ks.right

        # Slewed parameters
        self.params = MaterialParams()

        # One-pole lowpass state for tone control
        self.lp_zL = 0
        self.lp_zR = 0

        # Strategy dispatch table (material_id -> processing method)
        self._strategies: dict[int, MaterialStrategy] = {
            0: self._clean_glue,
            1: self._tape_bus,
            2: self._rank_grit,
            3: self._tropical_cut,
            4: self._cat_fold,
            5: self._cat_cusp,
            6: self._delay_line,
            7: self._chorus,
            8: self._reverb,
            # 9: SVF handled separately (has extra params)
            10: self._karplus_strong,
        }
    
    def step_block(self, L: List[int], R: List[int], params: ParamFrame,
                   material_id: int, cat_id: int = 0,
                   filter_cutoff_q15: int = Q15_ONE,
                   filter_reso_q15: int = 0,
                   filter_mode: int = 0) -> Tuple[List[int], List[int]]:
        """
        Apply selected material program to stereo block.

        material_id:
            0 = clean (transparent)
            1 = tape bus
            2 = rank grit
            3 = tropical cut
            4 = catastrophe snap (fold)
            5 = catastrophe A4 (cusp)
            6 = delay line
            7 = chorus/flanger
            8 = reverb
            9 = SVF filter (TPT)
            10 = Karplus-Strong pluck

        Filter parameters (used when material_id=9):
            filter_cutoff_q15: Cutoff frequency (Q15: 0=20Hz, 32767=20kHz, log scale)
            filter_reso_q15: Resonance (Q15: 0=flat, 32767=self-oscillation)
            filter_mode: 0=LP, 1=HP, 2=BP, 3=Notch
        """
        # Slew parameters
        self.params.slew_from(params)

        # SVF filter has extra parameters, handle separately
        if material_id == 9:
            return self._svf_filter(L, R, filter_cutoff_q15, filter_reso_q15, filter_mode)

        # Use strategy dispatch table, default to clean_glue
        strategy = self._strategies.get(material_id, self._clean_glue)
        return strategy(L, R)
    
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
        rate_hz = 0.1 + (self.params.drive_q15 / Q15_ONE) * 4.9
        lfo_inc = int(rate_hz / 48000 * (1 << 32))

        # tone: modulation depth (0-480 samples = 0-10ms)
        depth = (self.params.tone_q15 * 480) >> 15

        # pressure: wet/dry mix
        mix_q15 = self.params.pressure_q15

        # Stereo offset: fixed phase difference between L and R
        # bias_q15 = 16384 (0.5) gives 90 degree offset = pi/2
        # Map to radians: stereo_phase_offset = bias_q15 / 32767 * pi
        stereo_phase_offset = (self.params.bias_q15 / Q15_ONE) * math.pi

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

    # =========================================================================
    # REVERB (material_id=8)
    # =========================================================================
    #
    # FPGA IMPLEMENTATION NOTES:
    # --------------------------
    # Total delay buffer per channel: ~7600 samples = ~15KB (Q15)
    # Stereo total: ~30KB - TOO LARGE FOR BRAM
    #
    # Recommended FPGA approach:
    # 1. Use external SRAM or SDRAM for delay lines
    # 2. Prefetch next samples during processing
    # 3. Or use reduced delay times (halve for ~15KB total)
    # 4. Consider mono reverb bus with stereo decorrelation at output
    #
    # Alternative: Freeverb-style with 8 combs + 4 allpass = more memory
    # This Schroeder design is more FPGA-friendly (4 combs + 2 allpass)

    def _comb_filter(self, x: int, buffer: List[int], idx: int, delay: int,
                     feedback_q15: int) -> Tuple[int, int]:
        """
        Comb filter: y[n] = x[n-D] + g * y[n-D]

        Returns (output, new_idx)
        """
        # Read delayed sample
        read_idx = idx
        delayed = buffer[read_idx]

        # Output is just the delayed sample
        output = delayed

        # Write input + feedback to buffer
        fb = mul_q15_sat(delayed, feedback_q15)
        buffer[idx] = sat16(x + fb)

        # Advance index
        new_idx = (idx + 1) % delay

        return output, new_idx

    def _allpass_filter(self, x: int, buffer: List[int], idx: int, delay: int,
                        g_q15: int) -> Tuple[int, int]:
        """
        Allpass filter: y[n] = -g*x[n] + x[n-D] + g*y[n-D]

        Returns (output, new_idx)
        """
        # Read delayed sample
        delayed = buffer[idx]

        # Allpass output: -g*x + delayed + g*delayed_feedback
        # Simplified: output = delayed - g * (x - delayed)
        diff = sat16(x - delayed)
        output = sat16(delayed - mul_q15_sat(g_q15, diff))

        # Write input to buffer (for next iteration)
        buffer[idx] = x

        # Advance index
        new_idx = (idx + 1) % delay

        return output, new_idx

    def _reverb_sample(self, x: int, state: ReverbState,
                       decay_q15: int, damping_q15: int) -> int:
        """
        Single-sample Schroeder reverb.

        Args:
            x: Input sample (Q15)
            state: ReverbState with comb and allpass buffers
            decay_q15: Reverb decay/size (Q15, 0-32767)
            damping_q15: High frequency damping (Q15, 0-32767)

        Returns:
            Reverb output (Q15)
        """
        # Scale input to prevent clipping in feedback loops
        x_scaled = x >> 2

        # Comb filter feedback based on decay
        # Higher decay = longer reverb tail
        # Typical comb feedback: 0.7-0.9 for natural reverb
        fb = (decay_q15 * 28000) >> 15  # Scale to ~0.85 max

        # Process 4 parallel comb filters
        c1, state.comb_idx[0] = self._comb_filter(
            x_scaled, state.comb1, state.comb_idx[0],
            state.COMB_DELAYS[0], fb)
        c2, state.comb_idx[1] = self._comb_filter(
            x_scaled, state.comb2, state.comb_idx[1],
            state.COMB_DELAYS[1], fb)
        c3, state.comb_idx[2] = self._comb_filter(
            x_scaled, state.comb3, state.comb_idx[2],
            state.COMB_DELAYS[2], fb)
        c4, state.comb_idx[3] = self._comb_filter(
            x_scaled, state.comb4, state.comb_idx[3],
            state.COMB_DELAYS[3], fb)

        # Sum comb outputs (with scaling to prevent overflow)
        comb_sum = (c1 + c2 + c3 + c4) >> 2

        # Apply damping (simple lowpass via averaging with previous)
        # Higher damping = more muffled reverb
        # TODO: Could add proper one-pole lowpass here

        # Series allpass filters for diffusion
        # Allpass coefficient ~0.5 for good diffusion
        ap_g = Q15_HALF

        ap1_out, state.ap_idx[0] = self._allpass_filter(
            comb_sum, state.ap1, state.ap_idx[0],
            state.AP_DELAYS[0], ap_g)
        ap2_out, state.ap_idx[1] = self._allpass_filter(
            ap1_out, state.ap2, state.ap_idx[1],
            state.AP_DELAYS[1], ap_g)

        # Scale output back up
        return sat16(ap2_out << 1)

    def _reverb(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """
        Reverb processing.

        Uses drive_q15 as decay/size (0-32767: small room to large hall)
        Uses tone_q15 as damping (0=bright, 32767=dark)
        Uses pressure_q15 as wet/dry mix
        """
        outL = []
        outR = []

        # Map parameters
        decay_q15 = self.params.drive_q15
        damping_q15 = self.params.tone_q15
        mix_q15 = self.params.pressure_q15

        for i in range(self.B):
            # Process reverb
            wetL = self._reverb_sample(L[i], self.reverb_L, decay_q15, damping_q15)
            wetR = self._reverb_sample(R[i], self.reverb_R, decay_q15, damping_q15)

            # Mix dry/wet
            dryL = mul_q15_sat(L[i], Q15_ONE - mix_q15)
            dryR = mul_q15_sat(R[i], Q15_ONE - mix_q15)
            wetL = mul_q15_sat(wetL, mix_q15)
            wetR = mul_q15_sat(wetR, mix_q15)

            outL.append(sat16(dryL + wetL))
            outR.append(sat16(dryR + wetR))

        return outL, outR

    # =========================================================================
    # TPT SVF FILTER (material_id=9)
    # =========================================================================
    #
    # Topology-Preserving Transform State Variable Filter
    #
    # Based on Vadim Zavalishin's "The Art of VA Filter Design"
    # This is the standard TPT SVF used in modern synths (Serum, Vital, etc.)
    #
    # Benefits:
    # - Stable at all frequencies up to Nyquist
    # - Correct frequency response (no cramping)
    # - Self-oscillation at high resonance
    # - Clean, musical sound
    #
    # FPGA NOTES:
    # - 2 state variables per channel (4 total for stereo)
    # - Requires tan() lookup or polynomial approximation
    # - ~10 multiplies per sample per channel
    # - Can be pipelined for high throughput

    def _svf_coefficients(self, cutoff_q15: int, reso_q15: int, sample_rate: int = 48000):
        """
        Compute TPT SVF coefficients from cutoff and resonance.

        Args:
            cutoff_q15: Cutoff frequency (Q15: 0=20Hz, 32767=20kHz log scale)
            reso_q15: Resonance (Q15: 0=flat, 32767=near self-oscillation)
            sample_rate: Sample rate in Hz

        Returns:
            (g, k) where:
            - g: Frequency coefficient (tan(pi * fc / fs))
            - k: Damping coefficient (2 - 2*Q, where Q = 0.5 to 25)
        """
        import math

        # Map cutoff_q15 to frequency (log scale: 20Hz to 20kHz)
        # log2(20) ≈ 4.32, log2(20000) ≈ 14.29
        # Range: ~10 octaves
        log_min = math.log2(20)
        log_max = math.log2(20000)
        log_freq = log_min + (cutoff_q15 / Q15_ONE) * (log_max - log_min)
        fc = 2 ** log_freq

        # Clamp to valid range
        fc = max(20, min(fc, sample_rate * 0.49))

        # TPT coefficient: g = tan(pi * fc / fs)
        g = math.tan(math.pi * fc / sample_rate)

        # Map resonance to Q factor
        # reso_q15 = 0 -> Q = 0.5 (no resonance)
        # reso_q15 = 32767 -> Q = 25 (near self-oscillation)
        # k = 2 - 2*Q = 2 - 1/Q (damping)
        # Q = 0.5 + (reso/32767) * 24.5
        Q = 0.5 + (reso_q15 / Q15_ONE) * 24.5
        k = 2.0 - 1.0 / Q

        # Clamp k to prevent instability (k must be > 0 for stability)
        k = max(0.01, k)

        return g, k

    def _svf_sample(self, x: int, state: SVFState, g: float, k: float,
                    mode: int) -> int:
        """
        Process single sample through TPT SVF.

        TPT SVF equations:
            v1 = (x - ic2eq) * g / (1 + g*(g + k))
            v2 = ic2eq + g * v1

            lp = v2 + ic2eq
            bp = v1 + ic1eq
            hp = x - k*bp - lp

            ic1eq = 2*v1 - ic1eq
            ic2eq = 2*v2 - ic2eq

        Args:
            x: Input sample (Q15)
            state: SVFState with integrator states
            g: Frequency coefficient
            k: Damping coefficient
            mode: 0=LP, 1=HP, 2=BP, 3=Notch

        Returns:
            Filtered output (Q15)
        """
        # Convert input to float for processing
        x_f = x / 32767.0

        # Convert state to float
        ic1eq = state.ic1eq / 32767.0
        ic2eq = state.ic2eq / 32767.0

        # TPT SVF core
        # v1 = g * (x - ic2eq) / (1 + g*(g + k))
        denom = 1.0 + g * (g + k)
        v1 = g * (x_f - ic2eq) / denom
        v2 = ic2eq + g * v1

        # Compute outputs
        lp = v2 + ic2eq
        bp = v1 + ic1eq
        hp = x_f - k * bp - lp
        notch = lp + hp  # Notch = LP + HP

        # Update state
        state.ic1eq = int((2.0 * v1 - ic1eq) * Q15_ONE)
        state.ic2eq = int((2.0 * v2 - ic2eq) * Q15_ONE)

        # Clamp state to prevent runaway
        state.ic1eq = max(-Q15_ONE, min(Q15_ONE, state.ic1eq))
        state.ic2eq = max(-Q15_ONE, min(Q15_ONE, state.ic2eq))

        # Select output based on mode
        if mode == 0:
            out = lp
        elif mode == 1:
            out = hp
        elif mode == 2:
            out = bp
        else:  # mode == 3
            out = notch

        # Convert back to Q15
        return sat16(int(out * Q15_ONE))

    def _svf_filter(self, L: List[int], R: List[int],
                    cutoff_q15: int, reso_q15: int, mode: int) -> Tuple[List[int], List[int]]:
        """
        TPT SVF filter processing.

        Args:
            L, R: Input stereo block (Q15)
            cutoff_q15: Cutoff frequency (Q15 log scale)
            reso_q15: Resonance (Q15)
            mode: 0=LP, 1=HP, 2=BP, 3=Notch

        Returns:
            Filtered stereo block (Q15)
        """
        outL = []
        outR = []

        # Compute coefficients (constant for block)
        g, k = self._svf_coefficients(cutoff_q15, reso_q15)

        for i in range(self.B):
            yL = self._svf_sample(L[i], self.svf_L, g, k, mode)
            yR = self._svf_sample(R[i], self.svf_R, g, k, mode)

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    # =========================================================================
    # KARPLUS-STRONG (material_id=10)
    # =========================================================================
    #
    # Classic plucked string synthesis using a delay line with averaging filter.
    #
    # The input signal acts as an exciter - typically noise burst or impulse.
    # The delay line length determines pitch: delay = sample_rate / frequency
    # The averaging filter determines decay rate and brightness.
    #
    # Parameters mapped:
    #   drive_q15: Exciter mix (0=resonator only, 32767=full input)
    #   tone_q15: Damping (0=bright/long, 32767=dark/short decay)
    #   pressure_q15: Feedback amount (0=no feedback, 32767=infinite sustain)
    #   bias_q15: Pitch offset (for detuning/stereo spread)
    #
    # FPGA NOTES:
    # - Simple algorithm: 1 read, 1 write, 1 average per sample
    # - Memory: ~5KB per channel for full pitch range
    # - Deterministic: fixed operations per sample

    def _ks_sample(self, x: int, state: KarplusStrongState,
                   exciter_mix_q15: int, damping_q15: int,
                   feedback_q15: int) -> int:
        """
        Single-sample Karplus-Strong with exciter input.

        Args:
            x: Input/exciter sample (Q15)
            state: KarplusStrongState with delay buffer
            exciter_mix_q15: How much input goes into the delay (Q15)
            damping_q15: Filter coefficient for decay (Q15: 0=bright, 32767=dark)
            feedback_q15: Feedback gain (Q15: ~32000 for long sustain)

        Returns:
            Output sample (Q15)
        """
        delay = state.delay_samples

        # Read from delay line
        read_idx = (state.write_idx - delay) % state.max_samples
        if read_idx < 0:
            read_idx += state.max_samples
        delayed = state.buffer[read_idx]

        # Output is the delayed sample
        output = delayed

        # Averaging filter: new = (prev + current) / 2
        # This is the classic KS lowpass that causes decay
        # More damping = more low-pass = faster high-frequency decay
        #
        # Adjustable damping: lerp between current and average
        # damping_q15=0: output unchanged (bright, long sustain)
        # damping_q15=32767: full averaging (dark, fast decay)
        avg = (state.prev_sample + delayed) >> 1
        filtered = delayed + mul_q15_sat(avg - delayed, damping_q15)

        # Apply feedback gain (controls overall decay rate)
        feedback_signal = mul_q15_sat(filtered, feedback_q15)

        # Mix exciter input with feedback
        # exciter_mix_q15=0: pure resonator (no new energy)
        # exciter_mix_q15=32767: full input (continuous excitation)
        exciter = mul_q15_sat(x, exciter_mix_q15)
        write_val = sat16(exciter + feedback_signal)

        # Write to delay line
        state.buffer[state.write_idx] = write_val

        # Update state
        state.prev_sample = delayed
        state.write_idx = (state.write_idx + 1) % state.max_samples

        return output

    def _karplus_strong(self, L: List[int], R: List[int]) -> Tuple[List[int], List[int]]:
        """
        Karplus-Strong plucked string processing.

        Uses material params:
            drive_q15: Exciter amount (0=resonator, 32767=full input)
            tone_q15: Damping/brightness (0=bright, 32767=dark)
            pressure_q15: Feedback/sustain (0=short, 32767=long)
            bias_q15: Stereo detune offset (adds slight pitch difference L/R)
        """
        outL = []
        outR = []

        # Map parameters
        exciter_mix = self.params.drive_q15
        damping = self.params.tone_q15
        # Feedback should be high for musical sustain (~0.95-0.99)
        # Map pressure 0-32767 to feedback 0.9-0.999 range
        # fb = 29491 + (pressure/32767) * 3276 = ~0.9 to ~0.999
        feedback = 29491 + ((self.params.pressure_q15 * 3276) >> 15)

        for i in range(self.B):
            yL = self._ks_sample(L[i], self.ks_L, exciter_mix, damping, feedback)
            yR = self._ks_sample(R[i], self.ks_R, exciter_mix, damping, feedback)

            outL.append(yL)
            outR.append(yR)

        return outL, outR

    def set_karplus_pitch(self, freq_hz: float, sample_rate: int = 48000):
        """
        Set Karplus-Strong pitch by frequency.

        Args:
            freq_hz: Desired fundamental frequency (20-5000 Hz typical)
            sample_rate: Audio sample rate

        This sets the delay line length for both L and R channels.
        For stereo spread, manually adjust L/R delay_samples slightly.
        """
        delay_samples = int(sample_rate / freq_hz)
        delay_samples = max(10, min(delay_samples, self.ks_L.max_samples - 1))

        self.ks_L.delay_samples = delay_samples
        self.ks_R.delay_samples = delay_samples

    def excite_karplus(self, noise_level_q15: int = Q15_ONE):
        """
        Excite the Karplus-Strong string with a noise burst.

        This fills the delay line with noise to create the "pluck" attack.
        Call this on note-on for realistic plucked string sound.

        Args:
            noise_level_q15: Amplitude of noise burst (Q15)
        """
        import random

        for i in range(self.ks_L.delay_samples):
            noise = random.randint(-noise_level_q15, noise_level_q15)
            self.ks_L.buffer[i] = noise

        for i in range(self.ks_R.delay_samples):
            noise = random.randint(-noise_level_q15, noise_level_q15)
            self.ks_R.buffer[i] = noise
