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
