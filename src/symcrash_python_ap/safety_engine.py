"""
Safety Engine
=============

Makes audio blocks "legal" and block-invisible:
- Metrics measurement (RMS, peak, HF, DC)
- Regulation (RG smoothed tracking)
- Q projection (gain/filter adjustment)
- Limiter (peak protection)
"""

from typing import List, Tuple
from dataclasses import dataclass

from fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15, slew_q15_asym,
    sqrt_q15, soft_clip_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)
from frames import ParamFrame, Metrics


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
    Safety processing: metrics, regulation, limiting.
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
    
    def step_block(self, L: List[int], R: List[int], 
                   params: ParamFrame) -> Tuple[List[int], List[int]]:
        """Full safety processing chain."""
        # 1) Measure metrics
        m = self.measure_metrics(L, R)
        
        # 2) Update RG (smoothed tracking)
        self.update_rg(m, params.K_mode)
        
        # 3) Apply Q (gain + HF control)
        Lq, Rq = self.apply_q(L, R, params.K_mode)
        
        # 4) Apply limiter
        Lo, Ro = self.apply_limiter(Lq, Rq)
        
        # 5) Remove DC
        Lo, Ro = self.remove_dc(Lo, Ro)
        
        return Lo, Ro
    
    def get_metrics(self) -> Metrics:
        """Get current smoothed metrics."""
        return Metrics(
            rms_q15=self.rg.E_hat,
            peak_q15=self.rg.peak_hat,
            hf_energy_q15=self.rg.HF_hat,
            dc_offset_q15=self.rg.dc_hat,
            crest_factor_q15=0
        )
