"""
Wavelet Engine
==============

Renders VoiceFrames to audio via inverse wavelet packet synthesis.
Uses CDF 5/3 lifting with 5 tree types.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from fixed import (
    sat16, mul_q15_sat, shift_round, slew_q15,
    Q15_ONE, Q15_SHIFT, Q15_HALF
)
from frames import VoiceFrame
from wavelet_packet_trees import PacketTreeSet, BLOCK_SIZE, LEAF_COUNT, NODE_SIZE


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
        
        # Tree transition state per voice
        self.voice_tree_state: Dict[int, Dict] = {}
        
        # Crossfade duration (blocks)
        self.xfade_blocks = 32
    
    def render_voice(self, vf: VoiceFrame) -> List[int]:
        """Render single VoiceFrame to audio block."""
        voice_id = vf.voice_id

        # Use Option A tree control: tree_a_id, tree_b_id, tree_blend_q15
        # Fall back to legacy tree_id if tree_a_id is 0 and tree_id is set
        tree_a_id = vf.tree_a_id if vf.tree_a_id > 0 else vf.tree_id
        tree_b_id = vf.tree_b_id if vf.tree_b_id > 0 else tree_a_id
        blend_q15 = vf.tree_blend_q15

        # Synthesize with blending if needed
        if blend_q15 > 0 and tree_a_id != tree_b_id:
            # Voice Engine is morphing between trees
            x_q15 = self.trees.synthesize_crossfade(
                tree_a_id, tree_b_id,
                vf.coeff_q15, blend_q15
            )
        else:
            # Normal synthesis with primary tree
            x_q15 = self.trees.synthesize(tree_a_id, vf.coeff_q15)

        # Apply voice gain
        x_q15 = [mul_q15_sat(s, vf.gain_q15) for s in x_q15]

        # Ensure correct length
        if len(x_q15) < self.B:
            x_q15.extend([0] * (self.B - len(x_q15)))
        elif len(x_q15) > self.B:
            x_q15 = x_q15[:self.B]

        return x_q15
    
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
        """Mix multiple voice blocks to stereo output."""
        # Accumulate in int32
        accL = [0] * self.B
        accR = [0] * self.B
        
        for vf, x in voice_blocks:
            # Note: gain already applied in render_voice, only apply pan here
            L, R = self.apply_pan(x, vf.pan_q15)
            
            for i in range(self.B):
                accL[i] += L[i]
                accR[i] += R[i]
        
        # Scale and saturate
        # Divide by number of voices to prevent clipping
        n_voices = max(1, len(voice_blocks))
        shift = int(np.log2(n_voices)) if n_voices > 1 else 0
        
        outL = [sat16(accL[i] >> shift) for i in range(self.B)]
        outR = [sat16(accR[i] >> shift) for i in range(self.B)]
        
        return outL, outR
    
    def apply_pan(self, x: List[int], pan: int) -> Tuple[List[int], List[int]]:
        """Apply pan to mono signal, return stereo L/R. No gain change at center."""
        # Pan law: constant power approximation
        # pan: -32768 = full left, 0 = center, +32767 = full right
        
        # At center (pan=0): both channels get full signal
        # At extremes: one channel gets full, other gets zero
        
        if pan <= -16384:
            # Hard left
            return list(x), [0] * len(x)
        elif pan >= 16384:
            # Hard right
            return [0] * len(x), list(x)
        else:
            # Linear crossfade around center
            # pan=0: L=1, R=1 (mono compatible)
            # pan=-16384: L=1, R=0
            # pan=+16384: L=0, R=1
            
            # Scale pan to 0-1 range for right channel
            right_frac = (pan + 16384) / 32768  # 0 to 1
            left_frac = 1.0 - right_frac * 0.5   # 1 to 0.5 to 1
            right_frac = 0.5 + right_frac * 0.5  # 0.5 to 1
            
            # Actually simpler: both channels full at center
            # For now, just pass through at center
            if abs(pan) < 1000:
                return list(x), list(x)
            
            left_gain = int(Q15_ONE * (1.0 - max(0, pan) / 32768))
            right_gain = int(Q15_ONE * (1.0 + min(0, pan) / 32768))
            
            L = [mul_q15_sat(s, left_gain) for s in x]
            R = [mul_q15_sat(s, right_gain) for s in x]
            
            return L, R
