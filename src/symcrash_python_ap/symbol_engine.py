"""
Symbol Engine
=============

Maintains coefficient states and applies operator algebra.
Emits VoiceFrames for wavelet synthesis.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from fixed import (
    sat16, mul_q15_sat, shift_round_sat16, slew_q15,
    Q15_ONE, Q15_SHIFT, mac_q15
)
from frames import ParamFrame, VoiceFrame

# Import operator library
from operator_library import (
    build_operator_library, COEFF_LEN, BAND_SLICES, NUM_BANDS,
    OpType, Operator, from_q15, to_q15
)


class OperatorEngine:
    """
    Applies operators from the library with bit-accurate arithmetic.
    """
    
    def __init__(self, coeff_len: int = COEFF_LEN):
        self.coeff_len = coeff_len
        self.library = build_operator_library()
        self.tree_id = 0
        self.material_id = 1
        
        # Operator program state
        self.current_program: List[int] = [0]  # Start with identity
        
        # Slew state for coefficients
        self.slew_max_delta = 512  # Max change per block
    
    def set_program(self, op_ids: List[int]):
        """Set the operator program (sequence of op_ids)."""
        self.current_program = op_ids
    
    def apply_diag(self, c: List[int], diag: np.ndarray) -> List[int]:
        """Apply diagonal operator (bit-accurate)."""
        result = []
        for i in range(len(c)):
            val = mul_q15_sat(c[i], int(diag[i]))
            result.append(val)
        return result
    
    def apply_permute(self, c: List[int], perm: np.ndarray) -> List[int]:
        """Apply permutation operator."""
        result = [0] * len(c)
        for i, j in enumerate(perm):
            result[i] = c[int(j)]
        return result
    
    def apply_lowrank(self, c: List[int], U: np.ndarray, V: np.ndarray) -> List[int]:
        """Apply low-rank operator: c + Σ U_k (V_k · c)."""
        result = list(c)
        rank = U.shape[0]
        
        for k in range(rank):
            # Compute V_k · c (dot product in int32)
            dot = 0
            for i in range(len(c)):
                dot = mac_q15(dot, int(V[k, i]), c[i])
            dot = shift_round_sat16(dot, Q15_SHIFT)
            
            # Add U_k * dot to result
            for i in range(len(c)):
                contrib = mul_q15_sat(int(U[k, i]), dot)
                result[i] = sat16(result[i] + contrib)
        
        return result
    
    def apply_banded_tridiag(self, c: List[int], op: Operator) -> List[int]:
        """Apply tridiagonal banded operator."""
        result = [0] * len(c)
        
        for band_id, (start, end) in enumerate(BAND_SLICES):
            kernel_info = op.bands_kernel.get(band_id)
            if kernel_info is None or "kernel" not in kernel_info:
                for i in range(start, end):
                    result[i] = c[i]
                continue
            
            kernel = kernel_info["kernel"]
            a, b, a2 = int(kernel[0]), int(kernel[1]), int(kernel[2])
            
            for i in range(start, end):
                i_left = max(start, i - 1)
                i_right = min(end - 1, i + 1)
                
                # Tridiagonal: a*c[i-1] + b*c[i] + a*c[i+1]
                acc = 0
                acc = mac_q15(acc, a, c[i_left])
                acc = mac_q15(acc, b, c[i])
                acc = mac_q15(acc, a2, c[i_right])
                result[i] = shift_round_sat16(acc, Q15_SHIFT)
        
        return result
    
    def apply_inject(self, c: List[int], inject: np.ndarray, amplitude: int = Q15_ONE) -> List[int]:
        """Apply injection (add scaled vector)."""
        result = list(c)
        for i in range(len(c)):
            if inject[i] != 0:
                contrib = mul_q15_sat(int(inject[i]), amplitude)
                result[i] = sat16(result[i] + contrib)
        return result
    
    def apply_proj_energy_cap(self, c: List[int], op: Operator) -> List[int]:
        """Apply energy capping projection per band."""
        result = list(c)
        targets = op.proj_params.get("targets_q15")
        
        if targets is None:
            return result
        
        for band_id, (start, end) in enumerate(BAND_SLICES):
            # Measure band energy (in int32)
            energy = 0
            for i in range(start, end):
                energy += c[i] * c[i]
            
            # Target energy
            target_energy = 0
            for i in range(start, end):
                target_energy += int(targets[i]) * int(targets[i])
            
            # Scale if over budget
            if energy > target_energy and energy > 0:
                # Compute scale factor (approximate)
                scale = int(np.sqrt(target_energy / energy) * Q15_ONE)
                scale = min(Q15_ONE, scale)
                
                for i in range(start, end):
                    result[i] = mul_q15_sat(result[i], scale)
        
        return result
    
    def apply_operator(self, c: List[int], op_id: int, params: ParamFrame) -> List[int]:
        """Apply single operator by ID."""
        if op_id not in self.library:
            return c
        
        op = self.library[op_id]
        
        if op.op_type == OpType.DIAG:
            return self.apply_diag(c, op.diag)
        
        elif op.op_type == OpType.PERMUTE:
            return self.apply_permute(c, op.perm)
        
        elif op.op_type == OpType.LOWRANK:
            return self.apply_lowrank(c, op.U, op.V)
        
        elif op.op_type == OpType.BANDED:
            if op.bands_kernel and "kernel" in list(op.bands_kernel.values())[0]:
                return self.apply_banded_tridiag(c, op)
            return c
        
        elif op.op_type == OpType.INJECT:
            amp = params.gate_q15 if params else Q15_ONE
            return self.apply_inject(c, op.inject_vec, amp)
        
        elif op.op_type == OpType.PROJ:
            return self.apply_proj_energy_cap(c, op)
        
        elif op.op_type == OpType.MIX:
            # Apply both ops with weighted mix
            if op.mix_ops:
                c_a = self.apply_operator(c, op.mix_ops[0], params)
                c_b = self.apply_operator(c, op.mix_ops[1], params)
                w = int(op.mix_weight * Q15_ONE)
                w_inv = Q15_ONE - w
                result = []
                for i in range(len(c)):
                    mixed = mul_q15_sat(c_a[i], w_inv) + mul_q15_sat(c_b[i], w)
                    result.append(sat16(mixed))
                return result
            return c
        
        return c
    
    def apply_program(self, c: List[int], params: ParamFrame) -> List[int]:
        """Apply full operator program."""
        result = c
        for op_id in self.current_program:
            result = self.apply_operator(result, op_id, params)
        return result
    
    def apply_Q(self, c: List[int], params: ParamFrame) -> List[int]:
        """Apply constraint projection Q (FineClamp + NormalizeScale)."""
        result = self.apply_operator(c, 2, params)   # FineClamp
        result = self.apply_operator(result, 11, params)  # NormalizeScale
        return result
    
    def apply_kappa(self, c: List[int], params: ParamFrame) -> List[int]:
        """Apply compression operator kappa (keeps structure sparse)."""
        result = list(c)
        
        # Per-band top-k sparsification
        k_per_band = [4, 8, 16, 24, 32, 44]  # Keep this many per band
        
        for band_id, (start, end) in enumerate(BAND_SLICES):
            band_len = end - start
            k = min(k_per_band[band_id], band_len)
            
            # Get magnitudes
            mags = [(abs(c[i]), i) for i in range(start, end)]
            mags.sort(reverse=True)
            
            # Zero out below top-k
            keep_indices = set(idx for _, idx in mags[:k])
            for i in range(start, end):
                if i not in keep_indices:
                    result[i] = 0
        
        return result
    
    def slew_coeffs(self, c_old: List[int], c_new: List[int]) -> List[int]:
        """Slew coefficients to prevent discontinuities."""
        result = []
        for old, new in zip(c_old, c_new):
            result.append(slew_q15(old, new, self.slew_max_delta))
        return result


class SymbolEngine:
    """
    Maintains per-voice coefficient state and emits VoiceFrames.
    """
    
    def __init__(self, n_voices: int = 4, coeff_len: int = COEFF_LEN):
        self.n_voices = n_voices
        self.coeff_len = coeff_len
        
        # Per-voice coefficient state
        self.c_q15: List[List[int]] = [[0] * coeff_len for _ in range(n_voices)]
        
        # Operator engine
        self.ops = OperatorEngine(coeff_len)
        
        # Per-voice settings
        self.voice_gains = [0x4000] * n_voices  # 0.5 default
        self.voice_pans = [0] * n_voices
        self.voice_trees = [0] * n_voices
        self.voice_materials = [1] * n_voices
        
        # Active voices
        self.active_voices = set(range(min(2, n_voices)))  # Start with 2 voices
    
    def set_voice_program(self, voice_id: int, op_ids: List[int]):
        """Set operator program for a voice."""
        # For now, same program for all voices
        self.ops.set_program(op_ids)
    
    def inject_note(self, voice_id: int, note_coeffs: List[int]):
        """Inject note coefficients into a voice."""
        if voice_id < self.n_voices:
            for i in range(min(len(note_coeffs), self.coeff_len)):
                self.c_q15[voice_id][i] = sat16(
                    self.c_q15[voice_id][i] + note_coeffs[i]
                )
            self.active_voices.add(voice_id)
    
    def step_block(self, params: ParamFrame) -> List[VoiceFrame]:
        """Process one block, return VoiceFrames for active voices."""
        frames: List[VoiceFrame] = []
        
        for vid in self.active_voices:
            # 1) Apply operator program
            c_next = self.ops.apply_program(self.c_q15[vid], params)
            
            # 2) Apply constraint Q
            c_next = self.ops.apply_Q(c_next, params)
            
            # 3) Apply compression kappa
            c_next = self.ops.apply_kappa(c_next, params)
            
            # 4) Slew for smoothness
            c_next = self.ops.slew_coeffs(self.c_q15[vid], c_next)
            
            # 5) Update state
            self.c_q15[vid] = c_next
            
            # 6) Emit VoiceFrame
            frames.append(VoiceFrame(
                tag=0xA0,
                voice_id=vid,
                tree_id=self.voice_trees[vid],
                flags=0,
                gain_q15=self.voice_gains[vid],
                pan_q15=self.voice_pans[vid],
                material_id=self.voice_materials[vid],
                role_id=0,
                aux_flags=0,
                aux_param_q15=0,
                coeff_q15=list(c_next)
            ))
        
        return frames
    
    def get_voice_energy(self, voice_id: int) -> int:
        """Get total energy of voice (for gating decisions)."""
        if voice_id >= self.n_voices:
            return 0
        energy = sum(c * c for c in self.c_q15[voice_id])
        return min(Q15_ONE, energy >> 15)
