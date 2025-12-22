"""
Data Structures and Frame Types
===============================

Block-rate data contracts for the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParamFrame:
    """Block-rate scalar parameters."""
    K_mode: int = 1              # 0=clean, 1=warm, 2=crunch, 3=destroy
    drive_q15: int = 0x2000      # Drive amount
    tone_q15: int = 0x4000       # Tone control
    bias_q15: int = 0            # DC bias
    pressure_q15: int = 0        # Catastrophe pressure
    cut_q15: int = 0             # Tropical cut threshold
    morph_q15: int = 0           # Operator morph position
    gate_q15: int = 0x7FFF       # Gate/envelope


@dataclass
class VoiceFrame:
    """
    Per-voice coefficient frame.

    Tree control uses Option A (two tree IDs + blend Q15) for clean separation:
    - Voice Engine decides WHEN tree changes happen (musical timing)
    - Wavelet Engine handles HOW to blend coefficients (numerical)

    Tree IDs: 0=Balanced, 1=Harmonic, 2=Texture, 3=Impact, 4=Drone
    """
    tag: int = 0xA0
    voice_id: int = 0
    tree_id: int = 0             # Legacy: single tree ID (use tree_a_id instead)
    flags: int = 0
    gain_q15: int = 0x4000       # Voice gain (0.5 default)
    pan_q15: int = 0             # Pan (-1..+1 as Q1.15)
    material_id: int = 0
    role_id: int = 0
    aux_flags: int = 0
    aux_param_q15: int = 0
    coeff_q15: List[int] = field(default_factory=list)

    # Tree control - Option A (Voice Engine emits, Wavelet Engine consumes)
    tree_a_id: int = 0           # Primary tree ID (0-4)
    tree_b_id: int = 0           # Secondary tree ID (0-4)
    tree_blend_q15: int = 0      # Blend: 0 = pure A, 32767 = pure B

    # Musical envelopes (from Voice Engine ADSR)
    tilt_env_q15: int = 0x4000   # Brightness/tilt envelope
    density_env_q15: int = 0x4000  # Coefficient density envelope

    # Voice state (for Safety Engine de-click)
    is_starting: bool = False    # True on first block after note-on
    is_stopping: bool = False    # True when transitioning to FREE
    is_active: bool = True       # False when voice is done

    # Legacy compatibility: convert float betas to IDs
    @property
    def tree_a(self) -> float:
        """Legacy: return tree_a_id as beta (0-1)."""
        return self.tree_a_id / 4.0

    @tree_a.setter
    def tree_a(self, value: float):
        """Legacy: set tree_a_id from beta."""
        self.tree_a_id = int(value * 4.0)

    @property
    def tree_b(self) -> float:
        """Legacy: return tree_b_id as beta (0-1)."""
        return self.tree_b_id / 4.0

    @tree_b.setter
    def tree_b(self, value: float):
        """Legacy: set tree_b_id from beta."""
        self.tree_b_id = int(value * 4.0)

    @property
    def tree_blend(self) -> float:
        """Legacy: return tree_blend_q15 as float (0-1)."""
        return self.tree_blend_q15 / 32767.0

    @tree_blend.setter
    def tree_blend(self, value: float):
        """Legacy: set tree_blend_q15 from float."""
        self.tree_blend_q15 = int(value * 32767)


@dataclass
class BlockMarker:
    """Block boundary marker."""
    tag: int = 0xB0
    block_id: int = 0
    timestamp: int = 0


@dataclass 
class MaterialParams:
    """Slewed material parameters (internal state)."""
    drive_q15: int = 0x2000
    tone_q15: int = 0x4000
    bias_q15: int = 0
    pressure_q15: int = 0
    cut_q15: int = 0
    
    def slew_from(self, pf: ParamFrame, rate: int = 256):
        """Slew toward ParamFrame values."""
        from fixed import slew_q15
        self.drive_q15 = slew_q15(self.drive_q15, pf.drive_q15, rate)
        self.tone_q15 = slew_q15(self.tone_q15, pf.tone_q15, rate)
        self.bias_q15 = slew_q15(self.bias_q15, pf.bias_q15, rate)
        self.pressure_q15 = slew_q15(self.pressure_q15, pf.pressure_q15, rate)
        self.cut_q15 = slew_q15(self.cut_q15, pf.cut_q15, rate)


@dataclass
class Metrics:
    """Block metrics for safety/monitoring."""
    rms_q15: int = 0
    peak_q15: int = 0
    hf_energy_q15: int = 0
    dc_offset_q15: int = 0
    crest_factor_q15: int = 0
