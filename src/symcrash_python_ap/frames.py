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
    """Per-voice coefficient frame."""
    tag: int = 0xA0
    voice_id: int = 0
    tree_id: int = 0
    flags: int = 0
    gain_q15: int = 0x4000       # Voice gain (0.5 default)
    pan_q15: int = 0             # Pan (-1..+1 as Q1.15)
    material_id: int = 0
    role_id: int = 0
    aux_flags: int = 0
    aux_param_q15: int = 0
    coeff_q15: List[int] = field(default_factory=list)


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
