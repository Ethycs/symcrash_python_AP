"""
Cathedral of Stress - Operator Programs
=======================================

Defines operator scheduling for each segment of the torture test.
Maps operator IDs to the 24-operator library.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from enum import IntEnum

from test_config import (
    BLOCKS_PER_SEC, should_trigger, ramp, slew_int,
    SegmentTiming, SEGMENTS
)


# =============================================================================
# OPERATOR ID MAPPING
# =============================================================================

class OpID(IntEnum):
    """
    Operator IDs matching the 24-operator library.
    
    DIAG family (0-3):
    """
    IDENTITY = 0
    TILT = 1
    COMPRESS_K = 2
    INTRA_MIX = 3
    
    # BANDED family (4-7):
    DIFFUSE_UP = 4
    DIFFUSE_DOWN = 5
    BAND_SHEAR = 6
    LOWRANK_MORPH = 7
    
    # PERMUTE family (8-11):
    GLOBAL_PULSE = 8
    SWAP_BANDS = 9
    ROTATE_NODES = 10
    NORMALIZE_SCALE = 11
    
    # SPECIAL family (12-15):
    GLUE_PAIR = 12
    IMPULSE = 13
    NOISE_BAND = 14
    PRESSURE_RAMP = 15
    
    # ADVANCED family (16-19):
    GATE_MASK = 16
    QUASI_ROTATE = 17
    STURMIAN_GATE = 18
    EMERGENCY_SIMPLIFY = 19
    
    # CATASTROPHE family (20-23):
    FOLD_PRESSURE = 20
    CUSP_INJECTION = 21
    SWALLOWTAIL_MASK = 22
    SHEAF_PROJECT = 23


# =============================================================================
# OPERATOR SCHEDULE ENTRY
# =============================================================================

@dataclass
class OperatorEvent:
    """A single operator application event."""
    op_id: int
    strength: float = 1.0      # 0-1 operator strength
    target_bands: Optional[List[int]] = None  # Which bands to affect
    voice_mask: Optional[List[int]] = None    # Which voices to affect


@dataclass  
class OperatorProgram:
    """
    Operator program for a block.
    
    Contains backbone (continuous) operators and event (triggered) operators.
    """
    backbone: List[OperatorEvent]
    events: List[OperatorEvent]
    
    def all_operators(self) -> List[OperatorEvent]:
        return self.backbone + self.events


# =============================================================================
# SEGMENT A: COHERENT HARMONIC ORGAN
# =============================================================================

def get_segment_a_program(block: int, seg: SegmentTiming) -> OperatorProgram:
    """
    Segment A (0.00-1.00s): Coherent harmonic organ that starts to deform.
    
    Backbone: Tilt + NormalizeScale + LowRankMorph + slow DiffuseUp
    Events: GlobalPulse every 16 blocks, GluePair every 64 blocks
    """
    t = seg.progress(block)
    local_block = block - seg.start_block
    
    # Continuous backbone
    backbone = [
        OperatorEvent(OpID.TILT, strength=0.3 + 0.2 * t),
        OperatorEvent(OpID.NORMALIZE_SCALE, strength=0.8),
        OperatorEvent(OpID.LOWRANK_MORPH, strength=0.2 + 0.3 * t),
    ]
    
    # DiffuseUp every 8 blocks
    if should_trigger(local_block, 8):
        backbone.append(OperatorEvent(OpID.DIFFUSE_UP, strength=0.1))
    
    # Events
    events = []
    
    # GlobalPulse every 16 blocks (low amplitude, coarse bands)
    if should_trigger(local_block, 16):
        events.append(OperatorEvent(
            OpID.GLOBAL_PULSE,
            strength=0.15,
            target_bands=[0, 4, 8, 12, 16, 20, 24, 28],  # Coarse bands
        ))
    
    # GluePair every 64 blocks (toggle on/off)
    if should_trigger(local_block, 64):
        events.append(OperatorEvent(OpID.GLUE_PAIR, strength=0.5))
    
    return OperatorProgram(backbone=backbone, events=events)


# =============================================================================
# SEGMENT B: TRANSIENT STORM
# =============================================================================

def get_segment_b_program(block: int, seg: SegmentTiming) -> OperatorProgram:
    """
    Segment B (1.00-2.00s): Mid-band transient storm + structured chopping.
    
    Backbone: IntraMix + DiffuseDown + BandShear + RotateNodes
    Events: GateMask every 12, QuasiRotate every 24, Impulse every 32, NoiseBand every 48
    """
    t = seg.progress(block)
    local_block = block - seg.start_block
    
    backbone = [
        OperatorEvent(OpID.INTRA_MIX, strength=0.4),
        OperatorEvent(OpID.DIFFUSE_DOWN, strength=0.3),
    ]
    
    # BandShear every 4 blocks
    if should_trigger(local_block, 4):
        backbone.append(OperatorEvent(OpID.BAND_SHEAR, strength=0.2))
    
    # RotateNodes every 8 blocks
    if should_trigger(local_block, 8):
        backbone.append(OperatorEvent(OpID.ROTATE_NODES, strength=0.3))
    
    events = []
    
    # GateMask every 12 blocks (Sturmian gating)
    if should_trigger(local_block, 12):
        events.append(OperatorEvent(OpID.GATE_MASK, strength=0.6))
    
    # QuasiRotate every 24 blocks
    if should_trigger(local_block, 24):
        events.append(OperatorEvent(OpID.QUASI_ROTATE, strength=0.4))
    
    # Impulse every 32 blocks (band boundaries)
    if should_trigger(local_block, 32):
        events.append(OperatorEvent(
            OpID.IMPULSE,
            strength=0.5,
            target_bands=[7, 15, 23, 31],  # Band boundaries
        ))
    
    # NoiseBand every 48 blocks (top band)
    if should_trigger(local_block, 48):
        events.append(OperatorEvent(
            OpID.NOISE_BAND,
            strength=0.3,
            target_bands=[28, 29, 30, 31],  # Top band
        ))
    
    return OperatorProgram(backbone=backbone, events=events)


# =============================================================================
# SEGMENT C: CATASTROPHE CORRIDOR
# =============================================================================

def get_segment_c_program(block: int, seg: SegmentTiming) -> OperatorProgram:
    """
    Segment C (2.00-3.25s): Catastrophe corridor.
    
    Simpler structure to let catastrophe behavior shine.
    Reduced permutations (every 16 blocks).
    
    2.00-2.50s: Fold A2 depth ramps 0→1
    2.50-3.00s: Cusp A3 (introduce bias)
    3.00-3.25s: Swallowtail A4 briefly
    """
    t = seg.progress(block)
    local_block = block - seg.start_block
    
    # Sub-segment timing within C
    fold_end = int(0.5 / 1.25 * seg.n_blocks)      # First 0.5s
    cusp_end = int(1.0 / 1.25 * seg.n_blocks)      # Next 0.5s
    # swallowtail: remainder (0.25s)
    
    backbone = [
        OperatorEvent(OpID.NORMALIZE_SCALE, strength=0.7),
    ]
    
    # Pressure ramp (global)
    pressure_strength = min(1.0, t * 1.5)  # Ramps up over segment
    backbone.append(OperatorEvent(OpID.PRESSURE_RAMP, strength=pressure_strength))
    
    events = []
    
    if local_block < fold_end:
        # Fold phase
        fold_t = local_block / fold_end
        events.append(OperatorEvent(OpID.FOLD_PRESSURE, strength=fold_t))
    
    elif local_block < cusp_end:
        # Cusp phase
        cusp_t = (local_block - fold_end) / (cusp_end - fold_end)
        events.append(OperatorEvent(OpID.FOLD_PRESSURE, strength=1.0))
        events.append(OperatorEvent(OpID.CUSP_INJECTION, strength=cusp_t * 0.7))
    
    else:
        # Swallowtail phase (brief)
        swallow_t = (local_block - cusp_end) / (seg.n_blocks - cusp_end)
        events.append(OperatorEvent(OpID.CUSP_INJECTION, strength=0.7))
        events.append(OperatorEvent(
            OpID.SWALLOWTAIL_MASK,
            strength=0.5 * (1 - swallow_t),  # Retreat toward end
        ))
    
    # Reduced permutations (every 16 blocks)
    if should_trigger(local_block, 16):
        events.append(OperatorEvent(OpID.ROTATE_NODES, strength=0.2))
    
    return OperatorProgram(backbone=backbone, events=events)


# =============================================================================
# SEGMENT D: GAMELAN FRACTURE LATTICE
# =============================================================================

def get_segment_d_program(block: int, seg: SegmentTiming) -> OperatorProgram:
    """
    Segment D (3.25-4.25s): Gamelan fracture lattice.
    
    Inharmonic but structured. Tests sheaf constraints and topology.
    
    SwapBands at start, SheafProject continuous, alternating BandShear/QuasiRotate.
    """
    t = seg.progress(block)
    local_block = block - seg.start_block
    
    backbone = [
        OperatorEvent(OpID.SHEAF_PROJECT, strength=0.6),
        OperatorEvent(OpID.DIFFUSE_UP, strength=0.15),  # Keeps from pure sand
    ]
    
    events = []
    
    # SwapBands at segment start
    if local_block == 0:
        events.append(OperatorEvent(OpID.SWAP_BANDS, strength=1.0))
    
    # GluePair every 16 blocks
    if should_trigger(local_block, 16):
        events.append(OperatorEvent(OpID.GLUE_PAIR, strength=0.4))
    
    # Alternating BandShear and QuasiRotate every 8 blocks
    if should_trigger(local_block, 8):
        if (local_block // 8) % 2 == 0:
            events.append(OperatorEvent(OpID.BAND_SHEAR, strength=0.35))
        else:
            events.append(OperatorEvent(OpID.QUASI_ROTATE, strength=0.35))
    
    # NoiseBand gated by GateMask
    if should_trigger(local_block, 24):
        events.append(OperatorEvent(OpID.NOISE_BAND, strength=0.25))
        events.append(OperatorEvent(OpID.GATE_MASK, strength=0.5))
    
    return OperatorProgram(backbone=backbone, events=events)


# =============================================================================
# SEGMENT E: COLLAPSE TO COHERENCE
# =============================================================================

def get_segment_e_program(block: int, seg: SegmentTiming) -> OperatorProgram:
    """
    Segment E (4.25-5.00s): Collapse to single coherent voice.
    
    Emergency simplification, then tonal landing.
    Final GlobalPulse at 4.90s.
    """
    t = seg.progress(block)
    local_block = block - seg.start_block
    
    # Emergency simplify at ~4.30s (first 38 blocks = 0.05s into segment)
    emergency_block = int(0.05 * seg.n_blocks)
    final_pulse_block = int(0.87 * seg.n_blocks)  # ~4.90s
    
    backbone = [
        OperatorEvent(OpID.NORMALIZE_SCALE, strength=0.9),
        OperatorEvent(OpID.TILT, strength=0.5 - 0.3 * t),  # Reduce tilt
    ]
    
    events = []
    
    # Emergency simplify near start of segment
    if local_block == emergency_block:
        events.append(OperatorEvent(OpID.EMERGENCY_SIMPLIFY, strength=1.0))
    
    # Gradual LowRankMorph increase (convergence)
    backbone.append(OperatorEvent(OpID.LOWRANK_MORPH, strength=0.5 + 0.4 * t))
    
    # Final GlobalPulse (breath)
    if local_block == final_pulse_block:
        events.append(OperatorEvent(
            OpID.GLOBAL_PULSE,
            strength=0.3,
            target_bands=list(range(8)),  # Low bands only
        ))
    
    return OperatorProgram(backbone=backbone, events=events)


# =============================================================================
# MAIN DISPATCHER
# =============================================================================

def get_operator_program(block: int) -> OperatorProgram:
    """
    Get the operator program for a given block.
    
    Dispatches to segment-specific program generator.
    """
    from test_config import get_current_segment
    
    seg_name, seg = get_current_segment(block)
    
    if seg_name == 'A':
        return get_segment_a_program(block, seg)
    elif seg_name == 'B':
        return get_segment_b_program(block, seg)
    elif seg_name == 'C':
        return get_segment_c_program(block, seg)
    elif seg_name == 'D':
        return get_segment_d_program(block, seg)
    else:  # E
        return get_segment_e_program(block, seg)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OPERATOR PROGRAM SAMPLING")
    print("=" * 60)
    
    sample_blocks = [0, 375, 750, 1125, 1500, 1875, 2250, 2625, 3000, 3375, 3562]
    
    for block in sample_blocks:
        from test_config import get_current_segment
        seg_name, seg = get_current_segment(block)
        prog = get_operator_program(block)
        
        time_sec = block / BLOCKS_PER_SEC
        backbone_ids = [e.op_id.name for e in prog.backbone]
        event_ids = [e.op_id.name for e in prog.events]
        
        print(f"\nBlock {block} ({time_sec:.2f}s) - Segment {seg_name}:")
        print(f"  Backbone: {backbone_ids}")
        print(f"  Events: {event_ids if event_ids else '(none)'}")
