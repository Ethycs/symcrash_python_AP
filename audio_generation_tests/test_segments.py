"""
Cathedral of Stress - Segment Implementations
=============================================

State management and per-segment logic for the torture test.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from test_config import (
    N_VOICES, N_LEAVES, NODE_SIZE, BLOCKS_PER_SEC,
    TreeID, MaterialID, AtomID,
    VoiceConfig, MaterialProgram, AtomConfig,
    build_initial_voices, ramp, slew, slew_int,
    SEGMENTS, get_current_segment
)


# =============================================================================
# TEST STATE
# =============================================================================

@dataclass
class VoiceState:
    """Runtime state for a single voice."""
    config: VoiceConfig
    tree_id: int = 0
    tree_crossfade: float = 0.0  # 0-1 crossfade progress
    target_tree_id: int = 0
    amplitude: int = 0
    target_amplitude: int = 0
    active: bool = True


@dataclass
class TestState:
    """Global state for the torture test."""
    block: int = 0
    
    # Voice states
    voices: List[VoiceState] = field(default_factory=list)
    
    # Leaf atom mix amount (m dial)
    atom_mix: float = 0.15
    target_atom_mix: float = 0.15
    
    # Material settings
    material_id: int = MaterialID.TAPE
    drive_q15: int = 0x3000
    pressure_q15: int = 0
    bias_q15: int = 0
    
    # Drive boost for stress testing
    drive_boost_db: float = 0.0
    
    # Active voice count (for segment E fadeout)
    active_voice_count: int = N_VOICES
    
    def __post_init__(self):
        if not self.voices:
            configs = build_initial_voices()
            self.voices = [
                VoiceState(
                    config=cfg,
                    tree_id=cfg.tree_id,
                    target_tree_id=cfg.tree_id,
                    amplitude=cfg.amplitude,
                    target_amplitude=cfg.amplitude,
                )
                for cfg in configs
            ]


# =============================================================================
# SEGMENT A: COHERENT HARMONIC ORGAN
# =============================================================================

def update_segment_a(state: TestState, t: float) -> None:
    """
    Update state for segment A (0.00-1.00s).
    
    - Atom mix: 0.15 → 0.35
    - Voices stable on initial trees
    - Material: tape (gentle)
    """
    # Atom mix ramp
    state.target_atom_mix = 0.15 + 0.20 * t
    
    # Material: tape with gentle settings
    state.material_id = MaterialID.TAPE
    state.drive_q15 = 0x3000
    state.pressure_q15 = 0
    
    # All voices active
    for v in state.voices:
        v.active = True
        v.target_amplitude = v.config.amplitude


# =============================================================================
# SEGMENT B: TRANSIENT STORM
# =============================================================================

def update_segment_b(state: TestState, t: float) -> None:
    """
    Update state for segment B (1.00-2.00s).
    
    - Crossfade harmonic voices from Tree1 → Tree3 (Impact)
    - Atom mix: 0.35 → 0.55
    - Material: rank grit + subtle fold
    """
    # Tree crossfade for harmonic voices (first 64 blocks of segment = ~85ms)
    crossfade_progress = min(1.0, t * 8)  # Complete in first 0.125s
    
    for v in state.voices:
        if v.config.role == 'harmonic':
            v.target_tree_id = TreeID.IMPACT
            v.tree_crossfade = crossfade_progress
    
    # Atom mix ramp
    state.target_atom_mix = 0.35 + 0.20 * t
    
    # Material: rank grit with subtle fold
    state.material_id = MaterialID.RANK
    state.drive_q15 = 0x3800
    state.pressure_q15 = int(0x1000 * t)  # Gentle pressure ramp


# =============================================================================
# SEGMENT C: CATASTROPHE CORRIDOR
# =============================================================================

def update_segment_c(state: TestState, t: float) -> None:
    """
    Update state for segment C (2.00-3.25s).
    
    Catastrophe progression:
    - 0.00-0.40 (2.00-2.50s): Fold A2 depth 0→1
    - 0.40-0.80 (2.50-3.00s): Cusp A3
    - 0.80-1.00 (3.00-3.25s): Swallowtail A4, then retreat
    
    Atom mix: peaks at 0.85
    Drive boost: +6dB during 2.7-3.1s
    """
    # Atom mix: peak at 0.85 during cusp/swallowtail
    if t < 0.4:
        state.target_atom_mix = 0.55 + 0.30 * (t / 0.4)
    elif t < 0.8:
        state.target_atom_mix = 0.85
    else:
        state.target_atom_mix = 0.85 - 0.30 * ((t - 0.8) / 0.2)
    
    # Material: catastrophe progression with gentler settings
    # Use tape (1) as base, but with catastrophe-like pressure modulation
    if t < 0.4:
        # Fold phase - use tape with increasing pressure
        state.material_id = MaterialID.TAPE
        state.pressure_q15 = int(0x3000 * (t / 0.4))  # Gentler pressure
        state.bias_q15 = 0
    elif t < 0.8:
        # Cusp phase - tape with bias oscillation
        state.material_id = MaterialID.TAPE
        state.pressure_q15 = 0x3000
        cusp_t = (t - 0.4) / 0.4
        # Oscillating bias creates hysteresis effect
        state.bias_q15 = int(0x2000 * np.sin(cusp_t * np.pi * 4))
    else:
        # Retreat to clean
        state.material_id = MaterialID.TAPE
        retreat_t = (t - 0.8) / 0.2
        state.pressure_q15 = int(0x3000 * (1 - retreat_t))
        state.bias_q15 = 0
    
    # Drive boost: +6dB during 2.7-3.1s (t = 0.56 to t = 0.88)
    if 0.56 <= t <= 0.88:
        boost_t = (t - 0.56) / 0.32
        if boost_t < 0.5:
            state.drive_boost_db = 6.0 * (boost_t * 2)
        else:
            state.drive_boost_db = 6.0 * (1 - (boost_t - 0.5) * 2)
    else:
        state.drive_boost_db = 0.0
    
    # Apply drive boost
    base_drive = 0x4000  # Slightly higher base drive
    boost_linear = 10 ** (state.drive_boost_db / 20)
    state.drive_q15 = min(0x7FFF, int(base_drive * boost_linear))


# =============================================================================
# SEGMENT D: GAMELAN FRACTURE LATTICE
# =============================================================================

def update_segment_d(state: TestState, t: float) -> None:
    """
    Update state for segment D (3.25-4.25s).
    
    - 8 voices → Texture tree
    - 4 voices stay as drones on Field
    - Material: rank grit for metallic character
    - Atom mix: moderate
    """
    # Voice tree assignments
    for i, v in enumerate(state.voices):
        if i < 8:
            # First 8 voices → Texture
            v.target_tree_id = TreeID.TEXTURE
            v.tree_crossfade = min(1.0, t * 4)  # Quick crossfade
        else:
            # Last 4 → Field (drones)
            v.target_tree_id = TreeID.FIELD
    
    # Atom mix: moderate with movement
    state.target_atom_mix = 0.45 + 0.25 * np.sin(t * np.pi * 3)
    
    # Material: rank grit for metallic/gamelan character
    state.material_id = MaterialID.RANK
    state.drive_q15 = 0x5000  # Higher drive for crunch
    state.pressure_q15 = int(0x2000 * (0.5 + 0.3 * np.sin(t * np.pi * 4)))
    state.bias_q15 = int(0x1000 * np.sin(t * np.pi * 7))  # Asymmetry
    state.drive_boost_db = 0.0


# =============================================================================
# SEGMENT E: COLLAPSE TO COHERENCE
# =============================================================================

def update_segment_e(state: TestState, t: float) -> None:
    """
    Update state for segment E (4.25-5.00s).
    
    - Fade from 12 voices to 1 lead voice
    - Lead voice → Harmonic tree with DCT8 atom
    - Material: tape (gentle), drive down
    - Atom mix: return to 0.25
    """
    # Voice fadeout (over first 32 blocks ≈ 43ms)
    fadeout_progress = min(1.0, t * 4)
    
    # Calculate how many voices should be active
    target_active = max(1, int(N_VOICES * (1 - fadeout_progress * 0.92)))
    
    for i, v in enumerate(state.voices):
        if i == 0:
            # Lead voice stays active
            v.active = True
            v.target_tree_id = TreeID.HARMONIC
            v.tree_crossfade = fadeout_progress
            v.target_amplitude = v.config.amplitude
        elif i < target_active:
            # Still fading
            v.active = True
            v.target_amplitude = int(v.config.amplitude * (1 - fadeout_progress))
        else:
            # Fully faded
            v.active = False
            v.target_amplitude = 0
    
    # Atom mix: return to 0.25
    state.target_atom_mix = 0.55 - 0.30 * t
    
    # Material: gentle tape
    state.material_id = MaterialID.TAPE
    state.drive_q15 = int(0x3000 * (1 - 0.3 * t))  # Reduce drive
    state.pressure_q15 = 0
    state.bias_q15 = 0
    state.drive_boost_db = 0.0


# =============================================================================
# MAIN UPDATE DISPATCHER
# =============================================================================

def update_segment_state(state: TestState) -> None:
    """
    Update state based on current block.
    
    Slews continuous parameters and dispatches to segment-specific update.
    """
    seg_name, seg = get_current_segment(state.block)
    t = seg.progress(state.block)
    
    # Dispatch to segment-specific update
    if seg_name == 'A':
        update_segment_a(state, t)
    elif seg_name == 'B':
        update_segment_b(state, t)
    elif seg_name == 'C':
        update_segment_c(state, t)
    elif seg_name == 'D':
        update_segment_d(state, t)
    else:  # E
        update_segment_e(state, t)
    
    # Slew continuous parameters
    state.atom_mix = slew(state.atom_mix, state.target_atom_mix, rate=0.02)
    
    # Slew voice amplitudes and tree crossfades
    for v in state.voices:
        v.amplitude = slew_int(v.amplitude, v.target_amplitude, max_delta=256)
        
        # Update tree_id when crossfade completes
        if v.tree_crossfade >= 0.99:
            v.tree_id = v.target_tree_id
            v.tree_crossfade = 0.0


def get_material_program(state: TestState) -> MaterialProgram:
    """Get current material program from state."""
    return MaterialProgram(
        material_id=state.material_id,
        drive_q15=state.drive_q15,
        pressure_q15=state.pressure_q15,
        bias_q15=state.bias_q15,
        tone_q15=0x5000,
    )


def get_voice_configs(state: TestState) -> List[Tuple[VoiceConfig, int, float, int]]:
    """
    Get current voice configurations.
    
    Returns list of (config, tree_id, crossfade, amplitude) tuples.
    """
    result = []
    for v in state.voices:
        if v.active:
            result.append((
                v.config,
                v.tree_id,
                v.tree_crossfade,
                v.amplitude,
            ))
    return result


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SEGMENT STATE EVOLUTION")
    print("=" * 60)
    
    state = TestState()
    
    sample_blocks = [0, 375, 750, 1125, 1500, 1875, 2250, 2625, 3000, 3375, 3562]
    
    for block in sample_blocks:
        state.block = block
        update_segment_state(state)
        
        seg_name, seg = get_current_segment(block)
        time_sec = block / BLOCKS_PER_SEC
        
        active_count = sum(1 for v in state.voices if v.active)
        tree_ids = [v.tree_id for v in state.voices if v.active]
        
        print(f"\nBlock {block} ({time_sec:.2f}s) - Segment {seg_name}:")
        print(f"  Atom mix: {state.atom_mix:.2f}")
        print(f"  Material: {state.material_id}, drive=0x{state.drive_q15:04X}")
        print(f"  Pressure: 0x{state.pressure_q15:04X}, bias: 0x{state.bias_q15:04X}")
        print(f"  Active voices: {active_count}, trees: {set(tree_ids)}")
        if state.drive_boost_db > 0:
            print(f"  Drive boost: +{state.drive_boost_db:.1f} dB")
