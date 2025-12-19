"""
Cathedral of Stress - Configuration
===================================

Constants, timing, and voice configurations for the 5-second torture test.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

FS = 48000           # Sample rate
BLOCK_SIZE = 64      # Samples per block
BLOCKS_PER_SEC = FS // BLOCK_SIZE  # 750 blocks/sec
TOTAL_DURATION = 5.0  # seconds
TOTAL_BLOCKS = int(TOTAL_DURATION * BLOCKS_PER_SEC)  # 3750 blocks

N_VOICES = 12        # Polyphony count
N_LEAVES = 32        # Wavelet packet leaves
NODE_SIZE = 8        # Coefficients per leaf
COEFF_LEN = N_LEAVES * NODE_SIZE  # Total coefficient count (256)


# =============================================================================
# SEGMENT TIMING
# =============================================================================

@dataclass
class SegmentTiming:
    """Timing for a test segment."""
    name: str
    start_sec: float
    end_sec: float
    description: str
    
    @property
    def start_block(self) -> int:
        return int(self.start_sec * BLOCKS_PER_SEC)
    
    @property
    def end_block(self) -> int:
        return int(self.end_sec * BLOCKS_PER_SEC)
    
    @property
    def n_blocks(self) -> int:
        return self.end_block - self.start_block
    
    def progress(self, block: int) -> float:
        """Return 0-1 progress within this segment."""
        if block < self.start_block:
            return 0.0
        if block >= self.end_block:
            return 1.0
        return (block - self.start_block) / self.n_blocks


SEGMENTS = {
    'A': SegmentTiming(
        name='A',
        start_sec=0.00,
        end_sec=1.00,
        description='Coherent harmonic organ that starts to deform'
    ),
    'B': SegmentTiming(
        name='B',
        start_sec=1.00,
        end_sec=2.00,
        description='Mid-band transient storm + structured chopping'
    ),
    'C': SegmentTiming(
        name='C',
        start_sec=2.00,
        end_sec=3.25,
        description='Catastrophe corridor: fold → cusp → swallowtail'
    ),
    'D': SegmentTiming(
        name='D',
        start_sec=3.25,
        end_sec=4.25,
        description='Gamelan fracture lattice (inharmonic but structured)'
    ),
    'E': SegmentTiming(
        name='E',
        start_sec=4.25,
        end_sec=5.00,
        description='Collapse to single coherent voice + perfect landing'
    ),
}


def get_current_segment(block: int) -> Tuple[str, SegmentTiming]:
    """Get the current segment for a given block number."""
    for name, seg in SEGMENTS.items():
        if seg.start_block <= block < seg.end_block:
            return name, seg
    return 'E', SEGMENTS['E']  # Default to final segment


# =============================================================================
# TREE CONFIGURATION
# =============================================================================

class TreeID:
    """Tree type identifiers."""
    BALANCED = 0   # Tree0 - Even frequency distribution
    HARMONIC = 1   # Tree1 - Optimized for pitched content
    TEXTURE = 2    # Tree2 - Noise/texture focused
    IMPACT = 3     # Tree3 - Transient/attack focused
    FIELD = 4      # Tree4 - Ambient/drone focused (aka Drone)


# =============================================================================
# VOICE CONFIGURATION
# =============================================================================

@dataclass
class VoiceConfig:
    """Configuration for a single voice."""
    voice_id: int
    role: str           # 'harmonic', 'drone', 'texture', 'lead'
    tree_id: int
    base_freq: float    # Hz
    amplitude: int      # Q15
    pan: int = 0        # Q15, -32768 to 32767


def build_initial_voices() -> List[VoiceConfig]:
    """
    Build initial voice configuration for segment A.
    
    6 voices: harmonic stack (octaves + 5ths) → Tree1 (Harmonic)
    4 voices: drones/fields → Tree4 (Field)
    2 voices: quiet noise bed → Tree2 (Texture)
    """
    voices = []
    
    # Harmonic stack (voices 0-5)
    harmonic_freqs = [
        110.0,   # A2 - fundamental
        220.0,   # A3 - octave
        165.0,   # E3 - fifth
        330.0,   # E4 - fifth + octave
        440.0,   # A4 - 2 octaves
        247.5,   # B3 - major 2nd (adds tension)
    ]
    for i, freq in enumerate(harmonic_freqs):
        voices.append(VoiceConfig(
            voice_id=i,
            role='harmonic',
            tree_id=TreeID.HARMONIC,
            base_freq=freq,
            amplitude=0x3000,
            pan=int((i - 2.5) * 4000),  # Spread across stereo
        ))
    
    # Drone voices (voices 6-9)
    drone_freqs = [55.0, 82.5, 27.5, 41.25]  # Low A, E, sub-A, sub-E
    for i, freq in enumerate(drone_freqs):
        voices.append(VoiceConfig(
            voice_id=6 + i,
            role='drone',
            tree_id=TreeID.FIELD,
            base_freq=freq,
            amplitude=0x2800,
            pan=int((i - 1.5) * 6000),
        ))
    
    # Texture voices (voices 10-11)
    for i in range(2):
        voices.append(VoiceConfig(
            voice_id=10 + i,
            role='texture',
            tree_id=TreeID.TEXTURE,
            base_freq=1000.0 + i * 500,  # Noise bands
            amplitude=0x1000,
            pan=int((i - 0.5) * 16000),  # Wide stereo
        ))
    
    return voices


# =============================================================================
# MATERIAL CONFIGURATION
# =============================================================================

class MaterialID:
    """Material type identifiers."""
    CLEAN = 0
    TAPE = 1
    RANK = 2
    STEEL = 3      # Steel hysteresis (new)
    FOLD = 4       # Catastrophe A2
    CUSP = 5       # Catastrophe A3
    SWALLOWTAIL = 6  # Catastrophe A4 (new)


@dataclass
class MaterialProgram:
    """Material settings for a block."""
    material_id: int
    drive_q15: int = 0x3000
    pressure_q15: int = 0
    bias_q15: int = 0
    tone_q15: int = 0x5000


# =============================================================================
# LEAF ATOM CONFIGURATION
# =============================================================================

class AtomID:
    """Leaf atom matrix identifiers."""
    IDENTITY = 0    # M0 = I (no transform)
    DCT8 = 1        # M1 = DCT basis (harmonic analysis)
    CHIRP = 2       # M2 = Chirp-ish (time-varying)
    SPIKE = 3       # M3 = Spike/asymmetric


@dataclass
class AtomConfig:
    """Atom configuration for complex leaves."""
    atom_id: int
    mix_amount: float  # 0-1, how much to blend with identity


# =============================================================================
# SLEW UTILITIES
# =============================================================================

def slew(current: float, target: float, rate: float = 0.01) -> float:
    """Slew toward target at given rate."""
    delta = target - current
    if abs(delta) < rate:
        return target
    return current + np.sign(delta) * rate


def slew_int(current: int, target: int, max_delta: int = 512) -> int:
    """Slew integer toward target."""
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + (max_delta if delta > 0 else -max_delta)


# =============================================================================
# EVENT SCHEDULING
# =============================================================================

def should_trigger(block: int, interval: int, offset: int = 0) -> bool:
    """Check if an event should trigger at this block."""
    return (block - offset) % interval == 0


def ramp(block: int, start_block: int, end_block: int) -> float:
    """Linear ramp from 0 to 1 over block range."""
    if block <= start_block:
        return 0.0
    if block >= end_block:
        return 1.0
    return (block - start_block) / (end_block - start_block)
