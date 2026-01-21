"""
Wavetable Synthesis Engine (Phase 4)
====================================

Per WSE-STD-210 Architecture:
  Wavetable atlas (2D array of waveforms) → 2D bilinear interpolation
  → coefficient generation → 512-coefficient output

Key Features:
  - 2D wavetable atlas: frames × position (e.g., 64 frames × 256 samples)
  - Bilinear interpolation between 4 nearest wave frames
  - Coefficient-domain output compatible with CCS pipeline
  - Position modulation for timbral morphing
  - Frame scanning for evolving timbres

Wavetable Structure:
  Atlas: 2D grid of single-cycle waveforms
    - X axis: position (0.0-1.0) - timbral variation
    - Y axis: frame index (0, 1, 2, ...) - evolution/morphing

  Each cell contains one period of audio (typically 256 or 512 samples).
  Interpolation happens in both dimensions for smooth timbral transitions.

All arithmetic in Q15/Q16 fixed-point per WSE-STD-001.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

from .tables import VM_K, COEFF_LEN
from ..wavelet.fast_lifting import fwpt_jit

# Import canonical constants from primitives (single source of truth)
from ..primitives import NUM_LEAVES, BLOCK_SIZE, Q15_ONE, Q16_ONE


# =============================================================================
# CONSTANTS
# =============================================================================

# Default wavetable dimensions
DEFAULT_FRAME_SIZE = 256       # Samples per wave frame (one cycle)
DEFAULT_NUM_FRAMES = 64        # Number of frames in atlas (Y dimension)
DEFAULT_NUM_POSITIONS = 1      # Number of positions in atlas (X dimension, for morphing)

# Q16 fixed-point for position
Q16_SCALE = 65536


# =============================================================================
# WAVETABLE ATLAS
# =============================================================================

@dataclass
class WavetableAtlas:
    """
    2D wavetable atlas structure.

    Stores a grid of single-cycle waveforms for interpolated playback.
    The atlas can be 1D (single position, multiple frames) or 2D
    (multiple positions × multiple frames) for timbral morphing.

    Shape: [num_positions, num_frames, frame_size]
      - num_positions: X dimension (timbral variation, typically 1-16)
      - num_frames: Y dimension (evolution, typically 16-256)
      - frame_size: samples per cycle (typically 256 or 512)

    Attributes:
        data: 3D int16 array of waveform samples
        num_positions: Number of X positions (morph targets)
        num_frames: Number of Y frames (evolution stages)
        frame_size: Samples per wave cycle
        name: Human-readable atlas name
        base_note: MIDI note at which frame_size = one period (for pitch calc)
    """
    data: np.ndarray = field(default_factory=lambda: np.zeros(
        (1, DEFAULT_NUM_FRAMES, DEFAULT_FRAME_SIZE), dtype=np.int16
    ))
    num_positions: int = 1
    num_frames: int = DEFAULT_NUM_FRAMES
    frame_size: int = DEFAULT_FRAME_SIZE
    name: str = "Default"
    base_note: int = 60  # Middle C

    def __post_init__(self):
        """Validate atlas dimensions."""
        if self.data.ndim != 3:
            raise ValueError(f"Atlas data must be 3D, got {self.data.ndim}D")
        self.num_positions, self.num_frames, self.frame_size = self.data.shape

    @classmethod
    def from_2d_array(cls, frames: np.ndarray, name: str = "Imported",
                      base_note: int = 60) -> 'WavetableAtlas':
        """
        Create atlas from 2D array [num_frames, frame_size].

        Single-position atlas (no X-axis morphing).
        """
        if frames.ndim != 2:
            raise ValueError(f"Expected 2D array, got {frames.ndim}D")
        # Expand to 3D with single position
        data = frames[np.newaxis, :, :].astype(np.int16)
        return cls(data=data, name=name, base_note=base_note)

    @classmethod
    def from_waveforms(cls, waveforms: List[np.ndarray], name: str = "Combined",
                       base_note: int = 60) -> 'WavetableAtlas':
        """
        Create atlas from list of single-cycle waveforms.

        All waveforms must have the same length.
        """
        if not waveforms:
            raise ValueError("Must provide at least one waveform")
        frame_size = len(waveforms[0])
        if not all(len(w) == frame_size for w in waveforms):
            raise ValueError("All waveforms must have same length")

        frames = np.array(waveforms, dtype=np.int16)
        return cls.from_2d_array(frames, name, base_note)

    def get_sample(self, position_q16: int, frame_q16: int, phase_q16: int) -> int:
        """
        Get interpolated sample from atlas using 2D bilinear interpolation.

        Args:
            position_q16: X position (0 to num_positions-1) in Q16
            frame_q16: Y frame (0 to num_frames-1) in Q16
            phase_q16: Phase within wave (0 to 65535 maps to 0-frame_size)

        Returns:
            Interpolated sample (int16)
        """
        # Extract integer and fractional parts
        pos_int = (position_q16 >> 16) % self.num_positions
        pos_frac = position_q16 & 0xFFFF

        frame_int = (frame_q16 >> 16) % self.num_frames
        frame_frac = frame_q16 & 0xFFFF

        # Scale phase to frame size
        # phase_q16 = 0..65535 maps to 0..frame_size-1
        phase_scaled = (phase_q16 * self.frame_size) >> 16
        phase_int = phase_scaled % self.frame_size
        phase_next = (phase_int + 1) % self.frame_size

        # Get next position and frame indices (with wrapping)
        pos_next = (pos_int + 1) % self.num_positions
        frame_next = (frame_int + 1) % self.num_frames

        # 4-corner samples for bilinear interpolation (ignoring phase interpolation for now)
        # s00 = position_int, frame_int
        # s01 = position_int, frame_next
        # s10 = position_next, frame_int
        # s11 = position_next, frame_next
        s00 = int(self.data[pos_int, frame_int, phase_int])
        s01 = int(self.data[pos_int, frame_next, phase_int])
        s10 = int(self.data[pos_next, frame_int, phase_int])
        s11 = int(self.data[pos_next, frame_next, phase_int])

        # Also get samples at phase_next for phase interpolation
        s00_n = int(self.data[pos_int, frame_int, phase_next])
        s01_n = int(self.data[pos_int, frame_next, phase_next])
        s10_n = int(self.data[pos_next, frame_int, phase_next])
        s11_n = int(self.data[pos_next, frame_next, phase_next])

        # Phase interpolation fraction (sub-sample)
        phase_frac = ((phase_q16 * self.frame_size) & 0xFFFF)

        # Interpolate along phase axis first
        def lerp_q16(a: int, b: int, frac: int) -> int:
            """Linear interpolation with Q16 fraction."""
            return a + (((b - a) * frac) >> 16)

        s00 = lerp_q16(s00, s00_n, phase_frac)
        s01 = lerp_q16(s01, s01_n, phase_frac)
        s10 = lerp_q16(s10, s10_n, phase_frac)
        s11 = lerp_q16(s11, s11_n, phase_frac)

        # Bilinear interpolation: first along frame axis, then position axis
        # (could be done in either order)
        s0 = lerp_q16(s00, s01, frame_frac)  # Interpolate along frame at pos_int
        s1 = lerp_q16(s10, s11, frame_frac)  # Interpolate along frame at pos_next
        result = lerp_q16(s0, s1, pos_frac)  # Interpolate along position

        return max(-32768, min(32767, result))


def create_basic_waves() -> WavetableAtlas:
    """
    Create a basic wavetable with fundamental waveforms.

    Frames: sine → triangle → saw → square (morphable)
    """
    frame_size = DEFAULT_FRAME_SIZE
    num_frames = 4  # sine, triangle, saw, square

    frames = np.zeros((num_frames, frame_size), dtype=np.int16)

    # Phase array (0 to 2π)
    phase = np.linspace(0, 2 * np.pi, frame_size, endpoint=False)

    # Frame 0: Sine
    frames[0] = (np.sin(phase) * 32767).astype(np.int16)

    # Frame 1: Triangle
    tri = 2 * np.abs(2 * (np.linspace(0, 1, frame_size, endpoint=False) - 0.5)) - 1
    frames[1] = (tri * 32767).astype(np.int16)

    # Frame 2: Saw (ramp down)
    saw = 1 - 2 * np.linspace(0, 1, frame_size, endpoint=False)
    frames[2] = (saw * 32767).astype(np.int16)

    # Frame 3: Square
    sq = np.where(phase < np.pi, 1.0, -1.0)
    frames[3] = (sq * 32767).astype(np.int16)

    return WavetableAtlas.from_2d_array(frames, name="BasicWaves")


def create_harmonic_series(num_frames: int = 32, max_harmonics: int = 32) -> WavetableAtlas:
    """
    Create a wavetable with progressive harmonic content.

    Frame 0: fundamental only
    Frame N: fundamental + harmonics up to N

    Useful for creating evolving timbres and band-limited oscillators.
    """
    frame_size = DEFAULT_FRAME_SIZE
    frames = np.zeros((num_frames, frame_size), dtype=np.float64)

    phase = np.linspace(0, 2 * np.pi, frame_size, endpoint=False)

    for f in range(num_frames):
        # Number of harmonics in this frame
        num_harmonics = 1 + (f * max_harmonics) // num_frames

        waveform = np.zeros(frame_size)
        for h in range(1, num_harmonics + 1):
            # Saw-like spectrum: 1/h amplitude
            waveform += np.sin(h * phase) / h

        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        frames[f] = waveform

    return WavetableAtlas.from_2d_array(
        (frames * 32767).astype(np.int16),
        name="HarmonicSeries"
    )


# =============================================================================
# WAVETABLE BANK
# =============================================================================

@dataclass
class WavetableBank:
    """
    Collection of wavetable atlases.

    Provides indexed access to multiple wavetables for preset selection.

    Attributes:
        atlases: List of WavetableAtlas objects
        names: Mapping of name -> index for lookup
    """
    atlases: List[WavetableAtlas] = field(default_factory=list)

    def __post_init__(self):
        """Build name index."""
        self._name_index = {a.name: i for i, a in enumerate(self.atlases)}

    def add(self, atlas: WavetableAtlas) -> int:
        """Add atlas and return its index."""
        idx = len(self.atlases)
        self.atlases.append(atlas)
        self._name_index[atlas.name] = idx
        return idx

    def get(self, index: int) -> Optional[WavetableAtlas]:
        """Get atlas by index."""
        if 0 <= index < len(self.atlases):
            return self.atlases[index]
        return None

    def get_by_name(self, name: str) -> Optional[WavetableAtlas]:
        """Get atlas by name."""
        idx = self._name_index.get(name)
        if idx is not None:
            return self.atlases[idx]
        return None

    def __len__(self) -> int:
        return len(self.atlases)


def create_default_bank() -> WavetableBank:
    """Create a bank with basic wavetables."""
    bank = WavetableBank()
    bank.add(create_basic_waves())
    bank.add(create_harmonic_series())
    return bank


# =============================================================================
# WAVETABLE OSCILLATOR STATE
# =============================================================================

@dataclass
class WavetableState:
    """
    Wavetable oscillator state (coefficient-native).

    Tracks phase, position, and frame for wavetable playback.
    All values in Q16 fixed-point for smooth interpolation.

    Attributes:
        atlas: Reference to wavetable atlas
        phase_q32: Current phase (0 to 2^32-1 wraps to 0-frame_size)
        position_q16: X position in atlas (0 to num_positions-1 in Q16)
        frame_q16: Y frame in atlas (0 to num_frames-1 in Q16)
        frame_rate_q16: Frame scanning speed (Q16, 0 = static)
        phase_inc_q32: Phase increment per sample (derived from pitch)
    """
    atlas: Optional[WavetableAtlas] = None

    # Phase tracking (Q32 for precision)
    phase_q32: int = 0
    phase_inc_q32: int = 0  # Set from pitch

    # Position and frame (Q16)
    position_q16: int = 0        # X morph position
    frame_q16: int = 0           # Y frame position
    frame_rate_q16: int = 0      # Frame scanning rate (Q16, positive = forward)

    # Modulation targets (applied per block)
    position_mod_q15: int = 0    # Added to position
    frame_mod_q15: int = 0       # Added to frame

    # Playback state
    is_playing: bool = False

    def reset(self, phase: int = 0):
        """Reset oscillator to initial state."""
        self.phase_q32 = phase
        self.is_playing = True

    def set_pitch_hz(self, freq_hz: float, sample_rate: int = 48000):
        """
        Set oscillator pitch from frequency.

        Phase increment = (freq_hz / sample_rate) * 2^32
        """
        if self.atlas is None:
            return

        # Calculate phase increment for one wave cycle
        # phase_inc_q32 = (freq_hz * 2^32) / sample_rate
        self.phase_inc_q32 = int((freq_hz * (1 << 32)) / sample_rate)

    def set_pitch_note(self, midi_note: int, sample_rate: int = 48000):
        """Set oscillator pitch from MIDI note."""
        freq_hz = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        self.set_pitch_hz(freq_hz, sample_rate)

    def set_position(self, position: float):
        """Set X morph position (0.0 to 1.0)."""
        if self.atlas is None:
            return
        # Map 0.0-1.0 to 0 to (num_positions-1) in Q16
        max_pos = (self.atlas.num_positions - 1) << 16
        self.position_q16 = int(max(0, min(1.0, position)) * max_pos)

    def set_frame(self, frame: float):
        """Set Y frame position (0.0 to 1.0)."""
        if self.atlas is None:
            return
        # Map 0.0-1.0 to 0 to (num_frames-1) in Q16
        max_frame = (self.atlas.num_frames - 1) << 16
        self.frame_q16 = int(max(0, min(1.0, frame)) * max_frame)

    def set_frame_rate(self, rate_hz: float, sample_rate: int = 48000):
        """
        Set frame scanning rate in Hz.

        Positive = forward scan through frames
        Negative = reverse scan
        0 = static (no scanning)
        """
        if self.atlas is None:
            return
        # rate_hz cycles per second through all frames
        # Q16 increment per sample = (rate_hz * num_frames * 65536) / sample_rate
        frames_per_sample = rate_hz * self.atlas.num_frames / sample_rate
        self.frame_rate_q16 = int(frames_per_sample * Q16_SCALE)


# =============================================================================
# COEFFICIENT GENERATION
# =============================================================================

def wavetable_generate_block(state: WavetableState) -> np.ndarray:
    """
    Generate one block of audio at sample rate from wavetable.

    Args:
        state: WavetableState with atlas and phase

    Returns:
        audio_block: [BLOCK_SIZE] int16 samples
    """
    output = np.zeros(BLOCK_SIZE, dtype=np.int16)

    if state.atlas is None or not state.is_playing:
        return output

    atlas = state.atlas

    # Apply modulation to position and frame
    mod_position = state.position_q16 + ((state.position_mod_q15 * atlas.num_positions) >> 15)
    mod_frame = state.frame_q16 + ((state.frame_mod_q15 * atlas.num_frames) >> 15)

    # Clamp modulated values
    max_pos = (atlas.num_positions - 1) << 16
    max_frame = (atlas.num_frames - 1) << 16
    mod_position = max(0, min(max_pos, mod_position))
    mod_frame = max(0, min(max_frame, mod_frame))

    for i in range(BLOCK_SIZE):
        # Convert Q32 phase to Q16 for atlas lookup
        # phase_q32 wraps at 2^32, we want 0-65535 for one cycle
        phase_q16 = (state.phase_q32 >> 16) & 0xFFFF

        # Get interpolated sample
        sample = atlas.get_sample(mod_position, mod_frame, phase_q16)
        output[i] = sample

        # Advance phase
        state.phase_q32 = (state.phase_q32 + state.phase_inc_q32) & 0xFFFFFFFF

        # Advance frame (scanning)
        if state.frame_rate_q16 != 0:
            state.frame_q16 = (state.frame_q16 + state.frame_rate_q16)
            # Wrap frame to valid range
            while state.frame_q16 >= max_frame + (1 << 16):
                state.frame_q16 -= max_frame + (1 << 16)
            while state.frame_q16 < 0:
                state.frame_q16 += max_frame + (1 << 16)

    return output


def wavetable_generate_coeffs(state: WavetableState, output_512: np.ndarray) -> None:
    """
    Generate coefficients from wavetable playback.

    Generates one block of audio, then transforms to coefficient domain.

    Args:
        state: WavetableState with atlas and phase
        output_512: Output array [512] to fill with coefficients
    """
    if state.atlas is None or not state.is_playing:
        output_512[:] = 0
        return

    # Generate audio block
    audio_block = wavetable_generate_block(state)

    # Transform to coefficients via FWPT (analysis transform)
    coeffs_64 = fwpt_jit(audio_block.tolist(), headroom_bits=1)

    # Replicate across K time slots
    for k in range(VM_K):
        for leaf in range(NUM_LEAVES):
            output_512[leaf * VM_K + k] = np.clip(coeffs_64[leaf], -32768, 32767)


def wavetable_generate_coeffs_fast(state: WavetableState, output_512: np.ndarray) -> None:
    """
    Fast coefficient generation using pre-computed wavetable coefficients.

    Requires atlas to have pre-computed coefficient frames (future optimization).
    Falls back to standard generation if not available.

    Args:
        state: WavetableState with atlas
        output_512: Output array [512] to fill with coefficients
    """
    # For now, use standard generation
    # Future: add coeff_atlas with pre-transformed frames
    wavetable_generate_coeffs(state, output_512)


# =============================================================================
# WAVETABLE SPEC (for InstrumentIR)
# =============================================================================

@dataclass
class WavetableSpec:
    """
    Wavetable specification for InstrumentIR.

    Defines wavetable oscillator configuration in the IR layer.
    """
    # Atlas reference (by name or index)
    atlas_name: str = "BasicWaves"
    atlas_index: Optional[int] = None

    # Initial position and frame
    position: float = 0.0         # X morph position (0.0-1.0)
    frame: float = 0.0            # Y frame position (0.0-1.0)

    # Frame scanning
    frame_rate_hz: float = 0.0    # Scanning rate (0 = static)
    frame_loop: bool = True       # Loop frame scanning

    # Modulation routing (target names)
    position_mod_source: Optional[str] = None  # e.g., "LFO1", "ENV2"
    frame_mod_source: Optional[str] = None

    # Morph envelope
    morph_time_ms: float = 0.0    # Time to morph between positions

    def to_dict(self) -> dict:
        """Serialize to dictionary for YAML/JSON."""
        return {
            'atlas_name': self.atlas_name,
            'atlas_index': self.atlas_index,
            'position': self.position,
            'frame': self.frame,
            'frame_rate_hz': self.frame_rate_hz,
            'frame_loop': self.frame_loop,
            'position_mod_source': self.position_mod_source,
            'frame_mod_source': self.frame_mod_source,
            'morph_time_ms': self.morph_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WavetableSpec':
        """Deserialize from dictionary."""
        return cls(
            atlas_name=data.get('atlas_name', 'BasicWaves'),
            atlas_index=data.get('atlas_index'),
            position=data.get('position', 0.0),
            frame=data.get('frame', 0.0),
            frame_rate_hz=data.get('frame_rate_hz', 0.0),
            frame_loop=data.get('frame_loop', True),
            position_mod_source=data.get('position_mod_source'),
            frame_mod_source=data.get('frame_mod_source'),
            morph_time_ms=data.get('morph_time_ms', 0.0),
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Atlas and Bank
    'WavetableAtlas',
    'WavetableBank',
    'create_basic_waves',
    'create_harmonic_series',
    'create_default_bank',
    # State
    'WavetableState',
    # Coefficient generation
    'wavetable_generate_block',
    'wavetable_generate_coeffs',
    'wavetable_generate_coeffs_fast',
    # IR Spec
    'WavetableSpec',
    # Constants
    'DEFAULT_FRAME_SIZE',
    'DEFAULT_NUM_FRAMES',
]
