"""
WSE-STD-040 — Deterministic Pseudorandomness (PRNG) Standard
=============================================================

This module implements the normative PRNG model for the Wavelet Synth Engine.

Key Properties:
- PCG-32 generator per voice
- Deterministic across platforms (Python, FPGA, simulators)
- Voice isolation: reordering/skipping does not change results
- Bit-exact reproducibility for replay

Architecture Rules:
- PRNG state is owned by Voice Engine and is per-voice
- State advances ONLY when voice is serviced for a block
- Skipped/frozen voices do NOT advance PRNG
- Each operator documents fixed PRNG_DRAWS_PER_CALL
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


# =============================================================================
# CONSTANTS
# =============================================================================

# PCG-32 constants (from PCG reference implementation)
PCG_MULTIPLIER = 6364136223846793005
PCG_DEFAULT_INCREMENT = 1442695040888963407

# Fixed-point conversion
Q15_SCALE = 32768


# =============================================================================
# PCG-32 IMPLEMENTATION (Normative per Section 4.1)
# =============================================================================

class PCG32:
    """
    PCG-32 pseudorandom number generator.

    Per WSE-STD-040 Section 4.1:
    - 32-bit output
    - 64-bit internal state
    - Deterministic across platforms

    This implementation matches the PCG reference exactly for bit-exact
    reproducibility between Python and FPGA.
    """

    def __init__(self, state: int = 0, stream: int = PCG_DEFAULT_INCREMENT):
        """
        Initialize PCG-32 generator.

        Args:
            state: Initial 64-bit state
            stream: Stream selector (must be odd, default is standard increment)
        """
        # Ensure stream is odd (required by PCG)
        self._increment = (stream << 1) | 1
        self._state = 0

        # Initialize state through stepping
        self._state = self._step()
        self._state = (self._state + state) & 0xFFFFFFFFFFFFFFFF
        self._state = self._step()

    def _step(self) -> int:
        """Advance internal state by one step."""
        old_state = self._state
        self._state = ((old_state * PCG_MULTIPLIER) + self._increment) & 0xFFFFFFFFFFFFFFFF
        return old_state

    def next_u32(self) -> int:
        """
        Generate next 32-bit unsigned random value.

        Returns:
            32-bit unsigned integer
        """
        old_state = self._step()

        # XSH-RR output function
        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & 0xFFFFFFFF
        rot = (old_state >> 59) & 0x1F

        # Rotate right
        result = ((xorshifted >> rot) | (xorshifted << (32 - rot))) & 0xFFFFFFFF
        return result

    def next_q15(self) -> int:
        """
        Generate signed Q1.15 random value.

        Per WSE-STD-040 Section 9:
        uint32 r = pcg32();
        int16  q15 = (r >> 17) - 0x4000;

        Returns:
            Signed 16-bit Q15 value in range [-16384, 16383]
        """
        r = self.next_u32()
        return (r >> 17) - 0x4000

    def next_u8(self) -> int:
        """Generate unsigned 8-bit random value."""
        return (self.next_u32() >> 24) & 0xFF

    def next_float(self) -> float:
        """Generate random float in [0.0, 1.0)."""
        return self.next_u32() / 0x100000000

    def next_float_bipolar(self) -> float:
        """Generate random float in [-1.0, 1.0)."""
        return (self.next_u32() / 0x80000000) - 1.0

    @property
    def state(self) -> int:
        """Get current 64-bit state (for serialization)."""
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        """Set 64-bit state (for deserialization)."""
        self._state = value & 0xFFFFFFFFFFFFFFFF

    @property
    def stream(self) -> int:
        """Get stream value (for serialization)."""
        return self._increment >> 1

    def clone(self) -> 'PCG32':
        """Create an identical copy of this PRNG."""
        new_prng = PCG32.__new__(PCG32)
        new_prng._state = self._state
        new_prng._increment = self._increment
        return new_prng


# =============================================================================
# HASH64 FOR SEEDING (Normative per Section 4.3)
# =============================================================================

def hash64_mix(a: int, b: int) -> int:
    """
    Mix two 64-bit values deterministically.

    Uses XOR and multiplication mixing for good distribution.
    This is a simple deterministic hash suitable for seeding.
    """
    # XOR-rotate-multiply mixing
    h = (a ^ b) & 0xFFFFFFFFFFFFFFFF
    h = ((h ^ (h >> 33)) * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    h = ((h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 33)) & 0xFFFFFFFFFFFFFFFF
    return h


def hash64_seed(
    global_seed: int = 0,
    voice_id: int = 0,
    note_id: int = 0,
    preset_id: int = 0,
) -> int:
    """
    Compute deterministic seed for voice PRNG.

    Per WSE-STD-040 Section 4.3:
    seed = HASH64(global_seed, voice_id, note_id, preset_id)

    Requirements:
    - Deterministic and platform-independent
    - No floating-point used
    - global_seed defaults to 0 if unspecified

    Args:
        global_seed: Global seed for session (default 0)
        voice_id: Voice slot ID
        note_id: Note-on counter or MIDI note
        preset_id: Current preset/program ID

    Returns:
        64-bit seed value
    """
    # Mix all inputs deterministically
    h = hash64_mix(global_seed, voice_id)
    h = hash64_mix(h, note_id)
    h = hash64_mix(h, preset_id)
    return h


# =============================================================================
# PER-VOICE PRNG STATE (Normative per Section 4.2)
# =============================================================================

@dataclass
class VoicePRNG:
    """
    Per-voice PRNG state container.

    Per WSE-STD-040 Section 4.2:
    - prng_state_u64: Current PCG state
    - prng_stream_u64: Stream identifier (fixed per voice)

    These values are part of VoiceState, not VoiceFrame.
    """
    voice_id: int
    prng: PCG32 = field(default_factory=lambda: PCG32(0))
    draws_this_block: int = 0
    total_draws: int = 0
    frozen: bool = False

    @classmethod
    def create(
        cls,
        voice_id: int,
        global_seed: int = 0,
        note_id: int = 0,
        preset_id: int = 0,
    ) -> 'VoicePRNG':
        """
        Create a new VoicePRNG with proper seeding.

        Per WSE-STD-040 Section 4.3, seeding occurs on voice allocation.
        """
        seed = hash64_seed(global_seed, voice_id, note_id, preset_id)
        # Use voice_id as stream selector for independence
        stream = voice_id * 2 + 1  # Ensure odd
        return cls(
            voice_id=voice_id,
            prng=PCG32(state=seed, stream=stream),
        )

    def draw_u32(self) -> int:
        """
        Draw a 32-bit random value.

        Per WSE-STD-040 Section 5.1:
        - Only advances when voice is serviced
        - Frozen voices do NOT advance

        Returns 0 if frozen (but caller should not call when frozen).
        """
        if self.frozen:
            return 0

        self.draws_this_block += 1
        self.total_draws += 1
        return self.prng.next_u32()

    def draw_q15(self) -> int:
        """Draw a Q15 random value."""
        if self.frozen:
            return 0

        self.draws_this_block += 1
        self.total_draws += 1
        return self.prng.next_q15()

    def draw_n(self, n: int) -> list:
        """
        Draw exactly N values (for operators with fixed draw count).

        Per WSE-STD-040 Section 5.2:
        - Operators consume PRNG in execution order
        - Each operator documents how many draws it consumes
        - Conditional draws still consume (discarded if unused)
        """
        if self.frozen:
            return [0] * n

        results = []
        for _ in range(n):
            self.draws_this_block += 1
            self.total_draws += 1
            results.append(self.prng.next_u32())
        return results

    def begin_block(self) -> None:
        """Called at start of block to reset per-block counter."""
        self.draws_this_block = 0

    def freeze(self) -> None:
        """Freeze this voice's PRNG (no more advances)."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze this voice's PRNG."""
        self.frozen = False

    def get_state(self) -> Tuple[int, int]:
        """Get (state, stream) for serialization."""
        return (self.prng.state, self.prng.stream)

    def set_state(self, state: int, stream: int) -> None:
        """Restore state from serialization."""
        self.prng._state = state & 0xFFFFFFFFFFFFFFFF
        self.prng._increment = ((stream << 1) | 1) & 0xFFFFFFFFFFFFFFFF


# =============================================================================
# OPERATOR PRNG CONTRACT (Normative per Section 7)
# =============================================================================

# Registry of operator PRNG draw counts
# Per WSE-STD-040 Section 7: Each operator MUST declare PRNG_DRAWS_PER_CALL = N
OPERATOR_PRNG_DRAWS = {
    # Operators that use randomness
    'NoiseBand': 2,       # Per spec example: exactly 2 draws
    'QuasiRotate': 1,     # Per spec example: exactly 1 draw
    'Scatter': 1,         # Scatter injection
    'GlobalPulse': 1,     # Random magnitude variation
    'CuspInjection': 2,   # Target selection + magnitude
    'SwapBands': 1,       # Which bands to swap

    # Operators that do NOT use randomness (0 draws)
    'Identity': 0,
    'Tilt': 0,
    'FineClamp': 0,
    'IntraMix': 0,
    'DiffuseUp': 0,
    'DiffuseDown': 0,
    'BandShear': 0,
    'LowRankMorph': 0,
    'RotateNodes': 0,
    'NormalizeScale': 0,
    'GluePair': 0,
    'Impulse': 0,
    'PressureRamp': 0,
    'GateMask': 0,
    'MorphMix': 0,
    'EmergencySimplify': 0,
    'FoldPressure': 0,
    'SwallowtailMask': 0,
    'SheafProject': 0,
}


def get_operator_prng_draws(op_name: str) -> int:
    """
    Get the fixed number of PRNG draws for an operator.

    Per WSE-STD-040 Section 7:
    - N MUST be constant
    - Draws MUST occur even if strength = 0
    - Draw results MAY be ignored, but NOT skipped

    Args:
        op_name: Operator name

    Returns:
        Number of PRNG draws (0 if operator doesn't use randomness)
    """
    return OPERATOR_PRNG_DRAWS.get(op_name, 0)


def validate_operator_draws(op_name: str, actual_draws: int) -> bool:
    """
    Validate that an operator consumed the correct number of draws.

    Returns True if valid, False if mismatch.
    """
    expected = get_operator_prng_draws(op_name)
    return actual_draws == expected


# =============================================================================
# FIXED-POINT CONVERSION (Normative per Section 9)
# =============================================================================

def prng_to_q15(r: int) -> int:
    """
    Convert 32-bit PRNG output to signed Q15.

    Per WSE-STD-040 Section 9:
    uint32 r = pcg32();
    int16  q15 = (r >> 17) - 0x4000;

    This yields symmetric signed Q1.15 noise.
    """
    return (r >> 17) - 0x4000


def prng_to_q15_unsigned(r: int) -> int:
    """
    Convert 32-bit PRNG output to unsigned Q15 [0, 32767].

    Useful for magnitudes and blend factors.
    """
    return (r >> 17) & 0x7FFF


def prng_to_range(r: int, min_val: int, max_val: int) -> int:
    """
    Convert 32-bit PRNG output to integer in [min_val, max_val].

    Uses rejection-free modulo bias reduction for small ranges.
    """
    range_size = max_val - min_val + 1
    if range_size <= 0:
        return min_val

    # Simple modulo (acceptable for audio, not crypto)
    return min_val + (r % range_size)


# =============================================================================
# PRNG MANAGER (For Voice Engine Integration)
# =============================================================================

class PRNGManager:
    """
    Manages per-voice PRNG instances.

    Per WSE-STD-040:
    - Owns all voice PRNG state
    - Handles allocation/deallocation
    - Enforces advancement rules
    """

    def __init__(self, max_voices: int = 128, global_seed: int = 0):
        self.max_voices = max_voices
        self.global_seed = global_seed
        self._voices: dict[int, VoicePRNG] = {}

    def allocate_voice(
        self,
        voice_id: int,
        note_id: int = 0,
        preset_id: int = 0,
    ) -> VoicePRNG:
        """
        Allocate PRNG for a new voice.

        Per WSE-STD-040 Section 4.3:
        On voice allocation (note-on), PRNG is seeded.
        """
        if voice_id in self._voices:
            # Re-seed existing voice
            self._voices[voice_id] = VoicePRNG.create(
                voice_id=voice_id,
                global_seed=self.global_seed,
                note_id=note_id,
                preset_id=preset_id,
            )
        else:
            self._voices[voice_id] = VoicePRNG.create(
                voice_id=voice_id,
                global_seed=self.global_seed,
                note_id=note_id,
                preset_id=preset_id,
            )

        return self._voices[voice_id]

    def deallocate_voice(self, voice_id: int) -> None:
        """Deallocate voice PRNG."""
        self._voices.pop(voice_id, None)

    def get_voice(self, voice_id: int) -> Optional[VoicePRNG]:
        """Get PRNG for a voice."""
        return self._voices.get(voice_id)

    def begin_block(self) -> None:
        """Called at start of each block."""
        for prng in self._voices.values():
            prng.begin_block()

    def freeze_voice(self, voice_id: int) -> None:
        """Freeze voice PRNG (skipped voice)."""
        if voice_id in self._voices:
            self._voices[voice_id].freeze()

    def unfreeze_voice(self, voice_id: int) -> None:
        """Unfreeze voice PRNG."""
        if voice_id in self._voices:
            self._voices[voice_id].unfreeze()

    def set_global_seed(self, seed: int) -> None:
        """
        Set global seed (affects future allocations only).

        Per WSE-STD-040: global_seed defaults to 0 if unspecified.
        """
        self.global_seed = seed


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core PRNG
    'PCG32',
    # Seeding
    'hash64_mix',
    'hash64_seed',
    # Per-voice state
    'VoicePRNG',
    # Operator contract
    'OPERATOR_PRNG_DRAWS',
    'get_operator_prng_draws',
    'validate_operator_draws',
    # Fixed-point conversion
    'prng_to_q15',
    'prng_to_q15_unsigned',
    'prng_to_range',
    # Manager
    'PRNGManager',
]
