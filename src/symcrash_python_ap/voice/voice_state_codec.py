"""
Voice State Codec (WSE-STD-VOICEENGINEREGISTRY v0 §7)
======================================================

Binary serialization/deserialization for VMState.

Enables:
- Snapshot/checkpoint creation
- Bit-identical resume
- Deterministic replay
- State persistence

Format: Little-endian, integer-only, fixed-length arrays
"""

import struct
import numpy as np
from typing import Tuple

from ..vm.state import VMState, EnvelopeState, EnvelopePhase
from ..vm.coeff_lut_streaming import CCSState
from ..vm.tables import COEFF_LEN


# =============================================================================
# CONSTANTS
# =============================================================================

VMSTATE_VERSION = 0
ENVELOPE_SIZE = 24  # bytes: BBHIhHII (20) + hH (4) = 24
VMSTATE_HEADER_SIZE = 4  # version + voice_id
COEFF_ARRAY_SIZE = COEFF_LEN * 2  # 256 * 2 = 512 bytes


# =============================================================================
# ENVELOPE STATE CODEC
# =============================================================================

def envelope_to_bytes(env: EnvelopeState) -> bytes:
    """
    Serialize EnvelopeState to 24 bytes.

    Format (little-endian):
        u8  phase           (1)
        u8  padding         (1)
        u16 padding         (2)
        u32 phase_sample    (4)
        i16 level_q15       (2)
        u16 padding         (2)
        u32 attack_samples  (4)
        u32 decay_samples   (4)  = 20 bytes
        i16 sustain_level   (2)
        u16 padding         (2)  = 4 bytes
        Total: 24 bytes
    """
    return struct.pack(
        '<BBHIhHII',
        int(env.phase),
        0,  # padding
        0,  # padding
        env.phase_sample,
        env.level_q15,
        0,  # padding
        env.attack_samples,
        env.decay_samples,
    ) + struct.pack('<hH', env.sustain_level_q15, 0)  # sustain + padding


def envelope_from_bytes(data: bytes) -> EnvelopeState:
    """Deserialize EnvelopeState from 24 bytes."""
    # First 20 bytes: BBHIhHII = 1+1+2+4+2+2+4+4 = 20 bytes
    v1 = struct.unpack('<BBHIhHII', data[:20])
    # Last 4 bytes: hH = 2+2 = 4 bytes
    v2 = struct.unpack('<hH', data[20:24])

    return EnvelopeState(
        phase=EnvelopePhase(v1[0]),
        phase_sample=v1[3],
        level_q15=v1[4],
        attack_samples=v1[6],
        decay_samples=v1[7],
        sustain_level_q15=v2[0],
        release_samples=0,  # Not serialized (reconstructed on note-on)
    )


# =============================================================================
# VMSTATE CODEC
# =============================================================================

def vmstate_to_bytes(state: VMState) -> bytes:
    """
    Serialize VMState to binary format.

    Format (little-endian):
        u16 version
        u16 voice_id

        # Carrier oscillator (20 bytes)
        u32 phase
        u32 phase_inc
        u32 target_phase_inc
        u32 pitch_slew_rate
        i8  last_note
        i8  padding
        i16 last_cents

        # Harmonics (32 bytes)
        i16 harmonic_amps_q15[16]

        # Envelope (24 bytes)
        EnvelopeState (packed)

        # DC block filter (4 bytes)
        i16 dc_block_x1
        i16 dc_block_y1

        # Coefficient state (512 bytes)
        i16 coeff_state[256]

        # PRNG (4 bytes)
        u32 prng_state

        # Tree state (4 bytes)
        u8  tree_a_id
        u8  tree_b_id
        u16 padding

        # Latency tracking (8 bytes)
        u32 blocks_since_start
        u8  wet_ready
        u8  padding
        u16 padding

        # Previous values (4 bytes)
        i16 eta_prev_q15
        i16 gain_prev_q15

        # Wet path state flags (4 bytes)
        u8  mask_state_initialized
        u8  wet_seam_initialized
        u8  wet_ola_initialized
        u8  use_coeff_osc

        # Wet seam tracking (2 bytes)
        i16 wet_prev_end

        # CCS State (will be added separately if needed)
        # Mode B State (will be added separately if needed)

    Total size (excluding CCS/ModeB): ~620 bytes
    """
    parts = []

    # Header
    parts.append(struct.pack('<HH', VMSTATE_VERSION, state.voice_id))

    # Carrier oscillator
    parts.append(struct.pack(
        '<IIIIbbh',
        state.phase & 0xFFFFFFFF,  # Ensure 32-bit
        state.phase_inc & 0xFFFFFFFF,
        state.target_phase_inc & 0xFFFFFFFF,
        state.pitch_slew_rate & 0xFFFFFFFF,
        state.last_note,
        0,  # padding
        state.last_cents,
    ))

    # Harmonics
    harmonics_bytes = struct.pack('<16h', *state.harmonic_amps_q15[:16])
    parts.append(harmonics_bytes)

    # Envelope
    parts.append(envelope_to_bytes(state.envelope))

    # DC block filter
    parts.append(struct.pack('<hh', state.dc_block_x1, state.dc_block_y1))

    # Coefficient state (256 i16 values)
    coeff_bytes = state.coeff_state.astype(np.int16).tobytes()
    parts.append(coeff_bytes)

    # PRNG
    parts.append(struct.pack('<I', state.prng_state & 0xFFFFFFFF))

    # Tree state
    parts.append(struct.pack('<BBH', state.tree_a_id, state.tree_b_id, 0))

    # Latency tracking
    parts.append(struct.pack('<IBB', state.blocks_since_start, int(state.wet_ready), 0))

    # Previous values
    parts.append(struct.pack('<hh', state.eta_prev_q15, state.gain_prev_q15))

    # Wet path flags
    parts.append(struct.pack(
        '<BBBB',
        int(state.mask_state_initialized),
        int(state.wet_seam_initialized),
        int(state.wet_ola_initialized),
        int(state.use_coeff_osc),
    ))

    # Wet seam
    parts.append(struct.pack('<h', state.wet_prev_end))

    return b''.join(parts)


def vmstate_from_bytes(data: bytes) -> VMState:
    """
    Deserialize VMState from binary format.

    Args:
        data: Binary data (should be ~620 bytes minimum)

    Returns:
        VMState instance

    Raises:
        struct.error if data is malformed
    """
    offset = 0

    # Header
    version, voice_id = struct.unpack_from('<HH', data, offset)
    offset += 4

    if version != VMSTATE_VERSION:
        raise ValueError(f"Unsupported VMState version: {version}")

    # Carrier oscillator
    phase, phase_inc, target_phase_inc, pitch_slew_rate, last_note, _, last_cents = struct.unpack_from(
        '<IIIIbbh', data, offset
    )
    offset += 20

    # Harmonics
    harmonic_amps = list(struct.unpack_from('<16h', data, offset))
    offset += 32

    # Envelope
    envelope_data = data[offset:offset + ENVELOPE_SIZE]
    envelope = envelope_from_bytes(envelope_data)
    offset += ENVELOPE_SIZE

    # DC block filter
    dc_block_x1, dc_block_y1 = struct.unpack_from('<hh', data, offset)
    offset += 4

    # Coefficient state
    coeff_bytes = data[offset:offset + COEFF_ARRAY_SIZE]
    coeff_state = np.frombuffer(coeff_bytes, dtype=np.int16).copy()
    offset += COEFF_ARRAY_SIZE

    # PRNG
    prng_state, = struct.unpack_from('<I', data, offset)
    offset += 4

    # Tree state
    tree_a_id, tree_b_id, _ = struct.unpack_from('<BBH', data, offset)
    offset += 4

    # Latency tracking
    blocks_since_start, wet_ready_int, _ = struct.unpack_from('<IBB', data, offset)
    wet_ready = bool(wet_ready_int)
    offset += 6

    # Previous values
    eta_prev_q15, gain_prev_q15 = struct.unpack_from('<hh', data, offset)
    offset += 4

    # Wet path flags
    mask_init, seam_init, ola_init, use_coeff = struct.unpack_from('<BBBB', data, offset)
    offset += 4

    # Wet seam
    wet_prev_end, = struct.unpack_from('<h', data, offset)
    offset += 2

    # Construct VMState
    return VMState(
        voice_id=voice_id,
        phase=phase,
        phase_inc=phase_inc,
        target_phase_inc=target_phase_inc,
        pitch_slew_rate=pitch_slew_rate,
        last_note=last_note,
        last_cents=last_cents,
        harmonic_amps_q15=harmonic_amps,
        envelope=envelope,
        dc_block_x1=dc_block_x1,
        dc_block_y1=dc_block_y1,
        coeff_state=coeff_state,
        prng_state=prng_state,
        tree_a_id=tree_a_id,
        tree_b_id=tree_b_id,
        blocks_since_start=blocks_since_start,
        wet_ready=wet_ready,
        eta_prev_q15=eta_prev_q15,
        gain_prev_q15=gain_prev_q15,
        mask_state_initialized=bool(mask_init),
        wet_seam_initialized=bool(seam_init),
        wet_ola_initialized=bool(ola_init),
        use_coeff_osc=bool(use_coeff),
        wet_prev_end=wet_prev_end,
    )


def get_vmstate_size() -> int:
    """
    Get minimum size of serialized VMState.

    Note: Actual VMState may have additional fields (CCS state, Mode B state,
    beta state, etc.) that increase the size. This returns the core fields only.

    TODO: Add CCS/ModeB serialization for complete snapshot support.
    """
    return (
        4 +    # Header
        20 +   # Carrier oscillator
        32 +   # Harmonics
        24 +   # Envelope (ENVELOPE_SIZE)
        4 +    # DC block
        512 +  # Coefficients
        4 +    # PRNG
        4 +    # Tree
        6 +    # Latency
        4 +    # Previous values
        4 +    # Flags
        2      # Wet seam
    )  # = 620 bytes (minimum, core fields only)


__all__ = [
    'vmstate_to_bytes',
    'vmstate_from_bytes',
    'get_vmstate_size',
    'envelope_to_bytes',
    'envelope_from_bytes',
    'VMSTATE_VERSION',
]
