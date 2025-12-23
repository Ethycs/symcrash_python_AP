"""
Data Structures and Frame Types
===============================

Block-rate data contracts for the pipeline.

Includes:
- ParamFrame: Block-rate scalar parameters
- VoiceFrame: Per-voice coefficient frame (internal)
- VoiceFramePacketV0b: 136-byte wire format (per Wire Contract v0b)
- MaterialParams: Slewed material parameters
- Metrics: Block metrics for safety/monitoring
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import IntEnum
import struct


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


# =============================================================================
# VOICEFRAME PACKET V0b (136 bytes exact)
# Per docs/Core System/04 - Wire Contract v0b, VoiceFrame Packet v0b.md
# =============================================================================

# VoiceFramePacketV0b constants
VFP_MAGIC = 0x5646      # "VF" in ASCII
VFP_VERSION = 1
VFP_PACKET_LEN = 136
NUM_TREES = 5
Q15_ONE = 32767


class VoiceFlags(IntEnum):
    """Voice flags bits per Wire Contract v0b."""
    GATE = 0x0001           # Gate is active
    GATE_RISE = 0x0002      # Gate just rose (note-on)
    GATE_FALL = 0x0004      # Gate just fell (note-off)
    RETRIGGER = 0x0008      # Retrigger envelope
    LEGATO = 0x0010         # Legato mode active
    SUSTAIN = 0x0020        # Sustain pedal active


@dataclass
class VoiceFramePacketV0b:
    """
    VoiceFrame Packet v0b - exact 136-byte format.

    IMPORTANT: This is an INTERNAL/DEBUG format, NOT for real-time wire use.

    At 750 Hz block rate with 12 voices:
      136 bytes × 12 voices × 750 Hz = 1,224,000 bytes/s
      UART at 921600 bps = ~92,000 bytes/s capacity
      → Physically impossible for real-time transmission.

    For real-time wire transmission, use CTRL-DELTA packets (ctrl_delta.py):
      - Sparse parameter updates using MaskedUpdate + FieldMask
      - Typically < 30 bytes per update
      - Only changed parameters transmitted

    This format is used for:
      - Debug/diagnostic dumps
      - Logging and replay
      - Internal Python simulation
      - Test fixtures

    Per docs/Core System/04 - Wire Contract v0b, VoiceFrame Packet v0b.md
    All fields are big-endian per spec Section 2.

    Layout:
        0x0000  2     magic u16 = 0x5646 ("VF")
        0x0002  1     version u8 = 1
        0x0003  1     packet_len u8 = 136
        0x0004  2     program_id u16
        0x0006  2     voice_id u16
        0x0008  4     frame_seq u32
        0x000C  2     reject_code u16
        0x000E  2     hdr_pad u16 = 0
        0x0010  2     flags u16
        0x0012  1     regime_id u8
        0x0013  1     degradation_level u8
        0x0014  1     note_u8 u8
        0x0015  1     pad0 u8 = 0
        0x0016  2     cents_i16 i16
        0x0018  2     gain_q15 u16
        0x001A  2     activity_q15 u16
        0x001C  2     pan_q15 i16
        0x001E  2     sendA_q15 u16
        0x0020  2     sendB_q15 u16
        0x0022  16    w_family_q15[8] u16[8]
        0x0032  1     tree_a_id u8
        0x0033  1     tree_b_id u8
        0x0034  2     tree_blend_q15 u16
        0x0036  1     event_mask u8
        0x0037  1     pad1 u8 = 0
        0x0038  16    event_strength_q15[8] u16[8]
        0x0048  64    event_payload_u64[8] u64[8]
        0x0088  END (136 bytes)
    """
    # Header (16 bytes)
    magic: int = VFP_MAGIC
    version: int = VFP_VERSION
    packet_len: int = VFP_PACKET_LEN
    program_id: int = 0
    voice_id: int = 0
    frame_seq: int = 0
    reject_code: int = 0
    hdr_pad: int = 0

    # Control (8 bytes)
    flags: int = 0
    regime_id: int = 0
    degradation_level: int = 0
    note_u8: int = 60           # MIDI note (0..127)
    pad0: int = 0
    cents_i16: int = 0          # Fine tuning in cents

    # Gain/Pan/Send (10 bytes)
    gain_q15: int = Q15_ONE
    activity_q15: int = 0
    pan_q15: int = 0            # -32768..32767 (bipolar)
    sendA_q15: int = 0
    sendB_q15: int = 0

    # Family weights (16 bytes)
    w_family_q15: List[int] = field(default_factory=lambda: [0] * 8)

    # Tree control (4 bytes)
    tree_a_id: int = 0
    tree_b_id: int = 0
    tree_blend_q15: int = 0

    # Events (18 bytes header + 64 bytes payloads = 82 bytes)
    event_mask: int = 0
    pad1: int = 0
    event_strength_q15: List[int] = field(default_factory=lambda: [0] * 8)
    event_payload_u64: List[int] = field(default_factory=lambda: [0] * 8)

    def to_bytes(self) -> bytes:
        """
        Serialize to exact 136 bytes (big-endian per spec).
        """
        # Header (16 bytes)
        data = struct.pack('>HBBHHIHH',
            self.magic,
            self.version,
            self.packet_len,
            self.program_id,
            self.voice_id,
            self.frame_seq,
            self.reject_code,
            self.hdr_pad
        )

        # Control (8 bytes)
        data += struct.pack('>HBBBBh',
            self.flags,
            self.regime_id,
            self.degradation_level,
            self.note_u8,
            self.pad0,
            self.cents_i16
        )

        # Gain/Pan/Send (10 bytes)
        data += struct.pack('>HHhHH',
            self.gain_q15,
            self.activity_q15,
            self.pan_q15,
            self.sendA_q15,
            self.sendB_q15
        )

        # Family weights (16 bytes)
        for i in range(8):
            data += struct.pack('>H', self.w_family_q15[i] if i < len(self.w_family_q15) else 0)

        # Tree control (4 bytes)
        data += struct.pack('>BBHB',
            self.tree_a_id,
            self.tree_b_id,
            self.tree_blend_q15,
            self.event_mask
        )

        # Event pad (1 byte)
        data += struct.pack('>B', self.pad1)

        # Event strengths (16 bytes)
        for i in range(8):
            data += struct.pack('>H', self.event_strength_q15[i] if i < len(self.event_strength_q15) else 0)

        # Event payloads (64 bytes)
        for i in range(8):
            payload = self.event_payload_u64[i] if i < len(self.event_payload_u64) else 0
            data += struct.pack('>Q', payload)

        assert len(data) == VFP_PACKET_LEN, f"Packet size mismatch: {len(data)} != {VFP_PACKET_LEN}"
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VoiceFramePacketV0b':
        """
        Deserialize from 136 bytes (big-endian per spec).
        """
        if len(data) < VFP_PACKET_LEN:
            raise ValueError(f"Data too short: {len(data)} < {VFP_PACKET_LEN}")

        offset = 0

        # Header (16 bytes)
        magic, version, packet_len, program_id, voice_id, frame_seq, reject_code, hdr_pad = \
            struct.unpack('>HBBHHIHH', data[offset:offset+16])
        offset += 16

        # Control (8 bytes)
        flags, regime_id, degradation_level, note_u8, pad0, cents_i16 = \
            struct.unpack('>HBBBBh', data[offset:offset+8])
        offset += 8

        # Gain/Pan/Send (10 bytes)
        gain_q15, activity_q15, pan_q15, sendA_q15, sendB_q15 = \
            struct.unpack('>HHhHH', data[offset:offset+10])
        offset += 10

        # Family weights (16 bytes)
        w_family_q15 = list(struct.unpack('>8H', data[offset:offset+16]))
        offset += 16

        # Tree control + event_mask (5 bytes)
        tree_a_id, tree_b_id, tree_blend_q15, event_mask = \
            struct.unpack('>BBHB', data[offset:offset+5])
        offset += 5

        # Event pad (1 byte)
        pad1 = struct.unpack('>B', data[offset:offset+1])[0]
        offset += 1

        # Event strengths (16 bytes)
        event_strength_q15 = list(struct.unpack('>8H', data[offset:offset+16]))
        offset += 16

        # Event payloads (64 bytes)
        event_payload_u64 = list(struct.unpack('>8Q', data[offset:offset+64]))
        offset += 64

        return cls(
            magic=magic,
            version=version,
            packet_len=packet_len,
            program_id=program_id,
            voice_id=voice_id,
            frame_seq=frame_seq,
            reject_code=reject_code,
            hdr_pad=hdr_pad,
            flags=flags,
            regime_id=regime_id,
            degradation_level=degradation_level,
            note_u8=note_u8,
            pad0=pad0,
            cents_i16=cents_i16,
            gain_q15=gain_q15,
            activity_q15=activity_q15,
            pan_q15=pan_q15,
            sendA_q15=sendA_q15,
            sendB_q15=sendB_q15,
            w_family_q15=w_family_q15,
            tree_a_id=tree_a_id,
            tree_b_id=tree_b_id,
            tree_blend_q15=tree_blend_q15,
            event_mask=event_mask,
            pad1=pad1,
            event_strength_q15=event_strength_q15,
            event_payload_u64=event_payload_u64,
        )

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate packet structure and constraints.

        Returns (is_valid, error_message).
        """
        if self.magic != VFP_MAGIC:
            return False, f"Invalid magic: 0x{self.magic:04X} != 0x{VFP_MAGIC:04X}"

        if self.version != VFP_VERSION:
            return False, f"Invalid version: {self.version} != {VFP_VERSION}"

        if self.packet_len != VFP_PACKET_LEN:
            return False, f"Invalid packet_len: {self.packet_len} != {VFP_PACKET_LEN}"

        if self.hdr_pad != 0:
            return False, f"hdr_pad must be 0: {self.hdr_pad}"

        if self.pad0 != 0:
            return False, f"pad0 must be 0: {self.pad0}"

        if self.pad1 != 0:
            return False, f"pad1 must be 0: {self.pad1}"

        if self.note_u8 > 127:
            return False, f"note_u8 out of range: {self.note_u8} > 127"

        if self.regime_id > 15:
            return False, f"regime_id out of range: {self.regime_id} > 15"

        if self.tree_a_id >= NUM_TREES or self.tree_b_id >= NUM_TREES:
            return False, f"tree_id out of range: a={self.tree_a_id}, b={self.tree_b_id}"

        # Validate event payloads: lane_id in bits 63..60 must match index
        for i, payload in enumerate(self.event_payload_u64):
            if self.event_mask & (1 << i):
                lane_id = (payload >> 60) & 0x0F
                if lane_id != i:
                    return False, f"Lane {i} payload has mismatched lane_id: {lane_id}"

        return True, None

    @staticmethod
    def unpack_event_payload(payload_u64: int) -> dict:
        """
        Unpack a 64-bit event payload into its fields.

        Per Wire Contract v0b:
            63..60: lane_id
            59..56: variant
            55..48: strength_q8 (reserved=0)
            47..40: target_kind
            39..32: target_id
            31..24: pattern_id
            23..16: param0_q8
            15..8:  param1_q8
            7..0:   param2_q8
        """
        return {
            'lane_id': (payload_u64 >> 60) & 0x0F,
            'variant': (payload_u64 >> 56) & 0x0F,
            'strength_q8': (payload_u64 >> 48) & 0xFF,
            'target_kind': (payload_u64 >> 40) & 0xFF,
            'target_id': (payload_u64 >> 32) & 0xFF,
            'pattern_id': (payload_u64 >> 24) & 0xFF,
            'param0_q8': (payload_u64 >> 16) & 0xFF,
            'param1_q8': (payload_u64 >> 8) & 0xFF,
            'param2_q8': payload_u64 & 0xFF,
        }

    @staticmethod
    def pack_event_payload(lane_id: int, variant: int = 0, strength_q8: int = 0,
                           target_kind: int = 0, target_id: int = 0, pattern_id: int = 0,
                           param0_q8: int = 0, param1_q8: int = 0, param2_q8: int = 0) -> int:
        """Pack fields into a 64-bit event payload."""
        return (
            ((lane_id & 0x0F) << 60) |
            ((variant & 0x0F) << 56) |
            ((strength_q8 & 0xFF) << 48) |
            ((target_kind & 0xFF) << 40) |
            ((target_id & 0xFF) << 32) |
            ((pattern_id & 0xFF) << 24) |
            ((param0_q8 & 0xFF) << 16) |
            ((param1_q8 & 0xFF) << 8) |
            (param2_q8 & 0xFF)
        )

    def to_voice_frame(self) -> VoiceFrame:
        """
        Convert to internal VoiceFrame format.
        """
        return VoiceFrame(
            voice_id=self.voice_id,
            tree_id=self.tree_a_id,
            flags=self.flags,
            gain_q15=self.gain_q15,
            pan_q15=self.pan_q15,
            tree_a_id=self.tree_a_id,
            tree_b_id=self.tree_b_id,
            tree_blend_q15=self.tree_blend_q15,
        )

    @classmethod
    def from_voice_frame(cls, vf: VoiceFrame, program_id: int = 0,
                         frame_seq: int = 0, regime_id: int = 0,
                         note_u8: int = 60, cents_i16: int = 0) -> 'VoiceFramePacketV0b':
        """
        Convert from internal VoiceFrame format to 136-byte packet.
        """
        return cls(
            program_id=program_id,
            voice_id=vf.voice_id,
            frame_seq=frame_seq,
            flags=vf.flags,
            regime_id=regime_id,
            note_u8=note_u8,
            cents_i16=cents_i16,
            gain_q15=vf.gain_q15,
            pan_q15=vf.pan_q15,
            tree_a_id=vf.tree_a_id,
            tree_b_id=vf.tree_b_id,
            tree_blend_q15=vf.tree_blend_q15,
        )
