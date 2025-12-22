"""
WSE-STD-020 UART Protocol Implementation
========================================

Implements VoiceFrame ABI + Event Command Bus over UART (v0).

This module provides:
- VoiceFrame v0 ABI with all required fields
- BackboneEntry for operator control
- Payload64 format encoder/decoder
- UART packet protocol (framing, CRC, message types)
- Python UART listener for controller testing

Conformance: WSE-STD-020 v0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import IntEnum
import struct
import threading
import queue
import time

from fixed import sat16, mul_q15_sat, Q15_ONE, Q15_HALF


# =============================================================================
# CONSTANTS (per WSE-STD-020)
# =============================================================================

BLOCK_SAMPLES = 64
SYNC_WORD = 0xA55A
PROTOCOL_VERSION = 0

# Packet limits
MAX_PAYLOAD_SIZE = 512
MAX_BACKBONE_ENTRIES = 4
NUM_EVENT_LANES = 8
NUM_TREES = 5

# CRC-32C (Castagnoli) polynomial - reflected form for table-driven algorithm
CRC32C_POLY = 0x82F63B78


# =============================================================================
# EVENT LANES (per WSE-STD-020 Section 2.3)
# =============================================================================

class EventLane(IntEnum):
    """Event lane IDs per WSE-STD-020 Section 2.3."""
    EVT_OP_FIRE = 0      # Fire one-shot operator/event
    EVT_OP_TOGGLE = 1    # Toggle persistent op_enable bits
    EVT_INJECT = 2       # Injection (impulse/noise/burst)
    EVT_MATERIAL = 3     # Material engine override
    EVT_TREE = 4         # Tree morph or switch event
    EVT_CLOUD = 5        # Cloud/foreground reclassify hint
    EVT_VOICE = 6        # Note lifecycle / state events
    EVT_DEBUG = 7        # Debug marker / capture request


# =============================================================================
# MESSAGE TYPES (per WSE-STD-020 Section 6.3)
# =============================================================================

class MsgType(IntEnum):
    """UART message types per WSE-STD-020 Section 6.3."""
    VF_ABSOLUTE = 0x01      # Full VoiceFrame (absolute fields)
    VF_DELTA = 0x02         # Partial VoiceFrame update
    PARAM_ABSOLUTE = 0x03   # Set parameter register (absolute)
    PARAM_DELTA = 0x04      # Add delta to parameter register
    PRESET_LOAD = 0x05      # Load preset by ID/hash
    ACK = 0x06              # Ack a seq with status
    NACK = 0x07             # Nack with error code
    TELEMETRY = 0x08        # Optional: meters, voice counts
    CAPTURE = 0x09          # Optional: debug capture frames


# =============================================================================
# PAYLOAD MODES (per WSE-STD-020 Section 4)
# =============================================================================

class PayloadMode(IntEnum):
    """Payload modes per WSE-STD-020 Section 4."""
    COMMON = 0              # Single payload for all lanes
    PER_LANE_SPARSE = 1     # k (lane_id, payload64) pairs
    PER_LANE_FULL = 2       # payload_lane[8]


# =============================================================================
# TARGET KINDS (per WSE-STD-020 Section 3.1)
# =============================================================================

class TargetKind(IntEnum):
    """Target kinds for Payload64 per WSE-STD-020 Section 3.1."""
    PATTERN = 0         # Resolves via Pattern Table
    LEAF = 1            # Direct leaf index
    BAND_GROUP = 2      # Band group (low/mid/high)
    COEFF_MASK = 3      # Coefficient mask within leaf
    RESERVED = 4


# =============================================================================
# ERROR CODES (per WSE-STD-020 Section 9.2)
# =============================================================================

class ErrorCode(IntEnum):
    """NACK error codes per WSE-STD-020 Section 9.2."""
    BAD_CRC = 1
    BAD_LENGTH = 2
    UNKNOWN_MSG_TYPE = 3
    INVALID_VOICE_ID = 4
    INVALID_TREE_ID = 5
    INVALID_PAYLOAD_MODE = 6
    BACKLOG_OVERFLOW = 7
    BLOCK_TOO_OLD = 8


# =============================================================================
# ACK STATUS (per WSE-STD-020 Section 9.1)
# =============================================================================

class AckStatus(IntEnum):
    """ACK status codes per WSE-STD-020 Section 9.1."""
    OK = 0
    OK_QUEUED = 1
    OK_APPLIED_LATE = 2


# =============================================================================
# CRC-32C IMPLEMENTATION
# =============================================================================

def _make_crc32c_table() -> List[int]:
    """Generate CRC-32C lookup table."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ CRC32C_POLY
            else:
                crc >>= 1
        table.append(crc)
    return table

_CRC32C_TABLE = _make_crc32c_table()


def crc32c(data: bytes, initial: int = 0xFFFFFFFF) -> int:
    """
    Compute CRC-32C (Castagnoli) checksum.

    Per WSE-STD-020 Section 6.2: CRC-32 of [version..payload], NOT including sync.
    """
    crc = initial
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


# =============================================================================
# BACKBONE ENTRY (per WSE-STD-020 Section 2.2)
# =============================================================================

@dataclass
class BackboneEntry:
    """
    Backbone operator entry per WSE-STD-020 Section 2.2.

    Normative: Backbone operator order is SEMANTIC and MUST NOT be reordered.
    """
    op_id: int = 0              # 0..255 (v0 uses 0..23)
    strength_q15: int = 0       # 0..32767
    param0_q15: int = 0         # Operator-specific (or 0)
    param1_q15: int = 0         # Operator-specific (or 0)

    def to_bytes(self) -> bytes:
        """Serialize to 7 bytes (packed)."""
        return struct.pack('<BHHH',
            self.op_id,
            self.strength_q15,
            self.param0_q15,
            self.param1_q15
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'BackboneEntry':
        """Deserialize from bytes."""
        op_id, strength, p0, p1 = struct.unpack('<BHHH', data[:7])
        return cls(op_id=op_id, strength_q15=strength, param0_q15=p0, param1_q15=p1)

    @staticmethod
    def byte_size() -> int:
        return 7


# =============================================================================
# PAYLOAD64 FORMAT (per WSE-STD-020 Section 3)
# =============================================================================

@dataclass
class Payload64:
    """
    Payload64 format per WSE-STD-020 Section 3.

    64-bit packed selector for event targeting.
    """
    payload_type: int = 0       # bits 0..7
    lane_hint: int = 0          # bits 8..15 (only for sparse mode)
    pattern_id: int = 0         # bits 16..23
    target_kind: TargetKind = TargetKind.PATTERN  # bits 24..31
    target_id: int = 0          # bits 32..39 (e.g., leaf index)
    coeff_mask: int = 0         # bits 40..47 (bit j selects coeff j)
    variant0: int = 0           # bits 48..55 (e.g., impulse type)
    variant1: int = 0           # bits 56..63 (extra subtype)

    def pack(self) -> int:
        """Pack to uint64."""
        return (
            (self.payload_type & 0xFF) |
            ((self.lane_hint & 0xFF) << 8) |
            ((self.pattern_id & 0xFF) << 16) |
            ((int(self.target_kind) & 0xFF) << 24) |
            ((self.target_id & 0xFF) << 32) |
            ((self.coeff_mask & 0xFF) << 40) |
            ((self.variant0 & 0xFF) << 48) |
            ((self.variant1 & 0xFF) << 56)
        )

    @classmethod
    def unpack(cls, value: int) -> 'Payload64':
        """Unpack from uint64."""
        return cls(
            payload_type=(value >> 0) & 0xFF,
            lane_hint=(value >> 8) & 0xFF,
            pattern_id=(value >> 16) & 0xFF,
            target_kind=TargetKind((value >> 24) & 0xFF),
            target_id=(value >> 32) & 0xFF,
            coeff_mask=(value >> 40) & 0xFF,
            variant0=(value >> 48) & 0xFF,
            variant1=(value >> 56) & 0xFF,
        )

    def to_bytes(self) -> bytes:
        """Serialize to 8 bytes (little-endian)."""
        return struct.pack('<Q', self.pack())

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Payload64':
        """Deserialize from bytes."""
        value = struct.unpack('<Q', data[:8])[0]
        return cls.unpack(value)

    @staticmethod
    def for_leaf(leaf_id: int, coeff_mask: int = 0, variant0: int = 0) -> 'Payload64':
        """Create payload targeting a specific leaf (canonical index)."""
        return Payload64(
            target_kind=TargetKind.LEAF,
            target_id=leaf_id,
            coeff_mask=coeff_mask,
            variant0=variant0
        )

    @staticmethod
    def for_pattern(pattern_id: int, coeff_mask: int = 0) -> 'Payload64':
        """Create payload targeting a pattern from the Pattern Table."""
        return Payload64(
            target_kind=TargetKind.PATTERN,
            pattern_id=pattern_id,
            coeff_mask=coeff_mask
        )


# =============================================================================
# VOICEFRAME v0 ABI (per WSE-STD-020 Section 2.1)
# =============================================================================

@dataclass
class VoiceFrameV0:
    """
    VoiceFrame v0 ABI per WSE-STD-020 Section 2.1.

    Describes what Symbol+Wavelet engines must execute for a single voice
    on a single block.
    """
    # Identity
    voice_id: int = 0           # 0..MAX_VOICES-1
    block_index: int = 0        # Monotonic per audio stream (wraps allowed)

    # Regime control
    regime_id: int = 0          # Instrument regime selector
    gear_q15: int = 0           # Commitment/intensity (0..32767)

    # Tree control
    tree_id_a: int = 0          # Primary tree selector
    tree_id_b: int = 0          # Optional morph target (0 = same as A)
    tree_blend_q15: int = 0     # Blend A->B (0..32767)

    # Voice control
    gain_q15: int = 0x4000      # Voice output gain into mix (0..32767)
    traction_q15: int = Q15_ONE # Q clamp strength / legality hardness (0..32767)

    # Backbone operators
    backbone: List[BackboneEntry] = field(default_factory=list)

    # Event lanes
    event_mask: int = 0         # Bit i = lane i fires this block
    event_strength_q15: List[int] = field(default_factory=lambda: [0] * NUM_EVENT_LANES)

    # Payload
    payload_mode: PayloadMode = PayloadMode.COMMON
    payload_common: Payload64 = field(default_factory=Payload64)
    payload_per_lane: List[Payload64] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes per WSE-STD-020 Section 7.1 (VF_ABSOLUTE).

        Binary encoding for UART transmission.
        """
        # Fixed header
        data = struct.pack('<HHHHHHHHHB',
            self.voice_id,
            self.block_index,
            self.regime_id,
            self.gear_q15,
            self.tree_id_a,
            self.tree_id_b,
            self.tree_blend_q15,
            self.gain_q15,
            self.traction_q15,
            len(self.backbone)
        )

        # Backbone entries
        for entry in self.backbone[:MAX_BACKBONE_ENTRIES]:
            data += entry.to_bytes()

        # Event mask
        data += struct.pack('<B', self.event_mask)

        # Event strengths (8 x uint16)
        for i in range(NUM_EVENT_LANES):
            data += struct.pack('<H', self.event_strength_q15[i])

        # Payload mode
        data += struct.pack('<B', int(self.payload_mode))

        # Payload common
        data += self.payload_common.to_bytes()

        # Per-lane payloads
        if self.payload_mode == PayloadMode.PER_LANE_FULL:
            for i in range(NUM_EVENT_LANES):
                if i < len(self.payload_per_lane):
                    data += self.payload_per_lane[i].to_bytes()
                else:
                    data += Payload64().to_bytes()

        elif self.payload_mode == PayloadMode.PER_LANE_SPARSE:
            data += struct.pack('<B', len(self.payload_per_lane))
            for i, payload in enumerate(self.payload_per_lane):
                # lane_id is stored in payload.lane_hint
                data += struct.pack('<B', payload.lane_hint)
                data += payload.to_bytes()

        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VoiceFrameV0':
        """Deserialize from bytes."""
        offset = 0

        # Fixed header
        (voice_id, block_index, regime_id, gear_q15,
         tree_id_a, tree_id_b, tree_blend_q15,
         gain_q15, traction_q15, backbone_len) = struct.unpack('<HHHHHHHHHB', data[offset:offset+19])
        offset += 19

        # Backbone entries
        backbone = []
        for _ in range(backbone_len):
            entry = BackboneEntry.from_bytes(data[offset:])
            backbone.append(entry)
            offset += BackboneEntry.byte_size()

        # Event mask
        event_mask = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1

        # Event strengths
        event_strength_q15 = list(struct.unpack('<8H', data[offset:offset+16]))
        offset += 16

        # Payload mode
        payload_mode = PayloadMode(struct.unpack('<B', data[offset:offset+1])[0])
        offset += 1

        # Payload common
        payload_common = Payload64.from_bytes(data[offset:])
        offset += 8

        # Per-lane payloads
        payload_per_lane = []
        if payload_mode == PayloadMode.PER_LANE_FULL:
            for _ in range(NUM_EVENT_LANES):
                payload_per_lane.append(Payload64.from_bytes(data[offset:]))
                offset += 8

        elif payload_mode == PayloadMode.PER_LANE_SPARSE:
            k = struct.unpack('<B', data[offset:offset+1])[0]
            offset += 1
            for _ in range(k):
                lane_id = struct.unpack('<B', data[offset:offset+1])[0]
                offset += 1
                payload = Payload64.from_bytes(data[offset:])
                payload.lane_hint = lane_id
                payload_per_lane.append(payload)
                offset += 8

        return cls(
            voice_id=voice_id,
            block_index=block_index,
            regime_id=regime_id,
            gear_q15=gear_q15,
            tree_id_a=tree_id_a,
            tree_id_b=tree_id_b,
            tree_blend_q15=tree_blend_q15,
            gain_q15=gain_q15,
            traction_q15=traction_q15,
            backbone=backbone,
            event_mask=event_mask,
            event_strength_q15=event_strength_q15,
            payload_mode=payload_mode,
            payload_common=payload_common,
            payload_per_lane=payload_per_lane,
        )


# =============================================================================
# VOICEFRAME DELTA (per WSE-STD-020 Section 7.2)
# =============================================================================

class FieldMask(IntEnum):
    """Field mask bits for VF_DELTA per WSE-STD-020 Section 7.2."""
    REGIME_ID = 0
    GEAR_Q15 = 1
    TREE_ID_A = 2
    TREE_ID_B = 3
    TREE_BLEND_Q15 = 4
    GAIN_Q15 = 5
    TRACTION_Q15 = 6
    BACKBONE = 7
    EVENT_MASK = 8
    EVENT_STRENGTH = 9
    PAYLOAD_MODE = 10
    PAYLOAD_COMMON = 11
    PER_LANE_PAYLOADS = 12


@dataclass
class VoiceFrameDelta:
    """
    Partial VoiceFrame update per WSE-STD-020 Section 7.2.

    Only transmits changed fields to reduce bandwidth.
    """
    voice_id: int = 0
    block_index: int = 0
    field_mask: int = 0         # Which fields are present

    # Optional fields (only present if corresponding bit is set)
    regime_id: Optional[int] = None
    gear_q15: Optional[int] = None
    tree_id_a: Optional[int] = None
    tree_id_b: Optional[int] = None
    tree_blend_q15: Optional[int] = None
    gain_q15: Optional[int] = None
    traction_q15: Optional[int] = None
    backbone: Optional[List[BackboneEntry]] = None
    event_mask: Optional[int] = None
    event_strength_q15: Optional[List[int]] = None
    payload_mode: Optional[PayloadMode] = None
    payload_common: Optional[Payload64] = None
    payload_per_lane: Optional[List[Payload64]] = None

    def to_bytes(self) -> bytes:
        """Serialize to bytes per canonical field order."""
        data = struct.pack('<HHII',
            self.voice_id,
            self.block_index,
            self.field_mask,
            0  # field_mask1 (reserved)
        )

        # Canonical order (MUST match spec Section 7.2)
        if self.field_mask & (1 << FieldMask.REGIME_ID):
            data += struct.pack('<H', self.regime_id)
        if self.field_mask & (1 << FieldMask.GEAR_Q15):
            data += struct.pack('<H', self.gear_q15)
        if self.field_mask & (1 << FieldMask.TREE_ID_A):
            data += struct.pack('<H', self.tree_id_a)
        if self.field_mask & (1 << FieldMask.TREE_ID_B):
            data += struct.pack('<H', self.tree_id_b)
        if self.field_mask & (1 << FieldMask.TREE_BLEND_Q15):
            data += struct.pack('<H', self.tree_blend_q15)
        if self.field_mask & (1 << FieldMask.GAIN_Q15):
            data += struct.pack('<H', self.gain_q15)
        if self.field_mask & (1 << FieldMask.TRACTION_Q15):
            data += struct.pack('<H', self.traction_q15)
        if self.field_mask & (1 << FieldMask.BACKBONE):
            data += struct.pack('<B', len(self.backbone))
            for entry in self.backbone:
                data += entry.to_bytes()
        if self.field_mask & (1 << FieldMask.EVENT_MASK):
            data += struct.pack('<B', self.event_mask)
        if self.field_mask & (1 << FieldMask.EVENT_STRENGTH):
            for s in self.event_strength_q15:
                data += struct.pack('<H', s)
        if self.field_mask & (1 << FieldMask.PAYLOAD_MODE):
            data += struct.pack('<B', int(self.payload_mode))
        if self.field_mask & (1 << FieldMask.PAYLOAD_COMMON):
            data += self.payload_common.to_bytes()
        if self.field_mask & (1 << FieldMask.PER_LANE_PAYLOADS):
            # Use same encoding as VF_ABSOLUTE based on payload_mode
            if self.payload_mode == PayloadMode.PER_LANE_FULL:
                for p in self.payload_per_lane:
                    data += p.to_bytes()
            elif self.payload_mode == PayloadMode.PER_LANE_SPARSE:
                data += struct.pack('<B', len(self.payload_per_lane))
                for p in self.payload_per_lane:
                    data += struct.pack('<B', p.lane_hint)
                    data += p.to_bytes()

        return data


# =============================================================================
# UART PACKET (per WSE-STD-020 Section 6.2)
# =============================================================================

@dataclass
class UARTPacket:
    """
    UART packet per WSE-STD-020 Section 6.2.

    All multi-byte integers are little-endian.
    """
    version: int = PROTOCOL_VERSION
    msg_type: MsgType = MsgType.VF_ABSOLUTE
    seq: int = 0
    payload: bytes = b''

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes with sync word and CRC.

        Format:
        - uint16 sync (0xA55A)
        - uint8 version
        - uint8 msg_type
        - uint16 seq
        - uint16 length
        - uint8[] payload
        - uint32 crc32 (of version..payload)
        """
        header = struct.pack('<BBHH',
            self.version,
            int(self.msg_type),
            self.seq,
            len(self.payload)
        )

        # CRC covers version through payload
        crc_data = header + self.payload
        crc = crc32c(crc_data)

        return struct.pack('<H', SYNC_WORD) + crc_data + struct.pack('<I', crc)

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple[Optional['UARTPacket'], int]:
        """
        Deserialize from bytes.

        Returns: (packet or None if invalid, bytes consumed)
        """
        if len(data) < 12:  # Minimum packet size
            return None, 0

        # Find sync word
        sync_offset = 0
        while sync_offset < len(data) - 1:
            sync = struct.unpack('<H', data[sync_offset:sync_offset+2])[0]
            if sync == SYNC_WORD:
                break
            sync_offset += 1
        else:
            return None, len(data) - 1

        # Read header
        if len(data) < sync_offset + 10:
            return None, sync_offset

        header_start = sync_offset + 2
        version, msg_type, seq, length = struct.unpack(
            '<BBHH', data[header_start:header_start+6]
        )

        # Check if we have full packet
        total_size = 2 + 6 + length + 4  # sync + header + payload + crc
        if len(data) < sync_offset + total_size:
            return None, sync_offset

        # Extract payload
        payload_start = header_start + 6
        payload = data[payload_start:payload_start + length]

        # Verify CRC
        crc_start = payload_start + length
        crc_received = struct.unpack('<I', data[crc_start:crc_start+4])[0]

        crc_data = data[header_start:payload_start + length]
        crc_computed = crc32c(crc_data)

        if crc_received != crc_computed:
            # CRC mismatch - skip this sync and try again
            return None, sync_offset + 1

        return cls(
            version=version,
            msg_type=MsgType(msg_type),
            seq=seq,
            payload=payload
        ), sync_offset + total_size

    @staticmethod
    def make_ack(seq: int, status: AckStatus = AckStatus.OK, detail: int = 0) -> 'UARTPacket':
        """Create ACK packet."""
        payload = struct.pack('<HBB', seq, int(status), detail)
        return UARTPacket(msg_type=MsgType.ACK, seq=0, payload=payload)

    @staticmethod
    def make_nack(seq: int, error_code: ErrorCode, detail: int = 0) -> 'UARTPacket':
        """Create NACK packet."""
        payload = struct.pack('<HBB', seq, int(error_code), detail)
        return UARTPacket(msg_type=MsgType.NACK, seq=0, payload=payload)


# =============================================================================
# DETERMINISTIC COMPLETION (per WSE-STD-020 Section 5)
# =============================================================================

@dataclass
class VoiceLatch:
    """
    Per-voice command latch for deterministic completion per WSE-STD-020 Section 5.

    "keep previous" means voice-local state, not global.
    """
    regime_id: int = 0
    gear_q15: int = 0
    tree_id_a: int = 0
    tree_id_b: int = 0
    tree_blend_q15: int = 0
    gain_q15: int = 0x4000
    traction_q15: int = Q15_ONE
    backbone: List[BackboneEntry] = field(default_factory=list)
    event_mask: int = 0
    event_strength_q15: List[int] = field(default_factory=lambda: [0] * NUM_EVENT_LANES)
    payload_mode: PayloadMode = PayloadMode.COMMON
    payload_common: Payload64 = field(default_factory=Payload64)

    def apply_delta(self, delta: VoiceFrameDelta) -> VoiceFrameV0:
        """
        Apply delta update and return complete frame.

        Per WSE-STD-020 Section 5.1: Omitted fields use "keep previous".
        """
        # Update latch with delta fields
        if delta.field_mask & (1 << FieldMask.REGIME_ID):
            self.regime_id = delta.regime_id
        if delta.field_mask & (1 << FieldMask.GEAR_Q15):
            self.gear_q15 = delta.gear_q15
        if delta.field_mask & (1 << FieldMask.TREE_ID_A):
            self.tree_id_a = delta.tree_id_a
        if delta.field_mask & (1 << FieldMask.TREE_ID_B):
            self.tree_id_b = delta.tree_id_b
        if delta.field_mask & (1 << FieldMask.TREE_BLEND_Q15):
            self.tree_blend_q15 = delta.tree_blend_q15
        if delta.field_mask & (1 << FieldMask.GAIN_Q15):
            self.gain_q15 = delta.gain_q15
        if delta.field_mask & (1 << FieldMask.TRACTION_Q15):
            self.traction_q15 = delta.traction_q15
        if delta.field_mask & (1 << FieldMask.BACKBONE):
            self.backbone = delta.backbone
        if delta.field_mask & (1 << FieldMask.EVENT_MASK):
            self.event_mask = delta.event_mask
        if delta.field_mask & (1 << FieldMask.EVENT_STRENGTH):
            self.event_strength_q15 = delta.event_strength_q15
        if delta.field_mask & (1 << FieldMask.PAYLOAD_MODE):
            self.payload_mode = delta.payload_mode
        if delta.field_mask & (1 << FieldMask.PAYLOAD_COMMON):
            self.payload_common = delta.payload_common

        # tree_id_b defaults to tree_id_a if 0
        tree_b = self.tree_id_b if self.tree_id_b > 0 else self.tree_id_a

        return VoiceFrameV0(
            voice_id=delta.voice_id,
            block_index=delta.block_index,
            regime_id=self.regime_id,
            gear_q15=self.gear_q15,
            tree_id_a=self.tree_id_a,
            tree_id_b=tree_b,
            tree_blend_q15=self.tree_blend_q15,
            gain_q15=self.gain_q15,
            traction_q15=self.traction_q15,
            backbone=list(self.backbone),
            event_mask=self.event_mask,
            event_strength_q15=list(self.event_strength_q15),
            payload_mode=self.payload_mode,
            payload_common=self.payload_common,
        )

    def reset(self, preset_defaults: Optional[Dict] = None):
        """
        Reset latch on voice allocation per WSE-STD-020 Section 5.2.
        """
        defaults = preset_defaults or {}

        self.event_mask = 0
        self.gear_q15 = defaults.get('gear_q15', 0x4000)
        self.traction_q15 = defaults.get('traction_q15', Q15_ONE)
        self.payload_mode = PayloadMode.COMMON
        self.backbone = list(defaults.get('backbone', []))


# =============================================================================
# UART LISTENER FOR CONTROLLER TESTING
# =============================================================================

class UARTListener:
    """
    Python UART listener for controller testing.

    Receives packets from serial port and dispatches to handlers.
    Can be used to test FPGA or hardware controllers.
    """

    def __init__(self, port: str = None, baudrate: int = 921600):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.rx_thread = None
        self.rx_queue: queue.Queue = queue.Queue()

        # Message handlers
        self.handlers: Dict[MsgType, Callable] = {}

        # Per-voice latches for deterministic completion
        self.voice_latches: Dict[int, VoiceLatch] = {}

        # Sequence counter for outgoing packets
        self.tx_seq = 0

        # Statistics
        self.stats = {
            'packets_rx': 0,
            'packets_tx': 0,
            'crc_errors': 0,
            'unknown_msg_types': 0,
        }

    def connect(self):
        """Connect to serial port."""
        try:
            import serial
            self.serial = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            return True
        except Exception as e:
            print(f"Failed to connect to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial:
            self.serial.close()
            self.serial = None

    def start(self):
        """Start receive thread."""
        if self.serial is None:
            raise RuntimeError("Not connected")

        self.running = True
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.rx_thread.start()

    def stop(self):
        """Stop receive thread."""
        self.running = False
        if self.rx_thread:
            self.rx_thread.join(timeout=1.0)
            self.rx_thread = None

    def _rx_loop(self):
        """Receive loop (runs in thread)."""
        buffer = b''

        while self.running:
            try:
                # Read available data
                if self.serial and self.serial.in_waiting:
                    buffer += self.serial.read(self.serial.in_waiting)

                # Try to parse packets
                while len(buffer) >= 12:
                    packet, consumed = UARTPacket.from_bytes(buffer)

                    if packet is None:
                        if consumed > 0:
                            buffer = buffer[consumed:]
                            self.stats['crc_errors'] += 1
                        else:
                            break
                    else:
                        buffer = buffer[consumed:]
                        self.stats['packets_rx'] += 1
                        self.rx_queue.put(packet)

                time.sleep(0.001)

            except Exception as e:
                print(f"RX error: {e}")
                time.sleep(0.1)

    def send_packet(self, packet: UARTPacket):
        """Send packet to serial port."""
        if self.serial:
            packet.seq = self.tx_seq
            self.tx_seq = (self.tx_seq + 1) & 0xFFFF

            data = packet.to_bytes()
            self.serial.write(data)
            self.stats['packets_tx'] += 1

    def send_voice_frame(self, frame: VoiceFrameV0):
        """Send VoiceFrame as VF_ABSOLUTE message."""
        packet = UARTPacket(
            msg_type=MsgType.VF_ABSOLUTE,
            payload=frame.to_bytes()
        )
        self.send_packet(packet)

    def send_voice_delta(self, delta: VoiceFrameDelta):
        """Send VoiceFrame delta update."""
        packet = UARTPacket(
            msg_type=MsgType.VF_DELTA,
            payload=delta.to_bytes()
        )
        self.send_packet(packet)

    def send_param(self, reg_id: int, value: int, absolute: bool = True):
        """Send parameter update."""
        payload = struct.pack('<Hi', reg_id, value)
        packet = UARTPacket(
            msg_type=MsgType.PARAM_ABSOLUTE if absolute else MsgType.PARAM_DELTA,
            payload=payload
        )
        self.send_packet(packet)

    def get_voice_latch(self, voice_id: int) -> VoiceLatch:
        """Get or create voice latch."""
        if voice_id not in self.voice_latches:
            self.voice_latches[voice_id] = VoiceLatch()
        return self.voice_latches[voice_id]

    def process_messages(self, timeout: float = 0.0) -> List[UARTPacket]:
        """
        Process received messages, calling handlers.

        Returns list of processed packets.
        """
        packets = []
        deadline = time.time() + timeout

        while True:
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    packet = self.rx_queue.get_nowait()
                else:
                    packet = self.rx_queue.get(timeout=remaining)

                packets.append(packet)

                # Dispatch to handler
                handler = self.handlers.get(packet.msg_type)
                if handler:
                    handler(packet)
                elif packet.msg_type not in (MsgType.ACK, MsgType.NACK):
                    self.stats['unknown_msg_types'] += 1

            except queue.Empty:
                break

        return packets

    def register_handler(self, msg_type: MsgType, handler: Callable):
        """Register handler for message type."""
        self.handlers[msg_type] = handler

    def wait_for_ack(self, seq: int, timeout: float = 1.0) -> Optional[Tuple[AckStatus, int]]:
        """Wait for ACK to specific sequence number."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            packets = self.process_messages(timeout=0.1)
            for packet in packets:
                if packet.msg_type == MsgType.ACK:
                    ack_seq, status, detail = struct.unpack('<HBB', packet.payload)
                    if ack_seq == seq:
                        return AckStatus(status), detail
                elif packet.msg_type == MsgType.NACK:
                    nack_seq, error, detail = struct.unpack('<HBB', packet.payload)
                    if nack_seq == seq:
                        return None  # NACK received

        return None  # Timeout


# =============================================================================
# MOCK UART FOR TESTING (no hardware required)
# =============================================================================

class MockUART:
    """
    Mock UART interface for testing without hardware.

    Simulates bidirectional communication between "host" and "device".
    """

    def __init__(self):
        self.host_tx: queue.Queue = queue.Queue()
        self.device_tx: queue.Queue = queue.Queue()

        self.host_latches: Dict[int, VoiceLatch] = {}
        self.device_latches: Dict[int, VoiceLatch] = {}

        # Block index for scheduling
        self.current_block: int = 0

        # Queued frames per voice
        self.frame_queue: Dict[int, Dict[int, VoiceFrameV0]] = {}

    def host_send(self, packet: UARTPacket):
        """Host sends packet to device."""
        self.host_tx.put(packet)

    def device_receive(self, timeout: float = 0.1) -> Optional[UARTPacket]:
        """Device receives packet from host."""
        try:
            return self.host_tx.get(timeout=timeout)
        except queue.Empty:
            return None

    def device_send(self, packet: UARTPacket):
        """Device sends packet to host."""
        self.device_tx.put(packet)

    def host_receive(self, timeout: float = 0.1) -> Optional[UARTPacket]:
        """Host receives packet from device."""
        try:
            return self.device_tx.get(timeout=timeout)
        except queue.Empty:
            return None

    def device_process(self) -> Optional[VoiceFrameV0]:
        """
        Process one packet on device side.

        Simulates FPGA behavior: processes VF_ABSOLUTE/VF_DELTA,
        applies deterministic completion, sends ACK.

        Returns the completed VoiceFrame if applicable.
        """
        packet = self.device_receive(timeout=0)
        if packet is None:
            return None

        if packet.msg_type == MsgType.VF_ABSOLUTE:
            frame = VoiceFrameV0.from_bytes(packet.payload)

            # Update latch
            latch = self._get_device_latch(frame.voice_id)
            latch.regime_id = frame.regime_id
            latch.gear_q15 = frame.gear_q15
            latch.tree_id_a = frame.tree_id_a
            latch.tree_id_b = frame.tree_id_b
            latch.tree_blend_q15 = frame.tree_blend_q15
            latch.gain_q15 = frame.gain_q15
            latch.traction_q15 = frame.traction_q15
            latch.backbone = frame.backbone
            latch.event_mask = frame.event_mask
            latch.event_strength_q15 = frame.event_strength_q15
            latch.payload_mode = frame.payload_mode
            latch.payload_common = frame.payload_common

            # Send ACK
            self.device_send(UARTPacket.make_ack(packet.seq))

            # Queue or execute
            if frame.block_index <= self.current_block:
                return frame
            else:
                self._queue_frame(frame)
                return None

        elif packet.msg_type == MsgType.VF_DELTA:
            # Parse delta header
            voice_id, block_index, field_mask, _ = struct.unpack(
                '<HHII', packet.payload[:12]
            )

            # Build delta (simplified - full parsing would be needed)
            delta = VoiceFrameDelta(
                voice_id=voice_id,
                block_index=block_index,
                field_mask=field_mask
            )

            # Apply to latch
            latch = self._get_device_latch(voice_id)
            frame = latch.apply_delta(delta)

            # Send ACK
            self.device_send(UARTPacket.make_ack(packet.seq))

            return frame

        else:
            # Unknown message type
            self.device_send(UARTPacket.make_nack(
                packet.seq, ErrorCode.UNKNOWN_MSG_TYPE
            ))
            return None

    def _get_device_latch(self, voice_id: int) -> VoiceLatch:
        if voice_id not in self.device_latches:
            self.device_latches[voice_id] = VoiceLatch()
        return self.device_latches[voice_id]

    def _queue_frame(self, frame: VoiceFrameV0):
        """Queue frame for future block."""
        if frame.voice_id not in self.frame_queue:
            self.frame_queue[frame.voice_id] = {}
        self.frame_queue[frame.voice_id][frame.block_index] = frame

    def advance_block(self) -> List[VoiceFrameV0]:
        """
        Advance to next block, return due frames.
        """
        self.current_block += 1
        frames = []

        for voice_id, queue in list(self.frame_queue.items()):
            if self.current_block in queue:
                frames.append(queue.pop(self.current_block))

        return frames


# =============================================================================
# PATTERN TABLE (per WSE-STD-020 Section 3.3)
# =============================================================================

@dataclass
class PatternEntry:
    """Pattern table entry."""
    leaf_mask_u32: int = 0xFFFFFFFF  # Bit i selects leaf i
    coeff_mask_default_u8: int = 0xFF  # Default coeff mask


# Default pattern table
DEFAULT_PATTERN_TABLE: Dict[int, PatternEntry] = {
    0: PatternEntry(leaf_mask_u32=0xFFFFFFFF, coeff_mask_default_u8=0xFF),  # All leaves
    1: PatternEntry(leaf_mask_u32=0x000000FF, coeff_mask_default_u8=0xFF),  # Low band (leaves 0-7)
    2: PatternEntry(leaf_mask_u32=0x0000FF00, coeff_mask_default_u8=0xFF),  # Low-mid (8-15)
    3: PatternEntry(leaf_mask_u32=0x00FF0000, coeff_mask_default_u8=0xFF),  # High-mid (16-23)
    4: PatternEntry(leaf_mask_u32=0xFF000000, coeff_mask_default_u8=0xFF),  # High (24-31)
    5: PatternEntry(leaf_mask_u32=0x0000FFFF, coeff_mask_default_u8=0xFF),  # Low half
    6: PatternEntry(leaf_mask_u32=0xFFFF0000, coeff_mask_default_u8=0xFF),  # High half
    7: PatternEntry(leaf_mask_u32=0x55555555, coeff_mask_default_u8=0xFF),  # Odd leaves
    8: PatternEntry(leaf_mask_u32=0xAAAAAAAA, coeff_mask_default_u8=0xFF),  # Even leaves
}


def resolve_pattern(payload: Payload64, pattern_table: Dict[int, PatternEntry] = None) -> Tuple[int, int]:
    """
    Resolve payload to (leaf_mask, coeff_mask).

    Per WSE-STD-020 Section 3.3.
    """
    table = pattern_table or DEFAULT_PATTERN_TABLE

    if payload.target_kind == TargetKind.PATTERN:
        entry = table.get(payload.pattern_id, PatternEntry())
        leaf_mask = entry.leaf_mask_u32
        coeff_mask = payload.coeff_mask if payload.coeff_mask else entry.coeff_mask_default_u8

    elif payload.target_kind == TargetKind.LEAF:
        # Direct leaf selection
        leaf_mask = 1 << payload.target_id
        coeff_mask = payload.coeff_mask if payload.coeff_mask else 0xFF

    elif payload.target_kind == TargetKind.BAND_GROUP:
        # Band groups: 0=low(0-7), 1=mid-low(8-15), 2=mid-high(16-23), 3=high(24-31)
        group = payload.target_id & 0x03
        leaf_mask = 0xFF << (group * 8)
        coeff_mask = payload.coeff_mask if payload.coeff_mask else 0xFF

    elif payload.target_kind == TargetKind.COEFF_MASK:
        # All leaves, specific coeffs
        leaf_mask = 0xFFFFFFFF
        coeff_mask = payload.coeff_mask

    else:
        # Reserved - use defaults
        leaf_mask = 0xFFFFFFFF
        coeff_mask = 0xFF

    return leaf_mask, coeff_mask


# =============================================================================
# VOICEFRAME PACKET V0b - imported from frames.py
# =============================================================================

# Re-export for backwards compatibility
from frames import (
    VoiceFramePacketV0b,
    VoiceFlags,
    VFP_MAGIC,
    VFP_VERSION,
    VFP_PACKET_LEN,
)


# =============================================================================
# CONFORMANCE TEST
# =============================================================================

def run_conformance_test():
    """
    Run WSE-STD-020 conformance tests.
    """
    print("=" * 60)
    print("WSE-STD-020 Conformance Test")
    print("=" * 60)

    # Test 1: CRC-32C
    print("\n1. CRC-32C Implementation")
    test_data = b"123456789"
    expected_crc = 0xE3069283  # Known CRC-32C value
    computed_crc = crc32c(test_data)
    assert computed_crc == expected_crc, f"CRC mismatch: {computed_crc:08X} vs {expected_crc:08X}"
    print(f"   CRC-32C('123456789') = 0x{computed_crc:08X} (correct)")
    print("   PASS")

    # Test 2: Payload64 pack/unpack
    print("\n2. Payload64 Pack/Unpack")
    p = Payload64(
        payload_type=1,
        lane_hint=2,
        pattern_id=3,
        target_kind=TargetKind.LEAF,
        target_id=15,
        coeff_mask=0xAA,
        variant0=0x55,
        variant1=0xFF
    )
    packed = p.pack()
    unpacked = Payload64.unpack(packed)
    assert unpacked.payload_type == p.payload_type
    assert unpacked.target_kind == p.target_kind
    assert unpacked.target_id == p.target_id
    assert unpacked.coeff_mask == p.coeff_mask
    print(f"   Packed: 0x{packed:016X}")
    print(f"   Unpacked: target_kind={unpacked.target_kind.name}, target_id={unpacked.target_id}")
    print("   PASS")

    # Test 3: BackboneEntry serialization
    print("\n3. BackboneEntry Serialization")
    entry = BackboneEntry(op_id=5, strength_q15=0x6000, param0_q15=0x1000, param1_q15=0x2000)
    entry_bytes = entry.to_bytes()
    restored = BackboneEntry.from_bytes(entry_bytes)
    assert restored.op_id == entry.op_id
    assert restored.strength_q15 == entry.strength_q15
    print(f"   Entry: op={entry.op_id}, strength={entry.strength_q15}")
    print(f"   Bytes: {entry_bytes.hex()}")
    print("   PASS")

    # Test 4: VoiceFrameV0 serialization
    print("\n4. VoiceFrameV0 Serialization")
    frame = VoiceFrameV0(
        voice_id=5,
        block_index=100,
        regime_id=2,
        gear_q15=0x6000,
        tree_id_a=1,
        tree_id_b=3,
        tree_blend_q15=0x4000,
        gain_q15=0x5000,
        traction_q15=0x7FFF,
        backbone=[
            BackboneEntry(op_id=1, strength_q15=0x4000),
            BackboneEntry(op_id=5, strength_q15=0x6000),
        ],
        event_mask=0x05,
        event_strength_q15=[0x3000, 0, 0x5000, 0, 0, 0, 0, 0],
        payload_mode=PayloadMode.COMMON,
        payload_common=Payload64.for_leaf(15, variant0=1),
    )
    frame_bytes = frame.to_bytes()
    restored_frame = VoiceFrameV0.from_bytes(frame_bytes)
    assert restored_frame.voice_id == frame.voice_id
    assert restored_frame.block_index == frame.block_index
    assert len(restored_frame.backbone) == len(frame.backbone)
    assert restored_frame.event_mask == frame.event_mask
    print(f"   Frame size: {len(frame_bytes)} bytes")
    print(f"   Voice: {frame.voice_id}, Block: {frame.block_index}")
    print(f"   Backbone: {len(frame.backbone)} entries")
    print("   PASS")

    # Test 5: UART packet framing
    print("\n5. UART Packet Framing")
    packet = UARTPacket(
        msg_type=MsgType.VF_ABSOLUTE,
        seq=42,
        payload=frame_bytes
    )
    packet_bytes = packet.to_bytes()

    # Check sync word
    sync = struct.unpack('<H', packet_bytes[:2])[0]
    assert sync == SYNC_WORD, f"Sync mismatch: 0x{sync:04X}"

    # Parse back
    parsed, consumed = UARTPacket.from_bytes(packet_bytes)
    assert parsed is not None
    assert parsed.msg_type == MsgType.VF_ABSOLUTE
    assert parsed.seq == 42
    assert parsed.payload == frame_bytes
    print(f"   Packet size: {len(packet_bytes)} bytes")
    print(f"   Sync: 0x{sync:04X}")
    print(f"   Seq: {parsed.seq}, Type: {parsed.msg_type.name}")
    print("   PASS")

    # Test 6: Deterministic completion
    print("\n6. Deterministic Completion (Section 5)")
    latch = VoiceLatch()
    latch.regime_id = 2
    latch.tree_id_a = 1
    latch.gain_q15 = 0x6000

    delta = VoiceFrameDelta(
        voice_id=5,
        block_index=101,
        field_mask=(1 << FieldMask.EVENT_MASK),  # Only event_mask changed
        event_mask=0x03
    )

    completed = latch.apply_delta(delta)
    assert completed.regime_id == 2  # Kept from latch
    assert completed.tree_id_a == 1  # Kept from latch
    assert completed.gain_q15 == 0x6000  # Kept from latch
    assert completed.event_mask == 0x03  # Updated from delta
    print(f"   Delta field_mask: {delta.field_mask:04X}")
    print(f"   Completed: regime={completed.regime_id}, tree_a={completed.tree_id_a}")
    print(f"   Event mask updated: 0x{completed.event_mask:02X}")
    print("   PASS")

    # Test 7: Mock UART round-trip
    print("\n7. Mock UART Round-Trip")
    mock = MockUART()

    # Host sends frame
    tx_packet = UARTPacket(
        msg_type=MsgType.VF_ABSOLUTE,
        seq=1,
        payload=frame_bytes
    )
    mock.host_send(tx_packet)

    # Device processes (use longer timeout to ensure queue is read)
    result_frame = mock.device_process()
    # Frame may be None if block_index > current_block (queued for later)
    # That's OK - we test that ACK is sent

    # Host receives ACK
    ack = mock.host_receive(timeout=1.0)
    assert ack is not None, "Expected ACK from mock device"
    assert ack.msg_type == MsgType.ACK, f"Expected ACK, got {ack.msg_type}"
    print(f"   Sent VF_ABSOLUTE for voice {frame.voice_id}")
    print(f"   Device processed: frame={'queued' if result_frame is None else 'executed'}")
    print(f"   Received ACK")
    print("   PASS")

    # Test 8: Pattern resolution
    print("\n8. Pattern Table Resolution (Section 3.3)")
    p_all = Payload64.for_pattern(0)  # All leaves
    leaf_mask, coeff_mask = resolve_pattern(p_all)
    assert leaf_mask == 0xFFFFFFFF
    print(f"   Pattern 0 (all): leaf_mask=0x{leaf_mask:08X}")

    p_low = Payload64.for_pattern(1)  # Low band
    leaf_mask, coeff_mask = resolve_pattern(p_low)
    assert leaf_mask == 0x000000FF
    print(f"   Pattern 1 (low): leaf_mask=0x{leaf_mask:08X}")

    p_leaf = Payload64.for_leaf(15)  # Single leaf
    leaf_mask, coeff_mask = resolve_pattern(p_leaf)
    assert leaf_mask == (1 << 15)
    print(f"   Leaf 15: leaf_mask=0x{leaf_mask:08X}")
    print("   PASS")

    # Test 9: Event lanes
    print("\n9. Event Lanes (Section 2.3)")
    assert EventLane.EVT_OP_FIRE == 0
    assert EventLane.EVT_INJECT == 2
    assert EventLane.EVT_DEBUG == 7
    print(f"   EVT_OP_FIRE = {EventLane.EVT_OP_FIRE}")
    print(f"   EVT_INJECT = {EventLane.EVT_INJECT}")
    print(f"   EVT_DEBUG = {EventLane.EVT_DEBUG}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("All WSE-STD-020 conformance tests PASSED")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_conformance_test()
