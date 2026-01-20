"""
WSE-STD-100 Dual-UART Link Protocol Implementation
===================================================

Implements the Dual-UART Link Spec v1.0 for separating control (reliable, low
bandwidth) from data (high bandwidth, streaming) traffic.

This module provides:
- COBS encoding/decoding for robust byte-stuffing
- Frame header with magic, version, channel, message type, seq, length, CRC-32
- Control UART (C-UART): reliable command/event plane with ACK/retransmit
- Data UART (D-UART): high-throughput content plane with credit-based flow control
- Per-voice VOICE_DELTA and VOICE_BURST messages for sparse leaf updates

Conformance: WSE-STD-100 v1.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Union
from enum import IntEnum
import struct
import threading
import queue
import time

from ..fixed import Q15_ONE


# =============================================================================
# CONSTANTS (per WSE-STD-100)
# =============================================================================

# Import from primitives (single source of truth)
from ..primitives import BLOCK_SIZE as BLOCK_SAMPLES
FRAME_MAGIC = 0xDEC0  # 0xC0 0xDE in little-endian
PROTOCOL_VERSION = 0x01
COBS_DELIMITER = 0x00

# Channel identifiers
CHANNEL_CONTROL = 0
CHANNEL_DATA = 1

# Default parameters (Section 12)
DEFAULT_LEAD_BLOCKS = 8
DEFAULT_ACK_TIMEOUT_MS = 30
DEFAULT_HOLD_BLOCKS = 4
DEFAULT_DECAY_BLOCKS = 32
DEFAULT_MAX_RETRIES = 3

# Limits
MAX_PAYLOAD_SIZE = 512
MAX_VOICE_ID = 127
MAX_LEAF_ENTRIES = 64


# =============================================================================
# COBS ENCODING/DECODING (per WSE-STD-100 Section 4.1)
# =============================================================================

def cobs_encode(data: bytes) -> bytes:
    """
    Encode data using Consistent Overhead Byte Stuffing (COBS).

    COBS eliminates all zero bytes from the data, allowing 0x00 to be used
    as an unambiguous frame delimiter.

    Returns encoded data WITHOUT the trailing delimiter.
    """
    if not data:
        return b'\x01'

    output = bytearray()
    code_idx = 0
    output.append(0)  # Placeholder for first code byte
    code = 1

    for byte in data:
        if byte == 0:
            output[code_idx] = code
            code_idx = len(output)
            output.append(0)  # Placeholder for next code byte
            code = 1
        else:
            output.append(byte)
            code += 1
            if code == 0xFF:
                output[code_idx] = code
                code_idx = len(output)
                output.append(0)  # Placeholder for next code byte
                code = 1

    output[code_idx] = code
    return bytes(output)


def cobs_decode(data: bytes) -> Optional[bytes]:
    """
    Decode COBS-encoded data.

    Returns decoded data, or None if the encoding is invalid.
    """
    if not data:
        return None

    output = bytearray()
    idx = 0

    while idx < len(data):
        code = data[idx]
        if code == 0:
            return None  # Invalid: zero in COBS-encoded data

        idx += 1

        for _ in range(code - 1):
            if idx >= len(data):
                return None  # Unexpected end
            output.append(data[idx])
            idx += 1

        # Add zero byte between groups, but not after the last group
        if code < 0xFF and idx < len(data):
            output.append(0)

    return bytes(output)


# =============================================================================
# CRC-32 IMPLEMENTATION (same as WSE-STD-020)
# =============================================================================

CRC32C_POLY = 0x82F63B78


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

    Per WSE-STD-100 Section 4.2: CRC-32 of header (with crc32=0) + payload.
    """
    crc = initial
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


# =============================================================================
# CONTROL UART MESSAGE TYPES (per WSE-STD-100 Section 5 & 13.1)
# =============================================================================

class ControlMsgType(IntEnum):
    """Control UART message types per WSE-STD-100 Section 13.1."""
    HELLO = 0x01
    HELLO_ACK = 0x02
    GET_STATUS = 0x03
    STATUS = 0x04
    SET_INSTRUMENT = 0x10
    SET_TREE_TARGET = 0x11
    SET_MACRO = 0x12
    NOTE_EVENT = 0x20
    VOICE_POLICY = 0x21
    STREAM_CTRL = 0x30
    ACK = 0xF0
    NACK = 0xF1


# =============================================================================
# DATA UART MESSAGE TYPES (per WSE-STD-100 Section 6 & 13.2)
# =============================================================================

class DataMsgType(IntEnum):
    """Data UART message types per WSE-STD-100 Section 13.2."""
    DATA_STREAM_START = 0x01
    DATA_STREAM_STOP = 0x02
    VOICE_DELTA = 0x10
    VOICE_BURST = 0x11
    DATA_CREDIT = 0x20
    DATA_DROP_NOTICE = 0x21
    DATA_HEARTBEAT = 0xF0


# =============================================================================
# ERROR CODES (per WSE-STD-100 Section 11)
# =============================================================================

class ErrorCode(IntEnum):
    """NACK error codes per WSE-STD-100 Section 11."""
    BAD_CRC = 0x0001
    BAD_LENGTH = 0x0002
    UNKNOWN_MSG_TYPE = 0x0003
    SEQ_OUT_OF_WINDOW = 0x0004
    INVALID_VOICE_ID = 0x0005
    APPLY_BLOCK_TOO_FAR = 0x0006
    DUART_OVERFLOW = 0x0007


# =============================================================================
# VOICE_DELTA SUBTYPES (per WSE-STD-100 Section 7.1)
# =============================================================================

class VoiceDeltaSubtype(IntEnum):
    """VOICE_DELTA subtypes per WSE-STD-100 Section 7.1."""
    LEAF_GAINS = 0
    LEAF_COEFFS = 1
    EVENTS = 2
    MIXED = 3


# =============================================================================
# VOICE_BURST MODES (per WSE-STD-100 Section 7.2)
# =============================================================================

class VoiceBurstMode(IntEnum):
    """VOICE_BURST modes per WSE-STD-100 Section 7.2."""
    DENSE64 = 0
    RLE = 1
    TOP_K = 2


# =============================================================================
# NOTE_EVENT FLAGS (per WSE-STD-100 Section 5.3.1)
# =============================================================================

class NoteEventFlags(IntEnum):
    """NOTE_EVENT flags per WSE-STD-100 Section 5.3.1."""
    NONE = 0x00
    LEGATO = 0x01
    RETRIG = 0x02


# =============================================================================
# TREE MORPH CURVES (per WSE-STD-100 Section 5.3.2)
# =============================================================================

class TreeMorphCurve(IntEnum):
    """Tree morph curves per WSE-STD-100 Section 5.3.2."""
    EXPONENTIAL = 0
    MIN_JERK = 1


# =============================================================================
# FRAME HEADER (per WSE-STD-100 Section 4.2)
# =============================================================================

FRAME_HEADER_SIZE = 14  # Fixed header size in bytes


@dataclass
class FrameHeader:
    """
    Frame header per WSE-STD-100 Section 4.2.

    All multi-byte integers are little-endian.
    """
    magic: int = FRAME_MAGIC          # 2 bytes: 0xC0 0xDE
    version: int = PROTOCOL_VERSION   # 1 byte
    chan: int = CHANNEL_CONTROL       # 1 byte: 0=control, 1=data
    msg_type: int = 0                 # 1 byte: enumerated
    flags: int = 0                    # 1 byte: bitfield
    seq: int = 0                      # 2 bytes: wraps, per-channel
    length: int = 0                   # 2 bytes: payload length
    crc32: int = 0                    # 4 bytes: CRC-32 of header+payload

    def to_bytes_without_crc(self) -> bytes:
        """Serialize header without CRC (for CRC calculation)."""
        return struct.pack('<HBBBBHHH',
            self.magic,
            self.version,
            self.chan,
            self.msg_type,
            self.flags,
            self.seq,
            self.length,
            0  # Padding for alignment (2 bytes reserved in CRC position)
        )[:10]  # First 10 bytes only

    def to_bytes(self) -> bytes:
        """Serialize header with CRC."""
        return struct.pack('<HBBBBHHI',
            self.magic,
            self.version,
            self.chan,
            self.msg_type,
            self.flags,
            self.seq,
            self.length,
            self.crc32
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional['FrameHeader']:
        """Deserialize header from bytes."""
        if len(data) < FRAME_HEADER_SIZE:
            return None

        magic, version, chan, msg_type, flags, seq, length, crc32 = struct.unpack(
            '<HBBBBHHI', data[:FRAME_HEADER_SIZE]
        )

        return cls(
            magic=magic,
            version=version,
            chan=chan,
            msg_type=msg_type,
            flags=flags,
            seq=seq,
            length=length,
            crc32=crc32
        )


# =============================================================================
# DUAL UART PACKET (per WSE-STD-100 Section 4)
# =============================================================================

@dataclass
class DualUARTPacket:
    """
    Dual UART packet per WSE-STD-100 Section 4.

    Wraps a frame header and payload with COBS encoding.
    """
    chan: int = CHANNEL_CONTROL
    msg_type: int = 0
    flags: int = 0
    seq: int = 0
    payload: bytes = b''

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes with COBS encoding.

        Returns COBS-encoded frame + 0x00 delimiter.
        """
        # Build header
        header = FrameHeader(
            magic=FRAME_MAGIC,
            version=PROTOCOL_VERSION,
            chan=self.chan,
            msg_type=self.msg_type,
            flags=self.flags,
            seq=self.seq,
            length=len(self.payload),
            crc32=0  # Will be computed
        )

        # Compute CRC over header (with crc32=0) + payload
        header_bytes = header.to_bytes()[:10]  # First 10 bytes (before CRC field)
        crc_data = header_bytes + self.payload
        header.crc32 = crc32c(crc_data)

        # Full frame
        frame = header.to_bytes() + self.payload

        # COBS encode + delimiter
        return cobs_encode(frame) + bytes([COBS_DELIMITER])

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple[Optional['DualUARTPacket'], int]:
        """
        Deserialize from COBS-encoded bytes.

        Returns: (packet or None if invalid, bytes consumed)
        """
        # Find COBS delimiter
        delim_idx = -1
        for i, b in enumerate(data):
            if b == COBS_DELIMITER:
                delim_idx = i
                break

        if delim_idx < 0:
            return None, 0

        # COBS decode
        cobs_data = data[:delim_idx]
        if not cobs_data:
            return None, delim_idx + 1

        decoded = cobs_decode(cobs_data)
        if decoded is None or len(decoded) < FRAME_HEADER_SIZE:
            return None, delim_idx + 1

        # Parse header
        header = FrameHeader.from_bytes(decoded)
        if header is None:
            return None, delim_idx + 1

        # Validate magic
        if header.magic != FRAME_MAGIC:
            return None, delim_idx + 1

        # Check length
        expected_len = FRAME_HEADER_SIZE + header.length
        if len(decoded) < expected_len:
            return None, delim_idx + 1

        # Extract payload
        payload = decoded[FRAME_HEADER_SIZE:FRAME_HEADER_SIZE + header.length]

        # Verify CRC
        header_bytes = decoded[:10]  # First 10 bytes (before CRC field)
        crc_data = header_bytes + payload
        crc_computed = crc32c(crc_data)

        if crc_computed != header.crc32:
            return None, delim_idx + 1

        return cls(
            chan=header.chan,
            msg_type=header.msg_type,
            flags=header.flags,
            seq=header.seq,
            payload=payload
        ), delim_idx + 1


# =============================================================================
# CONTROL MESSAGE PAYLOADS (per WSE-STD-100 Section 5.3)
# =============================================================================

@dataclass
class NoteEvent:
    """
    NOTE_EVENT payload per WSE-STD-100 Section 5.3.1.

    Total: 9 bytes
    """
    apply_block_id: int = 0    # 4 bytes: when it takes effect
    note: int = 0              # 1 byte: 0-127
    vel: int = 0               # 1 byte: 0-127
    on: bool = True            # 1 byte: 1=on, 0=off
    chan: int = 0              # 1 byte: MIDI channel
    flags: int = 0             # 1 byte: NoteEventFlags

    def to_bytes(self) -> bytes:
        return struct.pack('<IBBBBB',
            self.apply_block_id,
            self.note & 0x7F,
            self.vel & 0x7F,
            1 if self.on else 0,
            self.chan & 0x0F,
            self.flags
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'NoteEvent':
        apply_block_id, note, vel, on, chan, flags = struct.unpack('<IBBBBB', data[:9])
        return cls(
            apply_block_id=apply_block_id,
            note=note,
            vel=vel,
            on=bool(on),
            chan=chan,
            flags=flags
        )

    @staticmethod
    def byte_size() -> int:
        return 9


@dataclass
class SetTreeTarget:
    """
    SET_TREE_TARGET payload per WSE-STD-100 Section 5.3.2.

    Total: 8 bytes
    """
    apply_block_id: int = 0    # 4 bytes
    tree_index: int = 0        # 1 byte: 0..2
    tau_ms: int = 1000         # 2 bytes: morph time constant (1000..5000 typical)
    curve: int = 0             # 1 byte: TreeMorphCurve

    def to_bytes(self) -> bytes:
        return struct.pack('<IBHB',
            self.apply_block_id,
            self.tree_index & 0x03,
            self.tau_ms,
            self.curve
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SetTreeTarget':
        apply_block_id, tree_index, tau_ms, curve = struct.unpack('<IBHB', data[:8])
        return cls(
            apply_block_id=apply_block_id,
            tree_index=tree_index,
            tau_ms=tau_ms,
            curve=curve
        )

    @staticmethod
    def byte_size() -> int:
        return 8


@dataclass
class Status:
    """
    STATUS payload per WSE-STD-100 Section 5.3.3.

    Total: 22 bytes
    """
    block_id: int = 0              # 4 bytes: current block
    data_credit_bytes: int = 0     # 4 bytes: D-UART credit
    active_voices: int = 0         # 2 bytes: currently active
    main_voices: int = 0           # 2 bytes: main voice count
    underrun_count: int = 0        # 4 bytes: cumulative
    late_apply_count: int = 0      # 4 bytes: cumulative
    last_error_code: int = 0       # 2 bytes: most recent error

    def to_bytes(self) -> bytes:
        return struct.pack('<IIHHIIH',
            self.block_id,
            self.data_credit_bytes,
            self.active_voices,
            self.main_voices,
            self.underrun_count,
            self.late_apply_count,
            self.last_error_code
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Status':
        (block_id, data_credit_bytes, active_voices, main_voices,
         underrun_count, late_apply_count, last_error_code) = struct.unpack('<IIHHIIH', data[:22])
        return cls(
            block_id=block_id,
            data_credit_bytes=data_credit_bytes,
            active_voices=active_voices,
            main_voices=main_voices,
            underrun_count=underrun_count,
            late_apply_count=late_apply_count,
            last_error_code=last_error_code
        )

    @staticmethod
    def byte_size() -> int:
        return 22


@dataclass
class HelloAck:
    """
    HELLO_ACK payload per WSE-STD-100 Section 5.2.

    Returns build IDs, buffer sizes, current block_id.
    """
    block_id: int = 0              # 4 bytes: current block
    credit_bytes: int = 0          # 4 bytes: initial D-UART credit
    build_id: int = 0              # 4 bytes: firmware build ID
    buffer_size: int = 0           # 2 bytes: D-UART buffer size
    max_voices: int = 128          # 2 bytes: max supported voices
    features: int = 0              # 4 bytes: feature flags

    def to_bytes(self) -> bytes:
        return struct.pack('<IIIHHI',
            self.block_id,
            self.credit_bytes,
            self.build_id,
            self.buffer_size,
            self.max_voices,
            self.features
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'HelloAck':
        block_id, credit_bytes, build_id, buffer_size, max_voices, features = struct.unpack(
            '<IIIHHI', data[:20]
        )
        return cls(
            block_id=block_id,
            credit_bytes=credit_bytes,
            build_id=build_id,
            buffer_size=buffer_size,
            max_voices=max_voices,
            features=features
        )

    @staticmethod
    def byte_size() -> int:
        return 20


@dataclass
class SetMacro:
    """
    SET_MACRO payload per WSE-STD-100 Section 5.2.

    Set macro knob value with apply_block_id.
    """
    apply_block_id: int = 0    # 4 bytes
    macro_id: int = 0          # 1 byte: macro index
    value_q15: int = 0         # 2 bytes: 0..32767 (0..1 range)
    reserved: int = 0          # 1 byte

    def to_bytes(self) -> bytes:
        return struct.pack('<IBHB',
            self.apply_block_id,
            self.macro_id,
            self.value_q15,
            self.reserved
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SetMacro':
        apply_block_id, macro_id, value_q15, reserved = struct.unpack('<IBHB', data[:8])
        return cls(
            apply_block_id=apply_block_id,
            macro_id=macro_id,
            value_q15=value_q15,
            reserved=reserved
        )

    @staticmethod
    def byte_size() -> int:
        return 8


@dataclass
class StreamCtrl:
    """
    STREAM_CTRL payload per WSE-STD-100 Section 5.2.

    Start/stop data streaming.
    """
    start: bool = True         # 1 byte: 1=start, 0=stop
    stream_mode: int = 0       # 1 byte: stream mode flags
    reserved: int = 0          # 2 bytes

    def to_bytes(self) -> bytes:
        return struct.pack('<BBH',
            1 if self.start else 0,
            self.stream_mode,
            self.reserved
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'StreamCtrl':
        start, stream_mode, reserved = struct.unpack('<BBH', data[:4])
        return cls(start=bool(start), stream_mode=stream_mode, reserved=reserved)

    @staticmethod
    def byte_size() -> int:
        return 4


# =============================================================================
# DATA MESSAGE PAYLOADS (per WSE-STD-100 Section 6 & 7)
# =============================================================================

@dataclass
class LeafGainEntry:
    """
    Leaf gain entry per WSE-STD-100 Section 7.1.

    Total: 3 bytes
    """
    leaf_idx: int = 0          # 1 byte: 0..63
    gain_q15: int = 0          # 2 bytes: signed Q1.15

    def to_bytes(self) -> bytes:
        return struct.pack('<Bh', self.leaf_idx, self.gain_q15)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LeafGainEntry':
        leaf_idx, gain_q15 = struct.unpack('<Bh', data[:3])
        return cls(leaf_idx=leaf_idx, gain_q15=gain_q15)

    @staticmethod
    def byte_size() -> int:
        return 3


@dataclass
class LeafCoeffEntry:
    """
    Leaf coefficient entry per WSE-STD-100 Section 7.1.

    Total: 3 bytes
    """
    leaf_idx: int = 0          # 1 byte: 0..63
    coeff_q15: int = 0         # 2 bytes: signed Q1.15

    def to_bytes(self) -> bytes:
        return struct.pack('<Bh', self.leaf_idx, self.coeff_q15)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'LeafCoeffEntry':
        leaf_idx, coeff_q15 = struct.unpack('<Bh', data[:3])
        return cls(leaf_idx=leaf_idx, coeff_q15=coeff_q15)

    @staticmethod
    def byte_size() -> int:
        return 3


@dataclass
class EventEntry:
    """
    Event entry per WSE-STD-100 Section 7.1.

    Total: 5 bytes
    """
    event_id: int = 0          # 1 byte
    mag_q15: int = 0           # 2 bytes: signed Q1.15
    dur_blocks: int = 0        # 1 byte
    flags: int = 0             # 1 byte

    def to_bytes(self) -> bytes:
        return struct.pack('<BhBB', self.event_id, self.mag_q15, self.dur_blocks, self.flags)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'EventEntry':
        event_id, mag_q15, dur_blocks, flags = struct.unpack('<BhBB', data[:5])
        return cls(event_id=event_id, mag_q15=mag_q15, dur_blocks=dur_blocks, flags=flags)

    @staticmethod
    def byte_size() -> int:
        return 5


@dataclass
class VoiceDelta:
    """
    VOICE_DELTA payload per WSE-STD-100 Section 7.1.

    Sparse per-voice updates (the workhorse message).
    """
    apply_block_id: int = 0    # 4 bytes: block to apply
    voice_id: int = 0          # 1 byte: 0..127
    subtype: int = 0           # 1 byte: VoiceDeltaSubtype
    count: int = 0             # 1 byte: number of entries
    reserved: int = 0          # 1 byte

    # Variable entries (depends on subtype)
    leaf_gains: List[LeafGainEntry] = field(default_factory=list)
    leaf_coeffs: List[LeafCoeffEntry] = field(default_factory=list)
    events: List[EventEntry] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        # Header
        data = struct.pack('<IBBBB',
            self.apply_block_id,
            self.voice_id,
            self.subtype,
            len(self._get_entries()),
            self.reserved
        )

        # Entries
        for entry in self._get_entries():
            data += entry.to_bytes()

        return data

    def _get_entries(self) -> List:
        """Get entries based on subtype."""
        if self.subtype == VoiceDeltaSubtype.LEAF_GAINS:
            return self.leaf_gains
        elif self.subtype == VoiceDeltaSubtype.LEAF_COEFFS:
            return self.leaf_coeffs
        elif self.subtype == VoiceDeltaSubtype.EVENTS:
            return self.events
        else:  # MIXED
            return self.leaf_gains + self.leaf_coeffs + self.events

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VoiceDelta':
        apply_block_id, voice_id, subtype, count, reserved = struct.unpack('<IBBBB', data[:8])

        delta = cls(
            apply_block_id=apply_block_id,
            voice_id=voice_id,
            subtype=subtype,
            count=count,
            reserved=reserved
        )

        offset = 8
        if subtype == VoiceDeltaSubtype.LEAF_GAINS:
            for _ in range(count):
                entry = LeafGainEntry.from_bytes(data[offset:])
                delta.leaf_gains.append(entry)
                offset += LeafGainEntry.byte_size()
        elif subtype == VoiceDeltaSubtype.LEAF_COEFFS:
            for _ in range(count):
                entry = LeafCoeffEntry.from_bytes(data[offset:])
                delta.leaf_coeffs.append(entry)
                offset += LeafCoeffEntry.byte_size()
        elif subtype == VoiceDeltaSubtype.EVENTS:
            for _ in range(count):
                entry = EventEntry.from_bytes(data[offset:])
                delta.events.append(entry)
                offset += EventEntry.byte_size()

        return delta

    @staticmethod
    def header_size() -> int:
        return 8


@dataclass
class VoiceBurst:
    """
    VOICE_BURST payload per WSE-STD-100 Section 7.2.

    Dense payload for many leaves at once (rare case).
    """
    apply_block_id: int = 0    # 4 bytes
    voice_id: int = 0          # 1 byte
    mode: int = 0              # 1 byte: VoiceBurstMode
    payload_len: int = 0       # 2 bytes
    payload: bytes = b''       # Variable

    def to_bytes(self) -> bytes:
        return struct.pack('<IBBH',
            self.apply_block_id,
            self.voice_id,
            self.mode,
            len(self.payload)
        ) + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> 'VoiceBurst':
        apply_block_id, voice_id, mode, payload_len = struct.unpack('<IBBH', data[:8])
        payload = data[8:8 + payload_len]
        return cls(
            apply_block_id=apply_block_id,
            voice_id=voice_id,
            mode=mode,
            payload_len=payload_len,
            payload=payload
        )

    @staticmethod
    def header_size() -> int:
        return 8

    @classmethod
    def dense64(cls, apply_block_id: int, voice_id: int, gains: List[int]) -> 'VoiceBurst':
        """Create Dense64 burst with 64 x int16 gains."""
        payload = struct.pack('<64h', *gains[:64])
        return cls(
            apply_block_id=apply_block_id,
            voice_id=voice_id,
            mode=VoiceBurstMode.DENSE64,
            payload_len=len(payload),
            payload=payload
        )


@dataclass
class DataCredit:
    """
    DATA_CREDIT payload per WSE-STD-100 Section 6.3.

    Credit-based flow control message from FPGA to Host.
    """
    block_id: int = 0          # 4 bytes: current
    credit_bytes: int = 0      # 4 bytes: bytes host may send
    rx_fill_bytes: int = 0     # 4 bytes: optional

    def to_bytes(self) -> bytes:
        return struct.pack('<III',
            self.block_id,
            self.credit_bytes,
            self.rx_fill_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DataCredit':
        block_id, credit_bytes, rx_fill_bytes = struct.unpack('<III', data[:12])
        return cls(
            block_id=block_id,
            credit_bytes=credit_bytes,
            rx_fill_bytes=rx_fill_bytes
        )

    @staticmethod
    def byte_size() -> int:
        return 12


@dataclass
class DataStreamStart:
    """
    DATA_STREAM_START payload per WSE-STD-100 Section 6.2.
    """
    base_block_id: int = 0     # 4 bytes

    def to_bytes(self) -> bytes:
        return struct.pack('<I', self.base_block_id)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DataStreamStart':
        base_block_id = struct.unpack('<I', data[:4])[0]
        return cls(base_block_id=base_block_id)

    @staticmethod
    def byte_size() -> int:
        return 4


# =============================================================================
# PER-VOICE STATE (per WSE-STD-100 Section 8)
# =============================================================================

@dataclass
class VoiceState:
    """
    Per-voice state for receiver-side application rules.

    Per WSE-STD-100 Section 8: tracks last values for hold/decay on loss.
    """
    voice_id: int = 0
    active: bool = False
    last_block_id: int = 0
    hold_counter: int = 0

    # Last leaf values (for hold/decay)
    leaf_gains: Dict[int, int] = field(default_factory=dict)
    leaf_coeffs: Dict[int, int] = field(default_factory=dict)

    def update(self, delta: VoiceDelta):
        """Apply delta update to voice state."""
        self.last_block_id = delta.apply_block_id
        self.hold_counter = 0

        for entry in delta.leaf_gains:
            self.leaf_gains[entry.leaf_idx] = entry.gain_q15

        for entry in delta.leaf_coeffs:
            self.leaf_coeffs[entry.leaf_idx] = entry.coeff_q15

    def tick_block(self, current_block_id: int, hold_blocks: int, decay_blocks: int):
        """
        Process block tick for loss handling.

        Per WSE-STD-100 Section 8.2:
        - Hold last values for HOLD_BLOCKS
        - Then decay toward zero over DECAY_BLOCKS
        """
        if not self.active:
            return

        blocks_since_update = current_block_id - self.last_block_id

        if blocks_since_update > hold_blocks:
            self.hold_counter += 1

            if self.hold_counter <= decay_blocks:
                # Decay factor (linear)
                decay = 1.0 - (self.hold_counter / decay_blocks)
                for idx in self.leaf_gains:
                    self.leaf_gains[idx] = int(self.leaf_gains[idx] * decay)
                for idx in self.leaf_coeffs:
                    self.leaf_coeffs[idx] = int(self.leaf_coeffs[idx] * decay)
            else:
                # Fully decayed - clear
                self.leaf_gains.clear()
                self.leaf_coeffs.clear()


# =============================================================================
# PENDING MESSAGE FOR RETRANSMIT (C-UART reliability)
# =============================================================================

@dataclass
class PendingMessage:
    """Pending message awaiting ACK for C-UART reliability."""
    seq: int
    packet: DualUARTPacket
    send_time: float
    retry_count: int = 0


# =============================================================================
# DUAL UART LINK (per WSE-STD-100 Sections 1-12)
# =============================================================================

class DualUARTLink:
    """
    Dual UART Link per WSE-STD-100.

    Manages both C-UART (control, reliable) and D-UART (data, streaming).
    """

    def __init__(
        self,
        c_uart_port: str = None,
        d_uart_port: str = None,
        c_uart_baud: int = 1000000,
        d_uart_baud: int = 3000000,
        lead_blocks: int = DEFAULT_LEAD_BLOCKS,
        ack_timeout_ms: int = DEFAULT_ACK_TIMEOUT_MS,
        hold_blocks: int = DEFAULT_HOLD_BLOCKS,
        decay_blocks: int = DEFAULT_DECAY_BLOCKS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        # Port configuration
        self.c_uart_port = c_uart_port
        self.d_uart_port = d_uart_port
        self.c_uart_baud = c_uart_baud
        self.d_uart_baud = d_uart_baud

        # Serial connections (None until connected)
        self.c_uart = None
        self.d_uart = None

        # Parameters
        self.lead_blocks = lead_blocks
        self.ack_timeout_s = ack_timeout_ms / 1000.0
        self.hold_blocks = hold_blocks
        self.decay_blocks = decay_blocks
        self.max_retries = max_retries

        # State
        self.running = False
        self.current_block_id = 0
        self.data_credit_bytes = 0
        self.streaming = False

        # Per-channel sequence counters
        self.c_uart_tx_seq = 0
        self.d_uart_tx_seq = 0

        # Pending ACKs for C-UART (seq -> PendingMessage)
        self.pending_acks: Dict[int, PendingMessage] = {}
        self.pending_lock = threading.Lock()

        # Per-voice state
        self.voice_states: Dict[int, VoiceState] = {}

        # RX queues
        self.c_uart_rx_queue: queue.Queue = queue.Queue()
        self.d_uart_rx_queue: queue.Queue = queue.Queue()

        # Message handlers
        self.control_handlers: Dict[int, Callable] = {}
        self.data_handlers: Dict[int, Callable] = {}

        # Threads
        self.c_uart_rx_thread = None
        self.d_uart_rx_thread = None
        self.retransmit_thread = None

        # Statistics
        self.stats = {
            'c_uart_tx': 0,
            'c_uart_rx': 0,
            'c_uart_retransmits': 0,
            'c_uart_timeouts': 0,
            'd_uart_tx': 0,
            'd_uart_rx': 0,
            'd_uart_drops': 0,
            'crc_errors': 0,
        }

    def connect(self) -> bool:
        """Connect to both UARTs."""
        try:
            import serial

            if self.c_uart_port:
                self.c_uart = serial.Serial(
                    self.c_uart_port,
                    self.c_uart_baud,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.001  # Reduced from 0.1 for lower latency
                )

            if self.d_uart_port:
                self.d_uart = serial.Serial(
                    self.d_uart_port,
                    self.d_uart_baud,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.001,  # Reduced from 0.1 for lower latency
                    rtscts=True  # RTS/CTS recommended for D-UART
                )

            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from both UARTs."""
        if self.c_uart:
            self.c_uart.close()
            self.c_uart = None
        if self.d_uart:
            self.d_uart.close()
            self.d_uart = None

    def start(self):
        """Start RX threads and retransmit thread."""
        self.running = True

        if self.c_uart:
            self.c_uart_rx_thread = threading.Thread(
                target=self._c_uart_rx_loop, daemon=True
            )
            self.c_uart_rx_thread.start()

        if self.d_uart:
            self.d_uart_rx_thread = threading.Thread(
                target=self._d_uart_rx_loop, daemon=True
            )
            self.d_uart_rx_thread.start()

        self.retransmit_thread = threading.Thread(
            target=self._retransmit_loop, daemon=True
        )
        self.retransmit_thread.start()

    def stop(self):
        """Stop all threads."""
        self.running = False

        if self.c_uart_rx_thread:
            self.c_uart_rx_thread.join(timeout=1.0)
        if self.d_uart_rx_thread:
            self.d_uart_rx_thread.join(timeout=1.0)
        if self.retransmit_thread:
            self.retransmit_thread.join(timeout=1.0)

    # =========================================================================
    # C-UART (Control) Operations
    # =========================================================================

    def send_control(self, msg_type: ControlMsgType, payload: bytes, wait_ack: bool = True) -> Optional[int]:
        """
        Send control message with optional ACK wait.

        Returns sequence number if sent, None on failure.
        """
        seq = self.c_uart_tx_seq
        self.c_uart_tx_seq = (self.c_uart_tx_seq + 1) & 0xFFFF

        packet = DualUARTPacket(
            chan=CHANNEL_CONTROL,
            msg_type=msg_type,
            seq=seq,
            payload=payload
        )

        if wait_ack:
            with self.pending_lock:
                self.pending_acks[seq] = PendingMessage(
                    seq=seq,
                    packet=packet,
                    send_time=time.time()
                )

        self._send_c_uart(packet)
        return seq

    def send_hello(self) -> Optional[int]:
        """Send HELLO message to initiate session."""
        return self.send_control(ControlMsgType.HELLO, b'')

    def send_note_event(self, note_event: NoteEvent) -> Optional[int]:
        """Send NOTE_EVENT message."""
        return self.send_control(ControlMsgType.NOTE_EVENT, note_event.to_bytes())

    def send_set_tree_target(self, tree_target: SetTreeTarget) -> Optional[int]:
        """Send SET_TREE_TARGET message."""
        return self.send_control(ControlMsgType.SET_TREE_TARGET, tree_target.to_bytes())

    def send_set_macro(self, macro: SetMacro) -> Optional[int]:
        """Send SET_MACRO message."""
        return self.send_control(ControlMsgType.SET_MACRO, macro.to_bytes())

    def send_stream_ctrl(self, ctrl: StreamCtrl) -> Optional[int]:
        """Send STREAM_CTRL message."""
        seq = self.send_control(ControlMsgType.STREAM_CTRL, ctrl.to_bytes())
        if ctrl.start:
            self.streaming = True
        else:
            self.streaming = False
        return seq

    def send_get_status(self) -> Optional[int]:
        """Send GET_STATUS message."""
        return self.send_control(ControlMsgType.GET_STATUS, b'')

    def _send_c_uart(self, packet: DualUARTPacket):
        """Send packet on C-UART."""
        if self.c_uart:
            data = packet.to_bytes()
            self.c_uart.write(data)
            self.stats['c_uart_tx'] += 1

    def _c_uart_rx_loop(self):
        """C-UART receive loop."""
        buffer = b''

        while self.running:
            try:
                if self.c_uart and self.c_uart.in_waiting:
                    buffer += self.c_uart.read(self.c_uart.in_waiting)

                # Parse packets
                while buffer:
                    packet, consumed = DualUARTPacket.from_bytes(buffer)

                    if packet is None:
                        if consumed > 0:
                            buffer = buffer[consumed:]
                            self.stats['crc_errors'] += 1
                        else:
                            break
                    else:
                        buffer = buffer[consumed:]
                        self.stats['c_uart_rx'] += 1
                        self._handle_control_packet(packet)

                time.sleep(0.001)
            except Exception as e:
                print(f"C-UART RX error: {e}")
                time.sleep(0.1)

    def _handle_control_packet(self, packet: DualUARTPacket):
        """Handle received control packet."""
        # Handle ACK/NACK
        if packet.msg_type == ControlMsgType.ACK:
            acked_seq = struct.unpack('<H', packet.payload[:2])[0]
            with self.pending_lock:
                if acked_seq in self.pending_acks:
                    del self.pending_acks[acked_seq]

        elif packet.msg_type == ControlMsgType.NACK:
            nacked_seq, error_code = struct.unpack('<HH', packet.payload[:4])
            with self.pending_lock:
                if nacked_seq in self.pending_acks:
                    del self.pending_acks[nacked_seq]
            print(f"NACK received for seq {nacked_seq}: error {error_code}")

        elif packet.msg_type == ControlMsgType.HELLO_ACK:
            hello_ack = HelloAck.from_bytes(packet.payload)
            self.current_block_id = hello_ack.block_id
            self.data_credit_bytes = hello_ack.credit_bytes

        elif packet.msg_type == ControlMsgType.STATUS:
            status = Status.from_bytes(packet.payload)
            self.current_block_id = status.block_id
            self.data_credit_bytes = status.data_credit_bytes

        # Dispatch to handler
        handler = self.control_handlers.get(packet.msg_type)
        if handler:
            handler(packet)

        self.c_uart_rx_queue.put(packet)

    def _retransmit_loop(self):
        """Retransmit loop for unacked control messages."""
        while self.running:
            try:
                now = time.time()
                to_retransmit = []
                to_timeout = []

                with self.pending_lock:
                    for seq, pending in list(self.pending_acks.items()):
                        if now - pending.send_time > self.ack_timeout_s:
                            if pending.retry_count < self.max_retries:
                                to_retransmit.append(pending)
                                pending.retry_count += 1
                                pending.send_time = now
                            else:
                                to_timeout.append(seq)

                # Retransmit
                for pending in to_retransmit:
                    self._send_c_uart(pending.packet)
                    self.stats['c_uart_retransmits'] += 1

                # Remove timed out
                with self.pending_lock:
                    for seq in to_timeout:
                        if seq in self.pending_acks:
                            del self.pending_acks[seq]
                            self.stats['c_uart_timeouts'] += 1

                time.sleep(0.01)
            except Exception as e:
                print(f"Retransmit error: {e}")
                time.sleep(0.1)

    # =========================================================================
    # D-UART (Data) Operations
    # =========================================================================

    def send_data(self, msg_type: DataMsgType, payload: bytes) -> bool:
        """
        Send data message (best-effort, no ACK).

        Returns True if credit available and sent.
        """
        if len(payload) > self.data_credit_bytes:
            self.stats['d_uart_drops'] += 1
            return False

        seq = self.d_uart_tx_seq
        self.d_uart_tx_seq = (self.d_uart_tx_seq + 1) & 0xFFFF

        packet = DualUARTPacket(
            chan=CHANNEL_DATA,
            msg_type=msg_type,
            seq=seq,
            payload=payload
        )

        self._send_d_uart(packet)
        self.data_credit_bytes -= len(payload) + FRAME_HEADER_SIZE
        return True

    def send_voice_delta(self, delta: VoiceDelta) -> bool:
        """Send VOICE_DELTA message."""
        return self.send_data(DataMsgType.VOICE_DELTA, delta.to_bytes())

    def send_voice_burst(self, burst: VoiceBurst) -> bool:
        """Send VOICE_BURST message."""
        return self.send_data(DataMsgType.VOICE_BURST, burst.to_bytes())

    def send_data_stream_start(self, base_block_id: int) -> bool:
        """Send DATA_STREAM_START message."""
        start = DataStreamStart(base_block_id=base_block_id)
        return self.send_data(DataMsgType.DATA_STREAM_START, start.to_bytes())

    def send_data_stream_stop(self) -> bool:
        """Send DATA_STREAM_STOP message."""
        return self.send_data(DataMsgType.DATA_STREAM_STOP, b'')

    def _send_d_uart(self, packet: DualUARTPacket):
        """Send packet on D-UART."""
        if self.d_uart:
            data = packet.to_bytes()
            self.d_uart.write(data)
            self.stats['d_uart_tx'] += 1

    def _d_uart_rx_loop(self):
        """D-UART receive loop."""
        buffer = b''

        while self.running:
            try:
                if self.d_uart and self.d_uart.in_waiting:
                    buffer += self.d_uart.read(self.d_uart.in_waiting)

                # Parse packets
                while buffer:
                    packet, consumed = DualUARTPacket.from_bytes(buffer)

                    if packet is None:
                        if consumed > 0:
                            buffer = buffer[consumed:]
                            self.stats['crc_errors'] += 1
                        else:
                            break
                    else:
                        buffer = buffer[consumed:]
                        self.stats['d_uart_rx'] += 1
                        self._handle_data_packet(packet)

                time.sleep(0.001)
            except Exception as e:
                print(f"D-UART RX error: {e}")
                time.sleep(0.1)

    def _handle_data_packet(self, packet: DualUARTPacket):
        """Handle received data packet."""
        if packet.msg_type == DataMsgType.DATA_CREDIT:
            credit = DataCredit.from_bytes(packet.payload)
            self.data_credit_bytes = credit.credit_bytes
            self.current_block_id = credit.block_id

        elif packet.msg_type == DataMsgType.DATA_DROP_NOTICE:
            self.stats['d_uart_drops'] += 1

        # Dispatch to handler
        handler = self.data_handlers.get(packet.msg_type)
        if handler:
            handler(packet)

        self.d_uart_rx_queue.put(packet)

    # =========================================================================
    # Session Management (per WSE-STD-100 Section 9.3)
    # =========================================================================

    def start_session(self, timeout: float = 2.0) -> bool:
        """
        Start session per WSE-STD-100 Section 9.3.

        1. Host sends HELLO on C-UART
        2. FPGA replies HELLO_ACK with block_id and credit_bytes
        3. Host sends STREAM_CTRL(start) on C-UART
        4. Host sends DATA_STREAM_START on D-UART with base_block_id
        """
        # Step 1: Send HELLO
        self.send_hello()

        # Step 2: Wait for HELLO_ACK
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                packet = self.c_uart_rx_queue.get(timeout=0.1)
                if packet.msg_type == ControlMsgType.HELLO_ACK:
                    break
            except queue.Empty:
                pass
        else:
            return False  # Timeout

        # Step 3: Send STREAM_CTRL(start)
        self.send_stream_ctrl(StreamCtrl(start=True))

        # Step 4: Send DATA_STREAM_START
        base_block_id = self.current_block_id + self.lead_blocks
        self.send_data_stream_start(base_block_id)

        return True

    def stop_session(self):
        """Stop streaming session."""
        self.send_stream_ctrl(StreamCtrl(start=False))
        self.send_data_stream_stop()
        self.streaming = False

    # =========================================================================
    # Voice Management
    # =========================================================================

    def get_voice_state(self, voice_id: int) -> VoiceState:
        """Get or create voice state."""
        if voice_id not in self.voice_states:
            self.voice_states[voice_id] = VoiceState(voice_id=voice_id)
        return self.voice_states[voice_id]

    def schedule_block_id(self) -> int:
        """Get block_id for scheduling (current + lead_blocks)."""
        return self.current_block_id + self.lead_blocks

    # =========================================================================
    # Handler Registration
    # =========================================================================

    def register_control_handler(self, msg_type: ControlMsgType, handler: Callable):
        """Register handler for control message type."""
        self.control_handlers[msg_type] = handler

    def register_data_handler(self, msg_type: DataMsgType, handler: Callable):
        """Register handler for data message type."""
        self.data_handlers[msg_type] = handler


# =============================================================================
# MOCK DUAL UART FOR TESTING (no hardware required)
# =============================================================================

class MockDualUART:
    """
    Mock Dual UART interface for testing without hardware.

    Simulates bidirectional communication between Host and FPGA.
    """

    def __init__(self):
        # Host -> FPGA queues
        self.c_uart_host_tx: queue.Queue = queue.Queue()
        self.d_uart_host_tx: queue.Queue = queue.Queue()

        # FPGA -> Host queues
        self.c_uart_fpga_tx: queue.Queue = queue.Queue()
        self.d_uart_fpga_tx: queue.Queue = queue.Queue()

        # FPGA state
        self.fpga_block_id = 0
        self.fpga_credit_bytes = 4096
        self.fpga_active_voices: Dict[int, bool] = {}
        self.fpga_voice_states: Dict[int, VoiceState] = {}

        # Statistics
        self.stats = {
            'c_uart_packets': 0,
            'd_uart_packets': 0,
            'late_applies': 0,
            'invalid_voice_ids': 0,
        }

    def host_send_control(self, packet: DualUARTPacket):
        """Host sends control packet."""
        self.c_uart_host_tx.put(packet)

    def host_send_data(self, packet: DualUARTPacket):
        """Host sends data packet."""
        self.d_uart_host_tx.put(packet)

    def host_receive_control(self, timeout: float = 0.1) -> Optional[DualUARTPacket]:
        """Host receives control packet."""
        try:
            return self.c_uart_fpga_tx.get(timeout=timeout)
        except queue.Empty:
            return None

    def host_receive_data(self, timeout: float = 0.1) -> Optional[DualUARTPacket]:
        """Host receives data packet."""
        try:
            return self.d_uart_fpga_tx.get(timeout=timeout)
        except queue.Empty:
            return None

    def fpga_process(self):
        """
        Process messages on FPGA side.

        Simulates FPGA behavior: processes control/data messages,
        sends ACKs, generates credits.
        """
        # Process control messages
        try:
            packet = self.c_uart_host_tx.get_nowait()
            self._fpga_handle_control(packet)
        except queue.Empty:
            pass

        # Process data messages
        try:
            packet = self.d_uart_host_tx.get_nowait()
            self._fpga_handle_data(packet)
        except queue.Empty:
            pass

    def _fpga_handle_control(self, packet: DualUARTPacket):
        """Handle control message on FPGA side."""
        self.stats['c_uart_packets'] += 1

        if packet.msg_type == ControlMsgType.HELLO:
            # Send HELLO_ACK
            ack = HelloAck(
                block_id=self.fpga_block_id,
                credit_bytes=self.fpga_credit_bytes,
                build_id=0x12345678,
                buffer_size=4096,
                max_voices=128,
                features=0
            )
            self._fpga_send_control(ControlMsgType.HELLO_ACK, ack.to_bytes(), packet.seq)

        elif packet.msg_type == ControlMsgType.GET_STATUS:
            # Send STATUS
            status = Status(
                block_id=self.fpga_block_id,
                data_credit_bytes=self.fpga_credit_bytes,
                active_voices=sum(1 for v in self.fpga_active_voices.values() if v),
                main_voices=0,
                underrun_count=0,
                late_apply_count=self.stats['late_applies'],
                last_error_code=0
            )
            self._fpga_send_control(ControlMsgType.STATUS, status.to_bytes(), packet.seq)

        elif packet.msg_type == ControlMsgType.NOTE_EVENT:
            note_event = NoteEvent.from_bytes(packet.payload)

            # Check late apply
            if note_event.apply_block_id <= self.fpga_block_id:
                self.stats['late_applies'] += 1

            # Activate/deactivate voice (simplified)
            # In real implementation, voice allocation would happen here

            # Send ACK
            self._fpga_send_ack(packet.seq)

        elif packet.msg_type == ControlMsgType.SET_TREE_TARGET:
            # Send ACK
            self._fpga_send_ack(packet.seq)

        elif packet.msg_type == ControlMsgType.STREAM_CTRL:
            # Send ACK
            self._fpga_send_ack(packet.seq)

        else:
            # Send ACK for other known messages
            self._fpga_send_ack(packet.seq)

    def _fpga_handle_data(self, packet: DualUARTPacket):
        """Handle data message on FPGA side (no ACK)."""
        self.stats['d_uart_packets'] += 1

        if packet.msg_type == DataMsgType.VOICE_DELTA:
            delta = VoiceDelta.from_bytes(packet.payload)

            # Validate voice_id
            if delta.voice_id > MAX_VOICE_ID:
                self.stats['invalid_voice_ids'] += 1
                return

            # Check late apply
            if delta.apply_block_id <= self.fpga_block_id:
                self.stats['late_applies'] += 1

            # Update voice state
            if delta.voice_id not in self.fpga_voice_states:
                self.fpga_voice_states[delta.voice_id] = VoiceState(voice_id=delta.voice_id)
            self.fpga_voice_states[delta.voice_id].update(delta)

        elif packet.msg_type == DataMsgType.VOICE_BURST:
            burst = VoiceBurst.from_bytes(packet.payload)

            if burst.voice_id > MAX_VOICE_ID:
                self.stats['invalid_voice_ids'] += 1
                return

    def _fpga_send_control(self, msg_type: ControlMsgType, payload: bytes, ack_seq: int):
        """Send control message from FPGA."""
        packet = DualUARTPacket(
            chan=CHANNEL_CONTROL,
            msg_type=msg_type,
            seq=0,
            payload=payload
        )
        self.c_uart_fpga_tx.put(packet)

    def _fpga_send_ack(self, seq: int):
        """Send ACK from FPGA."""
        payload = struct.pack('<HBB', seq, 0, 0)  # seq, status=OK, detail=0
        packet = DualUARTPacket(
            chan=CHANNEL_CONTROL,
            msg_type=ControlMsgType.ACK,
            seq=0,
            payload=payload
        )
        self.c_uart_fpga_tx.put(packet)

    def fpga_send_credit(self):
        """Send DATA_CREDIT from FPGA."""
        credit = DataCredit(
            block_id=self.fpga_block_id,
            credit_bytes=self.fpga_credit_bytes,
            rx_fill_bytes=0
        )
        packet = DualUARTPacket(
            chan=CHANNEL_DATA,
            msg_type=DataMsgType.DATA_CREDIT,
            seq=0,
            payload=credit.to_bytes()
        )
        self.d_uart_fpga_tx.put(packet)

    def advance_block(self):
        """Advance FPGA block counter."""
        self.fpga_block_id += 1


# =============================================================================
# CONFORMANCE TEST
# =============================================================================

def run_conformance_test():
    """
    Run WSE-STD-100 conformance tests.
    """
    print("=" * 60)
    print("WSE-STD-100 Dual-UART Link Conformance Test")
    print("=" * 60)

    # Test 1: COBS encoding/decoding
    print("\n1. COBS Encoding/Decoding")
    test_data = bytes([0x11, 0x22, 0x00, 0x33])
    encoded = cobs_encode(test_data)
    decoded = cobs_decode(encoded)
    assert decoded == test_data, f"COBS mismatch: {decoded} vs {test_data}"
    print(f"   Original: {test_data.hex()}")
    print(f"   Encoded:  {encoded.hex()}")
    print(f"   Decoded:  {decoded.hex()}")
    print("   PASS")

    # Test 2: CRC-32C
    print("\n2. CRC-32C Implementation")
    test_crc_data = b"123456789"
    expected_crc = 0xE3069283
    computed_crc = crc32c(test_crc_data)
    assert computed_crc == expected_crc, f"CRC mismatch: {computed_crc:08X} vs {expected_crc:08X}"
    print(f"   CRC-32C('123456789') = 0x{computed_crc:08X}")
    print("   PASS")

    # Test 3: Frame header serialization
    print("\n3. Frame Header Serialization")
    header = FrameHeader(
        magic=FRAME_MAGIC,
        version=PROTOCOL_VERSION,
        chan=CHANNEL_CONTROL,
        msg_type=ControlMsgType.NOTE_EVENT,
        flags=0,
        seq=42,
        length=9,
        crc32=0x12345678
    )
    header_bytes = header.to_bytes()
    restored = FrameHeader.from_bytes(header_bytes)
    assert restored.magic == header.magic
    assert restored.msg_type == header.msg_type
    assert restored.seq == header.seq
    print(f"   Header size: {len(header_bytes)} bytes")
    print(f"   Magic: 0x{restored.magic:04X}")
    print("   PASS")

    # Test 4: DualUARTPacket round-trip
    print("\n4. DualUARTPacket Round-Trip")
    note = NoteEvent(
        apply_block_id=1000,
        note=60,
        vel=100,
        on=True,
        chan=0,
        flags=0
    )
    packet = DualUARTPacket(
        chan=CHANNEL_CONTROL,
        msg_type=ControlMsgType.NOTE_EVENT,
        seq=1,
        payload=note.to_bytes()
    )
    packet_bytes = packet.to_bytes()
    restored_packet, consumed = DualUARTPacket.from_bytes(packet_bytes)
    assert restored_packet is not None
    assert restored_packet.msg_type == ControlMsgType.NOTE_EVENT
    assert restored_packet.seq == 1
    assert restored_packet.payload == note.to_bytes()
    print(f"   Packet size: {len(packet_bytes)} bytes (COBS encoded)")
    print(f"   Payload: {restored_packet.payload.hex()}")
    print("   PASS")

    # Test 5: NoteEvent serialization
    print("\n5. NoteEvent Serialization")
    note = NoteEvent(apply_block_id=8000, note=60, vel=100, on=True, chan=0, flags=0)
    note_bytes = note.to_bytes()
    restored_note = NoteEvent.from_bytes(note_bytes)
    assert restored_note.apply_block_id == note.apply_block_id
    assert restored_note.note == note.note
    assert restored_note.vel == note.vel
    print(f"   Note: {note.note}, Vel: {note.vel}, Block: {note.apply_block_id}")
    print(f"   Bytes: {note_bytes.hex()}")
    print("   PASS")

    # Test 6: VoiceDelta with leaf gains
    print("\n6. VoiceDelta Serialization")
    delta = VoiceDelta(
        apply_block_id=8016,
        voice_id=5,
        subtype=VoiceDeltaSubtype.LEAF_GAINS,
        leaf_gains=[
            LeafGainEntry(leaf_idx=3, gain_q15=0x4000),
            LeafGainEntry(leaf_idx=7, gain_q15=0x7F80),
            LeafGainEntry(leaf_idx=15, gain_q15=0x2000),
        ]
    )
    delta_bytes = delta.to_bytes()
    restored_delta = VoiceDelta.from_bytes(delta_bytes)
    assert restored_delta.voice_id == delta.voice_id
    assert len(restored_delta.leaf_gains) == 3
    print(f"   Voice: {delta.voice_id}, Entries: {len(delta.leaf_gains)}")
    print(f"   Bytes: {delta_bytes.hex()}")
    print("   PASS")

    # Test 7: Mock UART round-trip
    print("\n7. Mock UART Round-Trip")
    mock = MockDualUART()

    # Send HELLO
    hello_packet = DualUARTPacket(
        chan=CHANNEL_CONTROL,
        msg_type=ControlMsgType.HELLO,
        seq=0,
        payload=b''
    )
    mock.host_send_control(hello_packet)
    mock.fpga_process()

    # Receive HELLO_ACK
    ack = mock.host_receive_control(timeout=1.0)
    assert ack is not None
    assert ack.msg_type == ControlMsgType.HELLO_ACK
    hello_ack = HelloAck.from_bytes(ack.payload)
    print(f"   Sent HELLO")
    print(f"   Received HELLO_ACK: block_id={hello_ack.block_id}, credit={hello_ack.credit_bytes}")
    print("   PASS")

    # Test 8: Data channel flow
    print("\n8. Data Channel Flow")
    mock.fpga_block_id = 100

    delta_packet = DualUARTPacket(
        chan=CHANNEL_DATA,
        msg_type=DataMsgType.VOICE_DELTA,
        seq=0,
        payload=delta.to_bytes()
    )
    mock.host_send_data(delta_packet)
    mock.fpga_process()

    assert 5 in mock.fpga_voice_states
    voice_state = mock.fpga_voice_states[5]
    assert 3 in voice_state.leaf_gains
    print(f"   Sent VOICE_DELTA for voice 5")
    print(f"   FPGA voice state updated: leaf 3 gain = {voice_state.leaf_gains[3]}")
    print("   PASS")

    # Test 9: Credit-based flow control
    print("\n9. Credit-Based Flow Control")
    mock.fpga_send_credit()
    credit_packet = mock.host_receive_data(timeout=1.0)
    assert credit_packet is not None
    assert credit_packet.msg_type == DataMsgType.DATA_CREDIT
    credit = DataCredit.from_bytes(credit_packet.payload)
    assert credit.credit_bytes == mock.fpga_credit_bytes
    print(f"   Received DATA_CREDIT: {credit.credit_bytes} bytes")
    print("   PASS")

    # Test 10: Voice state hold/decay
    print("\n10. Voice State Hold/Decay (Section 8)")
    voice = VoiceState(voice_id=0, active=True)
    voice.leaf_gains[0] = 32767
    voice.last_block_id = 100

    # Within hold period
    voice.tick_block(104, hold_blocks=4, decay_blocks=32)
    assert voice.leaf_gains[0] == 32767
    print(f"   Block 104 (within hold): gain = {voice.leaf_gains[0]}")

    # Start decay
    voice.tick_block(106, hold_blocks=4, decay_blocks=32)
    assert voice.leaf_gains[0] < 32767
    print(f"   Block 106 (decay started): gain = {voice.leaf_gains[0]}")
    print("   PASS")

    print("\n" + "=" * 60)
    print("All WSE-STD-100 conformance tests PASSED")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_conformance_test()
