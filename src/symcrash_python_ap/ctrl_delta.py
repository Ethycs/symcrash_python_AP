"""
CTRL-DELTA Wire Protocol
========================

Sparse parameter update format for UART transmission.

This is the PRIMARY wire format for block-rate parameter updates.
VoiceFramePacket (136 bytes) is internal/debug only - too large for real-time.

Key insight from architecture discussion:
- VoiceFramePacket per-voice per-block = ~102 kB/s (impossible at 921600 bps)
- CTRL-DELTA sends only changed parameters = typically < 20 bytes/voice/block

Packet Format:
    0x00  1     magic u8 = 0xCD ("CD" for CTRL-DELTA)
    0x01  1     version u8 = 1
    0x02  1     scope_type u8 (0=voice, 1=global, 2=program)
    0x03  1     scope_id u8 (voice_id for scope_type=0)
    0x04  4     field_mask u32 (which fields are present)
    0x08  1     mode u8 (0=ABS, 1=DELTA)
    0x09  1     payload_len u8 (bytes in payload)
    0x0A  N     payload (variable length, field values in mask order)

Maximum packet size: 10 + 255 = 265 bytes (but typically < 30 bytes)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import IntEnum
import struct

from parameter_registry import (
    FieldMask, MaskedUpdate, UpdateMode,
    FIELD_TO_PARAMS, get_registry
)


# =============================================================================
# CONSTANTS
# =============================================================================

CTRL_DELTA_MAGIC = 0xCD
CTRL_DELTA_VERSION = 1
CTRL_DELTA_HEADER_LEN = 10


class ScopeType(IntEnum):
    """Target scope for parameter update."""
    VOICE = 0     # Per-voice parameters (scope_id = voice_id)
    GLOBAL = 1    # Global parameters (scope_id = 0)
    PROGRAM = 2   # Per-program parameters (scope_id = program_id)


# Field sizes in bytes for payload encoding
FIELD_SIZES: Dict[int, int] = {
    # Voice Domain - single values
    FieldMask.GAIN_Q15: 2,
    FieldMask.ACTIVITY_Q15: 2,
    FieldMask.PAN_Q15: 2,
    FieldMask.SEND_A_Q15: 2,
    FieldMask.SEND_B_Q15: 2,
    FieldMask.NOTE_U8: 1,
    FieldMask.CENTS_I16: 2,
    FieldMask.REGIME_ID: 1,
    FieldMask.DEGRADATION: 1,
    FieldMask.TREE_A_ID: 1,
    FieldMask.TREE_B_ID: 1,
    FieldMask.TREE_BLEND_Q15: 2,
    FieldMask.FAMILY_WEIGHTS: 16,     # 8 x i16
    FieldMask.EVENT_MASK: 1,
    FieldMask.EVENT_STRENGTHS: 16,    # 8 x i16

    # Symbol Domain
    FieldMask.K_MODE: 1,
    FieldMask.DRIVE_Q15: 2,
    FieldMask.TONE_Q15: 2,
    FieldMask.BIAS_Q15: 2,
    FieldMask.PRESSURE_Q15: 2,

    # Safety Domain
    FieldMask.ENERGY_CAP_Q15: 2,
    FieldMask.GEAR_STRENGTH_Q15: 2,
    FieldMask.TRACTION_Q15: 2,
}


# =============================================================================
# CTRL-DELTA PACKET
# =============================================================================

@dataclass
class CtrlDeltaPacket:
    """
    Sparse parameter update packet for UART transmission.

    This is the PRIMARY wire format - much smaller than VoiceFramePacket.
    Only changed parameters are transmitted.
    """
    scope_type: ScopeType = ScopeType.VOICE
    scope_id: int = 0                  # voice_id for VOICE scope
    field_mask: int = 0                # Which fields are present
    mode: UpdateMode = UpdateMode.ABS  # ABS or DELTA
    values: Dict[int, Any] = field(default_factory=dict)  # FieldMask bit → value

    @classmethod
    def from_masked_update(
        cls,
        update: MaskedUpdate,
        scope_type: ScopeType = ScopeType.VOICE,
        scope_id: int = 0,
    ) -> 'CtrlDeltaPacket':
        """Create packet from MaskedUpdate."""
        return cls(
            scope_type=scope_type,
            scope_id=scope_id,
            field_mask=update.field_mask,
            mode=update.mode,
            values=dict(update.values),
        )

    def to_masked_update(self) -> MaskedUpdate:
        """Convert to MaskedUpdate for application."""
        return MaskedUpdate(
            field_mask=self.field_mask,
            values=dict(self.values),
            mode=self.mode,
        )

    def to_bytes(self) -> bytes:
        """
        Serialize to bytes for UART transmission.

        Returns variable-length packet with header + payload.
        """
        # Build payload first to know length
        payload = self._encode_payload()

        # Header (10 bytes)
        header = struct.pack('>BBBBIBBB',
            CTRL_DELTA_MAGIC,
            CTRL_DELTA_VERSION,
            self.scope_type,
            self.scope_id,
            self.field_mask,
            1 if self.mode == UpdateMode.DELTA else 0,
            len(payload),
            0,  # padding for alignment
        )

        # Wait, that's 11 bytes. Let me fix the format.
        header = struct.pack('>BBBBIB',
            CTRL_DELTA_MAGIC,
            CTRL_DELTA_VERSION,
            self.scope_type,
            self.scope_id,
            self.field_mask,
            (1 if self.mode == UpdateMode.DELTA else 0) | (len(payload) << 1),
        )

        # That's still not right. Let me be more careful.
        # Header layout:
        # 0: magic (1)
        # 1: version (1)
        # 2: scope_type (1)
        # 3: scope_id (1)
        # 4-7: field_mask (4)
        # 8: mode (1)
        # 9: payload_len (1)

        header = struct.pack('>BBBBIBBB',
            CTRL_DELTA_MAGIC,        # 0
            CTRL_DELTA_VERSION,      # 1
            self.scope_type,         # 2
            self.scope_id,           # 3
            self.field_mask,         # 4-7
            1 if self.mode == UpdateMode.DELTA else 0,  # 8
            len(payload),            # 9
            0,                       # 10 (pad to get clean alignment, but spec says 10 bytes)
        )
        # Actually header is 11 bytes with padding. Let me match spec exactly.

        header = bytes([
            CTRL_DELTA_MAGIC,                                  # 0
            CTRL_DELTA_VERSION,                                # 1
            self.scope_type,                                   # 2
            self.scope_id,                                     # 3
            (self.field_mask >> 24) & 0xFF,                   # 4
            (self.field_mask >> 16) & 0xFF,                   # 5
            (self.field_mask >> 8) & 0xFF,                    # 6
            self.field_mask & 0xFF,                           # 7
            1 if self.mode == UpdateMode.DELTA else 0,        # 8
            len(payload),                                      # 9
        ])

        return header + payload

    def _encode_payload(self) -> bytes:
        """Encode field values as payload bytes."""
        payload = bytearray()

        # Encode fields in FieldMask order (ascending bit position)
        for fm in FieldMask:
            if not (self.field_mask & (1 << fm)):
                continue

            value = self.values.get(int(fm))
            if value is None:
                continue

            size = FIELD_SIZES.get(int(fm), 2)

            if isinstance(value, list):
                # Group field (FAMILY_WEIGHTS, EVENT_STRENGTHS)
                for v in value:
                    if size // len(value) == 2:
                        payload.extend(struct.pack('>h', v))
                    else:
                        payload.append(v & 0xFF)
            elif size == 1:
                payload.append(value & 0xFF)
            elif size == 2:
                payload.extend(struct.pack('>h', value))
            else:
                # Fallback for unknown sizes
                payload.extend(struct.pack('>h', value))

        return bytes(payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'CtrlDeltaPacket':
        """
        Deserialize from bytes.

        Raises ValueError if packet is malformed.
        """
        if len(data) < CTRL_DELTA_HEADER_LEN:
            raise ValueError(f"Packet too short: {len(data)} < {CTRL_DELTA_HEADER_LEN}")

        # Parse header
        magic = data[0]
        version = data[1]
        scope_type = ScopeType(data[2])
        scope_id = data[3]
        field_mask = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
        mode_byte = data[8]
        payload_len = data[9]

        if magic != CTRL_DELTA_MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:02X} != 0x{CTRL_DELTA_MAGIC:02X}")

        if version != CTRL_DELTA_VERSION:
            raise ValueError(f"Invalid version: {version} != {CTRL_DELTA_VERSION}")

        if len(data) < CTRL_DELTA_HEADER_LEN + payload_len:
            raise ValueError(f"Payload truncated: {len(data)} < {CTRL_DELTA_HEADER_LEN + payload_len}")

        mode = UpdateMode.DELTA if mode_byte else UpdateMode.ABS
        payload = data[CTRL_DELTA_HEADER_LEN:CTRL_DELTA_HEADER_LEN + payload_len]

        # Decode payload
        values = cls._decode_payload(field_mask, payload)

        return cls(
            scope_type=scope_type,
            scope_id=scope_id,
            field_mask=field_mask,
            mode=mode,
            values=values,
        )

    @staticmethod
    def _decode_payload(field_mask: int, payload: bytes) -> Dict[int, Any]:
        """Decode payload bytes into field values."""
        values: Dict[int, Any] = {}
        offset = 0

        for fm in FieldMask:
            if not (field_mask & (1 << fm)):
                continue

            size = FIELD_SIZES.get(int(fm), 2)

            if offset + size > len(payload):
                break  # Truncated payload

            if fm in (FieldMask.FAMILY_WEIGHTS, FieldMask.EVENT_STRENGTHS):
                # Group field - 8 x i16
                group = []
                for i in range(8):
                    if offset + 2 <= len(payload):
                        val = struct.unpack('>h', payload[offset:offset+2])[0]
                        group.append(val)
                        offset += 2
                values[int(fm)] = group
            elif size == 1:
                values[int(fm)] = payload[offset]
                offset += 1
            elif size == 2:
                values[int(fm)] = struct.unpack('>h', payload[offset:offset+2])[0]
                offset += 2
            else:
                values[int(fm)] = struct.unpack('>h', payload[offset:offset+2])[0]
                offset += 2

        return values

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate packet structure.

        Returns (is_valid, error_message).
        """
        if self.scope_type not in ScopeType:
            return False, f"Invalid scope_type: {self.scope_type}"

        if self.scope_id > 255:
            return False, f"scope_id out of range: {self.scope_id}"

        # Check all values are present for set fields
        for fm in FieldMask:
            if self.field_mask & (1 << fm):
                if int(fm) not in self.values:
                    return False, f"Field {fm.name} set in mask but no value"

        return True, None

    def compute_size(self) -> int:
        """Compute total packet size in bytes."""
        payload_size = 0
        for fm in FieldMask:
            if self.field_mask & (1 << fm):
                payload_size += FIELD_SIZES.get(int(fm), 2)
        return CTRL_DELTA_HEADER_LEN + payload_size


# =============================================================================
# BANDWIDTH ESTIMATION
# =============================================================================

def estimate_bandwidth(
    n_voices: int,
    updates_per_voice: int,
    avg_fields_per_update: int,
    block_rate: float = 750.0,
) -> float:
    """
    Estimate UART bandwidth usage for CTRL-DELTA.

    Args:
        n_voices: Number of active voices
        updates_per_voice: Updates per voice per block (typically 0-1)
        avg_fields_per_update: Average fields changed per update
        block_rate: Blocks per second (48000 Hz / 64 samples = 750 Hz)

    Returns:
        Estimated bytes per second
    """
    # Average field size (mostly i16 = 2 bytes)
    avg_field_size = 2.0

    # Packet overhead
    overhead = CTRL_DELTA_HEADER_LEN

    # Average packet size
    avg_packet_size = overhead + avg_fields_per_update * avg_field_size

    # Total bandwidth
    packets_per_second = n_voices * updates_per_voice * block_rate
    return packets_per_second * avg_packet_size


def bandwidth_comparison() -> Dict[str, float]:
    """
    Compare CTRL-DELTA vs VoiceFramePacket bandwidth.

    Returns dict with bandwidth estimates for typical scenarios.
    """
    block_rate = 750.0  # 48kHz / 64 samples

    return {
        # VoiceFramePacket: 136 bytes * 12 voices * 750 Hz = 1,224,000 bytes/s
        # IMPOSSIBLE at 921600 bps (~92 kB/s usable)
        "vfp_12_voices_bps": 136 * 12 * block_rate,

        # CTRL-DELTA: Typical update = 10 header + 4 fields * 2 bytes = 18 bytes
        # 12 voices, 50% update rate = 12 * 0.5 * 18 * 750 = 81,000 bytes/s
        "ctrl_delta_12_voices_50pct_bps": 12 * 0.5 * 18 * block_rate,

        # CTRL-DELTA: Sparse update = 10 header + 2 fields * 2 bytes = 14 bytes
        # 12 voices, 25% update rate = 12 * 0.25 * 14 * 750 = 31,500 bytes/s
        "ctrl_delta_12_voices_25pct_bps": 12 * 0.25 * 14 * block_rate,

        # Maximum UART capacity at 921600 bps (8N1 = 10 bits per byte)
        "uart_921600_capacity_bps": 921600 / 10,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CTRL_DELTA_MAGIC',
    'CTRL_DELTA_VERSION',
    'CTRL_DELTA_HEADER_LEN',
    'ScopeType',
    'FIELD_SIZES',
    'CtrlDeltaPacket',
    'estimate_bandwidth',
    'bandwidth_comparison',
]
