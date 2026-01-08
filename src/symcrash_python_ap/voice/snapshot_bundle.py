"""
Snapshot Bundle (WSE-STD-VOICEENGINEREGISTRY v0 §7)
====================================================

Chunked binary container for engine state snapshots.

Snapshot Format:
- Header: "WSES" magic + version + endian marker + chunk count + CRC
- Chunks: type + version + voice_id + payload + CRC
- Chunk types: CORE=1, CMMM=2, CCS=3, IDWPT=4, GLOBAL=5

Enables bit-identical resume of engine state for:
- Deterministic replay
- Checkpoint/restore
- Regression testing
"""

import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import IntEnum

from ..vm.state import VMState
from .voice_state_codec import vmstate_to_bytes, vmstate_from_bytes


# =============================================================================
# CONSTANTS
# =============================================================================

SNAPSHOT_MAGIC = b'WSES'
SNAPSHOT_VERSION = 0
ENDIAN_MARKER = 0xFFFE  # Little-endian marker


class ChunkType(IntEnum):
    """Chunk type identifiers (WSE-STD-VOICEENGINEREGISTRY §7.3)."""
    CORE = 1      # Core VMState
    CMMM = 2      # CMMM time state
    CCS = 3       # CCS oscillator state
    IDWPT = 4     # IDWPT wavelet state
    GLOBAL = 5    # Global engine state


VOICE_ID_GLOBAL = 0xFFFF  # Special voice_id for global chunks


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SnapshotHeader:
    """Snapshot file header."""
    magic: bytes = SNAPSHOT_MAGIC
    version: int = SNAPSHOT_VERSION
    endian: int = ENDIAN_MARKER
    chunk_count: int = 0
    global_crc32: int = 0

    def to_bytes(self) -> bytes:
        """Serialize header to 16 bytes."""
        return struct.pack(
            '<4sHHHHI',
            self.magic,
            self.version,
            self.endian,
            self.chunk_count,
            0,  # reserved
            self.global_crc32,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SnapshotHeader':
        """Deserialize header from 16 bytes."""
        magic, version, endian, chunk_count, _, global_crc = struct.unpack('<4sHHHHI', data)

        if magic != SNAPSHOT_MAGIC:
            raise ValueError(f"Invalid snapshot magic: {magic}")
        if endian != ENDIAN_MARKER:
            raise ValueError(f"Endian mismatch: {endian:04X}")

        return cls(
            magic=magic,
            version=version,
            endian=endian,
            chunk_count=chunk_count,
            global_crc32=global_crc,
        )


@dataclass
class SnapshotChunk:
    """
    Single snapshot chunk.

    Each chunk has:
    - Type identifier (CORE, CMMM, CCS, IDWPT, GLOBAL)
    - Version for format evolution
    - Voice ID (or 0xFFFF for global chunks)
    - Payload (arbitrary bytes)
    - CRC32 for integrity
    """
    chunk_type: ChunkType
    chunk_version: int
    voice_id: int
    payload: bytes
    chunk_crc32: int = 0

    def compute_crc(self) -> int:
        """Compute CRC32 of chunk data (excluding CRC field itself)."""
        data = struct.pack(
            '<HHHHI',
            int(self.chunk_type),
            self.chunk_version,
            self.voice_id,
            0,  # reserved
            len(self.payload),
        ) + self.payload

        return zlib.crc32(data) & 0xFFFFFFFF

    def to_bytes(self) -> bytes:
        """
        Serialize chunk.

        Format:
            u16 chunk_type
            u16 chunk_version
            u16 voice_id
            u16 reserved
            u32 payload_len
            u8  payload[payload_len]
            u32 chunk_crc32
        """
        # Compute CRC if not set
        if self.chunk_crc32 == 0:
            self.chunk_crc32 = self.compute_crc()

        header = struct.pack(
            '<HHHHI',
            int(self.chunk_type),
            self.chunk_version,
            self.voice_id,
            0,  # reserved
            len(self.payload),
        )

        crc_bytes = struct.pack('<I', self.chunk_crc32)

        return header + self.payload + crc_bytes

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple['SnapshotChunk', int]:
        """
        Deserialize chunk from data.

        Args:
            data: Binary data
            offset: Starting offset

        Returns:
            (SnapshotChunk, new_offset)
        """
        # Read header (12 bytes)
        chunk_type, chunk_version, voice_id, _, payload_len = struct.unpack_from(
            '<HHHHI', data, offset
        )
        offset += 12

        # Read payload
        payload = data[offset:offset + payload_len]
        offset += payload_len

        # Read CRC
        chunk_crc32, = struct.unpack_from('<I', data, offset)
        offset += 4

        chunk = cls(
            chunk_type=ChunkType(chunk_type),
            chunk_version=chunk_version,
            voice_id=voice_id,
            payload=payload,
            chunk_crc32=chunk_crc32,
        )

        # Verify CRC
        computed_crc = chunk.compute_crc()
        if computed_crc != chunk_crc32:
            raise ValueError(
                f"CRC mismatch in chunk: got 0x{chunk_crc32:08X}, "
                f"expected 0x{computed_crc:08X}"
            )

        return chunk, offset


# =============================================================================
# SNAPSHOT BUNDLE
# =============================================================================

class SnapshotBundle:
    """
    Complete snapshot container with header + chunks.

    Per WSE-STD-VOICEENGINEREGISTRY v0 §7.
    """

    def __init__(self):
        self.header = SnapshotHeader()
        self.chunks: List[SnapshotChunk] = []

    def add_voice_state(self, voice_id: int, vm_state: VMState):
        """Add VMState as CORE chunk."""
        payload = vmstate_to_bytes(vm_state)
        chunk = SnapshotChunk(
            chunk_type=ChunkType.CORE,
            chunk_version=0,
            voice_id=voice_id,
            payload=payload,
        )
        self.chunks.append(chunk)

    def get_voice_state(self, voice_id: int) -> Optional[VMState]:
        """Get VMState for specific voice."""
        for chunk in self.chunks:
            if chunk.chunk_type == ChunkType.CORE and chunk.voice_id == voice_id:
                return vmstate_from_bytes(chunk.payload)
        return None

    def add_global_state(self, payload: bytes):
        """Add global state chunk."""
        chunk = SnapshotChunk(
            chunk_type=ChunkType.GLOBAL,
            chunk_version=0,
            voice_id=VOICE_ID_GLOBAL,
            payload=payload,
        )
        self.chunks.append(chunk)

    def get_global_state(self) -> Optional[bytes]:
        """Get global state payload."""
        for chunk in self.chunks:
            if chunk.chunk_type == ChunkType.GLOBAL:
                return chunk.payload
        return None

    def list_voice_ids(self) -> List[int]:
        """Get list of voice IDs in snapshot."""
        voice_ids = []
        for chunk in self.chunks:
            if chunk.voice_id != VOICE_ID_GLOBAL and chunk.voice_id not in voice_ids:
                voice_ids.append(chunk.voice_id)
        return sorted(voice_ids)

    def to_bytes(self) -> bytes:
        """
        Serialize complete snapshot to binary.

        Returns:
            Binary data ready to write to file
        """
        # Update header
        self.header.chunk_count = len(self.chunks)

        # Serialize all chunks
        chunk_bytes = b''.join(chunk.to_bytes() for chunk in self.chunks)

        # Compute global CRC (header + all chunks, excluding global_crc field)
        header_for_crc = struct.pack(
            '<4sHHHH',
            self.header.magic,
            self.header.version,
            self.header.endian,
            self.header.chunk_count,
            0,  # reserved
        )
        crc_data = header_for_crc + chunk_bytes
        self.header.global_crc32 = zlib.crc32(crc_data) & 0xFFFFFFFF

        # Serialize header with CRC
        header_bytes = self.header.to_bytes()

        return header_bytes + chunk_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SnapshotBundle':
        """
        Deserialize snapshot from binary data.

        Args:
            data: Binary snapshot data

        Returns:
            SnapshotBundle instance

        Raises:
            ValueError: If magic/version mismatch or CRC failure
        """
        bundle = cls()

        # Read header (16 bytes)
        bundle.header = SnapshotHeader.from_bytes(data[:16])
        offset = 16

        # Verify global CRC
        header_for_crc = data[:12]  # Exclude global_crc32 field
        chunk_data = data[16:]
        computed_crc = zlib.crc32(header_for_crc + chunk_data) & 0xFFFFFFFF

        if computed_crc != bundle.header.global_crc32:
            raise ValueError(
                f"Global CRC mismatch: got 0x{bundle.header.global_crc32:08X}, "
                f"expected 0x{computed_crc:08X}"
            )

        # Read chunks
        for _ in range(bundle.header.chunk_count):
            chunk, offset = SnapshotChunk.from_bytes(data, offset)
            bundle.chunks.append(chunk)

        return bundle

    def save(self, filepath: str):
        """Save snapshot to file."""
        data = self.to_bytes()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def load(cls, filepath: str) -> 'SnapshotBundle':
        """Load snapshot from file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.from_bytes(data)


__all__ = [
    'ChunkType',
    'SnapshotHeader',
    'SnapshotChunk',
    'SnapshotBundle',
    'SNAPSHOT_MAGIC',
    'SNAPSHOT_VERSION',
    'VOICE_ID_GLOBAL',
]
