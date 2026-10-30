"""Validated WSES v0 binary snapshot container (VoiceEngineRegistry section 7).

The container enforces framing, identity, versions and both CRC layers. Unknown
chunk types are validated and skipped. CORE v1 contains complete VM state;
CORE v0 remains readable as a legacy partial payload.
"""
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

from ..vm.state import VMState
from .voice_state_codec import vmstate_to_bytes, vmstate_from_bytes, _bytes, _integer, VMSTATE_VERSION


SNAPSHOT_MAGIC = b'WSES'
SNAPSHOT_VERSION = 0
ENDIAN_MARKER = 0xFFFE
HEADER_SIZE = 16
CHUNK_HEADER_SIZE = 12
CHUNK_CRC_SIZE = 4
VOICE_ID_GLOBAL = 0xFFFF


class ChunkType(IntEnum):
    CORE = 1
    CMMM = 2
    CCS = 3
    IDWPT = 4
    GLOBAL = 5


@dataclass
class SnapshotHeader:
    magic: bytes = SNAPSHOT_MAGIC
    version: int = SNAPSHOT_VERSION
    endian: int = ENDIAN_MARKER
    chunk_count: int = 0
    global_crc32: int = 0

    def to_bytes(self) -> bytes:
        if self.magic != SNAPSHOT_MAGIC:
            raise ValueError(f'Invalid snapshot magic: {self.magic!r}')
        _integer(self.version, SNAPSHOT_VERSION, SNAPSHOT_VERSION, 'Snapshot version')
        _integer(self.endian, ENDIAN_MARKER, ENDIAN_MARKER, 'Endian marker')
        count = _integer(self.chunk_count, 0, 0xFFFF, 'Chunk count')
        crc = _integer(self.global_crc32, 0, 0xFFFFFFFF, 'Global CRC')
        return struct.pack('<4sHHHHI', self.magic, self.version, self.endian, count, 0, crc)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SnapshotHeader':
        data = _bytes(data)
        if len(data) != HEADER_SIZE:
            raise ValueError('Snapshot header must contain exactly 16 bytes')
        magic, version, endian, count, reserved, crc = struct.unpack('<4sHHHHI', data)
        if magic != SNAPSHOT_MAGIC:
            raise ValueError(f'Invalid snapshot magic: {magic!r}')
        if version != SNAPSHOT_VERSION:
            raise ValueError(f'Unsupported snapshot version: {version}')
        if endian != ENDIAN_MARKER:
            raise ValueError(f'Endian mismatch: {endian:04X}')
        if reserved:
            raise ValueError('Nonzero reserved snapshot header field')
        return cls(magic, version, endian, count, crc)


@dataclass
class SnapshotChunk:
    chunk_type: int
    chunk_version: int
    voice_id: int
    payload: bytes
    chunk_crc32: int = 0

    def _body(self):
        kind = _integer(self.chunk_type, 0, 0xFFFF, 'Chunk type')
        version = _integer(self.chunk_version, 0, 0xFFFF, 'Chunk version')
        voice = _integer(self.voice_id, 0, 0xFFFF, 'Chunk voice ID')
        payload = _bytes(self.payload)
        length = _integer(len(payload), 0, 0xFFFFFFFF, 'Chunk payload size')
        return struct.pack('<HHHHI', kind, version, voice, 0, length) + payload

    def compute_crc(self) -> int:
        return zlib.crc32(self._body()) & 0xFFFFFFFF

    def to_bytes(self) -> bytes:
        body = self._body()
        # A chunk is mutable. Cached checksums are observational, never input
        # to a later serialization after payload/version/identity edits.
        self.chunk_crc32 = zlib.crc32(body) & 0xFFFFFFFF
        return body + struct.pack('<I', self.chunk_crc32)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple['SnapshotChunk', int]:
        data = _bytes(data)
        offset = _integer(offset, 0, len(data), 'Chunk offset')
        start = offset
        if len(data) - start < CHUNK_HEADER_SIZE + CHUNK_CRC_SIZE:
            raise ValueError('Truncated snapshot chunk header or CRC')
        kind, version, voice, reserved, length = struct.unpack_from('<HHHHI', data, start)
        if reserved:
            raise ValueError('Nonzero reserved snapshot chunk field')
        payload_start = start + CHUNK_HEADER_SIZE
        crc_at = payload_start + length
        if crc_at + CHUNK_CRC_SIZE > len(data):
            raise ValueError('Truncated snapshot chunk payload or CRC')
        saved_crc = struct.unpack_from('<I', data, crc_at)[0]
        actual_crc = zlib.crc32(data[start:crc_at]) & 0xFFFFFFFF
        if saved_crc != actual_crc:
            raise ValueError(f'CRC mismatch in chunk: got 0x{saved_crc:08X}, expected 0x{actual_crc:08X}')
        try:
            kind = ChunkType(kind)
        except ValueError:
            pass  # The bundle loader skips unknown types after validating CRC.
        return cls(kind, version, voice, data[payload_start:crc_at], saved_crc), crc_at + CHUNK_CRC_SIZE


def _validate_records(chunks):
    seen = set()
    cores = {}
    for chunk in chunks:
        if not isinstance(chunk, SnapshotChunk):
            raise ValueError('Snapshot records must be SnapshotChunk instances')
        # Validate even records inserted directly through the public list.
        chunk._body()
        if chunk.chunk_type not in ChunkType._value2member_map_:
            continue
        supported = (0, 1) if chunk.chunk_type == ChunkType.CORE else (0,)
        if chunk.chunk_version not in supported:
            raise ValueError(f'Unsupported version {chunk.chunk_version} for chunk type {int(chunk.chunk_type)}')
        global_chunk = chunk.chunk_type == ChunkType.GLOBAL
        if global_chunk != (chunk.voice_id == VOICE_ID_GLOBAL):
            raise ValueError('Snapshot chunk has an invalid voice/global scope')
        key = (int(chunk.chunk_type), chunk.voice_id)
        if key in seen:
            raise ValueError(f'Duplicate snapshot chunk identity: {key}')
        seen.add(key)
        if chunk.chunk_type == ChunkType.CORE:
            state = vmstate_from_bytes(chunk.payload)
            if struct.unpack_from('<H', chunk.payload)[0] != chunk.chunk_version:
                raise ValueError('CORE payload version does not match its chunk')
            if state.voice_id != chunk.voice_id:
                raise ValueError('CORE payload voice ID does not match its chunk')
            cores[state.voice_id] = state
    return cores


def _voice_selection(voice_ids, available):
    if voice_ids is None:
        return sorted(_integer(vid, 0, 0xFFFE, 'Voice ID') for vid in available)
    selected = [_integer(vid, 0, 0xFFFE, 'Voice ID') for vid in voice_ids]
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate voice ID in snapshot selection')
    if not set(selected) <= set(available):
        raise ValueError('Selected voice ID is missing from snapshot source')
    return sorted(selected)


class SnapshotBundle:
    """WSES v0 envelope; known payload coverage is separate from framing."""
    def __init__(self):
        self.header = SnapshotHeader()
        self.chunks: List[SnapshotChunk] = []

    def _append(self, chunk):
        _validate_records([*self.chunks, chunk])
        self.chunks.append(chunk)

    def add_voice_state(self, voice_id: int, vm_state: VMState, *, version: int = VMSTATE_VERSION):
        """Add complete CORE v1, or request the partial legacy payload explicitly."""
        _integer(voice_id, 0, 0xFFFE, 'Voice ID')
        if voice_id != vm_state.voice_id:
            raise ValueError('CORE payload voice ID does not match its chunk')
        self._append(SnapshotChunk(ChunkType.CORE, version, voice_id, vmstate_to_bytes(vm_state, version=version)))

    def get_voice_state(self, voice_id: int) -> Optional[VMState]:
        _integer(voice_id, 0, 0xFFFE, 'Voice ID')
        return _validate_records(self.chunks).get(voice_id)

    @classmethod
    def from_executor(cls, executor, voice_ids=None):
        """Capture all VM voices, or an explicit subset, in canonical ID order."""
        selected = _voice_selection(voice_ids, executor.states)
        result = cls()
        for vid in selected:
            state = executor.states[vid]
            if state.voice_id != vid:
                raise ValueError('VM executor key and state voice ID disagree')
            result.chunks.append(SnapshotChunk(ChunkType.CORE, VMSTATE_VERSION, vid, vmstate_to_bytes(state)))
        _validate_records(result.chunks)
        return result

    def restore_executor(self, executor, voice_ids=None):
        """Atomically replace all VM voices, or merge an explicitly selected subset.

        This consumes complete CORE v1 only. Legacy subsets and other owners'
        chunks cannot silently stand in for a complete VM checkpoint.
        """
        states = _validate_records(self.chunks)
        if any(c.chunk_type != ChunkType.CORE or c.chunk_version != VMSTATE_VERSION for c in self.chunks):
            raise ValueError('VM executor restore requires complete CORE v1 records only')
        selected = _voice_selection(voice_ids, states)
        restored = {} if voice_ids is None else dict(executor.states)
        restored.update((vid, states[vid]) for vid in selected)
        executor.states = restored

    def add_global_state(self, payload: bytes):
        """Add an opaque GLOBAL v0 payload; its host schema is caller-owned."""
        self._append(SnapshotChunk(ChunkType.GLOBAL, 0, VOICE_ID_GLOBAL, _bytes(payload)))

    def get_global_state(self) -> Optional[bytes]:
        _validate_records(self.chunks)
        for chunk in self.chunks:
            if chunk.chunk_type == ChunkType.GLOBAL:
                return _bytes(chunk.payload)
        return None

    def list_voice_ids(self) -> List[int]:
        _validate_records(self.chunks)
        return sorted({chunk.voice_id for chunk in self.chunks
                       if chunk.chunk_type in ChunkType._value2member_map_
                       and chunk.voice_id != VOICE_ID_GLOBAL})

    def to_bytes(self) -> bytes:
        _validate_records(self.chunks)
        count = _integer(len(self.chunks), 0, 0xFFFF, 'Chunk count')
        # Validate header before changing its cached count/CRC.
        header = SnapshotHeader(self.header.magic, self.header.version, self.header.endian, count)
        header_for_crc = header.to_bytes()[:12]
        body = b''.join(chunk.to_bytes() for chunk in self.chunks)
        header.global_crc32 = zlib.crc32(header_for_crc + body) & 0xFFFFFFFF
        self.header = header
        return header.to_bytes() + body

    @classmethod
    def from_bytes(cls, data: bytes) -> 'SnapshotBundle':
        data = _bytes(data)
        header = SnapshotHeader.from_bytes(data[:HEADER_SIZE])
        actual_crc = zlib.crc32(data[:12] + data[16:]) & 0xFFFFFFFF
        if actual_crc != header.global_crc32:
            raise ValueError(f'Global CRC mismatch: got 0x{header.global_crc32:08X}, expected 0x{actual_crc:08X}')
        if header.chunk_count * (CHUNK_HEADER_SIZE + CHUNK_CRC_SIZE) > len(data) - HEADER_SIZE:
            raise ValueError('Chunk count exceeds the available snapshot data')
        offset, known = HEADER_SIZE, []
        for _ in range(header.chunk_count):
            chunk, offset = SnapshotChunk.from_bytes(data, offset)
            if chunk.chunk_type in ChunkType._value2member_map_:
                known.append(chunk)
        if offset != len(data):
            raise ValueError('Trailing snapshot data or incorrect chunk count')
        _validate_records(known)
        result = cls()
        result.header, result.chunks = header, known
        return result

    def save(self, filepath: str):
        data = self.to_bytes()
        with open(filepath, 'wb') as stream:
            stream.write(data)

    @classmethod
    def load(cls, filepath: str) -> 'SnapshotBundle':
        with open(filepath, 'rb') as stream:
            return cls.from_bytes(stream.read())


__all__ = ['ChunkType', 'SnapshotHeader', 'SnapshotChunk', 'SnapshotBundle',
           'SNAPSHOT_MAGIC', 'SNAPSHOT_VERSION', 'VOICE_ID_GLOBAL']
