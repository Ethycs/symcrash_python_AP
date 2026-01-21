"""
Capsule Binary Codec (WSE-STD-CAPSULE-COMPAT v0 §3-4)
======================================================

Binary serialization for .wsecap files.

File Format:
- Header: "WSEC" magic + version + kind + flags + chunk_count + CRC
- Chunks: identity, program, control_defaults, voice_params, assets, seeds
- Each chunk: type + version + length + payload + CRC

Packing follows spec §3.2 for control_defaults.bin and §4 for voice_params.bin.
"""

import struct
import json
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .capsule import (
    CapsuleV0, CapsuleKind, AssetType,
    CapsuleIdentity, EngineCompat, ProgramReference,
    CapsuleDefaults, AssetRef, SeedConfig,
)

# Import registry for value type lookup
from .central_registry import get_central_registry
from ..voice.voice_engine_registry import ValueType


# =============================================================================
# CONSTANTS
# =============================================================================

CAPSULE_MAGIC = b'WSEC'
CAPSULE_VERSION = 0


class ChunkType:
    """Chunk type identifiers."""
    IDENTITY = 1
    PROGRAM = 2
    CONTROL_DEFAULTS = 3
    VOICE_PARAMS = 4
    ASSETS = 5
    SEEDS = 6


# =============================================================================
# CONTROL DEFAULTS BINARY PACKING (§3.2)
# =============================================================================

def pack_control_defaults(defaults: Dict[int, int]) -> bytes:
    """
    Pack control defaults to binary per §3.2.

    Format:
        u16 count
        repeat count times:
            u8  value_type (always 0 for now)
            u8  control_id
            i16 value

    Sorted by control_id ascending.
    """
    if not defaults:
        return struct.pack('<H', 0)

    # Sort by control_id
    sorted_items = sorted(defaults.items(), key=lambda x: x[0])

    parts = [struct.pack('<H', len(sorted_items))]

    for control_id, value in sorted_items:
        parts.append(struct.pack(
            '<BBh',
            0,  # value_type (reserved)
            control_id & 0xFF,
            value,
        ))

    return b''.join(parts)


def unpack_control_defaults(data: bytes) -> Dict[int, int]:
    """
    Unpack control defaults from binary.

    Returns:
        Dict mapping control_id -> value
    """
    if len(data) < 2:
        return {}

    count, = struct.unpack_from('<H', data, 0)
    offset = 2
    defaults = {}

    for _ in range(count):
        if offset + 4 > len(data):
            break
        _, control_id, value = struct.unpack_from('<BBh', data, offset)
        offset += 4
        defaults[control_id] = value

    return defaults


# =============================================================================
# VOICE PARAMS BINARY PACKING (§4, same as VoiceEngineRegistry §6.4)
# =============================================================================

def pack_voice_params(defaults: Dict[int, int]) -> bytes:
    """
    Pack voice param defaults to binary per §4.

    Format per 16-bit entry (6 bytes):
        u8  persistence (0=PER_VOICE, 1=PROGRAM, 2=GLOBAL)
        u8  value_type
        u16 param_id
        i16/u16 value (signed or unsigned based on registry value_type)

    Format per 32-bit entry (8 bytes):
        u8  persistence
        u8  value_type (5 = U32)
        u16 param_id
        u32 value

    Sorted by param_id ascending. Uses VoiceEngineRegistry to determine
    whether each parameter should be packed as signed, unsigned 16-bit, or 32-bit.
    """
    if not defaults:
        return b''

    # Get registry for value type lookup
    registry = get_central_registry().voice_engine

    # Sort by param_id
    sorted_items = sorted(defaults.items(), key=lambda x: x[0])

    parts = []
    for param_id, value in sorted_items:
        # Look up value type from registry
        entry = registry.get(param_id)

        if entry is not None and entry.value_type == ValueType.U32:
            # 32-bit unsigned: 8-byte entry
            parts.append(struct.pack(
                '<BBHI',
                0,  # persistence (default PER_VOICE)
                int(ValueType.U32),
                param_id,
                value & 0xFFFFFFFF,  # Ensure u32 range
            ))
        elif entry is not None and entry.value_type in (ValueType.U16, ValueType.Q15U):
            # 16-bit unsigned: 6-byte entry
            parts.append(struct.pack(
                '<BBHH',
                0,  # persistence (default PER_VOICE)
                int(entry.value_type),
                param_id,
                value & 0xFFFF,  # Ensure u16 range
            ))
        else:
            # 16-bit signed: 6-byte entry (default for unknown params)
            value_type_id = int(entry.value_type) if entry else 0
            parts.append(struct.pack(
                '<BBHh',
                0,  # persistence (default PER_VOICE)
                value_type_id,
                param_id,
                value,
            ))

    return b''.join(parts)


def unpack_voice_params(data: bytes) -> Dict[int, int]:
    """
    Unpack voice param defaults from binary.

    Uses the value_type field stored in the binary to determine
    whether to interpret the value as signed, unsigned 16-bit, or 32-bit.

    Returns:
        Dict mapping param_id -> value
    """
    if not data:
        return {}

    offset = 0
    defaults = {}

    while offset + 4 <= len(data):  # At minimum need header (4 bytes)
        # Read header fields
        _, value_type, param_id = struct.unpack_from('<BBH', data, offset)

        # Determine entry size and value format from stored value_type
        if value_type == int(ValueType.U32):
            # 32-bit unsigned: 8-byte entry
            if offset + 8 > len(data):
                break
            value, = struct.unpack_from('<I', data, offset + 4)
            offset += 8
        elif value_type in (int(ValueType.U16), int(ValueType.Q15U)):
            # 16-bit unsigned: 6-byte entry
            if offset + 6 > len(data):
                break
            value, = struct.unpack_from('<H', data, offset + 4)
            offset += 6
        else:
            # 16-bit signed: 6-byte entry
            if offset + 6 > len(data):
                break
            value, = struct.unpack_from('<h', data, offset + 4)
            offset += 6

        defaults[param_id] = value

    return defaults


# =============================================================================
# CHUNK SERIALIZATION
# =============================================================================

def pack_chunk(chunk_type: int, payload: bytes) -> bytes:
    """
    Pack a single chunk.

    Format:
        u16 chunk_type
        u16 chunk_version (always 0)
        u32 payload_length
        bytes payload
        u32 crc32
    """
    header = struct.pack('<HHII', chunk_type, 0, len(payload), 0)
    crc = zlib.crc32(header[:8] + payload) & 0xFFFFFFFF

    return struct.pack('<HHI', chunk_type, 0, len(payload)) + payload + struct.pack('<I', crc)


def unpack_chunk(data: bytes, offset: int) -> Tuple[int, bytes, int]:
    """
    Unpack a single chunk.

    Returns:
        (chunk_type, payload, new_offset)

    Raises:
        ValueError if CRC mismatch
    """
    chunk_type, version, payload_len = struct.unpack_from('<HHI', data, offset)
    offset += 8

    payload = data[offset:offset + payload_len]
    offset += payload_len

    crc, = struct.unpack_from('<I', data, offset)
    offset += 4

    # Verify CRC
    header_for_crc = struct.pack('<HHI', chunk_type, version, payload_len)
    computed_crc = zlib.crc32(header_for_crc + payload) & 0xFFFFFFFF

    if computed_crc != crc:
        raise ValueError(f"CRC mismatch in chunk type {chunk_type}: got 0x{crc:08X}, expected 0x{computed_crc:08X}")

    return chunk_type, payload, offset


# =============================================================================
# IDENTITY CHUNK
# =============================================================================

def pack_identity(identity: CapsuleIdentity) -> bytes:
    """Pack identity as JSON payload."""
    data = {
        'capsule_id': identity.capsule_id,
        'kind': int(identity.kind),
        'name': identity.name,
        'author': identity.author,
        'tags': identity.tags,
        'engine_compat': {
            'control_registry_abi': identity.engine_compat.control_registry_abi,
            'voice_registry_abi': identity.engine_compat.voice_registry_abi,
            'voiceframe_abi': identity.engine_compat.voiceframe_abi,
            'sample_rate_hz': identity.engine_compat.sample_rate_hz,
            'block_size': identity.engine_compat.block_size,
        },
    }
    return json.dumps(data).encode('utf-8')


def unpack_identity(payload: bytes) -> CapsuleIdentity:
    """Unpack identity from JSON payload."""
    data = json.loads(payload.decode('utf-8'))

    compat_data = data.get('engine_compat', {})
    engine_compat = EngineCompat(
        control_registry_abi=compat_data.get('control_registry_abi', 0),
        voice_registry_abi=compat_data.get('voice_registry_abi', 0),
        voiceframe_abi=compat_data.get('voiceframe_abi', 0),
        sample_rate_hz=compat_data.get('sample_rate_hz', 48000),
        block_size=compat_data.get('block_size', 64),
    )

    return CapsuleIdentity(
        capsule_id=data['capsule_id'],
        kind=CapsuleKind(data['kind']),
        name=data['name'],
        author=data.get('author', ''),
        tags=data.get('tags', []),
        engine_compat=engine_compat,
    )


# =============================================================================
# PROGRAM CHUNK
# =============================================================================

def pack_program(program: ProgramReference) -> bytes:
    """Pack program reference as JSON payload."""
    data = {
        'program_id': program.program_id,
        'instrument_ir_id': program.instrument_ir_id,
        'regime_graph_id': program.regime_graph_id,
        'tree_family_id': program.tree_family_id,
    }
    return json.dumps(data).encode('utf-8')


def unpack_program(payload: bytes) -> ProgramReference:
    """Unpack program reference from JSON payload."""
    data = json.loads(payload.decode('utf-8'))

    return ProgramReference(
        program_id=data.get('program_id'),
        instrument_ir_id=data.get('instrument_ir_id'),
        regime_graph_id=data.get('regime_graph_id'),
        tree_family_id=data.get('tree_family_id'),
    )


# =============================================================================
# ASSETS CHUNK
# =============================================================================

def pack_assets(assets: List[AssetRef]) -> bytes:
    """Pack assets as JSON payload."""
    data = [
        {
            'asset_type': int(a.asset_type),
            'asset_id': a.asset_id,
            'asset_hash': a.asset_hash,
            'asset_path': a.asset_path,
        }
        for a in assets
    ]
    return json.dumps(data).encode('utf-8')


def unpack_assets(payload: bytes) -> List[AssetRef]:
    """Unpack assets from JSON payload."""
    data = json.loads(payload.decode('utf-8'))

    return [
        AssetRef(
            asset_type=AssetType(a['asset_type']),
            asset_id=a['asset_id'],
            asset_hash=a.get('asset_hash'),
            asset_path=a.get('asset_path'),
        )
        for a in data
    ]


# =============================================================================
# SEEDS CHUNK
# =============================================================================

def pack_seeds(seeds: SeedConfig) -> bytes:
    """Pack seeds as binary."""
    return struct.pack('<IH', seeds.master_seed, seeds.derivation_rule_id)


def unpack_seeds(payload: bytes) -> SeedConfig:
    """Unpack seeds from binary."""
    master_seed, derivation_rule_id = struct.unpack_from('<IH', payload, 0)
    return SeedConfig(master_seed=master_seed, derivation_rule_id=derivation_rule_id)


# =============================================================================
# FULL CAPSULE SERIALIZATION
# =============================================================================

def capsule_to_bytes(capsule: CapsuleV0) -> bytes:
    """
    Serialize complete capsule to binary (.wsecap format).

    Format:
        Header (16 bytes):
            u8[4] magic "WSEC"
            u16   version
            u8    kind
            u8    flags
            u16   chunk_count
            u16   reserved
            u32   global_crc32

        Chunks (variable):
            [identity, program, control_defaults, voice_params, assets, seeds]
    """
    chunks = []

    # Chunk 1: Identity (always present)
    chunks.append(pack_chunk(ChunkType.IDENTITY, pack_identity(capsule.identity)))

    # Chunk 2: Program (always present for INSTRUMENT/SYNTH)
    if capsule.identity.kind != CapsuleKind.ASSET:
        chunks.append(pack_chunk(ChunkType.PROGRAM, pack_program(capsule.program)))

    # Chunk 3: Control defaults (if any)
    if capsule.defaults.control_defaults:
        payload = pack_control_defaults(capsule.defaults.control_defaults)
        chunks.append(pack_chunk(ChunkType.CONTROL_DEFAULTS, payload))

    # Chunk 4: Voice params (if any)
    if capsule.defaults.voice_param_defaults:
        payload = pack_voice_params(capsule.defaults.voice_param_defaults)
        chunks.append(pack_chunk(ChunkType.VOICE_PARAMS, payload))

    # Chunk 5: Assets (if any)
    if capsule.assets:
        chunks.append(pack_chunk(ChunkType.ASSETS, pack_assets(capsule.assets)))

    # Chunk 6: Seeds
    chunks.append(pack_chunk(ChunkType.SEEDS, pack_seeds(capsule.seeds)))

    # Combine chunks
    chunk_bytes = b''.join(chunks)

    # Compute header CRC
    header_for_crc = struct.pack(
        '<4sHBBHH',
        CAPSULE_MAGIC,
        CAPSULE_VERSION,
        int(capsule.identity.kind),
        0,  # flags
        len(chunks),
        0,  # reserved
    )
    global_crc = zlib.crc32(header_for_crc + chunk_bytes) & 0xFFFFFFFF

    # Final header
    header = struct.pack(
        '<4sHBBHHI',
        CAPSULE_MAGIC,
        CAPSULE_VERSION,
        int(capsule.identity.kind),
        0,  # flags
        len(chunks),
        0,  # reserved
        global_crc,
    )

    return header + chunk_bytes


def capsule_from_bytes(data: bytes) -> CapsuleV0:
    """
    Deserialize capsule from binary (.wsecap format).

    Args:
        data: Binary capsule data

    Returns:
        CapsuleV0 instance

    Raises:
        ValueError if magic/version mismatch or CRC failure
    """
    # Read header (16 bytes)
    magic = data[:4]
    if magic != CAPSULE_MAGIC:
        raise ValueError(f"Invalid capsule magic: {magic}")

    version, kind, flags, chunk_count, reserved, global_crc = struct.unpack_from('<HBBHHI', data, 4)

    if version != CAPSULE_VERSION:
        raise ValueError(f"Unsupported capsule version: {version}")

    # Verify global CRC
    header_for_crc = data[:12]
    chunk_bytes = data[16:]
    computed_crc = zlib.crc32(header_for_crc + chunk_bytes) & 0xFFFFFFFF

    if computed_crc != global_crc:
        raise ValueError(f"Global CRC mismatch: got 0x{global_crc:08X}, expected 0x{computed_crc:08X}")

    # Parse chunks
    offset = 16
    identity = None
    program = ProgramReference()
    defaults = CapsuleDefaults()
    assets = []
    seeds = SeedConfig()

    for _ in range(chunk_count):
        chunk_type, payload, offset = unpack_chunk(data, offset)

        if chunk_type == ChunkType.IDENTITY:
            identity = unpack_identity(payload)
        elif chunk_type == ChunkType.PROGRAM:
            program = unpack_program(payload)
        elif chunk_type == ChunkType.CONTROL_DEFAULTS:
            defaults.control_defaults = unpack_control_defaults(payload)
        elif chunk_type == ChunkType.VOICE_PARAMS:
            defaults.voice_param_defaults = unpack_voice_params(payload)
        elif chunk_type == ChunkType.ASSETS:
            assets = unpack_assets(payload)
        elif chunk_type == ChunkType.SEEDS:
            seeds = unpack_seeds(payload)

    if identity is None:
        raise ValueError("Missing identity chunk")

    return CapsuleV0(
        identity=identity,
        program=program,
        defaults=defaults,
        assets=assets,
        seeds=seeds,
    )


# =============================================================================
# FILE I/O
# =============================================================================

def save_capsule(capsule: CapsuleV0, filepath: str):
    """Save capsule to .wsecap file."""
    data = capsule_to_bytes(capsule)
    with open(filepath, 'wb') as f:
        f.write(data)


def load_capsule(filepath: str) -> CapsuleV0:
    """Load capsule from .wsecap file."""
    with open(filepath, 'rb') as f:
        data = f.read()
    return capsule_from_bytes(data)


# =============================================================================
# BUNDLE DIRECTORY FORMAT (Alternative)
# =============================================================================

def save_capsule_bundle(capsule: CapsuleV0, dirpath: str):
    """
    Save capsule as a bundle directory.

    Creates:
    - manifest.json
    - control_defaults.bin (if any)
    - voice_params.bin (if any)
    """
    bundle_dir = Path(dirpath)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Save control_defaults.bin
    control_defaults_crc = None
    if capsule.defaults.control_defaults:
        data = pack_control_defaults(capsule.defaults.control_defaults)
        control_defaults_crc = zlib.crc32(data) & 0xFFFFFFFF
        with open(bundle_dir / "control_defaults.bin", 'wb') as f:
            f.write(data)

    # Save voice_params.bin
    voice_params_crc = None
    if capsule.defaults.voice_param_defaults:
        data = pack_voice_params(capsule.defaults.voice_param_defaults)
        voice_params_crc = zlib.crc32(data) & 0xFFFFFFFF
        with open(bundle_dir / "voice_params.bin", 'wb') as f:
            f.write(data)

    # Build manifest
    manifest = {
        'format': 'WSE_CAPSULE_V0',
        'capsule_version': CAPSULE_VERSION,
        'identity': {
            'capsule_id': capsule.identity.capsule_id,
            'kind': capsule.identity.kind.name,
            'name': capsule.identity.name,
            'author': capsule.identity.author,
            'tags': capsule.identity.tags,
        },
        'engine_compat': {
            'control_registry_abi': capsule.identity.engine_compat.control_registry_abi,
            'voice_registry_abi': capsule.identity.engine_compat.voice_registry_abi,
            'voiceframe_abi': capsule.identity.engine_compat.voiceframe_abi,
            'sample_rate_hz': capsule.identity.engine_compat.sample_rate_hz,
            'block_size': capsule.identity.engine_compat.block_size,
        },
        'program': {
            'program_id': capsule.program.program_id,
            'instrument_ir_id': capsule.program.instrument_ir_id,
            'regime_graph_id': capsule.program.regime_graph_id,
            'tree_family_id': capsule.program.tree_family_id,
        },
        'seeds': {
            'master_seed': capsule.seeds.master_seed,
            'derivation_rule_id': capsule.seeds.derivation_rule_id,
        },
        'files': {},
        'crc': {},
    }

    if control_defaults_crc is not None:
        manifest['files']['control_defaults'] = 'control_defaults.bin'
        manifest['crc']['control_defaults'] = f'0x{control_defaults_crc:08X}'

    if voice_params_crc is not None:
        manifest['files']['voice_params'] = 'voice_params.bin'
        manifest['crc']['voice_params'] = f'0x{voice_params_crc:08X}'

    if capsule.assets:
        manifest['assets'] = [
            {
                'type': a.asset_type.name,
                'id': a.asset_id,
                'hash': a.asset_hash,
                'path': a.asset_path,
            }
            for a in capsule.assets
        ]

    with open(bundle_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)


def load_capsule_bundle(dirpath: str) -> CapsuleV0:
    """Load capsule from bundle directory."""
    bundle_dir = Path(dirpath)

    # Load manifest
    with open(bundle_dir / "manifest.json", 'r') as f:
        manifest = json.load(f)

    if manifest.get('format') != 'WSE_CAPSULE_V0':
        raise ValueError(f"Unsupported format: {manifest.get('format')}")

    # Parse identity
    ident = manifest['identity']
    compat = manifest.get('engine_compat', {})
    identity = CapsuleIdentity(
        capsule_id=ident['capsule_id'],
        kind=CapsuleKind[ident['kind']],
        name=ident['name'],
        author=ident.get('author', ''),
        tags=ident.get('tags', []),
        engine_compat=EngineCompat(
            control_registry_abi=compat.get('control_registry_abi', 0),
            voice_registry_abi=compat.get('voice_registry_abi', 0),
            voiceframe_abi=compat.get('voiceframe_abi', 0),
            sample_rate_hz=compat.get('sample_rate_hz', 48000),
            block_size=compat.get('block_size', 64),
        ),
    )

    # Parse program
    prog = manifest.get('program', {})
    program = ProgramReference(
        program_id=prog.get('program_id'),
        instrument_ir_id=prog.get('instrument_ir_id'),
        regime_graph_id=prog.get('regime_graph_id'),
        tree_family_id=prog.get('tree_family_id'),
    )

    # Parse seeds
    seed_data = manifest.get('seeds', {})
    seeds = SeedConfig(
        master_seed=seed_data.get('master_seed', 0x12345678),
        derivation_rule_id=seed_data.get('derivation_rule_id', 0),
    )

    # Load control_defaults.bin
    control_defaults = {}
    files = manifest.get('files', {})
    crcs = manifest.get('crc', {})

    if 'control_defaults' in files:
        path = bundle_dir / files['control_defaults']
        with open(path, 'rb') as f:
            data = f.read()

        # Verify CRC if present
        if 'control_defaults' in crcs:
            expected_crc = int(crcs['control_defaults'], 16)
            computed_crc = zlib.crc32(data) & 0xFFFFFFFF
            if computed_crc != expected_crc:
                raise ValueError(f"control_defaults CRC mismatch")

        control_defaults = unpack_control_defaults(data)

    # Load voice_params.bin
    voice_param_defaults = {}
    if 'voice_params' in files:
        path = bundle_dir / files['voice_params']
        with open(path, 'rb') as f:
            data = f.read()

        if 'voice_params' in crcs:
            expected_crc = int(crcs['voice_params'], 16)
            computed_crc = zlib.crc32(data) & 0xFFFFFFFF
            if computed_crc != expected_crc:
                raise ValueError(f"voice_params CRC mismatch")

        voice_param_defaults = unpack_voice_params(data)

    defaults = CapsuleDefaults(
        control_defaults=control_defaults,
        voice_param_defaults=voice_param_defaults,
    )

    # Parse assets
    assets = []
    for a in manifest.get('assets', []):
        assets.append(AssetRef(
            asset_type=AssetType[a['type']],
            asset_id=a['id'],
            asset_hash=a.get('hash'),
            asset_path=a.get('path'),
        ))

    return CapsuleV0(
        identity=identity,
        program=program,
        defaults=defaults,
        assets=assets,
        seeds=seeds,
    )


__all__ = [
    'CAPSULE_MAGIC',
    'CAPSULE_VERSION',
    'pack_control_defaults',
    'unpack_control_defaults',
    'pack_voice_params',
    'unpack_voice_params',
    'capsule_to_bytes',
    'capsule_from_bytes',
    'save_capsule',
    'load_capsule',
    'save_capsule_bundle',
    'load_capsule_bundle',
]
