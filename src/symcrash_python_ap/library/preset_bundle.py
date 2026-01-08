"""
Preset Bundle (WSE-STD-VOICEENGINEREGISTRY v0 §6)
==================================================

Program presets (capsules) for instrument identity and default overrides.

ProgramPreset contains:
- Instrument identity (program_id, regime_graph_id)
- Registry default overrides (GLOBAL/PROGRAM/PER_VOICE)
- Seed policy for determinism
- Metadata (ABI versions, sample rate, block size)

Does NOT contain runtime state (phase, envelope position, coeffs).

Bundle structure:
- manifest.json (metadata, file references, CRCs)
- voice_params.bin (packed overrides, sorted by persistence/domain/param_id)
- control_defaults.bin (optional, ControlRegistry overrides)
"""

import json
import struct
import zlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from pathlib import Path

from ..voice.voice_engine_registry import Persistence, get_registry


# =============================================================================
# PROGRAM PRESET
# =============================================================================

@dataclass
class ProgramPreset:
    """
    Program preset (capsule) per WSE-STD-VOICEENGINEREGISTRY v0 §6.

    Portable "instrument identity" + default overrides.
    """

    # Identity
    program_id: int
    name: str
    regime_graph_id: Optional[int] = None

    # Default overrides (param_id → value)
    global_overrides: Dict[int, int] = field(default_factory=dict)
    program_overrides: Dict[int, int] = field(default_factory=dict)
    per_voice_init_overrides: Dict[int, int] = field(default_factory=dict)

    # Seed policy
    master_seed: int = 0x12345678
    derivation_rule: str = "deterministic"

    # Metadata
    sample_rate: int = 48000
    block_size: int = 64
    control_registry_abi: int = 0
    voice_registry_abi: int = 0
    voiceframe_abi: int = 0

    def pack_voice_params(self) -> bytes:
        """
        Pack voice parameter overrides to binary (WSE-STD-VOICEENGINEREGISTRY §6.4).

        Format per entry:
            u8  persistence (0=PER_VOICE, 1=PROGRAM, 2=GLOBAL)
            u8  value_type
            u16 param_id
            i16 value

        Entries sorted by: (persistence asc, domain asc, param_id asc)
        """
        registry = get_registry()

        # Collect all overrides with metadata
        entries = []

        for param_id, value in self.global_overrides.items():
            entry = registry.get(param_id)
            if entry:
                entries.append((Persistence.GLOBAL, entry.domain, param_id, entry.value_type, value))

        for param_id, value in self.program_overrides.items():
            entry = registry.get(param_id)
            if entry:
                entries.append((Persistence.PROGRAM, entry.domain, param_id, entry.value_type, value))

        for param_id, value in self.per_voice_init_overrides.items():
            entry = registry.get(param_id)
            if entry:
                entries.append((Persistence.PER_VOICE, entry.domain, param_id, entry.value_type, value))

        # Sort by (persistence, domain, param_id)
        entries.sort(key=lambda e: (e[0], e[1], e[2]))

        # Pack
        packed = []
        for persistence, domain, param_id, value_type, value in entries:
            # Clamp to valid range
            value_clamped = registry.clamp(param_id, value)

            packed.append(struct.pack(
                '<BBHh',
                int(persistence),
                int(value_type),
                param_id,
                value_clamped,
            ))

        return b''.join(packed)

    @classmethod
    def unpack_voice_params(cls, data: bytes) -> Dict[str, Dict[int, int]]:
        """
        Unpack binary voice parameter overrides.

        Returns:
            Dict with keys: 'global', 'program', 'per_voice'
            Each maps param_id → value
        """
        overrides = {
            'global': {},
            'program': {},
            'per_voice': {},
        }

        offset = 0
        entry_size = 6  # u8 + u8 + u16 + i16

        while offset + entry_size <= len(data):
            persistence, value_type, param_id, value = struct.unpack_from('<BBHh', data, offset)
            offset += entry_size

            if persistence == Persistence.GLOBAL:
                overrides['global'][param_id] = value
            elif persistence == Persistence.PROGRAM:
                overrides['program'][param_id] = value
            elif persistence == Persistence.PER_VOICE:
                overrides['per_voice'][param_id] = value

        return overrides

    def save_bundle(self, dirpath: str):
        """
        Save preset as bundle directory.

        Creates:
        - manifest.json
        - voice_params.bin
        - control_defaults.bin (if needed)
        """
        bundle_dir = Path(dirpath)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Pack voice params
        voice_params_data = self.pack_voice_params()
        voice_params_path = bundle_dir / "voice_params.bin"
        with open(voice_params_path, 'wb') as f:
            f.write(voice_params_data)

        # Compute CRCs
        voice_params_crc = zlib.crc32(voice_params_data) & 0xFFFFFFFF

        # Create manifest
        manifest = {
            "format": "WSE_PRESET_V0",
            "preset_version": 0,
            "control_registry_abi": self.control_registry_abi,
            "voice_registry_abi": self.voice_registry_abi,
            "voiceframe_abi": self.voiceframe_abi,
            "sample_rate_hz": self.sample_rate,
            "block_size": self.block_size,
            "program_id": self.program_id,
            "name": self.name,
            "regime_graph_id": self.regime_graph_id,
            "seed": {
                "master_seed": self.master_seed,
                "derivation_rule": self.derivation_rule,
            },
            "files": {
                "voice_params": "voice_params.bin",
            },
            "crc": {
                "voice_params": f"0x{voice_params_crc:08X}",
            },
        }

        # Save manifest
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load_bundle(cls, dirpath: str) -> 'ProgramPreset':
        """
        Load preset from bundle directory.

        Args:
            dirpath: Path to bundle directory

        Returns:
            ProgramPreset instance

        Raises:
            FileNotFoundError, ValueError for malformed bundles
        """
        bundle_dir = Path(dirpath)

        # Load manifest
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Validate format
        if manifest.get("format") != "WSE_PRESET_V0":
            raise ValueError(f"Unsupported preset format: {manifest.get('format')}")

        # Load voice params
        voice_params_path = bundle_dir / manifest["files"]["voice_params"]
        with open(voice_params_path, 'rb') as f:
            voice_params_data = f.read()

        # Verify CRC
        computed_crc = zlib.crc32(voice_params_data) & 0xFFFFFFFF
        expected_crc_str = manifest["crc"]["voice_params"]
        expected_crc = int(expected_crc_str, 16)

        if computed_crc != expected_crc:
            raise ValueError(
                f"voice_params.bin CRC mismatch: got 0x{computed_crc:08X}, "
                f"expected {expected_crc_str}"
            )

        # Unpack overrides
        overrides = cls.unpack_voice_params(voice_params_data)

        # Extract seed info
        seed_info = manifest.get("seed", {})

        # Construct preset
        return cls(
            program_id=manifest["program_id"],
            name=manifest["name"],
            regime_graph_id=manifest.get("regime_graph_id"),
            global_overrides=overrides['global'],
            program_overrides=overrides['program'],
            per_voice_init_overrides=overrides['per_voice'],
            master_seed=seed_info.get("master_seed", 0x12345678),
            derivation_rule=seed_info.get("derivation_rule", "deterministic"),
            sample_rate=manifest["sample_rate_hz"],
            block_size=manifest["block_size"],
            control_registry_abi=manifest["control_registry_abi"],
            voice_registry_abi=manifest["voice_registry_abi"],
            voiceframe_abi=manifest["voiceframe_abi"],
        )


# =============================================================================
# PRESET LIBRARY HELPERS
# =============================================================================

def create_default_preset(program_id: int, name: str) -> ProgramPreset:
    """Create a basic preset with default values."""
    return ProgramPreset(
        program_id=program_id,
        name=name,
        global_overrides={},
        program_overrides={},
        per_voice_init_overrides={},
    )


def create_glitch_violin_preset() -> ProgramPreset:
    """Create default preset for Glitch Violin (program_id=1001)."""
    from .voice_engine_registry import VoiceParamID

    preset = ProgramPreset(
        program_id=1001,
        name="Glitch Violin",
        program_overrides={
            VoiceParamID.TREE_ID: 1,  # Harmonic tree
            VoiceParamID.OSC_MODE: 0,  # Mode A (LUT)
            VoiceParamID.H_REQUEST: 8,  # 8 harmonics
            VoiceParamID.TAIL_POLICY_HINT: 0,  # FULL tail
        },
    )
    return preset


__all__ = [
    'ProgramPreset',
    'create_default_preset',
    'create_glitch_violin_preset',
]
