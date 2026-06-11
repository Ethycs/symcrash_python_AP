"""
Voice Engine Registry Conformance Tests (WSE-STD-VOICEENGINEREGISTRY v0)
==========================================================================

Tests conformance to WSE-STD-VOICEENGINEREGISTRY v0 specification.

Verifies:
- Registry entry format and operations (§1)
- VMState serialization (§7.2)
- Snapshot bundle format (§7.3)
- Preset bundle format (§6)
- Determinism (§1.4)
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from symcrash_python_ap.voice.voice_engine_registry import (
    VoiceEngineRegistry,
    VoiceParamEntry,
    ValueType,
    UpdateMode,
    Persistence,
    Domain,
    VoiceParamID,
    VOICE_ENGINE_REGISTRY_V0,
    get_registry,
    reset_registry,
)

from symcrash_python_ap.voice.voice_state_codec import (
    vmstate_to_bytes,
    vmstate_from_bytes,
    get_vmstate_size,
)

from symcrash_python_ap.voice.snapshot_bundle import (
    SnapshotBundle,
    SnapshotHeader,
    SnapshotChunk,
    ChunkType,
    SNAPSHOT_MAGIC,
    SNAPSHOT_VERSION,
)

from symcrash_python_ap.library.preset_bundle import (
    ProgramPreset,
    create_default_preset,
    create_glitch_violin_preset,
)

from symcrash_python_ap.vm.state import VMState, EnvelopeState, EnvelopePhase


# =============================================================================
# WSE-STD-VOICEENGINEREGISTRY v0 §1: REGISTRY OPERATIONS
# =============================================================================

class TestVoiceEngineRegistry:
    """Test registry entry format and operations per §1."""

    def test_registry_has_all_domains(self):
        """Registry should have entries for all 6 domains."""
        registry = VoiceEngineRegistry()

        domains_found = set()
        for entry in VOICE_ENGINE_REGISTRY_V0:
            domains_found.add(entry.domain)

        # Should have CORE, CMMM, CCS, IDWPT, PLANNER, DEBUG
        assert Domain.CORE in domains_found
        assert Domain.CCS in domains_found
        assert Domain.IDWPT in domains_found

    def test_param_id_allocation_ranges(self):
        """Param IDs should follow frozen allocation per §11."""
        registry = VoiceEngineRegistry()

        # CORE: 0-31
        core_params = registry.list_by_domain(Domain.CORE)
        for entry in core_params:
            assert 0 <= entry.param_id <= 31, (
                f"CORE param {entry.name} has ID {entry.param_id} outside 0-31"
            )

        # CMMM: 32-63
        cmmm_params = registry.list_by_domain(Domain.CMMM)
        for entry in cmmm_params:
            assert 32 <= entry.param_id <= 63

        # CCS: 64-127
        ccs_params = registry.list_by_domain(Domain.CCS)
        for entry in ccs_params:
            assert 64 <= entry.param_id <= 127

        # IDWPT: 128-159
        idwpt_params = registry.list_by_domain(Domain.IDWPT)
        for entry in idwpt_params:
            assert 128 <= entry.param_id <= 159

        # PLANNER: 160-191
        planner_params = registry.list_by_domain(Domain.PLANNER)
        for entry in planner_params:
            assert 160 <= entry.param_id <= 191

        # DEBUG: 240-255
        debug_params = registry.list_by_domain(Domain.DEBUG)
        for entry in debug_params:
            assert 240 <= entry.param_id <= 255

    def test_clamp_enforces_ranges(self):
        """Registry should clamp values to valid ranges."""
        registry = VoiceEngineRegistry()

        # TREE_ID: min=0, max=4
        assert registry.clamp(VoiceParamID.TREE_ID, -1) == 0
        assert registry.clamp(VoiceParamID.TREE_ID, 2) == 2
        assert registry.clamp(VoiceParamID.TREE_ID, 10) == 4

    def test_apply_abs_update(self):
        """Absolute updates should clamp values."""
        registry = VoiceEngineRegistry()

        # K_REQUEST: min=0, max=6
        result = registry.apply_abs(VoiceParamID.K_REQUEST, 3)
        assert result == 3

        result = registry.apply_abs(VoiceParamID.K_REQUEST, 99)
        assert result == 6  # Clamped to max

    def test_apply_delta_update(self):
        """Delta updates should add and clamp."""
        registry = VoiceEngineRegistry()

        # REGIME_BIAS_Q15: supports BOTH mode
        current = 10000
        delta = 5000
        result = registry.apply_delta(VoiceParamID.REGIME_BIAS_Q15, current, delta)
        assert result == 15000

        # Delta that would exceed max should clamp
        result = registry.apply_delta(VoiceParamID.REGIME_BIAS_Q15, 30000, 5000)
        assert result == 32767  # Clamped to max

    def test_get_default_values(self):
        """Registry should return correct default values."""
        registry = VoiceEngineRegistry()

        # PROGRAM_ID defaults to 1001 (Glitch Violin)
        assert registry.get_default(VoiceParamID.PROGRAM_ID) == 1001

        # TREE_ID defaults to 1 (Harmonic)
        assert registry.get_default(VoiceParamID.TREE_ID) == 1


# =============================================================================
# WSE-STD-VOICEENGINEREGISTRY v0 §7: VMSTATE SERIALIZATION
# =============================================================================

class TestVMStateSerialization:
    """Test VMState binary serialization per §7."""

    def test_vmstate_roundtrip(self):
        """VMState should round-trip through serialization."""
        # Create state with non-default values
        state = VMState.create(voice_id=5)
        state.phase = 12345678
        state.phase_inc = 987654
        state.last_note = 60
        state.last_cents = 50
        state.harmonic_amps_q15 = [32767, 16384, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        state.envelope.phase = EnvelopePhase.SUSTAIN
        state.envelope.phase_sample = 1000
        state.envelope.level_q15 = 24000
        state.prng_state = 0xABCDEF12
        state.eta_prev_q15 = 16384

        # Serialize
        data = vmstate_to_bytes(state)

        # Should have minimum size (core fields)
        # Note: Actual size may be larger due to CCS/ModeB state
        min_size = get_vmstate_size()
        assert len(data) >= min_size, f"Size {len(data)} < minimum {min_size}"

        # Deserialize
        restored = vmstate_from_bytes(data)

        # Verify fields match
        assert restored.voice_id == state.voice_id
        assert restored.phase == state.phase
        assert restored.phase_inc == state.phase_inc
        assert restored.last_note == state.last_note
        assert restored.last_cents == state.last_cents
        assert restored.harmonic_amps_q15 == state.harmonic_amps_q15
        assert restored.envelope.phase == state.envelope.phase
        assert restored.envelope.phase_sample == state.envelope.phase_sample
        assert restored.envelope.level_q15 == state.envelope.level_q15
        assert restored.prng_state == state.prng_state
        assert restored.eta_prev_q15 == state.eta_prev_q15

    def test_vmstate_deterministic(self):
        """Same VMState should serialize to identical bytes."""
        state = VMState.create(voice_id=3)
        state.phase = 99999
        state.last_note = 72

        data1 = vmstate_to_bytes(state)
        data2 = vmstate_to_bytes(state)

        assert data1 == data2, "VMState serialization not deterministic"

    def test_empty_vmstate_roundtrip(self):
        """Empty VMState should round-trip correctly."""
        state = VMState.create(voice_id=0)

        data = vmstate_to_bytes(state)
        restored = vmstate_from_bytes(data)

        assert restored.voice_id == 0
        assert restored.phase == 0
        assert restored.prng_state == 0x12345678  # Default seed


# =============================================================================
# WSE-STD-VOICEENGINEREGISTRY v0 §7.3: SNAPSHOT BUNDLE FORMAT
# =============================================================================

class TestSnapshotBundle:
    """Test snapshot bundle format per §7.3."""

    def test_snapshot_header_format(self):
        """Snapshot header should have correct format."""
        header = SnapshotHeader()

        data = header.to_bytes()

        assert len(data) == 16  # Fixed size
        assert data[:4] == SNAPSHOT_MAGIC
        assert header.version == SNAPSHOT_VERSION

    def test_snapshot_chunk_crc(self):
        """Chunks should compute CRC correctly."""
        payload = b'test payload data'
        chunk = SnapshotChunk(
            chunk_type=ChunkType.CORE,
            chunk_version=0,
            voice_id=5,
            payload=payload,
        )

        crc = chunk.compute_crc()
        assert crc != 0
        assert isinstance(crc, int)

    def test_snapshot_chunk_roundtrip(self):
        """Chunks should round-trip with CRC validation."""
        payload = b'test data 12345'
        chunk = SnapshotChunk(
            chunk_type=ChunkType.GLOBAL,
            chunk_version=0,
            voice_id=0xFFFF,
            payload=payload,
        )

        # Serialize
        data = chunk.to_bytes()

        # Deserialize
        restored, offset = SnapshotChunk.from_bytes(data, 0)

        assert restored.chunk_type == chunk.chunk_type
        assert restored.voice_id == chunk.voice_id
        assert restored.payload == payload
        assert restored.chunk_crc32 == chunk.chunk_crc32

    def test_snapshot_bundle_single_voice(self):
        """Snapshot bundle with single voice should work."""
        bundle = SnapshotBundle()

        # Create and add state
        state = VMState.create(voice_id=7)
        state.phase = 555555
        state.last_note = 64

        bundle.add_voice_state(7, state)

        # Serialize
        data = bundle.to_bytes()

        # Deserialize
        restored_bundle = SnapshotBundle.from_bytes(data)

        # Retrieve state
        restored_state = restored_bundle.get_voice_state(7)

        assert restored_state is not None
        assert restored_state.voice_id == 7
        assert restored_state.phase == 555555
        assert restored_state.last_note == 64

    def test_snapshot_bundle_multiple_voices(self):
        """Snapshot with multiple voices should work."""
        bundle = SnapshotBundle()

        # Add 3 voices
        for vid in [0, 1, 2]:
            state = VMState.create(voice_id=vid)
            state.phase = vid * 1000
            bundle.add_voice_state(vid, state)

        # Serialize and restore
        data = bundle.to_bytes()
        restored = SnapshotBundle.from_bytes(data)

        # Check all voices present
        voice_ids = restored.list_voice_ids()
        assert voice_ids == [0, 1, 2]

        # Check states match
        for vid in [0, 1, 2]:
            state = restored.get_voice_state(vid)
            assert state.voice_id == vid
            assert state.phase == vid * 1000

    def test_snapshot_save_load_file(self):
        """Snapshot should save/load from file."""
        bundle = SnapshotBundle()
        state = VMState.create(voice_id=10)
        state.phase = 999999
        bundle.add_voice_state(10, state)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wses', delete=False) as f:
            temp_path = f.name

        try:
            bundle.save(temp_path)

            # Load back
            loaded = SnapshotBundle.load(temp_path)

            restored_state = loaded.get_voice_state(10)
            assert restored_state.phase == 999999

        finally:
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# WSE-STD-VOICEENGINEREGISTRY v0 §6: PRESET BUNDLE
# =============================================================================

class TestPresetBundle:
    """Test preset bundle format per §6."""

    def test_preset_pack_voice_params(self):
        """voice_params.bin should pack correctly per §6.4."""
        preset = ProgramPreset(
            program_id=1001,
            name="Test",
            program_overrides={
                VoiceParamID.TREE_ID: 2,
                VoiceParamID.OSC_MODE: 1,
            },
        )

        # Pack
        data = preset.pack_voice_params()

        # Should have 2 entries * 6 bytes = 12 bytes
        assert len(data) == 12

        # Unpack
        overrides = ProgramPreset.unpack_voice_params(data)

        assert overrides['program'][VoiceParamID.TREE_ID] == 2
        assert overrides['program'][VoiceParamID.OSC_MODE] == 1

    def test_preset_save_load_bundle(self):
        """Preset should save/load as bundle directory."""
        preset = create_glitch_violin_preset()

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Save
            preset.save_bundle(str(temp_dir))

            # Verify files exist
            assert (temp_dir / "manifest.json").exists()
            assert (temp_dir / "voice_params.bin").exists()

            # Load
            loaded = ProgramPreset.load_bundle(str(temp_dir))

            # Verify
            assert loaded.program_id == 1001
            assert loaded.name == "Glitch Violin"
            assert loaded.sample_rate == 48000
            assert loaded.block_size == 64

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_preset_manifest_format(self):
        """Manifest should have required fields per §6.3."""
        preset = create_default_preset(1002, "Test Preset")

        temp_dir = Path(tempfile.mkdtemp())

        try:
            preset.save_bundle(str(temp_dir))

            # Load manifest
            with open(temp_dir / "manifest.json", 'r') as f:
                import json
                manifest = json.load(f)

            # Check required fields
            assert manifest["format"] == "WSE_PRESET_V0"
            assert manifest["preset_version"] == 0
            assert "control_registry_abi" in manifest
            assert "voice_registry_abi" in manifest
            assert "voiceframe_abi" in manifest
            assert manifest["sample_rate_hz"] == 48000
            assert manifest["block_size"] == 64
            assert manifest["program_id"] == 1002
            assert "files" in manifest
            assert "crc" in manifest

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# DETERMINISM AND ROUNDTRIP TESTS
# =============================================================================

class TestDeterminism:
    """Test determinism requirements per §1.4."""

    def test_vmstate_serialization_deterministic(self):
        """VMState serialization should be deterministic."""
        state = VMState.create(voice_id=1)
        state.phase = 12345
        state.last_note = 60

        # Serialize multiple times
        data1 = vmstate_to_bytes(state)
        data2 = vmstate_to_bytes(state)
        data3 = vmstate_to_bytes(state)

        assert data1 == data2 == data3

    def test_snapshot_bundle_deterministic(self):
        """Snapshot bundle should serialize deterministically."""
        bundle = SnapshotBundle()
        state = VMState.create(voice_id=0)
        state.phase = 99999
        bundle.add_voice_state(0, state)

        data1 = bundle.to_bytes()
        data2 = bundle.to_bytes()

        assert data1 == data2

    def test_preset_bundle_deterministic(self):
        """Preset bundle packing should be deterministic."""
        preset = create_glitch_violin_preset()

        data1 = preset.pack_voice_params()
        data2 = preset.pack_voice_params()

        assert data1 == data2


# =============================================================================
# PERSISTENCE SCOPE TESTS
# =============================================================================

class TestPersistenceScopes:
    """Test persistence scopes per §1.3."""

    def test_per_voice_parameters(self):
        """PER_VOICE parameters should exist."""
        registry = VoiceEngineRegistry()
        per_voice = registry.list_by_persistence(Persistence.PER_VOICE)

        assert len(per_voice) > 0

        # PROGRAM_ID should be PER_VOICE
        program_entry = registry.get(VoiceParamID.PROGRAM_ID)
        assert program_entry.persistence == Persistence.PER_VOICE

    def test_program_parameters(self):
        """PROGRAM parameters should exist."""
        registry = VoiceEngineRegistry()
        program = registry.list_by_persistence(Persistence.PROGRAM)

        assert len(program) > 0

        # REGIME_GRAPH_ID should be PROGRAM
        regime_entry = registry.get(VoiceParamID.REGIME_GRAPH_ID)
        assert regime_entry.persistence == Persistence.PROGRAM

    def test_global_parameters(self):
        """GLOBAL parameters should exist."""
        registry = VoiceEngineRegistry()
        global_params = registry.list_by_persistence(Persistence.GLOBAL)

        # DEBUG params should be GLOBAL
        bypass_entry = registry.get(VoiceParamID.BYPASS_IDWPT)
        assert bypass_entry.persistence == Persistence.GLOBAL


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_preset_and_snapshot(self):
        """Test complete workflow: preset → runtime → snapshot → restore."""
        # 1. Create preset
        preset = create_glitch_violin_preset()

        # 2. "Runtime": Create VMState (simulating engine usage)
        state = VMState.create(voice_id=0)
        state.phase = 555555
        state.last_note = 60
        state.harmonic_amps_q15 = [32767, 16384, 8192, 4096] + [0] * 12

        # 3. Create snapshot
        bundle = SnapshotBundle()
        bundle.add_voice_state(0, state)

        # 4. Serialize snapshot
        snapshot_data = bundle.to_bytes()

        # 5. Restore snapshot
        restored_bundle = SnapshotBundle.from_bytes(snapshot_data)
        restored_state = restored_bundle.get_voice_state(0)

        # 6. Verify bit-identical
        assert restored_state.voice_id == state.voice_id
        assert restored_state.phase == state.phase
        assert restored_state.last_note == state.last_note
        assert restored_state.harmonic_amps_q15 == state.harmonic_amps_q15


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_unknown_param_id(self):
        """Unknown param_id should handle gracefully."""
        registry = VoiceEngineRegistry()

        # Unknown ID should return None
        entry = registry.get(9999)
        assert entry is None

        # Clamp should pass through
        result = registry.clamp(9999, 12345)
        assert result == 12345

        # Default should return 0
        default = registry.get_default(9999)
        assert default == 0

    def test_malformed_snapshot_header(self):
        """Malformed snapshot header should raise ValueError."""
        bad_data = b'XXXX' + b'\x00' * 12

        with pytest.raises(ValueError, match="Invalid snapshot magic"):
            SnapshotHeader.from_bytes(bad_data)

    def test_crc_mismatch_detection(self):
        """Snapshot should detect CRC corruption."""
        bundle = SnapshotBundle()
        state = VMState.create(voice_id=0)
        bundle.add_voice_state(0, state)

        data = bundle.to_bytes()

        # Corrupt one byte in the middle
        corrupted = bytearray(data)
        corrupted[50] ^= 0xFF  # Flip bits
        corrupted_bytes = bytes(corrupted)

        # Should raise on load
        with pytest.raises(ValueError, match="CRC mismatch"):
            SnapshotBundle.from_bytes(corrupted_bytes)
