"""
WSE-STD-CAPSULE-COMPAT v0 Conformance Tests
=============================================

Tests capsule compatibility contract implementation.

Conformance Checklist (§8):
- [ ] Capsule bundle includes compat ABI versions for ControlRegistry, VoiceEngineRegistry, VoiceFrame
- [ ] Capsule defaults override only default values (never ranges/types/slew/bindings)
- [ ] Unknown IDs are ignored deterministically unless strict mode
- [ ] Capsule apply algorithm implemented exactly (§6)
- [ ] Clear ownership: cockpit writes VoiceFrame; voice engine reads VoiceFrame and applies engine params (§5.3)
"""

import os
import tempfile
import pytest

from symcrash_python_ap.library import (
    CapsuleKind,
    AssetType,
    EngineCompat,
    CapsuleIdentity,
    ProgramReference,
    AssetRef,
    SeedConfig,
    CapsuleDefaults,
    CapsuleV0,
    create_instrument_capsule,
    create_synth_capsule,
    create_asset_capsule,
    CAPSULE_MAGIC,
    CAPSULE_VERSION,
    pack_control_defaults,
    unpack_control_defaults,
    pack_voice_params,
    unpack_voice_params,
    capsule_to_bytes,
    capsule_from_bytes,
    save_capsule,
    load_capsule,
    save_capsule_bundle,
    load_capsule_bundle,
    CompatResult,
    ApplyResult,
    check_compatibility,
    apply_control_defaults,
    apply_voice_param_defaults,
    CapsuleLoader,
    apply_capsule_to_registries,
)
from symcrash_python_ap.control_registry import ControlRegistry, ControlID
from symcrash_python_ap.voice.voice_engine_registry import (
    VoiceEngineRegistry, VoiceParamID, get_registry
)


# =============================================================================
# §2: CAPSULE OBJECT MODEL
# =============================================================================

class TestCapsuleObjectModel:
    """Test CapsuleV0 data model per §2."""

    def test_capsule_kinds(self):
        """Three capsule kinds: INSTRUMENT, SYNTH, ASSET."""
        assert CapsuleKind.INSTRUMENT == 0
        assert CapsuleKind.SYNTH == 1
        assert CapsuleKind.ASSET == 2

    def test_asset_types(self):
        """Asset reference types per §2."""
        assert AssetType.IR == 0
        assert AssetType.WAVETABLE == 1
        assert AssetType.MATERIAL == 2
        assert AssetType.TABLE_ATLAS == 3
        assert AssetType.OTHER == 255

    def test_engine_compat_default_abis(self):
        """EngineCompat defaults to ABI v0."""
        compat = EngineCompat()
        assert compat.control_registry_abi == 0
        assert compat.voice_registry_abi == 0
        assert compat.voiceframe_abi == 0
        assert compat.sample_rate_hz == 48000
        assert compat.block_size == 64

    def test_capsule_identity_requires_fields(self):
        """CapsuleIdentity requires capsule_id and name."""
        identity = CapsuleIdentity(
            capsule_id="test-001",
            kind=CapsuleKind.INSTRUMENT,
            name="Test Instrument",
        )
        assert identity.capsule_id == "test-001"
        assert identity.name == "Test Instrument"

    def test_program_reference_program_id(self):
        """ProgramReference can use program_id."""
        prog = ProgramReference(program_id=1001)
        assert prog.program_id == 1001
        assert prog.instrument_ir_id is None

    def test_program_reference_ir_id(self):
        """ProgramReference can use instrument_ir_id."""
        prog = ProgramReference(instrument_ir_id="ir-violin-001")
        assert prog.program_id is None
        assert prog.instrument_ir_id == "ir-violin-001"

    def test_capsule_validation_requires_program(self):
        """Capsule validation requires program_id or instrument_ir_id."""
        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id="test",
                kind=CapsuleKind.INSTRUMENT,
                name="Test",
            ),
            program=ProgramReference(),  # Neither set
        )
        issues = capsule.validate()
        assert any("program_id" in i or "instrument_ir_id" in i for i in issues)

    def test_capsule_must_not_contain_runtime_histories(self):
        """Capsule MUST NOT contain runtime histories (§2)."""
        # CapsuleDefaults only contains default overrides, not runtime state
        defaults = CapsuleDefaults()
        assert not hasattr(defaults, 'coeff_state')
        assert not hasattr(defaults, 'phase')
        assert not hasattr(defaults, 'prng_state')
        assert not hasattr(defaults, 'adsr_state')


class TestCapsuleFactories:
    """Test factory functions."""

    def test_create_instrument_capsule(self):
        """create_instrument_capsule creates INSTRUMENT kind."""
        capsule = create_instrument_capsule(
            capsule_id="violin-001",
            name="Violin",
            program_id=1001,
            author="Test Author",
            tags=["strings", "classical"],
        )
        assert capsule.identity.kind == CapsuleKind.INSTRUMENT
        assert capsule.identity.capsule_id == "violin-001"
        assert capsule.program.program_id == 1001
        assert capsule.is_instrument

    def test_create_synth_capsule(self):
        """create_synth_capsule creates SYNTH kind."""
        capsule = create_synth_capsule(
            capsule_id="pad-001",
            name="Pad",
            program_id=1002,
        )
        assert capsule.identity.kind == CapsuleKind.SYNTH
        assert capsule.is_synth

    def test_create_asset_capsule(self):
        """create_asset_capsule creates ASSET kind."""
        assets = [
            AssetRef(asset_type=AssetType.IR, asset_id="ir-001"),
            AssetRef(asset_type=AssetType.WAVETABLE, asset_id="wt-001"),
        ]
        capsule = create_asset_capsule(
            capsule_id="asset-pack-001",
            name="Asset Pack",
            assets=assets,
        )
        assert capsule.identity.kind == CapsuleKind.ASSET
        assert capsule.is_asset
        assert len(capsule.assets) == 2


# =============================================================================
# §3: CONTROL DEFAULTS PACKING
# =============================================================================

class TestControlDefaultsPacking:
    """Test control_defaults.bin packing per §3.2."""

    def test_pack_empty_defaults(self):
        """Empty defaults pack to count=0."""
        data = pack_control_defaults({})
        assert len(data) == 2
        assert data == b'\x00\x00'

    def test_pack_single_default(self):
        """Single default packs correctly."""
        defaults = {0: 16384}  # TIMBRE = 50%
        data = pack_control_defaults(defaults)

        # count (2) + entry (4) = 6 bytes
        assert len(data) == 6

        # Unpack
        restored = unpack_control_defaults(data)
        assert restored == defaults

    def test_pack_multiple_defaults_sorted(self):
        """Multiple defaults sorted by control_id ascending."""
        defaults = {5: 100, 0: 16384, 16: 8}  # DETUNE, TIMBRE, ATK_BLOCKS
        data = pack_control_defaults(defaults)

        # count (2) + 3 entries (12) = 14 bytes
        assert len(data) == 14

        restored = unpack_control_defaults(data)
        assert restored == defaults

    def test_pack_roundtrip_all_controls(self):
        """All standard control IDs round-trip correctly."""
        defaults = {
            ControlID.TIMBRE: 20000,
            ControlID.TEXTURE: 15000,
            ControlID.DYNAMICS: 25000,
            ControlID.MORPH: 8000,
            ControlID.CHAOS: 5000,
            ControlID.ATK_BLOCKS: 16,
            ControlID.DCY_BLOCKS: 64,
            ControlID.SUS_Q15: 20000,
            ControlID.REL_BLOCKS: 128,
        }
        data = pack_control_defaults(defaults)
        restored = unpack_control_defaults(data)
        assert restored == defaults


# =============================================================================
# §4: VOICE PARAMS PACKING
# =============================================================================

class TestVoiceParamsPacking:
    """Test voice_params.bin packing per §4."""

    def test_pack_empty_params(self):
        """Empty params pack to empty bytes."""
        data = pack_voice_params({})
        assert data == b''

    def test_pack_single_param(self):
        """Single param packs correctly."""
        params = {VoiceParamID.TREE_ID: 2}
        data = pack_voice_params(params)

        # 6 bytes per entry
        assert len(data) == 6

        restored = unpack_voice_params(data)
        assert restored == params

    def test_pack_multiple_params(self):
        """Multiple params pack and restore correctly."""
        params = {
            VoiceParamID.PROGRAM_ID: 1001,
            VoiceParamID.TREE_ID: 1,
            VoiceParamID.H_REQUEST: 8,
        }
        data = pack_voice_params(params)
        restored = unpack_voice_params(data)
        assert restored == params


# =============================================================================
# §6: CAPSULE APPLY ALGORITHM
# =============================================================================

class TestCapsuleApplyAlgorithm:
    """Test Capsule Apply algorithm per §6."""

    def test_step1_validate_engine_compat(self):
        """Step 1: Validate engine_compat."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )

        # Compatible
        result = check_compatibility(
            capsule,
            current_sample_rate=48000,
            current_block_size=64,
        )
        assert result.compatible

        # Incompatible sample rate
        result = check_compatibility(
            capsule,
            current_sample_rate=44100,
            current_block_size=64,
        )
        assert not result.compatible
        assert any("Sample rate" in e for e in result.errors)

        # Incompatible block size
        result = check_compatibility(
            capsule,
            current_sample_rate=48000,
            current_block_size=128,
        )
        assert not result.compatible
        assert any("Block size" in e for e in result.errors)

    def test_step3_apply_voice_param_defaults(self):
        """Step 3: Apply voice_param_defaults with clamping."""
        registry = get_registry()

        # Valid param
        params = {VoiceParamID.TREE_ID: 2}
        applied, ignored, errors = apply_voice_param_defaults(params, registry)
        assert VoiceParamID.TREE_ID in applied
        assert applied[VoiceParamID.TREE_ID] == 2

        # Unknown param (ignored in non-strict)
        params = {9999: 100}
        applied, ignored, errors = apply_voice_param_defaults(params, registry, strict=False)
        assert 9999 in ignored
        assert len(errors) == 0

        # Unknown param (error in strict)
        applied, ignored, errors = apply_voice_param_defaults(params, registry, strict=True)
        assert 9999 in ignored
        assert len(errors) > 0

    def test_step4_apply_control_defaults(self):
        """Step 4: Apply control_defaults with clamping."""
        registry = ControlRegistry()

        # Valid control
        defaults = {ControlID.TIMBRE: 20000}
        applied, ignored, errors = apply_control_defaults(defaults, registry)
        assert ControlID.TIMBRE in applied
        assert applied[ControlID.TIMBRE] == 20000

        # Out of range - should be clamped
        defaults = {ControlID.TIMBRE: 50000}  # Max is 32767
        applied, ignored, errors = apply_control_defaults(defaults, registry)
        assert applied[ControlID.TIMBRE] == 32767

        # Unknown control (ignored in non-strict)
        defaults = {255: 100}
        applied, ignored, errors = apply_control_defaults(defaults, registry, strict=False)
        assert 255 in ignored

    def test_full_apply_algorithm(self):
        """Full capsule apply algorithm (§6)."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
            control_defaults={
                ControlID.TIMBRE: 16384,
                ControlID.ATK_BLOCKS: 16,
            },
            voice_param_defaults={
                VoiceParamID.TREE_ID: 1,
            },
        )

        loader = CapsuleLoader()
        result = loader.apply_capsule(capsule)

        assert result.success
        assert ControlID.TIMBRE in result.applied_control_defaults
        assert VoiceParamID.TREE_ID in result.applied_voice_params

        # Can retrieve applied values
        assert loader.get_program_id() == 1001
        assert loader.get_control_value(ControlID.TIMBRE) == 16384
        assert loader.get_voice_param_value(VoiceParamID.TREE_ID) == 1


# =============================================================================
# BINARY CODEC (.wsecap)
# =============================================================================

class TestCapsuleBinaryCodec:
    """Test .wsecap binary format."""

    def test_magic_and_version(self):
        """Capsule has correct magic and version."""
        assert CAPSULE_MAGIC == b'WSEC'
        assert CAPSULE_VERSION == 0

    def test_roundtrip_instrument_capsule(self):
        """Instrument capsule round-trips through binary."""
        capsule = create_instrument_capsule(
            capsule_id="violin-001",
            name="Glitch Violin",
            program_id=1001,
            author="Test",
            tags=["strings"],
            control_defaults={ControlID.TIMBRE: 20000},
            voice_param_defaults={VoiceParamID.TREE_ID: 1},
        )
        capsule.seeds.master_seed = 0xDEADBEEF

        data = capsule_to_bytes(capsule)

        # Check magic
        assert data[:4] == CAPSULE_MAGIC

        # Round-trip
        restored = capsule_from_bytes(data)

        assert restored.identity.capsule_id == "violin-001"
        assert restored.identity.name == "Glitch Violin"
        assert restored.identity.kind == CapsuleKind.INSTRUMENT
        assert restored.program.program_id == 1001
        assert restored.defaults.control_defaults[ControlID.TIMBRE] == 20000
        assert restored.defaults.voice_param_defaults[VoiceParamID.TREE_ID] == 1
        assert restored.seeds.master_seed == 0xDEADBEEF

    def test_roundtrip_synth_capsule(self):
        """Synth capsule round-trips through binary."""
        capsule = create_synth_capsule(
            capsule_id="pad-001",
            name="Warm Pad",
            program_id=1002,
        )

        data = capsule_to_bytes(capsule)
        restored = capsule_from_bytes(data)

        assert restored.identity.kind == CapsuleKind.SYNTH
        assert restored.program.program_id == 1002

    def test_roundtrip_asset_capsule(self):
        """Asset capsule round-trips through binary."""
        assets = [
            AssetRef(
                asset_type=AssetType.IR,
                asset_id="ir-001",
                asset_hash="abc123",
            ),
        ]
        capsule = create_asset_capsule(
            capsule_id="ir-pack",
            name="IR Pack",
            assets=assets,
        )

        data = capsule_to_bytes(capsule)
        restored = capsule_from_bytes(data)

        assert restored.identity.kind == CapsuleKind.ASSET
        assert len(restored.assets) == 1
        assert restored.assets[0].asset_id == "ir-001"

    def test_crc_validation(self):
        """CRC mismatch raises ValueError."""
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )
        data = bytearray(capsule_to_bytes(capsule))

        # Corrupt data
        data[20] ^= 0xFF

        with pytest.raises(ValueError, match="CRC"):
            capsule_from_bytes(bytes(data))

    def test_invalid_magic(self):
        """Invalid magic raises ValueError."""
        data = b'XXXX' + b'\x00' * 100

        with pytest.raises(ValueError, match="magic"):
            capsule_from_bytes(data)


# =============================================================================
# FILE I/O
# =============================================================================

class TestCapsuleFileIO:
    """Test file save/load."""

    def test_save_load_wsecap(self):
        """Save and load .wsecap file."""
        capsule = create_instrument_capsule(
            capsule_id="file-test",
            name="File Test",
            program_id=1001,
        )

        with tempfile.NamedTemporaryFile(suffix='.wsecap', delete=False) as f:
            temp_path = f.name

        try:
            save_capsule(capsule, temp_path)
            loaded = load_capsule(temp_path)

            assert loaded.identity.capsule_id == "file-test"
            assert loaded.program.program_id == 1001
        finally:
            os.unlink(temp_path)

    def test_save_load_bundle(self):
        """Save and load bundle directory."""
        capsule = create_instrument_capsule(
            capsule_id="bundle-test",
            name="Bundle Test",
            program_id=1001,
            control_defaults={ControlID.TIMBRE: 16384},
            voice_param_defaults={VoiceParamID.TREE_ID: 2},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = os.path.join(temp_dir, "test_bundle")

            save_capsule_bundle(capsule, bundle_path)

            # Check files exist
            assert os.path.exists(os.path.join(bundle_path, "manifest.json"))
            assert os.path.exists(os.path.join(bundle_path, "control_defaults.bin"))
            assert os.path.exists(os.path.join(bundle_path, "voice_params.bin"))

            # Load and verify
            loaded = load_capsule_bundle(bundle_path)

            assert loaded.identity.capsule_id == "bundle-test"
            assert loaded.defaults.control_defaults[ControlID.TIMBRE] == 16384
            assert loaded.defaults.voice_param_defaults[VoiceParamID.TREE_ID] == 2


# =============================================================================
# §8: CONFORMANCE CHECKLIST
# =============================================================================

class TestConformanceChecklist:
    """Verify conformance checklist items from §8."""

    def test_capsule_includes_abi_versions(self):
        """Capsule bundle includes compat ABI versions (§8 item 1)."""
        capsule = create_instrument_capsule(
            capsule_id="abi-test",
            name="ABI Test",
            program_id=1001,
        )

        # Engine compat is always present
        assert capsule.identity.engine_compat.control_registry_abi == 0
        assert capsule.identity.engine_compat.voice_registry_abi == 0
        assert capsule.identity.engine_compat.voiceframe_abi == 0

        # Round-trips through binary
        data = capsule_to_bytes(capsule)
        restored = capsule_from_bytes(data)

        assert restored.identity.engine_compat.control_registry_abi == 0

    def test_defaults_override_only_values(self):
        """Capsule defaults override only default values (§8 item 2)."""
        # CapsuleDefaults only contains value mappings
        defaults = CapsuleDefaults(
            control_defaults={0: 16384},
            voice_param_defaults={1: 2},
        )

        # No way to override ranges, types, slew policy, or binding targets
        assert not hasattr(defaults, 'control_ranges')
        assert not hasattr(defaults, 'value_types')
        assert not hasattr(defaults, 'slew_policies')
        assert not hasattr(defaults, 'binding_targets')

    def test_unknown_ids_ignored_deterministically(self):
        """Unknown IDs ignored deterministically unless strict (§8 item 3)."""
        registry = ControlRegistry()

        # Unknown ID 255
        defaults = {255: 100}

        # Non-strict: ignored silently
        applied, ignored, errors = apply_control_defaults(defaults, registry, strict=False)
        assert 255 in ignored
        assert len(errors) == 0

        # Strict: error
        applied, ignored, errors = apply_control_defaults(defaults, registry, strict=True)
        assert 255 in ignored
        assert len(errors) > 0

    def test_apply_algorithm_implemented(self):
        """Capsule apply algorithm implemented exactly (§8 item 4)."""
        capsule = create_instrument_capsule(
            capsule_id="apply-test",
            name="Apply Test",
            program_id=1001,
            control_defaults={ControlID.TIMBRE: 16384},
            voice_param_defaults={VoiceParamID.TREE_ID: 1},
        )

        loader = CapsuleLoader()

        # Step 1: Validate compat (implicit in apply_capsule)
        compat = loader.check_compat(capsule)
        assert compat.compatible

        # Steps 2-6: Apply
        result = loader.apply_capsule(capsule)
        assert result.success

        # Step 2: Program identity available
        assert loader.get_program_id() == 1001

        # Steps 3-4: Defaults applied
        assert loader.get_voice_param_value(VoiceParamID.TREE_ID) == 1
        assert loader.get_control_value(ControlID.TIMBRE) == 16384

        # Step 5: Seed available for deterministic allocation
        assert loader.get_master_seed() == 0x12345678

    def test_clear_ownership_separation(self):
        """Clear ownership: cockpit writes VoiceFrame; engine reads (§8 item 5)."""
        # ControlRegistry bindings target VoiceFrame emission
        control_registry = ControlRegistry()
        timbre_entry = control_registry.get_entry(ControlID.TIMBRE)
        assert "family_scale" in timbre_entry.binding_target

        # VoiceEngineRegistry bindings target engine internals
        voice_registry = get_registry()
        tree_entry = voice_registry.get(VoiceParamID.TREE_ID)
        assert tree_entry is not None
        assert "tree" in tree_entry.binding_target.lower()


# =============================================================================
# DETERMINISM
# =============================================================================

class TestDeterminism:
    """Test deterministic behavior."""

    def test_capsule_serialization_deterministic(self):
        """Same capsule serializes to identical bytes."""
        capsule = create_instrument_capsule(
            capsule_id="det-test",
            name="Determinism Test",
            program_id=1001,
            control_defaults={ControlID.TIMBRE: 16384, ControlID.ATK_BLOCKS: 8},
        )

        data1 = capsule_to_bytes(capsule)
        data2 = capsule_to_bytes(capsule)

        assert data1 == data2

    def test_control_defaults_packing_deterministic(self):
        """Control defaults pack deterministically (sorted by ID)."""
        # Same data, different insertion order
        defaults1 = {ControlID.ATK_BLOCKS: 8, ControlID.TIMBRE: 16384}
        defaults2 = {ControlID.TIMBRE: 16384, ControlID.ATK_BLOCKS: 8}

        data1 = pack_control_defaults(defaults1)
        data2 = pack_control_defaults(defaults2)

        assert data1 == data2

    def test_apply_result_deterministic(self):
        """Apply results are deterministic."""
        capsule = create_instrument_capsule(
            capsule_id="det-apply",
            name="Det Apply",
            program_id=1001,
            control_defaults={ControlID.TIMBRE: 16384},
        )

        loader1 = CapsuleLoader()
        result1 = loader1.apply_capsule(capsule)

        loader2 = CapsuleLoader()
        result2 = loader2.apply_capsule(capsule)

        assert result1.applied_control_defaults == result2.applied_control_defaults


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_capsule(self):
        """Capsule with no defaults."""
        capsule = create_instrument_capsule(
            capsule_id="empty",
            name="Empty",
            program_id=1001,
        )

        data = capsule_to_bytes(capsule)
        restored = capsule_from_bytes(data)

        assert restored.defaults.control_defaults == {}
        assert restored.defaults.voice_param_defaults == {}

    def test_max_control_values(self):
        """Control values at boundaries."""
        defaults = {
            ControlID.TIMBRE: 32767,  # Max Q15U
            ControlID.DETUNE: -2400,  # Min I16 for this control
        }
        data = pack_control_defaults(defaults)
        restored = unpack_control_defaults(data)
        assert restored == defaults

    def test_negative_values(self):
        """Negative values pack correctly."""
        defaults = {ControlID.DETUNE: -100}
        data = pack_control_defaults(defaults)
        restored = unpack_control_defaults(data)
        assert restored[ControlID.DETUNE] == -100

    def test_clamping_out_of_range(self):
        """Out-of-range values are clamped."""
        registry = ControlRegistry()

        # TIMBRE max is 32767
        defaults = {ControlID.TIMBRE: 50000}
        applied, _, _ = apply_control_defaults(defaults, registry)

        assert applied[ControlID.TIMBRE] == 32767

    def test_capsule_with_assets(self):
        """Capsule with asset references."""
        capsule = create_instrument_capsule(
            capsule_id="with-assets",
            name="With Assets",
            program_id=1001,
        )
        capsule.assets = [
            AssetRef(
                asset_type=AssetType.IR,
                asset_id="ir-001",
                asset_hash="sha256:abc",
                asset_path="/path/to/ir.wav",
            ),
            AssetRef(
                asset_type=AssetType.WAVETABLE,
                asset_id="wt-001",
            ),
        ]

        data = capsule_to_bytes(capsule)
        restored = capsule_from_bytes(data)

        assert len(restored.assets) == 2
        assert restored.assets[0].asset_type == AssetType.IR
        assert restored.assets[0].asset_hash == "sha256:abc"
        assert restored.assets[1].asset_type == AssetType.WAVETABLE
