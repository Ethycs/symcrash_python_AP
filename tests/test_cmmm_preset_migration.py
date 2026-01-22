"""
Tests for CMMM Enrichment Phase P5: Preset Migration
=====================================================

Tests capsule codec CMMM config support and preset migration.

Reference: CMMM-ENRICHMENT-DESIGN.md §9.7
"""

import pytest
import json
import tempfile
from pathlib import Path

from symcrash_python_ap.primitives import Q15_ONE, Q15_HALF


class TestCapsuleCodecVersion:
    """Tests for capsule codec version bump."""

    def test_capsule_version_is_1(self):
        """CAPSULE_VERSION should be 1 for CMMM enrichment."""
        from symcrash_python_ap.library.capsule_codec import CAPSULE_VERSION

        assert CAPSULE_VERSION == 1

    def test_cmmm_config_chunk_type_exists(self):
        """ChunkType.CMMM_CONFIG should be defined."""
        from symcrash_python_ap.library.capsule_codec import ChunkType

        assert hasattr(ChunkType, 'CMMM_CONFIG')
        assert ChunkType.CMMM_CONFIG == 7


class TestCmmmConfigPacking:
    """Tests for CMMM config pack/unpack."""

    def test_pack_cmmm_config_empty(self):
        """Packing None temperament should return valid JSON."""
        from symcrash_python_ap.library.capsule_codec import pack_cmmm_config

        result = pack_cmmm_config(None)
        assert isinstance(result, bytes)

        data = json.loads(result.decode('utf-8'))
        assert 'temperament' in data
        assert data['temperament'] == {}

    def test_pack_cmmm_config_with_values(self):
        """Packing temperament should preserve values."""
        from symcrash_python_ap.library.capsule_codec import pack_cmmm_config

        temperament = {
            'hysteresis_q15': 20000,
            'chaos_amount_q15': 1000,
        }
        result = pack_cmmm_config(temperament)

        data = json.loads(result.decode('utf-8'))
        assert data['temperament']['hysteresis_q15'] == 20000
        assert data['temperament']['chaos_amount_q15'] == 1000

    def test_unpack_cmmm_config_empty(self):
        """Unpacking empty payload should return defaults."""
        from symcrash_python_ap.library.capsule_codec import unpack_cmmm_config

        result = unpack_cmmm_config(b'')
        assert 'temperament' in result
        assert result['temperament'] == {}

    def test_unpack_cmmm_config_with_values(self):
        """Unpacking should preserve temperament values."""
        from symcrash_python_ap.library.capsule_codec import unpack_cmmm_config

        payload = json.dumps({
            'temperament': {
                'hysteresis_q15': 20000,
                'chaos_amount_q15': 1000,
            }
        }).encode('utf-8')

        result = unpack_cmmm_config(payload)
        assert result['temperament']['hysteresis_q15'] == 20000
        assert result['temperament']['chaos_amount_q15'] == 1000

    def test_pack_unpack_roundtrip(self):
        """Pack then unpack should preserve data."""
        from symcrash_python_ap.library.capsule_codec import (
            pack_cmmm_config, unpack_cmmm_config
        )

        original = {
            'hysteresis_q15': Q15_HALF,
            'stickiness_q15': Q15_HALF,
            'reactivity_q15': Q15_HALF,
            'escalation_bias_q15': Q15_HALF,
            'fatigue_sensitivity_q15': Q15_HALF,
            'tension_influence_q15': Q15_HALF,
            'chaos_amount_q15': 0,
        }

        packed = pack_cmmm_config(original)
        unpacked = unpack_cmmm_config(packed)

        for key, value in original.items():
            assert unpacked['temperament'][key] == value


class TestCapsuleV0CmmmConfig:
    """Tests for CapsuleV0 cmmm_config field."""

    def test_capsule_has_cmmm_config_field(self):
        """CapsuleV0 should have cmmm_config field."""
        from symcrash_python_ap.library.capsule import CapsuleV0

        assert 'cmmm_config' in CapsuleV0.__dataclass_fields__

    def test_capsule_cmmm_config_default_none(self):
        """CapsuleV0.cmmm_config should default to None."""
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )

        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test',
                kind=CapsuleKind.INSTRUMENT,
                name='Test',
            )
        )

        assert capsule.cmmm_config is None

    def test_capsule_cmmm_config_can_be_set(self):
        """CapsuleV0.cmmm_config should be settable."""
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )

        cmmm_config = {
            'temperament': {
                'chaos_amount_q15': 500,
            }
        }

        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test',
                kind=CapsuleKind.INSTRUMENT,
                name='Test',
            ),
            cmmm_config=cmmm_config,
        )

        assert capsule.cmmm_config == cmmm_config


class TestCapsuleSerialization:
    """Tests for capsule binary serialization with CMMM config."""

    def test_capsule_to_bytes_without_cmmm_config(self):
        """Capsule without cmmm_config should serialize."""
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )
        from symcrash_python_ap.library.capsule_codec import capsule_to_bytes

        capsule = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test',
                kind=CapsuleKind.INSTRUMENT,
                name='Test',
            )
        )

        data = capsule_to_bytes(capsule)
        assert isinstance(data, bytes)
        assert len(data) > 16  # At least header

    def test_capsule_to_bytes_with_cmmm_config(self):
        """Capsule with cmmm_config should serialize with extra chunk."""
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )
        from symcrash_python_ap.library.capsule_codec import capsule_to_bytes

        capsule_without = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test',
                kind=CapsuleKind.INSTRUMENT,
                name='Test',
            )
        )

        capsule_with = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test',
                kind=CapsuleKind.INSTRUMENT,
                name='Test',
            ),
            cmmm_config={'temperament': {'chaos_amount_q15': 500}},
        )

        data_without = capsule_to_bytes(capsule_without)
        data_with = capsule_to_bytes(capsule_with)

        # With CMMM config should be larger
        assert len(data_with) > len(data_without)

    def test_capsule_roundtrip_without_cmmm_config(self):
        """Capsule without cmmm_config should roundtrip."""
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )
        from symcrash_python_ap.library.capsule_codec import (
            capsule_to_bytes, capsule_from_bytes
        )

        original = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test_roundtrip',
                kind=CapsuleKind.INSTRUMENT,
                name='Test Roundtrip',
            )
        )

        data = capsule_to_bytes(original)
        restored = capsule_from_bytes(data)

        assert restored.identity.capsule_id == 'test_roundtrip'
        assert restored.cmmm_config is None

    def test_capsule_roundtrip_with_cmmm_config(self):
        """Capsule with cmmm_config should roundtrip preserving config."""
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )
        from symcrash_python_ap.library.capsule_codec import (
            capsule_to_bytes, capsule_from_bytes
        )

        cmmm_config = {
            'temperament': {
                'hysteresis_q15': 20000,
                'chaos_amount_q15': 500,
            }
        }

        original = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='test_cmmm',
                kind=CapsuleKind.INSTRUMENT,
                name='Test CMMM',
            ),
            cmmm_config=cmmm_config,
        )

        data = capsule_to_bytes(original)
        restored = capsule_from_bytes(data)

        assert restored.cmmm_config is not None
        assert restored.cmmm_config['temperament']['hysteresis_q15'] == 20000
        assert restored.cmmm_config['temperament']['chaos_amount_q15'] == 500


class TestControlConfigTemperamentParsing:
    """Tests for ControlConfig temperament parsing from YAML/dict."""

    def test_control_config_has_temperament(self):
        """ControlConfig should have temperament field."""
        from symcrash_python_ap.instrument_ir import ControlConfig

        config = ControlConfig()
        assert hasattr(config, 'temperament')

    def test_control_config_temperament_default(self):
        """ControlConfig.temperament should default to neutral TemperamentVector."""
        from symcrash_python_ap.instrument_ir import (
            ControlConfig, TemperamentVector
        )

        config = ControlConfig()
        assert isinstance(config.temperament, TemperamentVector)
        assert config.temperament.chaos_amount_q15 == 0  # Deterministic default

    def test_parse_control_config_without_temperament(self):
        """Parsing control config without temperament uses defaults."""
        from symcrash_python_ap.instrument_ir import _parse_control_config

        data = {
            'brightness': 0.7,
            'density': 0.5,
        }

        config = _parse_control_config(data)
        assert config.brightness == 0.7
        assert config.temperament.chaos_amount_q15 == 0

    def test_parse_control_config_with_temperament(self):
        """Parsing control config with temperament preserves values."""
        from symcrash_python_ap.instrument_ir import _parse_control_config

        data = {
            'brightness': 0.7,
            'temperament': {
                'hysteresis_q15': 20000,
                'chaos_amount_q15': 1000,
            }
        }

        config = _parse_control_config(data)
        assert config.brightness == 0.7
        assert config.temperament.hysteresis_q15 == 20000
        assert config.temperament.chaos_amount_q15 == 1000


class TestMigrationScript:
    """Tests for preset migration script."""

    def test_migrate_empty_preset(self):
        """Migrating empty preset should add control and temperament."""
        from tools.migrate_presets import migrate_preset_dict

        migrated, changes = migrate_preset_dict({})

        assert 'control' in migrated
        assert 'temperament' in migrated['control']
        assert migrated['control']['temperament']['chaos_amount_q15'] == 0
        assert len(changes) > 0

    def test_migrate_preset_with_control(self):
        """Migrating preset with control should add temperament."""
        from tools.migrate_presets import migrate_preset_dict

        original = {
            'control': {
                'brightness': 0.7,
            }
        }

        migrated, changes = migrate_preset_dict(original)

        assert migrated['control']['brightness'] == 0.7
        assert 'temperament' in migrated['control']
        assert len(changes) > 0

    def test_migrate_preset_preserves_existing_temperament(self):
        """Migrating preset should preserve existing temperament values."""
        from tools.migrate_presets import migrate_preset_dict

        original = {
            'control': {
                'temperament': {
                    'hysteresis_q15': 25000,
                }
            }
        }

        migrated, changes = migrate_preset_dict(original)

        assert migrated['control']['temperament']['hysteresis_q15'] == 25000
        # Should fill in missing fields
        assert migrated['control']['temperament']['chaos_amount_q15'] == 0

    def test_migrate_complete_temperament_no_changes(self):
        """Migrating preset with complete temperament should add minimal changes."""
        from tools.migrate_presets import migrate_preset_dict, DEFAULT_TEMPERAMENT

        original = {
            'control_config': {},
            'control': {
                'temperament': dict(DEFAULT_TEMPERAMENT)
            }
        }

        migrated, changes = migrate_preset_dict(original)

        # Should have no temperament-related changes
        temperament_changes = [c for c in changes if 'temperament' in c.lower()]
        assert len(temperament_changes) == 0

    def test_default_temperament_values(self):
        """DEFAULT_TEMPERAMENT should have all required fields."""
        from tools.migrate_presets import DEFAULT_TEMPERAMENT

        required_fields = [
            'hysteresis_q15',
            'stickiness_q15',
            'reactivity_q15',
            'escalation_bias_q15',
            'fatigue_sensitivity_q15',
            'tension_influence_q15',
            'chaos_amount_q15',
        ]

        for field in required_fields:
            assert field in DEFAULT_TEMPERAMENT

        # Verify deterministic default
        assert DEFAULT_TEMPERAMENT['chaos_amount_q15'] == 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with v0 presets."""

    def test_v0_capsule_loads_with_none_cmmm_config(self):
        """Loading v0 capsule should result in cmmm_config=None."""
        # This tests that capsules without CMMM_CONFIG chunk work
        from symcrash_python_ap.library.capsule import (
            CapsuleV0, CapsuleIdentity, CapsuleKind
        )
        from symcrash_python_ap.library.capsule_codec import (
            capsule_to_bytes, capsule_from_bytes
        )

        # Create capsule without cmmm_config (simulates v0)
        original = CapsuleV0(
            identity=CapsuleIdentity(
                capsule_id='v0_test',
                kind=CapsuleKind.INSTRUMENT,
                name='V0 Test',
            ),
            cmmm_config=None,
        )

        data = capsule_to_bytes(original)
        restored = capsule_from_bytes(data)

        assert restored.cmmm_config is None

    def test_temperament_defaults_are_neutral(self):
        """Default temperament should produce neutral/deterministic behavior."""
        from symcrash_python_ap.instrument_ir import TemperamentVector

        temp = TemperamentVector()

        # Neutral values
        assert temp.hysteresis_q15 == Q15_HALF
        assert temp.stickiness_q15 == Q15_HALF
        assert temp.reactivity_q15 == Q15_HALF

        # Deterministic
        assert temp.chaos_amount_q15 == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
