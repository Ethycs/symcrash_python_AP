"""
WSE-STD-PRESETLIB v0 Conformance Tests
=======================================

Tests for Preset Library Storage + Index standard.

Conformance Checklist (§9):
- [ ] Transactional import with CRC validation (§3)
- [ ] Strict vs lenient compatibility modes (§3.2)
- [ ] Deterministic resolution + dependency ordering (§4, §7)
- [ ] Export produces valid IPLC capsule bundles (§5)
- [ ] Metadata mutation supported (§6)
"""

import os
import tempfile
import pytest

from symcrash_python_ap.library import (
    CapsuleV0,
    CapsuleKind,
    AssetType,
    AssetRef,
    create_instrument_capsule,
    create_synth_capsule,
    capsule_to_bytes,
    save_capsule_bundle,
    load_capsule_bundle,
    CapsuleCompiler,
    ImportResult,
    AssetRecord,
    Query,
    ResolutionSelector,
    PresetLibraryV0,
    LifecycleState,
)
from symcrash_python_ap.control_registry import ControlID


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_library():
    """Create a temporary library directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield PresetLibraryV0(tmpdir)


@pytest.fixture
def sample_capsule():
    """Create a sample capsule for testing."""
    return create_instrument_capsule(
        capsule_id="test-violin-001",
        name="Test Violin",
        program_id=1001,
        author="Test Author",
        tags=["strings", "test"],
        control_defaults={ControlID.TIMBRE: 16384},
    )


@pytest.fixture
def compiled_capsule():
    """Create a compiled capsule."""
    compiler = CapsuleCompiler()
    result = compiler.compile_instrument(
        program_id=1001,
        name="Compiled Violin",
        control_defaults={ControlID.TIMBRE: 20000},
    )
    return result.capsule


# =============================================================================
# §3: IMPORT TRANSACTION
# =============================================================================

class TestImportTransaction:
    """Test transactional import per §3."""

    def test_import_capsule_from_bundle_dir(self, temp_library, sample_capsule):
        """Import capsule from bundle directory."""
        with tempfile.TemporaryDirectory() as bundle_dir:
            save_capsule_bundle(sample_capsule, bundle_dir)

            result = temp_library.import_capsule(bundle_dir)

            assert result.success
            assert result.capsule_id == "test-violin-001"
            assert result.revision_id == 1

    def test_import_capsule_from_bytes(self, temp_library, sample_capsule):
        """Import capsule from binary data."""
        data = capsule_to_bytes(sample_capsule)

        result = temp_library.import_capsule(data)

        assert result.success
        assert result.capsule_id == "test-violin-001"
        assert result.revision_id == 1

    def test_import_increments_revision(self, temp_library, sample_capsule):
        """Multiple imports increment revision_id."""
        data = capsule_to_bytes(sample_capsule)

        result1 = temp_library.import_capsule(data)
        result2 = temp_library.import_capsule(data)
        result3 = temp_library.import_capsule(data)

        assert result1.revision_id == 1
        assert result2.revision_id == 2
        assert result3.revision_id == 3

    def test_import_is_atomic(self, temp_library):
        """Failed import leaves no partial data."""
        # Try to import invalid data
        result = temp_library.import_capsule(b"invalid data")

        assert not result.success
        assert len(result.errors) > 0

        # Library should have no capsules
        records = temp_library.list_capsules()
        assert len(records) == 0

    def test_import_validates_crc(self, temp_library, sample_capsule):
        """Import validates CRC (done by codec)."""
        data = bytearray(capsule_to_bytes(sample_capsule))
        # Corrupt data
        data[50] ^= 0xFF

        result = temp_library.import_capsule(bytes(data))

        assert not result.success
        assert any("CRC" in e or "Failed" in e for e in result.errors)


# =============================================================================
# §3.2: COMPATIBILITY MODES
# =============================================================================

class TestCompatibilityModes:
    """Test strict vs lenient modes per §3.2."""

    def test_lenient_mode_accepts_incompatible(self):
        """Lenient mode imports incompatible capsules with flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibraryV0(tmpdir, mode="lenient", sample_rate=44100)

            # Capsule has 48000 Hz, library expects 44100
            capsule = create_instrument_capsule(
                capsule_id="test",
                name="Test",
                program_id=1001,
            )
            data = capsule_to_bytes(capsule)

            result = library.import_capsule(data)

            assert result.success
            assert len(result.warnings) > 0

            # Resolve should show incompatible
            record = library.resolve_capsule(
                "test",
                selector=ResolutionSelector.LATEST,
            )
            assert record is not None
            assert not record.compatible

    def test_strict_mode_rejects_incompatible(self):
        """Strict mode rejects incompatible capsules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibraryV0(tmpdir, mode="strict", sample_rate=44100)

            capsule = create_instrument_capsule(
                capsule_id="test",
                name="Test",
                program_id=1001,
            )
            data = capsule_to_bytes(capsule)

            result = library.import_capsule(data, mode="strict")

            assert not result.success
            assert len(result.errors) > 0


# =============================================================================
# §4: RESOLVE INTERFACE
# =============================================================================

class TestResolveInterface:
    """Test resolve interface per §4."""

    def test_resolve_latest(self, temp_library, sample_capsule):
        """resolve with LATEST returns highest revision."""
        data = capsule_to_bytes(sample_capsule)

        temp_library.import_capsule(data)  # rev 1
        temp_library.import_capsule(data)  # rev 2
        temp_library.import_capsule(data)  # rev 3

        record = temp_library.resolve_capsule(
            "test-violin-001",
            selector=ResolutionSelector.LATEST,
        )

        assert record is not None
        assert record.revision_id == 3

    def test_resolve_by_revision(self, temp_library, sample_capsule):
        """resolve with BY_REVISION returns specific revision."""
        data = capsule_to_bytes(sample_capsule)

        temp_library.import_capsule(data)  # rev 1
        temp_library.import_capsule(data)  # rev 2

        record = temp_library.resolve_capsule(
            "test-violin-001",
            selector=ResolutionSelector.BY_REVISION,
            revision_id=1,
        )

        assert record is not None
        assert record.revision_id == 1

    def test_resolve_latest_compatible(self):
        """resolve with LATEST_COMPATIBLE skips incompatible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibraryV0(tmpdir, mode="lenient")

            # Import compatible capsule (rev 1)
            capsule1 = create_instrument_capsule(
                capsule_id="test",
                name="Test",
                program_id=1001,
            )
            library.import_capsule(capsule_to_bytes(capsule1))

            # Mark rev 1 as incompatible (simulate)
            library._index.capsules["test"].revisions[1].compatible = False
            library._save_index()

            # Import another (rev 2, compatible)
            library.import_capsule(capsule_to_bytes(capsule1))

            # LATEST_COMPATIBLE should return rev 2
            record = library.resolve_capsule(
                "test",
                selector=ResolutionSelector.LATEST_COMPATIBLE,
            )

            assert record is not None
            assert record.revision_id == 2

    def test_resolve_unknown_returns_none(self, temp_library):
        """resolve returns None for unknown capsule."""
        record = temp_library.resolve_capsule("nonexistent")
        assert record is None

    def test_get_capsule_bundle(self, temp_library, sample_capsule):
        """get_capsule_bundle returns binary data."""
        data = capsule_to_bytes(sample_capsule)
        temp_library.import_capsule(data)

        record = temp_library.resolve_capsule("test-violin-001")
        bundle = temp_library.get_capsule_bundle(record)

        assert isinstance(bundle, bytes)
        assert len(bundle) > 0


# =============================================================================
# §4.3: QUERY INTERFACE
# =============================================================================

class TestQueryInterface:
    """Test query interface per §4.3."""

    def test_list_all_capsules(self, temp_library):
        """list_capsules returns all capsules."""
        capsule1 = create_instrument_capsule(
            capsule_id="violin",
            name="Violin",
            program_id=1001,
        )
        capsule2 = create_synth_capsule(
            capsule_id="pad",
            name="Pad",
            program_id=1002,
        )

        temp_library.import_capsule(capsule_to_bytes(capsule1))
        temp_library.import_capsule(capsule_to_bytes(capsule2))

        records = temp_library.list_capsules()

        assert len(records) == 2
        ids = {r.capsule_id for r in records}
        assert "violin" in ids
        assert "pad" in ids

    def test_query_by_kind(self, temp_library):
        """Query filters by kind."""
        capsule1 = create_instrument_capsule(
            capsule_id="violin",
            name="Violin",
            program_id=1001,
        )
        capsule2 = create_synth_capsule(
            capsule_id="pad",
            name="Pad",
            program_id=1002,
        )

        temp_library.import_capsule(capsule_to_bytes(capsule1))
        temp_library.import_capsule(capsule_to_bytes(capsule2))

        query = Query(kind=CapsuleKind.INSTRUMENT)
        records = temp_library.list_capsules(query)

        assert len(records) == 1
        assert records[0].capsule_id == "violin"

    def test_query_by_tags(self, temp_library):
        """Query filters by tags."""
        capsule1 = create_instrument_capsule(
            capsule_id="violin",
            name="Violin",
            program_id=1001,
            tags=["strings", "classical"],
        )
        capsule2 = create_instrument_capsule(
            capsule_id="guitar",
            name="Guitar",
            program_id=1002,
            tags=["strings", "folk"],
        )
        capsule3 = create_instrument_capsule(
            capsule_id="piano",
            name="Piano",
            program_id=1003,
            tags=["keys"],
        )

        temp_library.import_capsule(capsule_to_bytes(capsule1))
        temp_library.import_capsule(capsule_to_bytes(capsule2))
        temp_library.import_capsule(capsule_to_bytes(capsule3))

        query = Query(tags=["classical"])
        records = temp_library.list_capsules(query)

        assert len(records) == 1
        assert records[0].capsule_id == "violin"

    def test_query_by_name_substring(self, temp_library):
        """Query filters by name substring."""
        capsule1 = create_instrument_capsule(
            capsule_id="glitch-violin",
            name="Glitch Violin",
            program_id=1001,
        )
        capsule2 = create_instrument_capsule(
            capsule_id="warm-pad",
            name="Warm Pad",
            program_id=1002,
        )

        temp_library.import_capsule(capsule_to_bytes(capsule1))
        temp_library.import_capsule(capsule_to_bytes(capsule2))

        query = Query(name_substring="Violin")
        records = temp_library.list_capsules(query)

        assert len(records) == 1
        assert "Violin" in records[0].name

    def test_query_excludes_deprecated_by_default(self, temp_library, sample_capsule):
        """Query excludes deprecated by default."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))
        temp_library.set_flags("test-violin-001", 1, deprecated=True)

        records = temp_library.list_capsules()
        assert len(records) == 0

        # Include deprecated
        query = Query(include_deprecated=True)
        records = temp_library.list_capsules(query)
        assert len(records) == 1


# =============================================================================
# §4.4: DEPENDENCY RESOLUTION
# =============================================================================

class TestDependencyResolution:
    """Test dependency resolution per §4.4."""

    def test_resolve_dependencies_ordered(self, temp_library):
        """Dependencies ordered by (type, id, revision)."""
        # Import some assets
        temp_library.import_asset(b"asset1", "ir-001", AssetType.IR)
        temp_library.import_asset(b"asset2", "wt-001", AssetType.WAVETABLE)

        # Create capsule with asset refs
        capsule = create_instrument_capsule(
            capsule_id="test",
            name="Test",
            program_id=1001,
        )
        capsule.assets = [
            AssetRef(asset_type=AssetType.WAVETABLE, asset_id="wt-001"),
            AssetRef(asset_type=AssetType.IR, asset_id="ir-001"),
        ]

        temp_library.import_capsule(capsule_to_bytes(capsule))
        record = temp_library.resolve_capsule("test")

        deps = temp_library.resolve_dependencies(record)

        # IR (type 0) should come before WAVETABLE (type 1)
        assert len(deps) == 2
        assert deps[0].asset_type == AssetType.IR
        assert deps[1].asset_type == AssetType.WAVETABLE


# =============================================================================
# §5: EXPORT
# =============================================================================

class TestExport:
    """Test export per §5."""

    def test_export_produces_valid_bundle(self, temp_library, sample_capsule):
        """Export produces valid capsule bundle."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))
        record = temp_library.resolve_capsule("test-violin-001")

        with tempfile.TemporaryDirectory() as export_dir:
            temp_library.export_capsule(record, export_dir)

            # Verify files exist
            assert os.path.exists(os.path.join(export_dir, "manifest.json"))

            # Load and verify
            loaded = load_capsule_bundle(export_dir)
            assert loaded.identity.capsule_id == "test-violin-001"
            assert loaded.identity.name == "Test Violin"

    def test_export_roundtrip(self, temp_library, sample_capsule):
        """Export → import produces identical capsule."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))
        record1 = temp_library.resolve_capsule("test-violin-001")

        with tempfile.TemporaryDirectory() as export_dir:
            temp_library.export_capsule(record1, export_dir)

            # Import to new library
            with tempfile.TemporaryDirectory() as lib2_dir:
                library2 = PresetLibraryV0(lib2_dir)
                library2.import_capsule(export_dir)

                record2 = library2.resolve_capsule("test-violin-001")

                assert record2.capsule.identity.name == record1.capsule.identity.name
                assert record2.capsule.defaults.control_defaults == record1.capsule.defaults.control_defaults


# =============================================================================
# §6: MUTATION
# =============================================================================

class TestMutation:
    """Test metadata mutation per §6."""

    def test_set_deprecated_flag(self, temp_library, sample_capsule):
        """set_flags updates deprecated flag."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))

        temp_library.set_flags("test-violin-001", 1, deprecated=True)

        record = temp_library.resolve_capsule(
            "test-violin-001",
            selector=ResolutionSelector.LATEST,
        )
        assert record.deprecated

    def test_set_retired_flag(self, temp_library, sample_capsule):
        """set_flags updates retired flag."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))

        temp_library.set_flags("test-violin-001", 1, retired=True)

        # LATEST_COMPATIBLE skips retired
        record = temp_library.resolve_capsule(
            "test-violin-001",
            selector=ResolutionSelector.LATEST_COMPATIBLE,
        )
        assert record is None

        # But LATEST still finds it
        record = temp_library.resolve_capsule(
            "test-violin-001",
            selector=ResolutionSelector.LATEST,
        )
        assert record is not None
        assert record.retired

    def test_set_tags(self, temp_library, sample_capsule):
        """set_tags updates tags."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))

        temp_library.set_tags("test-violin-001", 1, ["new-tag", "another"])

        # Query by new tag
        query = Query(tags=["new-tag"], include_deprecated=True)
        records = temp_library.list_capsules(query)

        assert len(records) == 1

    def test_set_flags_unknown_capsule_raises(self, temp_library):
        """set_flags raises for unknown capsule."""
        with pytest.raises(KeyError):
            temp_library.set_flags("nonexistent", 1, deprecated=True)


# =============================================================================
# §7: DETERMINISM
# =============================================================================

class TestDeterminism:
    """Test determinism requirements per §7."""

    def test_list_order_deterministic(self, temp_library):
        """list_capsules returns deterministic order."""
        # Import in random order
        for i in [3, 1, 2]:
            capsule = create_instrument_capsule(
                capsule_id=f"capsule-{i}",
                name=f"Capsule {i}",
                program_id=1000 + i,
            )
            temp_library.import_capsule(capsule_to_bytes(capsule))

        records1 = temp_library.list_capsules()
        records2 = temp_library.list_capsules()

        ids1 = [r.capsule_id for r in records1]
        ids2 = [r.capsule_id for r in records2]

        assert ids1 == ids2
        # Should be sorted
        assert ids1 == sorted(ids1)

    def test_index_serialization_sorted(self, temp_library, sample_capsule):
        """Index serialization uses sorted keys."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))

        index_dict = temp_library._index.to_dict()

        # Check capsules dict is sorted
        capsule_keys = list(index_dict["capsules"].keys())
        assert capsule_keys == sorted(capsule_keys)


# =============================================================================
# CONFORMANCE CHECKLIST
# =============================================================================

class TestConformanceChecklist:
    """Verify conformance checklist items from §9."""

    def test_transactional_import(self, temp_library, sample_capsule):
        """§3: Transactional import with CRC validation."""
        # Valid import succeeds
        result = temp_library.import_capsule(capsule_to_bytes(sample_capsule))
        assert result.success

        # Invalid import leaves no partial data
        result = temp_library.import_capsule(b"invalid")
        assert not result.success

    def test_compatibility_modes(self):
        """§3.2: Strict vs lenient compatibility modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Lenient accepts incompatible
            lib_lenient = PresetLibraryV0(tmpdir, mode="lenient", sample_rate=44100)
            capsule = create_instrument_capsule(
                capsule_id="test",
                name="Test",
                program_id=1001,
            )
            result = lib_lenient.import_capsule(capsule_to_bytes(capsule))
            assert result.success

    def test_deterministic_resolution(self, temp_library, sample_capsule):
        """§4, §7: Deterministic resolution."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))

        record1 = temp_library.resolve_capsule("test-violin-001")
        record2 = temp_library.resolve_capsule("test-violin-001")

        assert record1.revision_id == record2.revision_id

    def test_export_produces_valid_bundles(self, temp_library, sample_capsule):
        """§5: Export produces valid IPLC capsule bundles."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))
        record = temp_library.resolve_capsule("test-violin-001")

        with tempfile.TemporaryDirectory() as export_dir:
            temp_library.export_capsule(record, export_dir)
            loaded = load_capsule_bundle(export_dir)
            assert loaded.identity.capsule_id == sample_capsule.identity.capsule_id

    def test_metadata_mutation(self, temp_library, sample_capsule):
        """§6: Metadata mutation supported."""
        temp_library.import_capsule(capsule_to_bytes(sample_capsule))

        # set_flags
        temp_library.set_flags("test-violin-001", 1, deprecated=True)
        record = temp_library.resolve_capsule(
            "test-violin-001",
            selector=ResolutionSelector.LATEST,
        )
        assert record.deprecated

        # set_tags
        temp_library.set_tags("test-violin-001", 1, ["updated"])
