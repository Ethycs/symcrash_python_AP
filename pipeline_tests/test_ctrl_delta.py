"""
CTRL-DELTA WIRE PROTOCOL TESTS
==============================

Tests for the sparse parameter update wire format.

Key property: CTRL-DELTA is the PRIMARY wire format.
VoiceFramePacket (136 bytes) is internal/debug only.
"""

import pytest
from typing import List, Dict


class TestCtrlDeltaPacket:
    """Basic packet construction and serialization."""

    def test_empty_packet_serializes(self):
        """Empty packet (no fields) serializes correctly."""
        from ctrl_delta import CtrlDeltaPacket, CTRL_DELTA_HEADER_LEN

        pkt = CtrlDeltaPacket()
        data = pkt.to_bytes()

        assert len(data) == CTRL_DELTA_HEADER_LEN
        assert data[0] == 0xCD  # Magic

    def test_single_field_roundtrip(self):
        """Single field survives encode/decode."""
        from ctrl_delta import CtrlDeltaPacket, ScopeType
        from parameter_registry import FieldMask

        pkt = CtrlDeltaPacket(
            scope_type=ScopeType.VOICE,
            scope_id=5,
        )
        pkt.values[int(FieldMask.GAIN_Q15)] = 16384
        pkt.field_mask = 1 << FieldMask.GAIN_Q15

        data = pkt.to_bytes()
        decoded = CtrlDeltaPacket.from_bytes(data)

        assert decoded.scope_type == ScopeType.VOICE
        assert decoded.scope_id == 5
        assert decoded.values[int(FieldMask.GAIN_Q15)] == 16384

    def test_multiple_fields_roundtrip(self):
        """Multiple fields survive encode/decode."""
        from ctrl_delta import CtrlDeltaPacket, ScopeType
        from parameter_registry import FieldMask, UpdateMode

        pkt = CtrlDeltaPacket(
            scope_type=ScopeType.VOICE,
            scope_id=3,
            mode=UpdateMode.ABS,
        )

        # Set multiple fields
        pkt.field_mask = (
            (1 << FieldMask.GAIN_Q15) |
            (1 << FieldMask.PAN_Q15) |
            (1 << FieldMask.TREE_A_ID)
        )
        pkt.values[int(FieldMask.GAIN_Q15)] = 20000
        pkt.values[int(FieldMask.PAN_Q15)] = -8000
        pkt.values[int(FieldMask.TREE_A_ID)] = 2

        data = pkt.to_bytes()
        decoded = CtrlDeltaPacket.from_bytes(data)

        assert decoded.values[int(FieldMask.GAIN_Q15)] == 20000
        assert decoded.values[int(FieldMask.PAN_Q15)] == -8000
        assert decoded.values[int(FieldMask.TREE_A_ID)] == 2

    def test_group_field_family_weights(self):
        """Group field (FAMILY_WEIGHTS) serializes correctly."""
        from ctrl_delta import CtrlDeltaPacket
        from parameter_registry import FieldMask

        weights = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

        pkt = CtrlDeltaPacket()
        pkt.field_mask = 1 << FieldMask.FAMILY_WEIGHTS
        pkt.values[int(FieldMask.FAMILY_WEIGHTS)] = weights

        data = pkt.to_bytes()
        decoded = CtrlDeltaPacket.from_bytes(data)

        assert decoded.values[int(FieldMask.FAMILY_WEIGHTS)] == weights

    def test_delta_mode_preserved(self):
        """DELTA mode flag survives roundtrip."""
        from ctrl_delta import CtrlDeltaPacket
        from parameter_registry import FieldMask, UpdateMode

        pkt = CtrlDeltaPacket(mode=UpdateMode.DELTA)
        pkt.field_mask = 1 << FieldMask.GAIN_Q15
        pkt.values[int(FieldMask.GAIN_Q15)] = 100  # Delta of +100

        data = pkt.to_bytes()
        decoded = CtrlDeltaPacket.from_bytes(data)

        assert decoded.mode == UpdateMode.DELTA


class TestMaskedUpdateConversion:
    """Conversion between MaskedUpdate and CtrlDeltaPacket."""

    def test_from_masked_update(self):
        """CtrlDeltaPacket.from_masked_update() works."""
        from ctrl_delta import CtrlDeltaPacket, ScopeType
        from parameter_registry import MaskedUpdate, FieldMask, UpdateMode

        update = MaskedUpdate(mode=UpdateMode.ABS)
        update.set(FieldMask.GAIN_Q15, 25000)
        update.set(FieldMask.PAN_Q15, 5000)

        pkt = CtrlDeltaPacket.from_masked_update(update, ScopeType.VOICE, 7)

        assert pkt.scope_type == ScopeType.VOICE
        assert pkt.scope_id == 7
        assert pkt.field_mask == update.field_mask
        assert pkt.values == update.values

    def test_to_masked_update(self):
        """CtrlDeltaPacket.to_masked_update() works."""
        from ctrl_delta import CtrlDeltaPacket
        from parameter_registry import FieldMask, UpdateMode

        pkt = CtrlDeltaPacket(mode=UpdateMode.DELTA)
        pkt.field_mask = 1 << FieldMask.DRIVE_Q15
        pkt.values[int(FieldMask.DRIVE_Q15)] = -500

        update = pkt.to_masked_update()

        assert update.mode == UpdateMode.DELTA
        assert update.has(FieldMask.DRIVE_Q15)
        assert update.get(FieldMask.DRIVE_Q15) == -500


class TestValidation:
    """Packet validation tests."""

    def test_valid_packet_passes(self):
        """Well-formed packet passes validation."""
        from ctrl_delta import CtrlDeltaPacket, ScopeType
        from parameter_registry import FieldMask

        pkt = CtrlDeltaPacket(scope_type=ScopeType.VOICE, scope_id=0)
        pkt.field_mask = 1 << FieldMask.GAIN_Q15
        pkt.values[int(FieldMask.GAIN_Q15)] = 16384

        valid, error = pkt.validate()
        assert valid is True
        assert error is None

    def test_missing_value_fails(self):
        """Field set in mask but no value fails validation."""
        from ctrl_delta import CtrlDeltaPacket
        from parameter_registry import FieldMask

        pkt = CtrlDeltaPacket()
        pkt.field_mask = 1 << FieldMask.GAIN_Q15
        # No value set for GAIN_Q15

        valid, error = pkt.validate()
        assert valid is False
        assert "GAIN_Q15" in error

    def test_invalid_magic_fails_decode(self):
        """Invalid magic byte fails decode."""
        from ctrl_delta import CtrlDeltaPacket

        bad_data = bytes([0xFF] + [0] * 9)  # Wrong magic

        with pytest.raises(ValueError, match="Invalid magic"):
            CtrlDeltaPacket.from_bytes(bad_data)

    def test_truncated_header_fails(self):
        """Truncated header fails decode."""
        from ctrl_delta import CtrlDeltaPacket

        short_data = bytes([0xCD, 1])  # Only 2 bytes

        with pytest.raises(ValueError, match="too short"):
            CtrlDeltaPacket.from_bytes(short_data)


class TestBandwidth:
    """Bandwidth estimation and comparison."""

    def test_ctrl_delta_smaller_than_vfp(self):
        """CTRL-DELTA is much smaller than VoiceFramePacket."""
        from ctrl_delta import CtrlDeltaPacket, bandwidth_comparison
        from parameter_registry import FieldMask

        # Create a typical update (4 fields)
        pkt = CtrlDeltaPacket()
        pkt.field_mask = (
            (1 << FieldMask.GAIN_Q15) |
            (1 << FieldMask.PAN_Q15) |
            (1 << FieldMask.TREE_A_ID) |
            (1 << FieldMask.TREE_BLEND_Q15)
        )
        pkt.values[int(FieldMask.GAIN_Q15)] = 16384
        pkt.values[int(FieldMask.PAN_Q15)] = 0
        pkt.values[int(FieldMask.TREE_A_ID)] = 0
        pkt.values[int(FieldMask.TREE_BLEND_Q15)] = 0

        size = pkt.compute_size()

        # VoiceFramePacket is 136 bytes
        # CTRL-DELTA with 4 fields should be ~17 bytes (10 header + 7 payload)
        assert size < 136
        assert size < 30  # Much smaller

    def test_bandwidth_comparison(self):
        """Bandwidth comparison shows CTRL-DELTA is feasible."""
        from ctrl_delta import bandwidth_comparison

        bw = bandwidth_comparison()

        # VoiceFramePacket is impossible (> 1 MB/s for 12 voices)
        assert bw["vfp_12_voices_bps"] > 1_000_000

        # UART capacity is ~92 kB/s
        uart_capacity = bw["uart_921600_capacity_bps"]
        assert uart_capacity < 100_000

        # CTRL-DELTA at 25% update rate fits easily
        assert bw["ctrl_delta_12_voices_25pct_bps"] < uart_capacity

    def test_estimate_bandwidth(self):
        """estimate_bandwidth() returns reasonable values."""
        from ctrl_delta import estimate_bandwidth

        # 12 voices, 0.5 updates/voice, 4 fields/update, 750 Hz blocks
        bw = estimate_bandwidth(
            n_voices=12,
            updates_per_voice=0.5,
            avg_fields_per_update=4,
            block_rate=750.0,
        )

        # Should be around 70-90 kB/s
        assert 50_000 < bw < 150_000


class TestIntegrationWithParameterBank:
    """Integration with ParameterBank for applying updates."""

    def test_apply_ctrl_delta_to_bank(self):
        """CTRL-DELTA can be applied to ParameterBank."""
        from ctrl_delta import CtrlDeltaPacket
        from parameter_registry import (
            ParameterRegistry, ParameterBank, FieldMask, UpdateMode
        )

        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        # Create update
        pkt = CtrlDeltaPacket(mode=UpdateMode.ABS)
        pkt.field_mask = 1 << FieldMask.GAIN_Q15
        pkt.values[int(FieldMask.GAIN_Q15)] = 20000

        # Convert to MaskedUpdate and apply
        update = pkt.to_masked_update()
        applied = bank.apply_masked(update)

        # Check value was applied
        assert 0x0001 in applied  # param_id for GAIN_Q15
        assert applied[0x0001] == 20000

    def test_delta_update_applied_correctly(self):
        """DELTA mode updates are applied correctly."""
        from ctrl_delta import CtrlDeltaPacket
        from parameter_registry import (
            ParameterRegistry, ParameterBank, FieldMask, UpdateMode
        )

        registry = ParameterRegistry()
        bank = ParameterBank(registry)

        # Set initial value
        bank.apply_abs(0x0001, 10000)  # GAIN_Q15

        # Apply delta
        pkt = CtrlDeltaPacket(mode=UpdateMode.DELTA)
        pkt.field_mask = 1 << FieldMask.GAIN_Q15
        pkt.values[int(FieldMask.GAIN_Q15)] = 5000

        update = pkt.to_masked_update()
        applied = bank.apply_masked(update)

        # Should be 10000 + 5000 = 15000
        assert applied[0x0001] == 15000
