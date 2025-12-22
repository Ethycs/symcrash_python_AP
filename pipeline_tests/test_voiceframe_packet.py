"""
VoiceFramePacketV0b Conformance Tests
=====================================

Tests for the 136-byte wire format per Wire Contract v0b.
"""

import pytest
from src.symcrash_python_ap.frames import (
    VoiceFramePacketV0b,
    VoiceFlags,
    VFP_MAGIC,
    VFP_VERSION,
    VFP_PACKET_LEN,
)


# =============================================================================
# PACKET SIZE TESTS
# =============================================================================

class TestPacketSize:
    """Verify exact 136-byte packet size."""

    def test_default_packet_size(self):
        """Default packet should be exactly 136 bytes."""
        pkt = VoiceFramePacketV0b()
        data = pkt.to_bytes()
        assert len(data) == VFP_PACKET_LEN

    def test_populated_packet_size(self):
        """Populated packet should still be exactly 136 bytes."""
        pkt = VoiceFramePacketV0b(
            program_id=1001,
            voice_id=5,
            frame_seq=12345,
            flags=VoiceFlags.GATE,
            regime_id=2,
            note_u8=60,
            gain_q15=0x7FFF,
            w_family_q15=[0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x7FFF],
        )
        data = pkt.to_bytes()
        assert len(data) == VFP_PACKET_LEN


# =============================================================================
# MAGIC/VERSION TESTS
# =============================================================================

class TestMagicVersion:
    """Verify magic word and version field."""

    def test_magic_is_vf(self):
        """Magic should be 0x5646 ('VF')."""
        pkt = VoiceFramePacketV0b()
        data = pkt.to_bytes()
        # Big-endian: first two bytes are magic
        magic = (data[0] << 8) | data[1]
        assert magic == 0x5646

    def test_version_is_one(self):
        """Version should be 1."""
        pkt = VoiceFramePacketV0b()
        data = pkt.to_bytes()
        # Third byte is version
        assert data[2] == 1

    def test_packet_len_field(self):
        """Packet length field should be 136."""
        pkt = VoiceFramePacketV0b()
        data = pkt.to_bytes()
        # Fourth byte is packet_len
        assert data[3] == 136


# =============================================================================
# ROUND-TRIP TESTS
# =============================================================================

class TestRoundTrip:
    """Verify pack/unpack round-trip."""

    def test_default_roundtrip(self):
        """Default packet should round-trip correctly."""
        pkt = VoiceFramePacketV0b()
        data = pkt.to_bytes()
        pkt2 = VoiceFramePacketV0b.from_bytes(data)

        assert pkt2.magic == VFP_MAGIC
        assert pkt2.version == VFP_VERSION
        assert pkt2.packet_len == VFP_PACKET_LEN
        assert pkt2.voice_id == pkt.voice_id

    def test_populated_roundtrip(self):
        """Populated packet should round-trip all fields."""
        pkt = VoiceFramePacketV0b(
            program_id=1001,
            voice_id=7,
            frame_seq=54321,
            reject_code=0,
            flags=VoiceFlags.GATE | VoiceFlags.GATE_RISE,
            regime_id=3,
            degradation_level=1,
            note_u8=72,
            cents_i16=-50,
            gain_q15=0x6000,
            activity_q15=0x5000,
            pan_q15=-0x4000,
            sendA_q15=0x2000,
            sendB_q15=0x1000,
            w_family_q15=[100, 200, 300, 400, 500, 600, 700, 800],
            tree_a_id=1,
            tree_b_id=3,
            tree_blend_q15=0x4000,
            event_mask=0b00001101,  # lanes 0, 2, 3
            event_strength_q15=[100, 0, 200, 300, 0, 0, 0, 0],
        )

        data = pkt.to_bytes()
        pkt2 = VoiceFramePacketV0b.from_bytes(data)

        assert pkt2.program_id == 1001
        assert pkt2.voice_id == 7
        assert pkt2.frame_seq == 54321
        assert pkt2.flags == (VoiceFlags.GATE | VoiceFlags.GATE_RISE)
        assert pkt2.regime_id == 3
        assert pkt2.degradation_level == 1
        assert pkt2.note_u8 == 72
        assert pkt2.cents_i16 == -50
        assert pkt2.gain_q15 == 0x6000
        assert pkt2.activity_q15 == 0x5000
        assert pkt2.pan_q15 == -0x4000
        assert pkt2.sendA_q15 == 0x2000
        assert pkt2.sendB_q15 == 0x1000
        assert pkt2.w_family_q15 == [100, 200, 300, 400, 500, 600, 700, 800]
        assert pkt2.tree_a_id == 1
        assert pkt2.tree_b_id == 3
        assert pkt2.tree_blend_q15 == 0x4000
        assert pkt2.event_mask == 0b00001101
        assert pkt2.event_strength_q15 == [100, 0, 200, 300, 0, 0, 0, 0]


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Verify packet validation."""

    def test_valid_default_packet(self):
        """Default packet should validate."""
        pkt = VoiceFramePacketV0b()
        valid, err = pkt.validate()
        assert valid, f"Validation failed: {err}"

    def test_invalid_magic(self):
        """Invalid magic should fail validation."""
        pkt = VoiceFramePacketV0b(magic=0x1234)
        valid, err = pkt.validate()
        assert not valid
        assert "magic" in err.lower()

    def test_invalid_version(self):
        """Invalid version should fail validation."""
        pkt = VoiceFramePacketV0b(version=99)
        valid, err = pkt.validate()
        assert not valid
        assert "version" in err.lower()

    def test_invalid_packet_len(self):
        """Invalid packet_len should fail validation."""
        pkt = VoiceFramePacketV0b(packet_len=100)
        valid, err = pkt.validate()
        assert not valid
        assert "packet_len" in err.lower()

    def test_nonzero_pad_fails(self):
        """Non-zero padding should fail validation."""
        pkt = VoiceFramePacketV0b(hdr_pad=1)
        valid, err = pkt.validate()
        assert not valid
        assert "hdr_pad" in err.lower()

    def test_note_out_of_range(self):
        """Note > 127 should fail validation."""
        pkt = VoiceFramePacketV0b(note_u8=200)
        valid, err = pkt.validate()
        assert not valid
        assert "note" in err.lower()

    def test_tree_id_out_of_range(self):
        """Tree ID >= NUM_TREES should fail validation."""
        pkt = VoiceFramePacketV0b(tree_a_id=10)
        valid, err = pkt.validate()
        assert not valid
        assert "tree" in err.lower()


# =============================================================================
# EVENT PAYLOAD TESTS
# =============================================================================

class TestEventPayload:
    """Test event payload pack/unpack."""

    def test_pack_unpack_payload(self):
        """Event payload should pack/unpack correctly."""
        packed = VoiceFramePacketV0b.pack_event_payload(
            lane_id=3,
            variant=2,
            strength_q8=128,
            target_kind=1,
            target_id=15,
            pattern_id=7,
            param0_q8=64,
            param1_q8=32,
            param2_q8=16,
        )

        fields = VoiceFramePacketV0b.unpack_event_payload(packed)

        assert fields['lane_id'] == 3
        assert fields['variant'] == 2
        assert fields['strength_q8'] == 128
        assert fields['target_kind'] == 1
        assert fields['target_id'] == 15
        assert fields['pattern_id'] == 7
        assert fields['param0_q8'] == 64
        assert fields['param1_q8'] == 32
        assert fields['param2_q8'] == 16

    def test_lane_id_mismatch_fails_validation(self):
        """Event payload with mismatched lane_id should fail validation."""
        pkt = VoiceFramePacketV0b(
            event_mask=0x01,  # Lane 0 active
            event_payload_u64=[
                VoiceFramePacketV0b.pack_event_payload(lane_id=5),  # Wrong lane_id!
                0, 0, 0, 0, 0, 0, 0
            ]
        )
        valid, err = pkt.validate()
        assert not valid
        assert "lane" in err.lower()

    def test_correct_lane_id_passes(self):
        """Event payload with correct lane_id should pass validation."""
        pkt = VoiceFramePacketV0b(
            event_mask=0x05,  # Lanes 0 and 2 active
            event_payload_u64=[
                VoiceFramePacketV0b.pack_event_payload(lane_id=0, variant=1),
                0,
                VoiceFramePacketV0b.pack_event_payload(lane_id=2, variant=2),
                0, 0, 0, 0, 0
            ]
        )
        valid, err = pkt.validate()
        assert valid, f"Validation failed: {err}"


# =============================================================================
# BIG-ENDIAN TESTS
# =============================================================================

class TestEndianness:
    """Verify big-endian byte order."""

    def test_program_id_big_endian(self):
        """program_id should be big-endian."""
        pkt = VoiceFramePacketV0b(program_id=0x1234)
        data = pkt.to_bytes()
        # program_id is at offset 4-5 (after magic, version, packet_len)
        assert data[4] == 0x12
        assert data[5] == 0x34

    def test_frame_seq_big_endian(self):
        """frame_seq (u32) should be big-endian."""
        pkt = VoiceFramePacketV0b(frame_seq=0x12345678)
        data = pkt.to_bytes()
        # frame_seq is at offset 8-11
        assert data[8] == 0x12
        assert data[9] == 0x34
        assert data[10] == 0x56
        assert data[11] == 0x78


# =============================================================================
# VOICE FLAGS TESTS
# =============================================================================

class TestVoiceFlags:
    """Test voice flags enum."""

    def test_gate_flag(self):
        """GATE flag should be 0x0001."""
        assert VoiceFlags.GATE == 0x0001

    def test_gate_rise_flag(self):
        """GATE_RISE flag should be 0x0002."""
        assert VoiceFlags.GATE_RISE == 0x0002

    def test_gate_fall_flag(self):
        """GATE_FALL flag should be 0x0004."""
        assert VoiceFlags.GATE_FALL == 0x0004

    def test_combined_flags(self):
        """Combined flags should work."""
        pkt = VoiceFramePacketV0b(
            flags=VoiceFlags.GATE | VoiceFlags.GATE_RISE | VoiceFlags.LEGATO
        )
        data = pkt.to_bytes()
        pkt2 = VoiceFramePacketV0b.from_bytes(data)

        assert pkt2.flags & VoiceFlags.GATE
        assert pkt2.flags & VoiceFlags.GATE_RISE
        assert pkt2.flags & VoiceFlags.LEGATO
        assert not (pkt2.flags & VoiceFlags.GATE_FALL)


# =============================================================================
# CONVERSION TESTS
# =============================================================================

class TestConversion:
    """Test conversion to/from internal VoiceFrame."""

    def test_to_voice_frame(self):
        """Should convert to internal VoiceFrame."""
        pkt = VoiceFramePacketV0b(
            voice_id=5,
            gain_q15=0x6000,
            tree_a_id=2,
            tree_b_id=3,
            tree_blend_q15=0x4000,
        )

        vf = pkt.to_voice_frame()

        assert vf.voice_id == 5
        assert vf.gain_q15 == 0x6000
        assert vf.tree_a_id == 2
        assert vf.tree_b_id == 3
        assert vf.tree_blend_q15 == 0x4000

    def test_from_voice_frame(self):
        """Should convert from internal VoiceFrame."""
        from src.symcrash_python_ap.frames import VoiceFrame

        vf = VoiceFrame(
            voice_id=7,
            gain_q15=0x5000,
            tree_a_id=1,
            tree_b_id=4,
            tree_blend_q15=0x2000,
        )

        pkt = VoiceFramePacketV0b.from_voice_frame(
            vf,
            program_id=1001,
            frame_seq=100,
            regime_id=2,
        )

        assert pkt.voice_id == 7
        assert pkt.program_id == 1001
        assert pkt.frame_seq == 100
        assert pkt.regime_id == 2
        assert pkt.gain_q15 == 0x5000
        assert pkt.tree_a_id == 1
        assert pkt.tree_b_id == 4
        assert pkt.tree_blend_q15 == 0x2000
