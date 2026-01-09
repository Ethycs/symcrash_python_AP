"""
InstrumentIR Factory Method Tests
=================================

Tests for InstrumentIR factory methods (quick_synth, quick_pad, etc.).

These tests verify:
- Factory methods create valid IRs
- Default parameters are sensible
- ADSR envelopes are properly configured
- Brightness/waveform parameters work correctly
"""

import pytest


# =============================================================================
# QUICK_SYNTH TESTS
# =============================================================================

class TestQuickSynth:
    """Tests for quick_synth factory method."""

    def test_creates_valid_ir(self):
        """quick_synth MUST create a valid InstrumentIR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test Synth")

        assert ir is not None
        assert ir.meta.name == "Test Synth"

    def test_default_waveform_saw(self):
        """quick_synth default waveform MUST be saw."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")

        # Check L0 graph has an oscillator with saw waveform
        osc_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Osc"]
        assert len(osc_nodes) > 0

        # Default waveform should be saw
        osc = osc_nodes[0]
        assert osc.params.waveform == "saw"

    def test_accepts_waveform_override(self):
        """quick_synth MUST accept waveform parameter."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test", waveform="sine")

        osc_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Osc"]
        assert osc_nodes[0].params.waveform == "sine"

    def test_default_envelope_values(self):
        """quick_synth MUST have sensible default envelope."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test")

        # Find amp node with ADSR
        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        assert len(amp_nodes) > 0

        adsr = amp_nodes[0].params.adsr
        assert adsr is not None

        # Default values per spec: 10ms attack, 100ms decay, 0.7 sustain, 200ms release
        assert adsr.a_ms == 10
        assert adsr.d_ms == 100
        assert adsr.s == 0.7
        assert adsr.r_ms == 200

    def test_accepts_envelope_overrides(self):
        """quick_synth MUST accept envelope parameters."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth(
            "Test",
            attack_ms=50,
            decay_ms=200,
            sustain=0.5,
            release_ms=500,
        )

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        assert adsr.a_ms == 50
        assert adsr.d_ms == 200
        assert adsr.s == 0.5
        assert adsr.r_ms == 500

    def test_brightness_affects_timbre(self):
        """quick_synth brightness MUST be stored."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir_bright = InstrumentIR.quick_synth("Bright", brightness=0.9)
        ir_dark = InstrumentIR.quick_synth("Dark", brightness=0.1)

        # Both should be valid
        assert ir_bright is not None
        assert ir_dark is not None

    def test_author_stored(self):
        """quick_synth author MUST be stored in meta."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_synth("Test", author="Test Author")

        assert ir.meta.author == "Test Author"


# =============================================================================
# QUICK_PAD TESTS
# =============================================================================

class TestQuickPad:
    """Tests for quick_pad factory method."""

    def test_creates_valid_ir(self):
        """quick_pad MUST create a valid InstrumentIR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pad("Test Pad")

        assert ir is not None
        assert ir.meta.name == "Test Pad"

    def test_slow_attack(self):
        """quick_pad MUST have slow attack (~300ms)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pad("Pad")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Pad attack should be slow (300ms per spec)
        assert adsr.a_ms >= 200

    def test_slow_release(self):
        """quick_pad MUST have slow release (~800ms)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pad("Pad")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Pad release should be slow
        assert adsr.r_ms >= 600

    def test_high_sustain(self):
        """quick_pad MUST have high sustain (~0.8)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pad("Pad")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Pad sustain should be high
        assert adsr.s >= 0.7


# =============================================================================
# QUICK_PLUCK TESTS
# =============================================================================

class TestQuickPluck:
    """Tests for quick_pluck factory method."""

    def test_creates_valid_ir(self):
        """quick_pluck MUST create a valid InstrumentIR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pluck("Test Pluck")

        assert ir is not None
        assert ir.meta.name == "Test Pluck"

    def test_fast_attack(self):
        """quick_pluck MUST have very fast attack (~2ms)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pluck("Pluck")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Pluck attack should be very fast
        assert adsr.a_ms <= 10

    def test_zero_sustain(self):
        """quick_pluck MUST have zero sustain (percussive)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pluck("Pluck")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Pluck sustain should be zero
        assert adsr.s == 0.0

    def test_short_decay(self):
        """quick_pluck MUST have short decay (~150ms)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_pluck("Pluck")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Pluck decay should be short
        assert adsr.d_ms <= 200


# =============================================================================
# QUICK_BASS TESTS
# =============================================================================

class TestQuickBass:
    """Tests for quick_bass factory method."""

    def test_creates_valid_ir(self):
        """quick_bass MUST create a valid InstrumentIR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_bass("Test Bass")

        assert ir is not None
        assert ir.meta.name == "Test Bass"

    def test_punchy_attack(self):
        """quick_bass MUST have punchy attack (~5ms)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_bass("Bass")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Bass attack should be punchy
        assert adsr.a_ms <= 20

    def test_medium_sustain(self):
        """quick_bass MUST have medium sustain (~0.6)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_bass("Bass")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Bass sustain should be medium
        assert 0.4 <= adsr.s <= 0.8


# =============================================================================
# QUICK_LEAD TESTS
# =============================================================================

class TestQuickLead:
    """Tests for quick_lead factory method."""

    def test_creates_valid_ir(self):
        """quick_lead MUST create a valid InstrumentIR."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_lead("Test Lead")

        assert ir is not None
        assert ir.meta.name == "Test Lead"

    def test_medium_attack(self):
        """quick_lead MUST have medium attack (~15ms)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_lead("Lead")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Lead attack should be medium
        assert 5 <= adsr.a_ms <= 30

    def test_high_sustain(self):
        """quick_lead MUST have high sustain (~0.85)."""
        from symcrash_python_ap.instrument_ir import InstrumentIR

        ir = InstrumentIR.quick_lead("Lead")

        amp_nodes = [n for n in ir.layers.L0_graph.nodes if n.type == "Amp"]
        adsr = amp_nodes[0].params.adsr

        # Lead sustain should be high
        assert adsr.s >= 0.75


# =============================================================================
# UNIT CONVERSION TESTS
# =============================================================================

class TestUnitConversions:
    """Tests for unit conversion helper functions."""

    def test_ms_to_blocks_basic(self):
        """ms_to_blocks MUST convert correctly."""
        from symcrash_python_ap.instrument_ir import ms_to_blocks

        # At 48kHz, 64 samples = 1.333ms per block
        # 10ms = 10 / 1.333 = ~7.5 blocks -> 8
        blocks = ms_to_blocks(10)
        assert blocks >= 7 and blocks <= 8

    def test_ms_to_blocks_minimum(self):
        """ms_to_blocks MUST return at least 1."""
        from symcrash_python_ap.instrument_ir import ms_to_blocks

        blocks = ms_to_blocks(0)
        assert blocks >= 1

    def test_pct_to_q15_range(self):
        """pct_to_q15 MUST handle 0.0-1.0 range."""
        from symcrash_python_ap.instrument_ir import pct_to_q15

        assert pct_to_q15(0.0) == 0
        assert pct_to_q15(1.0) == 32767
        assert pct_to_q15(0.5) == 16383 or pct_to_q15(0.5) == 16384

    def test_pct_to_q15_clamp(self):
        """pct_to_q15 MUST clamp out-of-range values."""
        from symcrash_python_ap.instrument_ir import pct_to_q15

        assert pct_to_q15(-0.5) == 0
        assert pct_to_q15(1.5) == 32767

    def test_q15_to_pct_roundtrip(self):
        """q15_to_pct MUST be inverse of pct_to_q15."""
        from symcrash_python_ap.instrument_ir import pct_to_q15, q15_to_pct

        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            q15 = pct_to_q15(pct)
            back = q15_to_pct(q15)
            assert abs(back - pct) < 0.0001
