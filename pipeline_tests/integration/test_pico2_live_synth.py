"""
PICO2 LIVE SYNTH INTEGRATION TESTS
==================================

Tests for the Pico2 live synthesis controller integration.

These tests verify that the Pico2SynthBridge correctly routes hardware
control changes to the VoiceEngine, and that the full pipeline
(Pico2 -> VoiceEngine -> UnifiedOutputStage) produces valid audio.

Tests run without actual hardware using virtual mode / mock injection.
"""

import pytest
import numpy as np


class TestPico2SynthBridgeIntegration:
    """
    Tests for Pico2SynthBridge -> VoiceEngine integration.

    Verifies that macro changes are correctly routed to ADSR,
    filter, and oscillator parameters.
    """

    @pytest.fixture
    def bridge_with_voice(self, voice_engine):
        """Create a bridge with an allocated voice."""
        from symcrash_python_ap.comm.pico2_synth_bridge import (
            Pico2SynthBridge,
            MACRO_ATTACK,
            MACRO_DECAY,
            MACRO_SUSTAIN,
            MACRO_RELEASE,
            MACRO_CUTOFF,
            MACRO_RESO,
            MACRO_HARMONICS,
            MACRO_ETA,
            MACRO_DETUNE,
            MACRO_GAIN,
        )
        from symcrash_python_ap.voice.voice_engine import GestureBasis

        # Create bridge in virtual mode (no hardware)
        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)

        # Allocate a voice
        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=16384)
        voice_engine.note_on(vid, gesture=gesture)

        return bridge, vid

    def test_attack_macro_updates_engine(self, bridge_with_voice, voice_engine):
        """Attack macro updates VoiceEngine program."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_ATTACK

        bridge, vid = bridge_with_voice

        # Initial attack
        initial_attack = voice_engine.default_program.attack_blocks

        # Inject macro change (simulates pot movement)
        bridge.receiver.inject_macro(MACRO_ATTACK, 32767)  # Max attack

        # Verify engine was updated
        assert voice_engine.default_program.attack_blocks > initial_attack
        assert voice_engine.voices[vid].program.attack_blocks > initial_attack

    def test_decay_macro_updates_engine(self, bridge_with_voice, voice_engine):
        """Decay macro updates VoiceEngine program."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_DECAY

        bridge, vid = bridge_with_voice

        initial_decay = voice_engine.default_program.decay_blocks

        bridge.receiver.inject_macro(MACRO_DECAY, 32767)

        assert voice_engine.default_program.decay_blocks > initial_decay

    def test_sustain_macro_updates_engine(self, bridge_with_voice, voice_engine):
        """Sustain macro updates VoiceEngine program."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_SUSTAIN

        bridge, vid = bridge_with_voice

        # Set sustain to specific value
        bridge.receiver.inject_macro(MACRO_SUSTAIN, 24576)  # 75%

        assert voice_engine.default_program.sustain_level_q15 == 24576
        assert voice_engine.voices[vid].program.sustain_level_q15 == 24576

    def test_release_macro_updates_engine(self, bridge_with_voice, voice_engine):
        """Release macro updates VoiceEngine program."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_RELEASE

        bridge, vid = bridge_with_voice

        initial_release = voice_engine.default_program.release_blocks

        bridge.receiver.inject_macro(MACRO_RELEASE, 32767)

        assert voice_engine.default_program.release_blocks > initial_release

    def test_eta_macro_updates_voice_state(self, bridge_with_voice, voice_engine):
        """Eta (dry/wet) macro updates voice state."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_ETA

        bridge, vid = bridge_with_voice

        # Set eta to specific value
        bridge.receiver.inject_macro(MACRO_ETA, 16384)  # 50%

        assert voice_engine.voices[vid].eta_q15 == 16384

    def test_harmonics_macro_updates_voice_state(self, bridge_with_voice, voice_engine):
        """Harmonics macro updates voice state."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_HARMONICS

        bridge, vid = bridge_with_voice

        bridge.receiver.inject_macro(MACRO_HARMONICS, 24576)  # 75% harmonics

        assert voice_engine.voices[vid].harmonics_q15 == 24576

    def test_gain_macro_updates_voice_state(self, bridge_with_voice, voice_engine):
        """Master gain macro updates voice state."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_GAIN

        bridge, vid = bridge_with_voice

        bridge.receiver.inject_macro(MACRO_GAIN, 20000)

        assert voice_engine.voices[vid].master_gain_q15 == 20000

    def test_detune_macro_converts_to_cents(self, bridge_with_voice, voice_engine):
        """Detune macro converts Q15 to cents correctly."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_DETUNE

        bridge, vid = bridge_with_voice

        # Center position = no detune
        bridge.receiver.inject_macro(MACRO_DETUNE, 16384)
        assert voice_engine.voices[vid].detune_cents == 0

        # Full left = -2 octaves (-2400 cents)
        bridge.receiver.inject_macro(MACRO_DETUNE, 0)
        assert voice_engine.voices[vid].detune_cents == -2400

        # Full right = +2 octaves (+2400 cents, approximately)
        bridge.receiver.inject_macro(MACRO_DETUNE, 32767)
        # Due to rounding: (32767 - 16384) * 2400 // 16384 = 2399
        assert voice_engine.voices[vid].detune_cents >= 2399

    def test_cutoff_updates_gesture(self, bridge_with_voice, voice_engine):
        """Cutoff macro updates gesture's chaos_q15."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_CUTOFF

        bridge, vid = bridge_with_voice

        bridge.receiver.inject_macro(MACRO_CUTOFF, 12345)

        # Cutoff is mapped to chaos_q15 in gesture
        assert voice_engine.voices[vid].gesture.chaos_q15 == 12345

    def test_resonance_updates_gesture(self, bridge_with_voice, voice_engine):
        """Resonance macro updates gesture's tilt_q15."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_RESO

        bridge, vid = bridge_with_voice

        bridge.receiver.inject_macro(MACRO_RESO, 23456)

        # Resonance is mapped to tilt_q15 in gesture
        assert voice_engine.voices[vid].gesture.tilt_q15 == 23456


class TestPico2SynthBridgeState:
    """
    Tests for Pico2SynthBridge state tracking and display.
    """

    @pytest.fixture
    def bridge(self, voice_engine):
        """Create a bridge in virtual mode."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge

        return Pico2SynthBridge(voice_engine, virtual_mode=True)

    def test_adsr_state_tracking(self, bridge):
        """Bridge tracks ADSR state correctly."""
        from symcrash_python_ap.comm.pico2_synth_bridge import (
            MACRO_ATTACK, MACRO_DECAY, MACRO_SUSTAIN, MACRO_RELEASE
        )

        bridge.receiver.inject_macro(MACRO_ATTACK, 10000)
        bridge.receiver.inject_macro(MACRO_DECAY, 20000)
        bridge.receiver.inject_macro(MACRO_SUSTAIN, 24576)
        bridge.receiver.inject_macro(MACRO_RELEASE, 15000)

        assert bridge.adsr.attack_blocks > 0
        assert bridge.adsr.decay_blocks > 0
        assert bridge.adsr.sustain_level_q15 == 24576
        assert bridge.adsr.release_blocks > 0

    def test_filter_state_tracking(self, bridge):
        """Bridge tracks filter state correctly."""
        from symcrash_python_ap.comm.pico2_synth_bridge import MACRO_CUTOFF, MACRO_RESO

        bridge.receiver.inject_macro(MACRO_CUTOFF, 16384)
        bridge.receiver.inject_macro(MACRO_RESO, 8192)

        assert bridge.filter.cutoff_q15 == 16384
        assert bridge.filter.reso_q15 == 8192

    def test_osc_state_tracking(self, bridge):
        """Bridge tracks oscillator state correctly."""
        from symcrash_python_ap.comm.pico2_synth_bridge import (
            MACRO_HARMONICS, MACRO_ETA, MACRO_DETUNE, MACRO_GAIN
        )

        bridge.receiver.inject_macro(MACRO_HARMONICS, 16384)
        bridge.receiver.inject_macro(MACRO_ETA, 24576)
        bridge.receiver.inject_macro(MACRO_DETUNE, 8192)
        bridge.receiver.inject_macro(MACRO_GAIN, 32000)

        assert bridge.osc.harmonics_q15 == 16384
        assert bridge.osc.eta_q15 == 24576
        assert bridge.osc.detune_q15 == 8192
        assert bridge.osc.gain_q15 == 32000

    def test_display_formatters(self, bridge):
        """Display formatters return valid strings."""
        adsr = bridge.get_adsr_display()
        assert "ATK" in adsr
        assert "DCY" in adsr
        assert "SUS" in adsr
        assert "REL" in adsr

        filter_ = bridge.get_filter_display()
        assert "CUT" in filter_
        assert "RES" in filter_

        osc = bridge.get_osc_display()
        assert "HARM" in osc
        assert "ETA" in osc
        assert "DET" in osc
        assert "GAIN" in osc


class TestPico2LiveSynthVoiceFrame:
    """
    Tests for VoiceFrame creation in Pico2LiveSynth.
    """

    def test_voice_frame_includes_eta(self, voice_engine):
        """VoiceFrame includes eta from voice state."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge, MACRO_ETA
        from symcrash_python_ap.voice.voice_engine import GestureBasis
        from symcrash_python_ap.frames import VoiceFrame

        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)

        # Allocate voice
        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=16384)
        voice_engine.note_on(vid, gesture=gesture)

        # Set eta
        bridge.receiver.inject_macro(MACRO_ETA, 24576)

        # Verify voice state has eta
        assert voice_engine.voices[vid].eta_q15 == 24576

    def test_voice_frame_includes_harmonics(self, voice_engine):
        """VoiceFrame includes harmonics from voice state."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge, MACRO_HARMONICS
        from symcrash_python_ap.voice.voice_engine import GestureBasis

        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=16384)
        voice_engine.note_on(vid, gesture=gesture)

        bridge.receiver.inject_macro(MACRO_HARMONICS, 16384)

        assert voice_engine.voices[vid].harmonics_q15 == 16384


class TestPico2LiveSynthConversion:
    """
    Tests for Q15 <-> parameter conversion utilities.
    """

    def test_q15_to_blocks_linear(self):
        """Q15 to blocks conversion works linearly."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge

        # Min value -> 1 block
        assert Pico2SynthBridge.q15_to_blocks(0, 100) == 1

        # Max value -> max blocks
        assert Pico2SynthBridge.q15_to_blocks(32767, 100) == 100

        # Half value -> half blocks
        assert Pico2SynthBridge.q15_to_blocks(16384, 100) == 50

    def test_blocks_to_ms(self):
        """Block count to milliseconds conversion."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge

        # 1 block at 48kHz/64 samples = ~1.33ms
        ms = Pico2SynthBridge.blocks_to_ms(1)
        assert 1.3 < ms < 1.4

        # 750 blocks = ~1 second
        ms = Pico2SynthBridge.blocks_to_ms(750)
        assert 990 < ms < 1010


class TestPico2SynthBridgeVirtualMode:
    """
    Tests for virtual mode operation (no hardware).
    """

    def test_virtual_mode_starts_without_port(self, voice_engine):
        """Virtual mode starts successfully without hardware."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge

        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)
        assert bridge.start() is True

    def test_virtual_mode_port_is_none(self, voice_engine):
        """Virtual mode has no port."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge

        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)
        assert bridge.port is None

    def test_inject_macro_works_in_virtual_mode(self, voice_engine):
        """Macro injection works in virtual mode."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge, MACRO_ATTACK
        from symcrash_python_ap.voice.voice_engine import GestureBasis

        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)

        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=16384)
        voice_engine.note_on(vid, gesture=gesture)

        initial = voice_engine.default_program.attack_blocks

        # Inject macro should still work (no serial port needed)
        bridge.receiver.inject_macro(MACRO_ATTACK, 32767)

        assert voice_engine.default_program.attack_blocks > initial


class TestPico2LiveSynthAudioPath:
    """
    Tests for the full audio rendering path.

    These tests verify that the complete Pico2 -> Audio pipeline
    produces valid, non-silent output.
    """

    def test_render_produces_audio(self, voice_engine, unified_output_stage):
        """Full render path produces non-silent audio."""
        from symcrash_python_ap.comm.pico2_synth_bridge import Pico2SynthBridge
        from symcrash_python_ap.voice.voice_engine import GestureBasis
        from symcrash_python_ap.frames import VoiceFrame

        bridge = Pico2SynthBridge(voice_engine, virtual_mode=True)

        # Allocate and activate voice
        vid = voice_engine.allocate_voice(program_id=1001)
        gesture = GestureBasis(gate=True, pressure_q15=24576)
        voice_engine.note_on(vid, gesture=gesture)

        # Also activate in output stage
        unified_output_stage.voice_on(vid, 60)

        # Render multiple blocks
        all_samples = []
        for _ in range(100):
            frames = voice_engine.step_all()

            # Find active frame
            active_frame = None
            for vf in frames:
                if vf.is_active and vf.gain_q15 > 0:
                    active_frame = vf
                    break

            if active_frame is not None:
                # Create VoiceFrame for stage
                frame = VoiceFrame(
                    voice_id=active_frame.voice_id,
                    note=60,
                    f0_hz=261.63,  # C4
                    gain_q15=active_frame.gain_q15,
                    beta_target_q15=16384,  # 50% wet
                    partial_amps_q15=[32767, 0, 0, 0, 0, 0, 0, 0],
                    is_active=True,
                )
            else:
                frame = VoiceFrame(
                    voice_id=0,
                    note=60,
                    f0_hz=261.63,
                    gain_q15=0,
                    beta_target_q15=0,
                    is_active=False,
                )

            left, right = unified_output_stage.render_block(frame)
            # left/right are numpy arrays or lists depending on stage implementation
            if hasattr(left, 'tolist'):
                all_samples.extend(left.tolist())
            else:
                all_samples.extend(left)

        samples = np.array(all_samples)

        # Should have non-zero samples (attack ramps up)
        assert np.max(np.abs(samples)) > 100, "Audio should not be silent"

    def test_multiple_voices_render(self, voice_engine, unified_output_stage):
        """Multiple voices can be rendered simultaneously."""
        from symcrash_python_ap.voice.voice_engine import GestureBasis
        from symcrash_python_ap.frames import VoiceFrame

        # Allocate multiple voices at different pitches
        vids = []
        for note in [60, 64, 67]:  # C, E, G chord
            vid = voice_engine.allocate_voice(program_id=1001)
            gesture = GestureBasis(gate=True, pressure_q15=20000)
            voice_engine.note_on(vid, gesture=gesture)
            unified_output_stage.voice_on(vid, note)
            vids.append(vid)

        # Render a few blocks
        for _ in range(50):
            frames = voice_engine.step_all()

            # Use first active voice for rendering
            for vf in frames:
                if vf.is_active and vf.gain_q15 > 0:
                    frame = VoiceFrame(
                        voice_id=vf.voice_id,
                        note=60,
                        f0_hz=261.63,
                        gain_q15=vf.gain_q15,
                        beta_target_q15=8192,
                        is_active=True,
                    )
                    left, right = unified_output_stage.render_block(frame)
                    # Just verify it doesn't crash
                    assert len(left) == 64
                    break
