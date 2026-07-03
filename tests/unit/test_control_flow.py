"""Quick test of control flow."""

from symcrash_python_ap.cockpit_engine import CockpitEngine
from symcrash_python_ap.voiceframe_emitter import VoiceFrameEmitter
from symcrash_python_ap.library.preset_compiler import MaterialConfig, LatchDefaults
from symcrash_python_ap.envelope_scheduler import ADSRSpec
from dataclasses import dataclass, field as dc_field


@dataclass
class TestPreset:
    preset_id: str = "test"
    preset_version: str = "1.0"
    author: str = "Test"
    description: str = "Test"
    voiceframe_std: str = "WSE-STD-020-v1"
    vm_std: str = "WSE-STD-VM-HYBRID-v1"
    operator_lib_id: str = "WSE-STD-030-v2"
    latch: LatchDefaults = dc_field(default_factory=lambda: LatchDefaults(
        tree_a_id=0, tree_b_id=0, tree_blend_q15=0
    ))
    amp_env: ADSRSpec = dc_field(default_factory=lambda: ADSRSpec(
        attack_ms=10.0, decay_ms=42.0, sustain=0.75, release_ms=85.0
    ))
    material: MaterialConfig = dc_field(default_factory=lambda: MaterialConfig(
        beta_start_q15=0, beta_target_q15=24576, beta_ramp_blocks=16
    ))
    backbone: list = dc_field(default_factory=list)
    harmonic_amps_q15: list = dc_field(default_factory=lambda: [32767, 16384, 10923, 8192])

    def backbone_ops(self):
        return []

    def backbone_families(self):
        # CompiledPreset contract: dict of op name -> strength (0.0-1.0).
        # Empty backbone -> empty dict.
        return {}


# Create instances
cockpit = CockpitEngine()
emitter = VoiceFrameEmitter(TestPreset(), voice_id=0)

# Send note-on
cockpit.midi_note_on(60, 100)
print("Note-on sent")

# Tick and check
frame = cockpit.tick()
print(f"ControlFrame: {len(frame.events)} events, {len(frame.voice_patches)} patches")
for event in frame.events:
    print(f"  Event: type={event.type}, a={event.a:04x}")

# Consume
emitter.consume_control_frame(frame)
print(f"Emitter active after consume: {emitter.is_active}")
print(f"Emitter note_on state: {emitter._is_note_on}")

# Emit VoiceFrame
vf = emitter.emit_frame()
print(f"VoiceFrame: gain={vf.gain_q15}, active={vf.is_active}")
print(f"  ops_enable: {vf.ops_enable}")
print(f"  families: {vf.families}")
print(f"  partial_amps_q15: {vf.partial_amps_q15}")

# Test subsequent blocks
print("\n--- Subsequent blocks ---")
for i in range(5):
    frame = cockpit.tick()
    emitter.consume_control_frame(frame)
    vf = emitter.emit_frame()
    # families is a Dict keyed by operator (frames.VoiceFrame contract)
    print(f"Block {i+1}: gain={vf.gain_q15}, active={vf.is_active}, families={vf.families}")
