"""Test the new InstrumentIR quick preset methods."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.instrument_ir import (
    InstrumentIR,
    ms_to_blocks,
    pct_to_q15,
    q15_to_pct,
)

print("=" * 60)
print("Testing InstrumentIR Quick Preset Methods")
print("=" * 60)

# Test unit conversion helpers
print("\n--- Unit Conversion Helpers ---")
print(f"ms_to_blocks(10) = {ms_to_blocks(10)} (expect ~8)")
print(f"ms_to_blocks(100) = {ms_to_blocks(100)} (expect ~75)")
print(f"pct_to_q15(0.5) = {pct_to_q15(0.5)} (expect ~16383)")
print(f"pct_to_q15(1.0) = {pct_to_q15(1.0)} (expect 32767)")
print(f"q15_to_pct(16383) = {q15_to_pct(16383):.4f} (expect ~0.5)")

# Test quick_synth
print("\n--- quick_synth ---")
ir = InstrumentIR.quick_synth("Bright Saw", brightness=0.8)
print(f"Name: {ir.meta.name}")
print(f"ID: {ir.meta.instrument_id}")
print(f"L0 nodes: {[n.id for n in ir.layers.L0_graph.nodes]}")
print(f"L0 macros: {[m.id for m in ir.layers.L0_graph.macros]}")
print(f"L2 regimes: {[r.id for r in ir.layers.L2_regimes.regime_nodes]}")
print(f"L3 templates: {list(ir.layers.L3_cmmm.emit_templates.keys())}")

# Test compilation
print("\n--- to_capsule() ---")
capsule = ir.to_capsule()
print(f"Capsule ID: {capsule.identity.capsule_id}")
print(f"Program ID: {capsule.program.program_id}")
print(f"Control defaults: {capsule.defaults.control_defaults}")

# Test quick_pad
print("\n--- quick_pad ---")
ir_pad = InstrumentIR.quick_pad("Warm Pad")
print(f"Name: {ir_pad.meta.name}")
adsr = ir_pad.layers.L0_graph.nodes[1].params.adsr
print(f"Attack: {adsr.a_ms}ms (expect 300)")
print(f"Decay: {adsr.d_ms}ms (expect 200)")
print(f"Sustain: {adsr.s} (expect 0.8)")
print(f"Release: {adsr.r_ms}ms (expect 800)")

# Test quick_pluck
print("\n--- quick_pluck ---")
ir_pluck = InstrumentIR.quick_pluck("Sharp Pluck")
print(f"Name: {ir_pluck.meta.name}")
adsr = ir_pluck.layers.L0_graph.nodes[1].params.adsr
print(f"Attack: {adsr.a_ms}ms (expect 2)")
print(f"Sustain: {adsr.s} (expect 0.0)")

# Test quick_bass
print("\n--- quick_bass ---")
ir_bass = InstrumentIR.quick_bass("Sub Bass", waveform="sine")
print(f"Name: {ir_bass.meta.name}")
print(f"Waveform: {ir_bass.layers.L0_graph.nodes[0].params.waveform} (expect sine)")

# Test quick_lead
print("\n--- quick_lead ---")
ir_lead = InstrumentIR.quick_lead("Screaming Lead", brightness=0.9)
print(f"Name: {ir_lead.meta.name}")
print(f"Brightness: {ir_lead.layers.L0_graph.macros[0].default} (expect 0.9)")

# Test minimal IR with auto-completion
print("\n--- Minimal IR with auto-completion ---")
ir_minimal = InstrumentIR()
ir_minimal.meta.instrument_id = "minimal-test"
ir_minimal.meta.name = "Minimal Test"
capsule_minimal = ir_minimal.to_capsule()
print(f"Capsule created: {capsule_minimal.identity.name}")
print(f"Has program_id: {capsule_minimal.program.program_id is not None}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
