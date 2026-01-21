"""Test OOB consolidation imports."""
from symcrash_python_ap import frames, instrument_ir, synth_master
from symcrash_python_ap.library import preset_compiler, capsule
from symcrash_python_ap.vm import composite_osc, tables
from symcrash_python_ap.budgets import pattern_table, tail_modes

print("All imports successful!")

# Verify values match primitives
from symcrash_python_ap.primitives import Q15_ONE, Q15_HALF, Q16_ONE, SAMPLE_RATE_HZ, BLOCK_SIZE

# Check instrument_ir
assert instrument_ir.pct_to_q15(1.0) == Q15_ONE
assert instrument_ir.q15_to_pct(Q15_ONE) == 1.0

# Check frames
vf = frames.VoiceFrame()
assert vf.cmmm_traction_q15 == Q15_ONE
assert vf.partial_amps_q15[0] == Q15_ONE

# Check preset_compiler
fm = preset_compiler.FM6Config()
assert fm.op_ratios[0] == Q16_ONE
assert fm.op_levels[0] == Q15_ONE

carrier = preset_compiler.CarrierConfig()
assert carrier.pulse_width_q15 == Q15_HALF

# Check capsule
ec = capsule.EngineCompat()
assert ec.sample_rate_hz == SAMPLE_RATE_HZ
assert ec.block_size == BLOCK_SIZE

# Check composite_osc
assert composite_osc.BLEND_PURE_B == Q15_ONE
assert composite_osc.BLEND_EQUAL == Q15_HALF

# Check synth_master
sc = synth_master.SynthConfig()
assert sc.sample_rate == SAMPLE_RATE_HZ
assert sc.block_size == BLOCK_SIZE

print("All value assertions passed!")
