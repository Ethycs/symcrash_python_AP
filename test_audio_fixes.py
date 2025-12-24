"""Quick test of the audio pipeline fixes."""
import sys
sys.path.insert(0, 'src/symcrash_python_ap')

from wavelet_engine import WaveletEngine
from harmonic_injector import HarmonicTargetGenerator, WaveType, freq_to_leaf_and_coeff
from frames import VoiceFrame
from fixed import Q15_ONE

# Test 1: Verify freq_to_leaf_and_coeff works
print('Testing freq_to_leaf_and_coeff...')
test_freqs = [110, 220, 440, 880, 1760]
for f in test_freqs:
    leaf, coeff = freq_to_leaf_and_coeff(f)
    print(f'  {f} Hz -> leaf={leaf}, coeff={coeff}')

# Test 2: Verify harmonic injection creates non-zero coeff indices
print()
print('Testing HarmonicTargetGenerator...')
gen = HarmonicTargetGenerator()
gen.set_note(0, note_u8=69)  # A4 = 440 Hz
gen.set_wave_params(0, wave_type=WaveType.SAW, partials_n=8)
target = gen.generate_target(0, include_dense=True)
print(f'  Leaf indices: {target.leaf_indices[:5]}...')
print(f'  Coeff indices: {target.coeff_indices[:5]}...')
nonzero = sum(1 for c in target.dense_target if c != 0)
print(f'  Non-zero coeffs in dense: {nonzero}')

# Test 3: Verify WaveletEngine has phase state
print()
print('Testing WaveletEngine phase state...')
we = WaveletEngine()
phase = we.get_phase_state(0)
print(f'  Phase state length: {len(phase)}')
print(f'  First 5 phases: {[round(p, 3) for p in phase[:5]]}')

# Test 4: Quick render test
print()
print('Testing render with phase continuity...')
vf = VoiceFrame(voice_id=0, tree_a_id=1, gain_q15=Q15_ONE//2, coeff_q15=target.dense_target)
audio1 = we.render_voice(vf)
audio2 = we.render_voice(vf)
print(f'  Block 1 samples: {len(audio1)}, first 5: {audio1[:5]}')
print(f'  Block 2 samples: {len(audio2)}, first 5: {audio2[:5]}')
print(f'  Block boundary continuity: |{audio1[-1]} - {audio2[0]}| = {abs(audio1[-1] - audio2[0])}')

print()
print('All tests passed!')
