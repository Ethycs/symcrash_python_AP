"""Analyze spur frequency relationships."""
# Check spur relationships
f0 = 440.0
spurs = [620, 1240, 1990, 130, 10010]

print('Spur analysis for A4 (440 Hz):')
for s in spurs:
    ratio = s / f0
    nearest_h = round(ratio)
    diff = abs(ratio - nearest_h)
    print(f'  {s} Hz: ratio={ratio:.3f}, nearest_h={nearest_h}, diff={diff:.3f}')

print()
print('Possible sources:')
print(f'  Block rate @ 48kHz: 48000/64 = 750 Hz')
print(f'  620 Hz = 440 + 180 Hz')
print(f'  1240 Hz = 2 * 620 Hz (2nd harmonic of spur)')
print(f'  130 Hz = 750 - 620 = intermod?')
print()
print('  The 620 Hz spur is likely block-rate aliasing:')
print(f'    750 - 130 = 620')
print(f'    440 - 310 = 130 (310 = 440 - 130)')
