"""Debug SineCore interpolation."""
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "symcrash_python_ap"))

from sine_core import sine_q15, Q_SCALE, PHASE_BITS

# Check specific phases
test_phases = [
    (0.125, 'sin(45)'),
    (0.25, 'sin(90)'),
    (0.375, 'sin(135)'),
    (0.5, 'sin(180)'),
    (0.625, 'sin(225)'),
    (0.75, 'sin(270)'),
    (0.875, 'sin(315)'),
]

print('Phase frac | Name       | Actual | Expected | Error')
print('-' * 60)
for frac, name in test_phases:
    phase = int(frac * (1 << PHASE_BITS))
    theta = 2 * math.pi * frac
    actual = sine_q15(phase)
    expected = Q_SCALE * math.sin(theta)
    error = actual - expected
    print(f'{frac:10.3f} | {name:10s} | {actual:6d} | {expected:8.1f} | {error:+7.1f}')

# Find the worst errors
print()
print('Finding worst errors...')
worst = []
for i in range(10000):
    frac = i / 10000
    phase = int(frac * (1 << PHASE_BITS))
    theta = 2 * math.pi * frac
    actual = sine_q15(phase)
    expected = Q_SCALE * math.sin(theta)
    error = abs(actual - expected)
    worst.append((error, frac, actual, expected))

worst.sort(reverse=True)
print('Top 10 worst errors:')
for error, frac, actual, expected in worst[:10]:
    print(f'  frac={frac:.4f}: actual={actual}, expected={expected:.1f}, error={error:.1f}')
