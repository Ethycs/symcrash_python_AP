"""Freeze Mode A metadata and legacy waveform inputs outside audio playback.

Native table increments come directly from the shared integer-cent pitch ROM.
The floating frequencies below are display metadata, never pitch arithmetic.
"""
import argparse
from pathlib import Path

import numpy as np


def source():
    frequencies = [440.0 * 2.0 ** ((cent - 6900) / 1200.0) for cent in range(12701)]
    pulse = [float(np.sin(h * (0.25 * np.pi))) for h in range(1, 33)]
    lines = ['"""Frozen Mode A metadata and legacy waveform inputs; generated offline.',
             '', 'Regenerate with tools/generate_mode_a_recipe.py. Playback uses these',
             'constants only as resource metadata; these are not the shared pitch ROM.',
             '"""', '']
    for name, values in [('PITCH_FREQUENCIES_HZ', frequencies), ('PULSE_AMPLITUDES', pulse)]:
        lines.append(name + ' = (')
        for i in range(0, len(values), 4):
            lines.append('    ' + ', '.join(repr(x) for x in values[i:i+4]) + ',')
        lines += [')', '']
    lines += ['LEGACY_FREQUENCY_HZ = 261.63',
              f'LEGACY_PHASE_INCREMENT = {int((261.63 / 48000) * 2**32)}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'src/symcrash_python_ap/vm/mode_a_recipe.py'
    expected = source()
    if args.check:
        if path.read_text(encoding='utf-8') != expected:
            raise SystemExit('Mode A recipe differs from its offline generator')
        print('Mode A recipe matches its offline generator.')
    else:
        path.write_text(expected, encoding='utf-8', newline='\n')
        print(path)


if __name__ == '__main__':
    main()
