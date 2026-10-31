"""Independent rational lifting oracle and reproducible production comparison.

The production CCS kernels keep wide intermediates between the two lifting
steps and saturate each tree-node result. The older scalar/HDL lift saturates
after each step; the two policies are reported separately, never conflated.
Headroom scaling remains the existing arithmetic downshift / saturated upshift.
"""
import argparse
from fractions import Fraction
import json

import numpy as np


def clip(value):
    return max(-32768, min(32767, int(value)))


def forward_step(samples, *, step_saturation=False, half_up=False):
    even, odd = samples[::2], samples[1::2]
    def reduce(value, divisor):
        return (value + divisor // 2) // divisor if half_up else round(Fraction(value, divisor))
    detail = [y - reduce(even[i] + even[min(i+1, len(even)-1)], 2)
              for i, y in enumerate(odd)]
    if step_saturation:
        detail = list(map(clip, detail))
    smooth = [x + reduce(detail[max(0, i-1)] + detail[i], 4)
              for i, x in enumerate(even)]
    return list(map(clip, smooth)), list(map(clip, detail))


def inverse_step(smooth, detail, *, step_saturation=False, half_up=False):
    def reduce(value, divisor):
        return (value + divisor // 2) // divisor if half_up else round(Fraction(value, divisor))
    even = [x - reduce(detail[max(0, i-1)] + detail[i], 4)
            for i, x in enumerate(smooth)]
    if step_saturation:
        even = list(map(clip, even))
    odd = [x + reduce(even[i] + even[min(i+1, len(even)-1)], 2)
           for i, x in enumerate(detail)]
    return [clip(x) for pair in zip(even, odd) for x in pair]


def forward(samples, headroom=0, **policy):
    def descend(node):
        if len(node) == 1:
            return node
        low, high = forward_step(node, **policy)
        return descend(low) + descend(high)
    return descend([int(x) // 2**headroom for x in samples])


def inverse(coefficients, headroom=0, **policy):
    def ascend(node):
        if len(node) == 1:
            return node
        mid = len(node) // 2
        return inverse_step(ascend(node[:mid]), ascend(node[mid:]), **policy)
    return [clip(x * 2**headroom) for x in ascend(list(map(int, coefficients)))]


def vectors():
    yield 'positive-ties', [0, 1, 0, 0] * 16
    yield 'negative-ties', [0, -1, 0, 0] * 16
    yield 'alternating-extrema', [-32768, 32767] * 32
    yield 'saturation-order', [32767, -32768, -32768, 32767] * 16
    for i in range(64):
        yield f'impulse-{i}', [32767 if j == i else 0 for j in range(64)]
    rng = np.random.default_rng(0xCDF53)
    for scale in (3, 8191, 32767):
        for index in range(8):
            yield f'random-{scale}-{index}', rng.integers(-scale, scale+1, 64).tolist()


def audit():
    from symcrash_python_ap.wavelet.fast_lifting import fwpt_jit, idwpt_jit
    rows = []
    for name, values in vectors():
        for direction, actual_fn, oracle in [('forward', fwpt_jit, forward),
                                             ('inverse', idwpt_jit, inverse)]:
            for headroom in (0, 1, 2, 3):
                actual = np.asarray(actual_fn(values, headroom_bits=headroom), dtype=np.int64)
                expected = np.asarray(oracle(values, headroom), dtype=np.int64)
                delta = actual - expected
                rows.append(dict(vector=name, direction=direction, headroom=headroom,
                    different=int(np.count_nonzero(delta)), max_delta=int(np.abs(delta).max())))
    return dict(scope='Rational arithmetic comparison; unchanged production node saturation and headroom',
        cases=len(rows), failing_cases=sum(r['different'] > 0 for r in rows),
        max_delta=max(r['max_delta'] for r in rows), rows=rows,
        saturation_example=dict(smooth=[32767], detail=[-32768],
            node_boundary=inverse_step([32767], [-32768]),
            step_boundary=inverse_step([32767], [-32768], step_saturation=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    report = audit()
    if args.summary:
        del report['rows']
    print(json.dumps(report, indent=2))
    return int(report['failing_cases'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
