"""Fail when default oscillator resource recipes exceed OrangeCrab memory limits.

This is a required storage check, not a complete device qualification. It does
not build resources or accept desktop paging as a substitute for target RAM.
"""
import argparse
from dataclasses import asdict
import json

from symcrash_python_ap.budgets.hardware_resources import (
    oscillator_allocations, check_allocations, COEFFICIENT_STORE_BYTES,
    EXTERNAL_RAM_BYTES, COEFFICIENT_WORKING_BYTES, BRAM_BYTES,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Emit a machine-readable result')
    args = parser.parse_args(argv)
    allocations = oscillator_allocations()
    failures = check_allocations(allocations)
    report = dict(target='OrangeCrab ECP5-25F',
        scope='Default oscillator image storage; other allocations and timing are not qualified',
        allocations=[asdict(item) for item in allocations],
        limits=dict(coefficients=COEFFICIENT_STORE_BYTES, external_total=EXTERNAL_RAM_BYTES,
                    coefficient_working=COEFFICIENT_WORKING_BYTES, bram_total=BRAM_BYTES),
        violations=[dict(asdict(item), excess_bytes=item.excess_bytes) for item in failures],
        storage_passed=not failures)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in allocations:
            print(f'{item.name}: {item.bytes:,} bytes')
        for item in failures:
            print(f'FAIL {item.region}: {item.used_bytes:,} > {item.limit_bytes:,} bytes '
                  f'(over by {item.excess_bytes:,})')
        if not failures:
            print('PASS default oscillator storage')
        print(report['scope'])
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
