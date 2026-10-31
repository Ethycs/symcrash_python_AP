"""Build and validate Mode A resources before starting audio playback."""
import argparse
from pathlib import Path
from time import perf_counter

from symcrash_python_ap.vm.mode_a_resource import PreparedModeAAtlas, default_resource_path
from symcrash_python_ap.vm.mode_a_resource_builder import build_mode_a_resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Resource path (default: presets/.build/resources)')
    parser.add_argument('--force', action='store_true', help='Atomically rebuild an existing resource')
    parser.add_argument('--workers', type=int, help='Offline build workers (default: up to 8)')
    args = parser.parse_args()
    target = Path(args.output) if args.output else default_resource_path()
    begin = perf_counter()
    if args.force or not target.exists():
        build_mode_a_resource(target, workers=args.workers,
            progress=lambda done, total: print(f'{done}/{total} tables', flush=True))
    atlas = PreparedModeAAtlas(target)
    print(f'Validated {len(atlas.keys)} tables, {target.stat().st_size} bytes in {perf_counter()-begin:.2f}s: {target}')


if __name__ == '__main__':
    main()
