"""Prepare the reproducible Mode B atlas before starting audio playback."""
import argparse
from time import perf_counter

from symcrash_python_ap.vm.basis_resource import PreparedBasisAtlas, default_resource_path
from symcrash_python_ap.vm.basis_resource_builder import build_basis_resource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=str, help='Resource path (default: presets/.build/resources)')
    parser.add_argument('--force', action='store_true', help='Rebuild an existing resource atomically')
    args = parser.parse_args()
    from pathlib import Path
    target = Path(args.output) if args.output else default_resource_path()
    start = perf_counter()
    if args.force or not target.exists():
        build_basis_resource(target, progress=lambda done, total: print(f'{done}/{total} source anchors', flush=True))
    atlas = PreparedBasisAtlas(target)
    print(f'Validated {len(atlas.keys)} source anchors, {target.stat().st_size} bytes in {perf_counter()-start:.2f}s: {target}')


if __name__ == '__main__':
    main()
