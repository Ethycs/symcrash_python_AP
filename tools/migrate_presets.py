"""
Preset Migration Script (CMMM Enrichment Phase P5)
===================================================

Migrates v0 presets to v1 format with CMMM enrichment defaults.

Usage:
    python -m tools.migrate_presets <input_dir> [--output-dir <output_dir>]
    python -m tools.migrate_presets --check <input_dir>  # Dry-run check

Migration Actions:
1. Adds default temperament values if missing
2. Bumps capsule version from 0 to 1
3. Preserves all existing fields

Reference: CMMM-ENRICHMENT-DESIGN.md §9.7
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Default temperament values (neutral behavior, deterministic)
DEFAULT_TEMPERAMENT = {
    'hysteresis_q15': 16384,          # Q15_HALF - neutral
    'stickiness_q15': 16384,          # Q15_HALF - neutral
    'reactivity_q15': 16384,          # Q15_HALF - neutral
    'escalation_bias_q15': 16384,     # Q15_HALF - neutral
    'fatigue_sensitivity_q15': 16384, # Q15_HALF - neutral
    'tension_influence_q15': 16384,   # Q15_HALF - neutral
    'chaos_amount_q15': 0,            # 0 - deterministic (default)
}


def migrate_preset_dict(preset: dict) -> Tuple[dict, List[str]]:
    """
    Migrate a preset dictionary from v0 to v1 format.

    Args:
        preset: Original preset dictionary

    Returns:
        (migrated_preset, changes): Migrated dict and list of changes made
    """
    result = dict(preset)  # Shallow copy
    changes = []

    # Ensure control_config exists
    if 'control_config' not in result:
        result['control_config'] = {}
        changes.append("Added control_config section")

    # Ensure control exists
    if 'control' not in result:
        result['control'] = {}
        changes.append("Added control section")

    # Add temperament defaults if missing
    control = result.get('control', {})
    if 'temperament' not in control:
        control['temperament'] = dict(DEFAULT_TEMPERAMENT)
        result['control'] = control
        changes.append("Added default temperament parameters")
    else:
        # Fill in any missing temperament fields
        temperament = control['temperament']
        for key, default_val in DEFAULT_TEMPERAMENT.items():
            if key not in temperament:
                temperament[key] = default_val
                changes.append(f"Added missing temperament field: {key}")
        control['temperament'] = temperament
        result['control'] = control

    # Update version if tracked in preset
    if 'version' in result:
        if result['version'] < 1:
            result['version'] = 1
            changes.append(f"Bumped version to 1")

    return result, changes


def migrate_yaml_preset(input_path: Path, output_path: Path) -> List[str]:
    """
    Migrate a YAML preset file.

    Args:
        input_path: Path to input YAML file
        output_path: Path to output YAML file

    Returns:
        List of changes made
    """
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required for YAML migration. Install with: pip install pyyaml")
        sys.exit(1)

    with open(input_path, 'r') as f:
        preset = yaml.safe_load(f)

    if preset is None:
        return ["Skipped: Empty YAML file"]

    migrated, changes = migrate_preset_dict(preset)

    if changes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(migrated, f, default_flow_style=False, sort_keys=False)

    return changes


def migrate_json_preset(input_path: Path, output_path: Path) -> List[str]:
    """
    Migrate a JSON preset file.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file

    Returns:
        List of changes made
    """
    with open(input_path, 'r') as f:
        preset = json.load(f)

    migrated, changes = migrate_preset_dict(preset)

    if changes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(migrated, f, indent=2)

    return changes


def migrate_directory(
    input_dir: Path,
    output_dir: Path = None,
    dry_run: bool = False,
    extensions: List[str] = None
) -> Dict[str, List[str]]:
    """
    Migrate all preset files in a directory.

    Args:
        input_dir: Directory containing presets
        output_dir: Output directory (None = in-place)
        dry_run: If True, don't write files
        extensions: File extensions to process

    Returns:
        Dict mapping filename to list of changes
    """
    if extensions is None:
        extensions = ['.yaml', '.yml', '.json']

    if output_dir is None:
        output_dir = input_dir

    results = {}

    for ext in extensions:
        for input_path in input_dir.rglob(f'*{ext}'):
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path

            try:
                if ext in ('.yaml', '.yml'):
                    changes = migrate_yaml_preset(input_path, output_path) if not dry_run else []
                    if dry_run:
                        # Still compute changes for dry run
                        try:
                            import yaml
                            with open(input_path, 'r') as f:
                                preset = yaml.safe_load(f)
                            if preset:
                                _, changes = migrate_preset_dict(preset)
                        except Exception:
                            changes = ["Error reading file"]
                elif ext == '.json':
                    changes = migrate_json_preset(input_path, output_path) if not dry_run else []
                    if dry_run:
                        with open(input_path, 'r') as f:
                            preset = json.load(f)
                        _, changes = migrate_preset_dict(preset)
                else:
                    continue

                if changes:
                    results[str(rel_path)] = changes

            except Exception as e:
                results[str(rel_path)] = [f"ERROR: {str(e)}"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Migrate presets to CMMM Enrichment v1 format"
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help="Input directory containing preset files"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help="Output directory (default: in-place)"
    )
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help="Dry-run: check what would be migrated without writing"
    )
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.yaml', '.yml', '.json'],
        help="File extensions to process (default: .yaml .yml .json)"
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    print(f"{'Checking' if args.check else 'Migrating'} presets in: {args.input_dir}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print()

    results = migrate_directory(
        args.input_dir,
        args.output_dir,
        dry_run=args.check,
        extensions=args.extensions
    )

    if not results:
        print("No changes needed - all presets are up to date.")
        return

    print(f"{'Would modify' if args.check else 'Modified'} {len(results)} file(s):\n")

    for filename, changes in sorted(results.items()):
        print(f"  {filename}:")
        for change in changes:
            print(f"    - {change}")
        print()

    if args.check:
        print(f"\nDry-run complete. Use without --check to apply changes.")


if __name__ == '__main__':
    main()
