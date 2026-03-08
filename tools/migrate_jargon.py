#!/usr/bin/env python3
"""
Migrate remaining internal jargon to sound-engineering vocabulary.

Replaces in all preset YAML files:
  - Emphasis family names: CONSTRAINT→stability, CALM→settling, etc.
  - Tree names: T1→harmonic, T2→texture, etc.
  - Character field: energy_intent→energy
  - Partials: Q15 integers→normalized floats (0.0–1.0)

See docs/jargon.md for the full mapping.
"""

import re
import sys
from pathlib import Path

# Emphasis families: internal → sound engineering
FAMILY_MAP = {
    'CONSTRAINT': 'stability',
    'CALM': 'settling',
    'COLOR': 'tone',
    'LOCAL': 'body',
    'TRANSPORT': 'spread',
    'COHERENCE': 'clarity',
    'EXCITE': 'energy',
    'TOPOLOGY': 'warp',
    'CATASTROPHE': 'chaos',
}

# Tree names: internal → sound engineering
TREE_MAP = {
    'T0': 'balanced',
    'T1': 'harmonic',
    'T2': 'texture',
    'T3': 'impact',
    'T4': 'drone',
}


def replace_emphasis_families(line: str) -> str:
    """Replace emphasis family names in a line."""
    for old, new in FAMILY_MAP.items():
        # Match: FAMILY_NAME: value (in YAML flow mappings)
        line = line.replace(f'{old}:', f'{new}:')
    return line


def replace_tree_names(line: str) -> str:
    """Replace tree names T1→harmonic etc. in trees: lines."""
    if 'trees:' not in line:
        return line
    for old, new in TREE_MAP.items():
        # Replace T1/T2/etc. with descriptive names in trees context
        # Match: "A: T1" or "B: T2" patterns
        line = re.sub(rf'\b{old}\b', new, line)
    return line


def replace_energy_intent(line: str) -> str:
    """Replace energy_intent with energy in character section."""
    return line.replace('energy_intent:', 'energy:')


def convert_partials_line(line: str) -> str:
    """Convert Q15 integer partials to normalized floats."""
    # Match: partials: [int, int, ...]
    m = re.match(r'^(\s+partials:\s*\[)([\d,\s]+)(\].*)$', line)
    if not m:
        return line
    prefix, values_str, suffix = m.groups()
    values = [int(v.strip()) for v in values_str.split(',') if v.strip()]
    # Check if these look like Q15 (any value > 1)
    if not any(v > 1 for v in values):
        return line  # Already floats
    floats = [round(v / 32767, 3) for v in values]
    float_strs = [str(f) for f in floats]
    return f"{prefix}{', '.join(float_strs)}{suffix}"


def migrate_file(path: Path, dry_run: bool = False) -> bool:
    """Migrate a single YAML file. Returns True if changes were made."""
    text = path.read_text(encoding='utf-8')
    lines = text.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        orig = line
        # Skip comments
        stripped = line.lstrip()
        if stripped.startswith('#'):
            new_lines.append(line)
            continue

        # Apply all replacements
        if 'emphasis:' in line:
            line = replace_emphasis_families(line)
        if 'trees:' in line:
            line = replace_tree_names(line)
        if 'energy_intent:' in line:
            line = replace_energy_intent(line)
        if 'partials:' in line and '[' in line:
            line = convert_partials_line(line)

        if line != orig:
            changed = True
        new_lines.append(line)

    if changed and not dry_run:
        path.write_text('\n'.join(new_lines), encoding='utf-8')

    return changed


def main():
    dry_run = '--dry-run' in sys.argv
    preset_dir = Path(__file__).parent.parent / 'presets' / 'instruments'

    if not preset_dir.exists():
        print(f"ERROR: {preset_dir} not found")
        sys.exit(1)

    yaml_files = sorted(preset_dir.glob('*.yaml'))
    print(f"Found {len(yaml_files)} preset files")
    if dry_run:
        print("DRY RUN - no files will be modified\n")

    changed_count = 0
    for path in yaml_files:
        changed = migrate_file(path, dry_run=dry_run)
        status = "CHANGED" if changed else "ok"
        print(f"  {status}: {path.name}")
        if changed:
            changed_count += 1

    print(f"\n{changed_count}/{len(yaml_files)} files {'would be ' if dry_run else ''}modified")


if __name__ == '__main__':
    main()
