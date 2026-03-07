"""
Migrate Presets to Sound-Design Format
======================================

Converts all presets from the layer-based format (layers.L0_graph.nodes[])
to the object-centric sound-design format (graph:, controls:, behavior:).

Usage:
    pixi run python tools/migrate_presets_sound_design.py presets/instruments/
    pixi run python tools/migrate_presets_sound_design.py presets/instruments/saw_lead_v0.yaml
"""

import sys
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. Install with: pip install pyyaml")
    sys.exit(1)


# Section order for new format
SECTION_ORDER = [
    'meta', 'audio', 'graph', 'controls', 'character',
    'envelopes', 'modulators', 'behavior', 'compile',
]


def extract_header_comments(text: str) -> str:
    """Extract header comment block from YAML text."""
    lines = text.split('\n')
    header = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            header.append(line)
        else:
            break
    return '\n'.join(header)


def translate_meta(meta: dict) -> dict:
    """Translate meta section: instrument_id → id, seed_u32 → seed."""
    result = {}
    if 'instrument_id' in meta:
        result['id'] = meta['instrument_id']
    if 'name' in meta:
        result['name'] = meta['name']
    if 'version' in meta:
        result['version'] = meta['version']
    if 'author' in meta:
        result['author'] = meta['author']
    if 'seed_u32' in meta:
        result['seed'] = meta['seed_u32']
    if 'voice_model' in meta:
        result['voice_model'] = meta['voice_model']
    return result


def translate_io(io_config: dict) -> dict:
    """Translate io → audio: sample_rate_hz → sample_rate."""
    result = {}
    if 'sample_rate_hz' in io_config:
        result['sample_rate'] = io_config['sample_rate_hz']
    if 'block_size' in io_config:
        result['block_size'] = io_config['block_size']
    return result


def translate_adsr_keys(adsr: dict) -> dict:
    """Translate a_ms/d_ms/s/r_ms → attack/decay/sustain/release."""
    result = {}
    if 'a_ms' in adsr:
        result['attack'] = adsr['a_ms']
    if 'd_ms' in adsr:
        result['decay'] = adsr['d_ms']
    if 's' in adsr:
        result['sustain'] = adsr['s']
    if 'r_ms' in adsr:
        result['release'] = adsr['r_ms']
    return result


def translate_graph(l0_graph: dict) -> dict:
    """Translate L0_graph (nodes/edges) → graph dict with output: on each node."""
    nodes = l0_graph.get('nodes', [])
    edges = l0_graph.get('edges', [])

    # Build edge map: src → dst
    edge_map = {}
    for edge in edges:
        edge_map[edge['src']] = edge['dst']

    graph = {}
    for node in nodes:
        node_id = node['id']
        node_type = node['type']
        params = node.get('params', {})

        entry = {'type': node_type}

        # Add output if this node has an edge
        if node_id in edge_map:
            entry['output'] = edge_map[node_id]

        if node_type == 'Amp':
            # Translate ADSR from params.adsr to flat keys
            adsr = params.get('adsr', {})
            if adsr:
                entry.update(translate_adsr_keys(adsr))
            # Copy any other params
            for k, v in params.items():
                if k != 'adsr':
                    entry[k] = v
        else:
            # All other node types: copy params flat
            entry.update(params)

        graph[node_id] = entry

    return graph


def translate_controls(macros: list) -> dict:
    """Translate macros list → controls dict (name → default)."""
    controls = {}
    for macro in macros:
        name = macro.get('name', '')
        default = macro.get('default', 0.5)
        # Simple format: just name → default
        controls[name] = default
    return controls


def translate_behavior_simple(emit_templates: dict) -> dict:
    """Translate emit_templates → behavior (simple format, no state machine)."""
    behavior = {}
    for template_name, template_data in emit_templates.items():
        # T_IDLE → idle, T_STABLE → stable, etc.
        state_name = template_name.replace('T_', '').lower()

        state = {}

        # ops_enable → processing
        if 'ops_enable' in template_data:
            state['processing'] = template_data['ops_enable']

        # families → emphasis (only if non-trivial)
        if 'families' in template_data:
            state['emphasis'] = template_data['families']

        # trees: beta → blend
        if 'trees' in template_data:
            trees = dict(template_data['trees'])
            if 'beta' in trees:
                trees['blend'] = trees.pop('beta')
            state['trees'] = trees

        # cmmm → resolution/safety/ramp/traction
        if 'cmmm' in template_data:
            cmmm = template_data['cmmm']
            if 'node_mask' in cmmm:
                mask = cmmm['node_mask']
                if isinstance(mask, int):
                    state['resolution'] = mask_to_resolution(mask)
                else:
                    state['resolution'] = mask
            if 'tail_mode' in cmmm:
                state['safety'] = cmmm['tail_mode']
            if 'mask_ramp_blocks' in cmmm:
                state['ramp'] = cmmm['mask_ramp_blocks']
            if 'traction' in cmmm:
                state['traction'] = cmmm['traction']
            elif 'traction_q15' in cmmm:
                state['traction'] = cmmm['traction_q15'] / 32767.0

        behavior[state_name] = state

    return behavior


def translate_behavior_complex(l2_regimes: dict, emit_templates: dict) -> dict:
    """Translate L2_regimes + emit_templates → behavior with states/transitions."""
    behavior = {}

    # Build regime_id → state_name mapping
    regime_nodes = l2_regimes.get('regime_nodes', [])
    transitions = l2_regimes.get('transitions', [])

    # Extract state names from regime IDs (R0_IDLE → idle, R2_STABLE → stable)
    regime_to_state = {}
    states = {}
    for node in regime_nodes:
        regime_id = node['id']
        # Extract state name: R0_IDLE → idle, R3_GLITCH → glitch
        parts = regime_id.split('_', 1)
        state_name = parts[1].lower() if len(parts) > 1 else regime_id.lower()
        regime_to_state[regime_id] = state_name

        state_def = {'kind': node.get('kind', '')}
        if node.get('dwell_ms') is not None:
            state_def['dwell'] = node['dwell_ms']
        states[state_name] = state_def

    behavior['states'] = states

    # Translate transitions
    new_transitions = []
    for t in transitions:
        from_regime = t.get('from', '')
        to_regime = t.get('to', '')
        from_state = regime_to_state.get(from_regime, from_regime)
        to_state = regime_to_state.get(to_regime, to_regime)

        trans = {'from': from_state, 'to': to_state, 'when': t.get('when', '')}
        if 'hysteresis' in t:
            trans['hysteresis'] = t['hysteresis']
        new_transitions.append(trans)

    behavior['transitions'] = new_transitions

    # Add emit templates as state processing dicts
    simple_behavior = translate_behavior_simple(emit_templates)
    behavior.update(simple_behavior)

    return behavior


def translate_envelopes(envelopes: list) -> dict:
    """Translate envelopes list → envelopes dict with flat ADSR."""
    result = {}
    for env in envelopes:
        env_id = env.get('id', '')
        entry = {}
        if 'adsr' in env:
            entry.update(translate_adsr_keys(env['adsr']))
        if 'targets' in env:
            entry['targets'] = env['targets']
        result[env_id] = entry
    return result


def translate_modulators(modulators: dict) -> dict:
    """Translate modulators.lfos list → modulators dict with short keys."""
    lfos = modulators.get('lfos', [])
    if not lfos:
        return {}

    result = {}
    for lfo in lfos:
        lfo_id = lfo.get('id', '')
        entry = {}

        if 'waveform' in lfo:
            entry['waveform'] = lfo['waveform']
        if 'rate_hz' in lfo:
            entry['rate'] = lfo['rate_hz']
        if 'depth' in lfo:
            entry['depth'] = lfo['depth']
        if 'delay_ms' in lfo:
            entry['delay'] = lfo['delay_ms']
        if 'ramp_ms' in lfo:
            entry['ramp'] = lfo['ramp_ms']
        if 'sync_mode' in lfo:
            entry['sync'] = lfo['sync_mode']

        # Translate targets: [{type: "pitch", amount: 1.0}] → [{pitch: 1.0}]
        if 'targets' in lfo:
            new_targets = []
            for t in lfo['targets']:
                if isinstance(t, dict) and 'type' in t:
                    new_targets.append({t['type']: t.get('amount', 1.0)})
                else:
                    new_targets.append(t)
            entry['targets'] = new_targets

        result[lfo_id] = entry

    return result


def translate_compile_material(material: dict) -> dict:
    """Translate material: beta_start → blend, beta_target → target, beta_ramp_blocks → ramp."""
    result = {}
    if 'beta_start' in material:
        result['blend'] = material['beta_start']
    elif 'beta_start_q15' in material:
        result['blend'] = material['beta_start_q15'] / 32767.0
    if 'beta_target' in material:
        result['target'] = material['beta_target']
    elif 'beta_target_q15' in material:
        result['target'] = material['beta_target_q15'] / 32767.0
    if 'beta_ramp_blocks' in material:
        result['ramp'] = material['beta_ramp_blocks']
    return result


def mask_to_resolution(mask: int) -> int:
    """Convert node_mask to K resolution value."""
    mask_to_k = {
        0x00000001: 1,
        0x0000000F: 4,
        0x000000FF: 8,
        0x00000FFF: 12,
        0x0000FFFF: 16,
        0x00FFFFFF: 24,
        0xFFFFFFFF: 32,
    }
    if isinstance(mask, int):
        return mask_to_k.get(mask & 0xFFFFFFFF, 32)
    return 32


def migrate_preset(filepath: Path) -> str:
    """Migrate a single preset file to sound-design format. Returns new YAML text."""
    text = filepath.read_text(encoding='utf-8')

    # Extract header comments
    header = extract_header_comments(text)

    # Parse YAML
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or 'instrument_ir_v0' not in data:
        raise ValueError(f"Not an instrument_ir_v0 file: {filepath}")

    ir = data['instrument_ir_v0']

    # Build new format
    new_ir = {}

    # 1. Meta
    new_ir['meta'] = translate_meta(ir.get('meta', {}))

    # 2. Audio
    new_ir['audio'] = translate_io(ir.get('io', {}))

    # 3. Graph
    l0_graph = ir.get('layers', {}).get('L0_graph', {})
    new_ir['graph'] = translate_graph(l0_graph)

    # 4. Controls
    macros = l0_graph.get('macros', [])
    if macros:
        new_ir['controls'] = translate_controls(macros)

    # 5. Character (from control:)
    if 'control' in ir:
        new_ir['character'] = ir['control']

    # 6. Envelopes
    if 'envelopes' in ir and ir['envelopes']:
        new_ir['envelopes'] = translate_envelopes(ir['envelopes'])

    # 7. Modulators
    if 'modulators' in ir and ir['modulators']:
        new_ir['modulators'] = translate_modulators(ir['modulators'])

    # 8. Behavior
    l2_regimes = ir.get('layers', {}).get('L2_regimes', {})
    l3_cmmm = ir.get('layers', {}).get('L3_cmmm', {})
    emit_templates = l3_cmmm.get('emit_templates', {})

    if l2_regimes.get('regime_nodes'):
        # Complex behavior with state machine
        new_ir['behavior'] = translate_behavior_complex(l2_regimes, emit_templates)
    elif emit_templates:
        # Simple behavior (just templates)
        new_ir['behavior'] = translate_behavior_simple(emit_templates)

    # 9. Compile
    if 'compile' in ir:
        compile_data = dict(ir['compile'])
        if 'material' in compile_data:
            compile_data['material'] = translate_compile_material(compile_data['material'])
        # Remove empty sub-sections that are just defaults
        new_compile = {}
        for k, v in compile_data.items():
            if v:  # Only include non-empty sections
                new_compile[k] = v
        if new_compile:
            new_ir['compile'] = new_compile

    # Render YAML
    output = render_yaml(new_ir, header)
    return output


def render_yaml(ir_data: dict, header: str) -> str:
    """Render the new format as clean YAML text."""
    lines = []

    # Header comments
    if header.strip():
        lines.append(header.rstrip())
        lines.append('')

    lines.append('instrument_ir_v0:')

    for section in SECTION_ORDER:
        if section not in ir_data:
            continue

        data = ir_data[section]
        lines.append(f'  {section}:')

        if section == 'meta':
            render_meta(lines, data)
        elif section == 'audio':
            render_audio(lines, data)
        elif section == 'graph':
            render_graph(lines, data)
        elif section == 'controls':
            render_controls(lines, data)
        elif section == 'character':
            render_character(lines, data)
        elif section == 'envelopes':
            render_envelopes(lines, data)
        elif section == 'modulators':
            render_modulators(lines, data)
        elif section == 'behavior':
            render_behavior(lines, data)
        elif section == 'compile':
            render_compile(lines, data)

        lines.append('')

    # Remove trailing blank lines
    while lines and lines[-1] == '':
        lines.pop()

    return '\n'.join(lines) + '\n'


def render_meta(lines, data):
    lines.append(f'    id: "{data.get("id", "")}"')
    lines.append(f'    name: "{data.get("name", "")}"')
    lines.append(f'    version: {data.get("version", 0)}')
    lines.append(f'    author: "{data.get("author", "")}"')
    seed = data.get('seed', 0)
    if isinstance(seed, int):
        lines.append(f'    seed: {hex(seed).upper().replace("X", "x")}')
    else:
        lines.append(f'    seed: {seed}')


def render_audio(lines, data):
    lines.append(f'    sample_rate: {data.get("sample_rate", 48000)}')
    lines.append(f'    block_size: {data.get("block_size", 64)}')


def render_graph(lines, data):
    for node_id, node_data in data.items():
        lines.append(f'    {node_id}:')
        node_type = node_data.get('type', '')
        lines.append(f'      type: {node_type}')

        if 'output' in node_data:
            lines.append(f'      output: {node_data["output"]}')

        if node_type == 'Amp':
            # Render ADSR keys
            for key in ['attack', 'decay', 'sustain', 'release']:
                if key in node_data:
                    lines.append(f'      {key}: {node_data[key]}')
        elif node_type == 'FM6':
            # Render FM6 params
            if 'algorithm' in node_data:
                lines.append(f'      algorithm: {node_data["algorithm"]}')
            if 'feedback' in node_data:
                lines.append(f'      feedback: {node_data["feedback"]}')
            if 'operators' in node_data:
                lines.append('      operators:')
                for op in node_data['operators']:
                    ratio = op.get('ratio', 1.0)
                    level = op.get('level', 1.0)
                    lines.append(f'        - ratio: {ratio}')
                    lines.append(f'          level: {level}')
        elif node_type == 'Osc':
            # Render Osc params
            if 'waveform' in node_data:
                lines.append(f'      waveform: {node_data["waveform"]}')
            if 'pw' in node_data:
                lines.append(f'      pw: {node_data["pw"]}')
            if 'partials' in node_data:
                partials = node_data['partials']
                lines.append(f'      partials: [{", ".join(str(p) for p in partials)}]')
            if 'harmonic_amps_q15' in node_data:
                amps = node_data['harmonic_amps_q15']
                lines.append(f'      harmonic_amps_q15: [{", ".join(str(a) for a in amps)}]')


def render_controls(lines, data):
    for name, value in data.items():
        lines.append(f'    {name}: {value}')


def render_character(lines, data):
    for key, value in data.items():
        lines.append(f'    {key}: {value}')


def render_envelopes(lines, data):
    for env_id, env_data in data.items():
        lines.append(f'    {env_id}:')
        for key in ['attack', 'decay', 'sustain', 'release']:
            if key in env_data:
                lines.append(f'      {key}: {env_data[key]}')
        if 'targets' in env_data:
            targets = env_data['targets']
            lines.append(f'      targets: [{", ".join(str(t) for t in targets)}]')


def render_modulators(lines, data):
    for lfo_id, lfo_data in data.items():
        lines.append(f'    {lfo_id}:')
        for key in ['waveform', 'rate', 'depth', 'delay', 'ramp', 'sync']:
            if key in lfo_data:
                lines.append(f'      {key}: {lfo_data[key]}')
        if 'targets' in lfo_data:
            lines.append('      targets:')
            for t in lfo_data['targets']:
                if isinstance(t, dict):
                    for target_type, amount in t.items():
                        lines.append(f'        - {target_type}: {amount}')


def render_behavior(lines, data):
    # States and transitions first (if complex)
    if 'states' in data:
        lines.append('    states:')
        for state_name, state_def in data['states'].items():
            parts = []
            parts.append(f'kind: {state_def["kind"]}')
            if 'dwell' in state_def:
                parts.append(f'dwell: {state_def["dwell"]}')
            lines.append(f'      {state_name}: {{{", ".join(parts)}}}')

        lines.append('')
        lines.append('    transitions:')
        for t in data['transitions']:
            lines.append(f'      - from: {t["from"]}')
            lines.append(f'        to: {t["to"]}')
            lines.append(f'        when: "{t["when"]}"')
            if 'hysteresis' in t:
                h = t['hysteresis']
                lines.append(f'        hysteresis: {{enter: {h["enter"]}, exit: {h["exit"]}}}')

        lines.append('')

    # State processing dicts
    reserved = {'states', 'transitions'}
    for state_name, state_def in data.items():
        if state_name in reserved:
            continue
        if not isinstance(state_def, dict):
            continue

        lines.append(f'    {state_name}:')

        if 'processing' in state_def:
            ops = state_def['processing']
            lines.append(f'      processing: [{", ".join(ops)}]')

        if 'emphasis' in state_def:
            emphasis = state_def['emphasis']
            parts = [f'{k}: {v}' for k, v in emphasis.items()]
            lines.append(f'      emphasis: {{{", ".join(parts)}}}')

        if 'trees' in state_def:
            trees = state_def['trees']
            parts = [f'{k}: {v}' for k, v in trees.items()]
            lines.append(f'      trees: {{{", ".join(parts)}}}')

        if 'resolution' in state_def:
            lines.append(f'      resolution: {state_def["resolution"]}')
        if 'safety' in state_def:
            lines.append(f'      safety: {state_def["safety"]}')
        if 'ramp' in state_def:
            lines.append(f'      ramp: {state_def["ramp"]}')
        if 'traction' in state_def:
            lines.append(f'      traction: {state_def["traction"]}')


def render_compile(lines, data):
    if 'material' in data:
        lines.append('    material:')
        material = data['material']
        if 'blend' in material:
            lines.append(f'      blend: {material["blend"]}')
        if 'target' in material:
            lines.append(f'      target: {material["target"]}')
        if 'ramp' in material:
            lines.append(f'      ramp: {material["ramp"]}')
    if 'completion_policy' in data:
        lines.append('    completion_policy:')
        cp = data['completion_policy']
        if 'default_tree' in cp:
            tree = cp['default_tree']
            parts = [f'{k}: {v}' for k, v in tree.items()]
            lines.append(f'      default_tree: {{{", ".join(parts)}}}')


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/migrate_presets_sound_design.py <path>")
        sys.exit(1)

    target = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob('*.yaml'))
    else:
        print(f"ERROR: {target} not found")
        sys.exit(1)

    ok = 0
    fail = 0

    for f in files:
        try:
            new_text = migrate_preset(f)

            if dry_run:
                print(f"--- {f.name} ---")
                print(new_text[:500])
                print("...")
            else:
                f.write_text(new_text, encoding='utf-8')

            # Verify: load through parser
            from symcrash_python_ap.instrument_ir import load_instrument
            ir = load_instrument(f) if not dry_run else None
            if ir:
                print(f"  OK    {f.name} ({len(ir.layers.L0_graph.nodes)} nodes, "
                      f"{len(ir.layers.L3_cmmm.emit_templates)} templates)")
            else:
                print(f"  OK    {f.name} (dry run)")
            ok += 1

        except Exception as e:
            print(f"  FAIL  {f.name}: {e}")
            fail += 1

    print(f"\n{ok}/{len(files)} migrated OK, {fail} failures")
    if fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
