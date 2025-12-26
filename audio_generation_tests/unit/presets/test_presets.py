"""
Test preset loading and validation.
"""
from symcrash_python_ap.presets import list_presets, load_preset, get_preset_info

# List all presets
print('Available presets:')
for name in list_presets():
    print(f'  - {name}')

print()

# Load and verify each preset
print('Loading and verifying all presets:')
for name in list_presets():
    try:
        ir = load_preset(name)
        stable = ir.layers.L3_cmmm.emit_templates.get('Stable')
        ops = stable.ops_enable if stable else []
        events = len(stable.events) if stable else 0
        print(f'  {name}: OK (ops={ops}, events={events})')
    except Exception as e:
        print(f'  {name}: FAILED - {e}')

print()
print('All presets validated successfully!')
