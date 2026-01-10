"""
Test Glitch Lead Preset Loading
===============================

Verifies that the glitch_lead_v0 preset with custom L2 regime graph loads correctly.
"""

from symcrash_python_ap.instrument_ir import load_instrument


def test_glitch_lead():
    ir = load_instrument('src/symcrash_python_ap/library/preset_data/glitch_lead_v0.yaml')
    print('Glitch Lead preset loaded successfully!')
    print(f'  Name: {ir.meta.name}')
    print(f'  ID: {ir.meta.instrument_id}')
    print()
    print('L2 Regime Graph:')
    print(f'  Regime nodes: {len(ir.layers.L2_regimes.regime_nodes)}')
    for node in ir.layers.L2_regimes.regime_nodes:
        dwell = f', dwell={node.dwell_ms}ms' if node.dwell_ms else ''
        print(f'    {node.id} ({node.kind}){dwell}')
    print()
    print(f'  Transitions: {len(ir.layers.L2_regimes.transitions)}')
    for t in ir.layers.L2_regimes.transitions:
        hyst = f' [hyst {t.hysteresis.enter}/{t.hysteresis.exit}]' if t.hysteresis else ''
        print(f'    {t.from_regime} -> {t.to} when "{t.when}"{hyst}')
    print()
    print('L3 Emit Templates:')
    for name, template in ir.layers.L3_cmmm.emit_templates.items():
        ops = ', '.join(template.ops_enable[:3])
        if len(template.ops_enable) > 3:
            ops += f'... (+{len(template.ops_enable) - 3} more)'
        print(f'  {name}: families={list(template.families.keys())}, ops=[{ops}]')

    # Verify structure
    assert len(ir.layers.L2_regimes.regime_nodes) == 5
    assert len(ir.layers.L2_regimes.transitions) == 6
    assert len(ir.layers.L3_cmmm.emit_templates) == 5

    # Verify glitch transition has hysteresis
    glitch_t = [t for t in ir.layers.L2_regimes.transitions if t.to == 'R3_GLITCH'][0]
    assert glitch_t.hysteresis is not None
    assert glitch_t.hysteresis.enter == 0.7
    assert glitch_t.hysteresis.exit == 0.3

    print('\nAll assertions passed!')


if __name__ == '__main__':
    test_glitch_lead()
