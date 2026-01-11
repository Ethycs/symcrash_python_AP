"""
Test L2 Regime Graph Parsing
============================

Verifies that L2 regime graphs can be parsed from YAML/dict format.
"""

from symcrash_python_ap.instrument_ir import load_instrument_from_dict


def test_l2_regime_parsing():
    """Test L2 regime parsing from dict."""
    data = {
        'meta': {'instrument_id': 'test', 'name': 'Test'},
        'layers': {
            'L2_regimes': {
                'regime_nodes': [
                    {'id': 'R0_IDLE', 'kind': 'Idle', 'emit_template': 'T_IDLE'},
                    {'id': 'R1_ATTACK', 'kind': 'Transient', 'emit_template': 'T_ATTACK', 'dwell_ms': 20},
                    {'id': 'R2_STABLE', 'kind': 'StableTone', 'emit_template': 'T_STABLE'},
                    {'id': 'R3_GLITCH', 'kind': 'GlitchChatter', 'emit_template': 'T_GLITCH', 'dwell_ms': 50},
                ],
                'transitions': [
                    {'from': 'R0_IDLE', 'to': 'R1_ATTACK', 'when': 'Gate rises'},
                    {'from': 'R1_ATTACK', 'to': 'R2_STABLE', 'when': 'dwell_done'},
                    {'from': 'R2_STABLE', 'to': 'R3_GLITCH', 'when': 'risk > 0.7', 'hysteresis': {'enter': 0.7, 'exit': 0.4}},
                    {'from': 'any', 'to': 'R0_IDLE', 'when': 'Gate falls'},
                ],
            },
        },
    }

    ir = load_instrument_from_dict(data)
    print('L2 Regime Graph parsed successfully!')
    print(f'  Regime nodes: {len(ir.layers.L2_regimes.regime_nodes)}')
    for node in ir.layers.L2_regimes.regime_nodes:
        print(f'    - {node.id} ({node.kind}), template={node.emit_template}, dwell={node.dwell_ms}')
    print(f'  Transitions: {len(ir.layers.L2_regimes.transitions)}')
    for t in ir.layers.L2_regimes.transitions:
        hyst = f', hysteresis={t.hysteresis.enter}/{t.hysteresis.exit}' if t.hysteresis else ''
        print(f'    - {t.from_regime} -> {t.to} when "{t.when}"{hyst}')

    # Verify counts
    assert len(ir.layers.L2_regimes.regime_nodes) == 4
    assert len(ir.layers.L2_regimes.transitions) == 4

    # Verify hysteresis parsing
    glitch_transition = ir.layers.L2_regimes.transitions[2]
    assert glitch_transition.hysteresis is not None
    assert glitch_transition.hysteresis.enter == 0.7
    assert glitch_transition.hysteresis.exit == 0.4

    print('\nAll assertions passed!')


if __name__ == '__main__':
    test_l2_regime_parsing()
