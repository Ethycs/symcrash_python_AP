"""
Test Editor Regime Graph Commands
==================================

Tests for the L2 regime graph-rewrite editor commands, typed regime graphs,
and V4 graph validator per problem.txt requirements.
"""

import pytest
from symcrash_python_ap.editor import (
    EditorSession,
    AddRegime,
    RemoveRegime,
    AddTransition,
    RemoveTransition,
    validate_regime_graph,
)
from symcrash_python_ap.instrument_ir import (
    InstrumentIR,
    InstrumentMeta,
    IOConfig,
    Layers,
    L0Graph,
    L1Lift,
    L2Regimes,
    L3Cmmm,
    RegimeNode,
    RegimeTransition,
    RegimeGraphSpec,
    EdgeSpec,
    Hysteresis,
)


def test_add_regime_legacy_mode():
    """Test AddRegime in legacy mode (no regime_graph_spec)."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                ]
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(AddRegime("R1_ATTACK", kind="Attack", emit_template="T_ATTACK"))
    
    assert result.ok
    assert len(ir.layers.L2_regimes.regime_nodes) == 2
    assert ir.layers.L2_regimes.regime_nodes[1].id == "R1_ATTACK"
    assert ir.layers.L2_regimes.regime_nodes[1].kind == "Attack"


def test_add_regime_duplicate_fails():
    """Test that adding duplicate regime fails."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                ]
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(AddRegime("R0_IDLE"))
    
    assert not result.ok
    assert "already exists" in result.message


def test_remove_regime_legacy_mode():
    """Test RemoveRegime in legacy mode with transitions cleanup."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                    RegimeNode(id="R2_SUSTAIN", kind="Sustain", emit_template="T_SUSTAIN"),
                ],
                transitions=[
                    RegimeTransition(from_regime="R0_IDLE", to="R1_ATTACK", when="gate_on"),
                    RegimeTransition(from_regime="R1_ATTACK", to="R2_SUSTAIN", when="attack_done"),
                    RegimeTransition(from_regime="R2_SUSTAIN", to="R0_IDLE", when="gate_off"),
                ]
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(RemoveRegime("R1_ATTACK"))
    
    assert result.ok
    assert len(ir.layers.L2_regimes.regime_nodes) == 2
    # Should remove 2 transitions involving R1_ATTACK
    assert len(ir.layers.L2_regimes.transitions) == 1


def test_remove_start_regime_fails():
    """Test that removing start regime fails."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                ]
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(RemoveRegime("R0_IDLE"))
    
    assert not result.ok
    assert "start regime" in result.message.lower()


def test_add_transition_legacy_mode():
    """Test AddTransition in legacy mode."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                ]
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(AddTransition("R0_IDLE", "R1_ATTACK", when="gate_on"))
    
    assert result.ok
    assert len(ir.layers.L2_regimes.transitions) == 1
    assert ir.layers.L2_regimes.transitions[0].from_regime == "R0_IDLE"
    assert ir.layers.L2_regimes.transitions[0].to == "R1_ATTACK"
    assert ir.layers.L2_regimes.transitions[0].when == "gate_on"


def test_add_transition_typed_mode():
    """Test AddTransition in typed mode (regime_graph_spec)."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                ],
                regime_graph_spec=RegimeGraphSpec(
                    start_regime="R0_IDLE",
                    edges=[]
                )
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(AddTransition("R0_IDLE", "R1_ATTACK", when="gate_on"))
    
    assert result.ok
    assert len(ir.layers.L2_regimes.regime_graph_spec.edges) == 1
    edge = ir.layers.L2_regimes.regime_graph_spec.edges[0]
    assert edge.src_regime == "R0_IDLE"
    assert edge.dst_regime == "R1_ATTACK"
    assert edge.guard == "gate_on"
    # Should not touch legacy transitions
    assert len(ir.layers.L2_regimes.transitions) == 0


def test_add_transition_any_wildcard():
    """Test AddTransition with 'any' wildcard source."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_PANIC", kind="Panic", emit_template="T_PANIC"),
                ],
                regime_graph_spec=RegimeGraphSpec(
                    start_regime="R0_IDLE",
                    edges=[]
                )
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(AddTransition("any", "R1_PANIC", when="emergency"))
    
    assert result.ok
    assert len(ir.layers.L2_regimes.regime_graph_spec.edges) == 1
    edge = ir.layers.L2_regimes.regime_graph_spec.edges[0]
    assert edge.src_regime == "any"
    assert edge.dst_regime == "R1_PANIC"


def test_remove_transition_typed_mode():
    """Test RemoveTransition in typed mode."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                ],
                regime_graph_spec=RegimeGraphSpec(
                    start_regime="R0_IDLE",
                    edges=[
                        EdgeSpec(src_regime="R0_IDLE", dst_regime="R1_ATTACK", guard="gate_on")
                    ]
                )
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(RemoveTransition("R0_IDLE", "R1_ATTACK"))
    
    assert result.ok
    assert len(ir.layers.L2_regimes.regime_graph_spec.edges) == 0


def test_undo_redo_exact():
    """Test that undo/redo produces exact state equality."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                ],
                regime_graph_spec=RegimeGraphSpec(
                    start_regime="R0_IDLE",
                    edges=[]
                )
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    import copy
    l2_before = copy.deepcopy(ir.layers.L2_regimes)
    
    session = EditorSession(ir)
    session.execute(AddRegime("R1_ATTACK"))
    session.execute(AddTransition("R0_IDLE", "R1_ATTACK"))
    
    # Undo both
    session.undo()
    session.undo()
    
    # Should be exactly equal
    assert ir.layers.L2_regimes == l2_before
    
    # Redo both
    session.redo()
    session.redo()
    
    # Should have the changes back
    assert len(ir.layers.L2_regimes.regime_nodes) == 2
    assert len(ir.layers.L2_regimes.regime_graph_spec.edges) == 1


def test_validate_regime_graph_empty():
    """Test that empty L2 is valid."""
    l2 = L2Regimes()
    errors = validate_regime_graph(l2)
    assert len(errors) == 0


def test_validate_regime_graph_duplicate_ids():
    """Test validator catches duplicate regime IDs."""
    l2 = L2Regimes(
        regime_nodes=[
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
        ]
    )
    errors = validate_regime_graph(l2)
    assert len(errors) > 0
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_regime_graph_invalid_start():
    """Test validator catches invalid start regime."""
    l2 = L2Regimes(
        regime_nodes=[
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
        ],
        regime_graph_spec=RegimeGraphSpec(
            start_regime="R99_INVALID",
            edges=[]
        )
    )
    errors = validate_regime_graph(l2)
    assert len(errors) > 0
    assert any("start regime" in e.lower() for e in errors)


def test_validate_regime_graph_invalid_edge_dst():
    """Test validator catches edges to non-existent destinations."""
    l2 = L2Regimes(
        regime_nodes=[
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
        ],
        regime_graph_spec=RegimeGraphSpec(
            start_regime="R0_IDLE",
            edges=[
                EdgeSpec(src_regime="R0_IDLE", dst_regime="R99_INVALID", guard="true")
            ]
        )
    )
    errors = validate_regime_graph(l2)
    assert len(errors) > 0
    assert any("destination" in e.lower() and "not a node" in e.lower() for e in errors)


def test_validate_regime_graph_invalid_edge_src():
    """Test validator catches edges from non-existent sources."""
    l2 = L2Regimes(
        regime_nodes=[
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
        ],
        regime_graph_spec=RegimeGraphSpec(
            start_regime="R0_IDLE",
            edges=[
                EdgeSpec(src_regime="R99_INVALID", dst_regime="R0_IDLE", guard="true")
            ]
        )
    )
    errors = validate_regime_graph(l2)
    assert len(errors) > 0
    assert any("source" in e.lower() and "not a node" in e.lower() for e in errors)


def test_validate_regime_graph_unreachable_nodes():
    """Test validator catches unreachable nodes."""
    l2 = L2Regimes(
        regime_nodes=[
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
            RegimeNode(id="R1_ORPHAN", kind="Orphan", emit_template="T_ORPHAN"),
        ],
        regime_graph_spec=RegimeGraphSpec(
            start_regime="R0_IDLE",
            edges=[]
        )
    )
    errors = validate_regime_graph(l2)
    assert len(errors) > 0
    assert any("not reachable" in e.lower() for e in errors)


def test_validate_regime_graph_any_makes_reachable():
    """Test that 'any' edges make destinations reachable."""
    l2 = L2Regimes(
        regime_nodes=[
            RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
            RegimeNode(id="R1_PANIC", kind="Panic", emit_template="T_PANIC"),
        ],
        regime_graph_spec=RegimeGraphSpec(
            start_regime="R0_IDLE",
            edges=[
                EdgeSpec(src_regime="any", dst_regime="R1_PANIC", guard="emergency")
            ]
        )
    )
    errors = validate_regime_graph(l2)
    # R1_PANIC should be reachable via 'any' edge
    assert len(errors) == 0


def test_atomic_batch_success():
    """Test atomic batch execution with success."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                ],
                regime_graph_spec=RegimeGraphSpec(start_regime="R0_IDLE", edges=[])
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    commands = [
        AddRegime("R1_ATTACK"),
        AddRegime("R2_SUSTAIN"),
        AddTransition("R0_IDLE", "R1_ATTACK"),
        AddTransition("R1_ATTACK", "R2_SUSTAIN"),
    ]
    
    results = session.execute_batch(commands, atomic=True)
    
    # All should succeed
    assert all(r.ok for r in results)
    assert len(ir.layers.L2_regimes.regime_nodes) == 3
    assert len(ir.layers.L2_regimes.regime_graph_spec.edges) == 2
    
    # Should be one history entry for the whole batch
    assert session.can_undo()
    session.undo()
    # All changes should be undone
    assert len(ir.layers.L2_regimes.regime_nodes) == 1


def test_atomic_batch_rollback_on_failure():
    """Test atomic batch rolls back on command failure."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                ],
                regime_graph_spec=RegimeGraphSpec(start_regime="R0_IDLE", edges=[])
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    commands = [
        AddRegime("R1_ATTACK"),
        AddRegime("R1_ATTACK"),  # Duplicate - will fail
    ]
    
    results = session.execute_batch(commands, atomic=True)
    
    # Second command should fail
    assert not results[1].ok
    # Entire batch should be rolled back
    assert len(ir.layers.L2_regimes.regime_nodes) == 1
    # No history entry
    assert not session.can_undo()


def test_atomic_batch_rollback_on_validation():
    """Test atomic batch rolls back on validation failure."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                ],
                regime_graph_spec=RegimeGraphSpec(
                    start_regime="R0_IDLE",
                    edges=[
                        EdgeSpec(src_regime="R0_IDLE", dst_regime="R1_ATTACK", guard="gate")
                    ]
                )
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    # Start with valid graph
    errors = validate_regime_graph(ir.layers.L2_regimes)
    assert len(errors) == 0
    
    session = EditorSession(ir)
    commands = [
        # This will make R1_ATTACK unreachable
        RemoveTransition("R0_IDLE", "R1_ATTACK"),
    ]
    
    results = session.execute_batch(commands, atomic=True)
    
    # Should have rolled back due to validation
    assert len(results) == 2  # Original result + failure result
    assert not results[-1].ok
    assert "validation" in results[-1].message.lower()
    
    # State should be unchanged
    assert len(ir.layers.L2_regimes.regime_graph_spec.edges) == 1


def test_envelope_override_removal():
    """Test that RemoveRegime removes envelope overrides."""
    ir = InstrumentIR(
        meta=InstrumentMeta(name="Test"),
        io=IOConfig(),
        layers=Layers(
            L0_graph=L0Graph(),
            L1_lift=L1Lift(),
            L2_regimes=L2Regimes(
                regime_nodes=[
                    RegimeNode(id="R0_IDLE", kind="Idle", emit_template="T_IDLE"),
                    RegimeNode(id="R1_ATTACK", kind="Attack", emit_template="T_ATTACK"),
                ],
                envelope_overrides={
                    "R1_ATTACK": []  # Some overrides
                }
            ),
            L3_cmmm=L3Cmmm()
        )
    )
    
    session = EditorSession(ir)
    result = session.execute(RemoveRegime("R1_ATTACK"))
    
    assert result.ok
    assert "R1_ATTACK" not in ir.layers.L2_regimes.envelope_overrides
    
    # Undo should restore it
    session.undo()
    assert "R1_ATTACK" in ir.layers.L2_regimes.envelope_overrides


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
