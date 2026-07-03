"""
Canonical table single-sourcing pins (synthesis-path dedup, 2026-07-02).

tables/ is the canonical home (per CMMM-SPEC); budgets/ re-exports.
These tests pin the active per-coefficient tick values (see the errata
note in tables/operator_costs.py: CMMM-SPEC §3.3 lists per-node units)
and the derivation chains so future drift is caught.
"""

from symcrash_python_ap.tables.operator_costs import (
    OPERATOR_COSTS as TABLES_COSTS,
    TAIL_CAPABLE_OPS,
    predict_ticks,
)
from symcrash_python_ap.budgets.operator_costs import (
    OPERATOR_COSTS as BUDGETS_COSTS,
)
from symcrash_python_ap.primitives import (
    K_LADDER_MASKS,
    TAIL_MODE_TABLE,
    TailMode,
)
from symcrash_python_ap.tables.tail_modes import TAIL_OPS
from symcrash_python_ap.tables.pattern_table import PATTERN_TABLE, K_STAIRCASE
from symcrash_python_ap.vm import tables as vm_tables


class TestOperatorCostSingleSource:
    def test_budgets_is_reexport(self):
        assert BUDGETS_COSTS is TABLES_COSTS

    def test_active_per_coefficient_values_pinned(self):
        # Per-coefficient tick unit: per-node terms carry NODE_LEN=8
        assert TABLES_COSTS[1].ticks_per_active_node == 8    # Tilt
        assert TABLES_COSTS[3].ticks_per_active_node == 24   # IntraMix
        assert TABLES_COSTS[6].ticks_per_total_node == 8     # BandShear
        assert TABLES_COSTS[7].ticks_per_total_node == 32    # LowRankMorph
        assert TABLES_COSTS[11].ticks_per_total_node == 24   # NormalizeScale
        assert TABLES_COSTS[11].nonlinear

    def test_all_24_ops_present(self):
        assert set(TABLES_COSTS.keys()) == set(range(24))

    def test_tail_capable_ops(self):
        assert TAIL_CAPABLE_OPS == frozenset({2, 11, 19, 23})

    def test_predict_ticks_uses_compute_ticks(self):
        ticks, nonlinear = predict_ticks([1], K_active_nodes=4)
        assert ticks == TABLES_COSTS[1].compute_ticks(4)
        assert not nonlinear


class TestKLadderSingleSource:
    def test_frozen_seven_step_mapping(self):
        # WSE-STD-DEGRADE-v0 §3.1: pattern_id -> K, no K20 rung
        expected = {0: 32, 1: 24, 2: 16, 3: 12, 4: 8, 5: 4, 6: 1}
        assert {
            pid: bin(mask).count('1') for pid, mask in K_LADDER_MASKS.items()
        } == expected

    def test_vm_pattern_table_derived_from_primitives(self):
        for pid, mask in K_LADDER_MASKS.items():
            assert vm_tables.PATTERN_TABLE[pid].mask_u32 == mask
            assert vm_tables.PATTERN_TABLE[pid].k_value == bin(mask).count('1')

    def test_tables_pattern_table_staircase_matches_primitives(self):
        for pid, mask in K_LADDER_MASKS.items():
            assert PATTERN_TABLE[pid].mask == mask
        assert K_STAIRCASE == [32, 24, 16, 12, 8, 4, 1]


class TestTailModeSingleSource:
    def test_tail_ops_derived_from_primitives(self):
        for mode_id, info in TAIL_MODE_TABLE.items():
            assert TAIL_OPS[TailMode(mode_id)] == list(info.operators)

    def test_emergency_is_clamp_plus_simplify(self):
        # WSE-STD-DEGRADE-v0 §4: EMERGENCY = [2, 19]
        assert TAIL_OPS[TailMode.EMERGENCY] == [2, 19]
