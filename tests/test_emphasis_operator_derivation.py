"""
Test control-axis → operator derivation (continuous, 7-axis model).

Verifies that derive_operators_from_emphasis() recruits operators continuously
from control-axis weights — with NO magic thresholds — and that legacy emphasis
and character names alias onto the canonical 7 axes.  See docs/jargon.md.
"""

import pytest
from symcrash_python_ap.instrument_ir import (
    derive_operators_from_emphasis,
    emphasis_to_family_base,
    canonical_axis,
    AXIS_OPERATORS,
)


class TestSafetyOperators:
    """Safety operators are always present, first, regardless of axes."""

    def test_empty_gives_safety_only(self):
        assert derive_operators_from_emphasis({}) == ['NormalizeScale', 'FineClamp']

    def test_safety_always_first(self):
        ops = derive_operators_from_emphasis({'chaos': 1.0})
        assert ops[0] == 'NormalizeScale'
        assert ops[1] == 'FineClamp'


class TestContinuousRecruitment:
    """Any nonzero axis weight recruits its operators — no threshold cliffs."""

    @pytest.mark.parametrize("axis,expected", [
        ('color', ['Tilt']),
        ('body', ['IntraMix']),
        ('spread', ['DiffuseDown', 'DiffuseUp']),
        ('snap', ['Impulse', 'GlobalPulse']),
        ('glue', ['GluePair', 'SheafProject']),
        ('warp', ['BandShear', 'SwapBands', 'RotateNodes']),
        ('chaos', ['FoldPressure']),
    ])
    def test_axis_recruits_its_operators(self, axis, expected):
        ops = derive_operators_from_emphasis({axis: 0.5})
        for op in expected:
            assert op in ops

    @pytest.mark.parametrize("weight", [0.01, 0.05, 0.2, 0.29, 0.3, 0.31, 0.7, 1.0])
    def test_no_threshold_cliff(self, weight):
        """The same operators are recruited at every nonzero weight (continuity).

        The historical model had a cliff at 0.3 (tone>0.3 → Tilt). The continuous
        model recruits Tilt at *any* nonzero weight; strength scales downstream.
        """
        ops = derive_operators_from_emphasis({'color': weight})
        assert 'Tilt' in ops

    def test_zero_weight_does_not_recruit(self):
        ops = derive_operators_from_emphasis({'color': 0.0})
        assert 'Tilt' not in ops

    def test_epsilon_floor_respected(self):
        """An explicit epsilon floor can gate recruitment if desired."""
        assert 'Tilt' not in derive_operators_from_emphasis({'color': 0.1}, epsilon=0.2)
        assert 'Tilt' in derive_operators_from_emphasis({'color': 0.3}, epsilon=0.2)


class TestLegacyAliases:
    """Legacy emphasis and character names alias onto the canonical 7 axes."""

    @pytest.mark.parametrize("legacy,canon", [
        ('tone', 'color'),
        ('energy', 'snap'),
        ('clarity', 'glue'),
        ('stability', 'glue'),
        ('settling', 'glue'),
        ('brightness', 'color'),
        ('density', 'body'),
        ('risk', 'chaos'),
        ('dream', 'warp'),
        ('slip', 'spread'),
    ])
    def test_alias_maps_to_canonical(self, legacy, canon):
        assert canonical_axis(legacy) == canon

    def test_legacy_emphasis_recruits_same_operators(self):
        legacy = derive_operators_from_emphasis({'tone': 0.5, 'energy': 0.9})
        canon = derive_operators_from_emphasis({'color': 0.5, 'snap': 0.9})
        assert legacy == canon

    def test_character_names_recruit_operators(self):
        # brightness→color→Tilt, risk→chaos→FoldPressure
        ops = derive_operators_from_emphasis({'brightness': 0.6, 'risk': 0.8})
        assert 'Tilt' in ops
        assert 'FoldPressure' in ops


class TestApplyOrder:
    """Derived operators preserve the normative apply order (WSE-STD-030)."""

    def test_order_is_stable_across_axes(self):
        ops = derive_operators_from_emphasis({
            'chaos': 1.0, 'color': 1.0, 'snap': 1.0, 'body': 1.0,
        })
        # Safety first, then color, body, snap, …, chaos last.
        assert ops[:2] == ['NormalizeScale', 'FineClamp']
        assert ops.index('Tilt') < ops.index('IntraMix') < ops.index('Impulse')
        assert ops.index('FoldPressure') == len(ops) - 1

    def test_unknown_names_ignored(self):
        ops = derive_operators_from_emphasis({'mystery': 1.0, 'color': 0.5})
        assert ops == ['NormalizeScale', 'FineClamp', 'Tilt']


class TestFamilyBase:
    """emphasis_to_family_base accepts canonical and legacy names."""

    def test_canonical_and_legacy_equivalent(self):
        assert emphasis_to_family_base({'color': 0.5}) == emphasis_to_family_base({'tone': 0.5})

    def test_glue_aggregates_additively(self):
        # clarity + stability + settling all fold into glue (slot 5)
        base = emphasis_to_family_base({'clarity': 0.3, 'stability': 0.3, 'settling': 0.3})
        assert base[5] > 0
        # near 0.9 of full scale (summed, clamped)
        assert base[5] == pytest.approx(int(0.9 * 32767), abs=3)


class TestEmphasisDrivesEmitTemplate:
    """Emphasis-only behavior states get continuously-derived operators."""

    def test_emphasis_without_processing_derives_ops(self):
        from symcrash_python_ap.instrument_ir import _translate_sound_design_format

        data = {
            'graph': {
                'osc': {'type': 'Osc', 'waveform': 'saw', 'output': 'amp'},
                'amp': {'type': 'Amp'},
            },
            'behavior': {
                'attack': {'emphasis': {'color': 0.9, 'snap': 1.0}},
            },
        }
        result = _translate_sound_design_format(data)
        templates = result['layers']['L3_cmmm']['emit_templates']
        ops = templates['T_ATTACK']['ops_enable']
        assert 'NormalizeScale' in ops
        assert 'Tilt' in ops
        assert 'Impulse' in ops
        assert 'GlobalPulse' in ops

    def test_explicit_processing_not_overridden(self):
        from symcrash_python_ap.instrument_ir import _translate_sound_design_format

        data = {
            'graph': {
                'osc': {'type': 'Osc', 'waveform': 'saw', 'output': 'amp'},
                'amp': {'type': 'Amp'},
            },
            'behavior': {
                'stable': {
                    'emphasis': {'color': 0.9},
                    'processing': ['Tilt', 'Compress_K'],
                },
            },
        }
        result = _translate_sound_design_format(data)
        templates = result['layers']['L3_cmmm']['emit_templates']
        assert templates['T_STABLE']['ops_enable'] == ['Tilt', 'Compress_K']
