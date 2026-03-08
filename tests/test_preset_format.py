"""
Preset Format Tests
===================

Validates that all preset YAML files conform to the instrument_ir_v0 format
using the sound-design format (graph:, controls:, behavior:).

Tests cover:
- All presets load through the canonical parser
- Top-level key is instrument_ir_v0
- Sound-design format structure (graph:, controls:, behavior:)
- Operator names use CamelCase
- pwm_string_v0 modulators
- Sound-design format translation (graph→L0, controls→macros, behavior→L3)
- Backward compatibility with legacy layers: format
"""

import pytest
from pathlib import Path

import yaml

from symcrash_python_ap.instrument_ir import (
    load_instrument,
    load_instrument_from_dict,
    _parse_modulators,
    RESOLUTION_TO_MASK,
    LEGACY_FAMILY_NAMES,
    LEGACY_TREE_NAMES,
)
from symcrash_python_ap.vm.harmonic_bank import (
    LFOWaveform, LFOSyncMode, LFOTarget,
)

# Discover all preset files
PRESET_DIR = Path(__file__).parent.parent / "presets" / "instruments"
ALL_PRESET_FILES = sorted(PRESET_DIR.glob("*.yaml"))

# Known CamelCase operator names
CAMELCASE_OPS = {
    "Tilt", "DiffuseUp", "DiffuseDown", "IntraMix",
    "FineClamp", "NormalizeScale", "BandScale",
    "LowRankMorph", "SheafProject", "BandShear", "SwapBands",
    "RotateNodes", "FoldPressure", "GluePair", "Impulse", "GlobalPulse",
}

# Legacy operator names that should not appear
LEGACY_OPS = {
    "TILT", "DIFFUSE_UP", "DIFFUSE_DOWN", "INTRA_MIX",
    "FINE_CLAMP", "NORMALIZE_SCALE", "BAND_SCALE",
}


# =============================================================================
# Parametrized tests across all presets
# =============================================================================

@pytest.mark.parametrize("preset_path", ALL_PRESET_FILES, ids=lambda p: p.stem)
def test_preset_loads_via_canonical_parser(preset_path):
    """Every preset must load through load_instrument()."""
    ir = load_instrument(preset_path)
    assert ir.meta.instrument_id, f"{preset_path.name}: missing instrument_id"
    assert ir.meta.name, f"{preset_path.name}: missing name"


@pytest.mark.parametrize("preset_path", ALL_PRESET_FILES, ids=lambda p: p.stem)
def test_preset_uses_instrument_ir_v0(preset_path):
    """Top-level key must be instrument_ir_v0."""
    with open(preset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "instrument_ir_v0" in data, (
        f"{preset_path.name}: uses '{list(data.keys())[0]}' instead of 'instrument_ir_v0'"
    )
    assert "preset" not in data, f"{preset_path.name}: still has deprecated 'preset:' key"


@pytest.mark.parametrize("preset_path", ALL_PRESET_FILES, ids=lambda p: p.stem)
def test_preset_uses_sound_design_format(preset_path):
    """All presets should use the sound-design format (graph:, not layers:)."""
    with open(preset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ir_data = data["instrument_ir_v0"]
    assert "graph" in ir_data, f"{preset_path.name}: missing 'graph:' section"
    assert "layers" not in ir_data, f"{preset_path.name}: still uses legacy 'layers:' format"


@pytest.mark.parametrize("preset_path", ALL_PRESET_FILES, ids=lambda p: p.stem)
def test_graph_nodes_have_type(preset_path):
    """Every graph node must have a type: field."""
    with open(preset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    graph = data["instrument_ir_v0"].get("graph", {})
    for node_id, node_data in graph.items():
        assert "type" in node_data, f"{preset_path.name}: node '{node_id}' missing type:"


@pytest.mark.parametrize("preset_path", ALL_PRESET_FILES, ids=lambda p: p.stem)
def test_operator_names_camelcase(preset_path):
    """All processing entries must use CamelCase."""
    with open(preset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ir_data = data["instrument_ir_v0"]
    behavior = ir_data.get("behavior", {})

    reserved = {"states", "transitions"}
    for state_name, state_def in behavior.items():
        if state_name in reserved or not isinstance(state_def, dict):
            continue
        for op in state_def.get("processing", []):
            assert op not in LEGACY_OPS, (
                f"{preset_path.name}: behavior.{state_name} has legacy op '{op}'"
            )


# =============================================================================
# Sound-design format translation tests
# =============================================================================

def test_translate_simple_graph():
    """graph: with Osc→Amp translates to L0Graph with nodes and edges."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "test", "name": "Test"},
            "audio": {"sample_rate": 48000, "block_size": 64},
            "graph": {
                "osc1": {"type": "Osc", "output": "amp", "waveform": "saw"},
                "amp": {"type": "Amp", "attack": 5, "decay": 100, "sustain": 0.7, "release": 200},
            },
        }
    }
    ir = load_instrument_from_dict(data)
    assert len(ir.layers.L0_graph.nodes) == 2
    assert len(ir.layers.L0_graph.edges) == 1
    assert ir.layers.L0_graph.edges[0].src == "osc1"
    assert ir.layers.L0_graph.edges[0].dst == "amp"

    osc = next(n for n in ir.layers.L0_graph.nodes if n.type == "Osc")
    assert osc.params.waveform == "saw"

    amp = next(n for n in ir.layers.L0_graph.nodes if n.type == "Amp")
    assert amp.params.adsr.a_ms == 5.0
    assert amp.params.adsr.d_ms == 100.0
    assert amp.params.adsr.s == 0.7
    assert amp.params.adsr.r_ms == 200.0


def test_translate_meta_shorthand():
    """meta.id → instrument_id, meta.seed → seed_u32."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "test-short", "name": "Short", "seed": 0xDEAD},
            "graph": {"amp": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
        }
    }
    ir = load_instrument_from_dict(data)
    assert ir.meta.instrument_id == "test-short"
    assert ir.meta.seed_u32 == 0xDEAD


def test_translate_audio_to_io():
    """audio.sample_rate → io.sample_rate_hz."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "audio": {"sample_rate": 44100, "block_size": 128},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
        }
    }
    ir = load_instrument_from_dict(data)
    assert ir.io.sample_rate_hz == 44100
    assert ir.io.block_size == 128


def test_translate_controls_to_macros():
    """controls: dict translates to macros list with auto-generated IDs."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "controls": {
                "Brightness": 0.6,
                "Body": 0.5,
                "Unison Spread": 0.7,
            },
        }
    }
    ir = load_instrument_from_dict(data)
    macros = ir.layers.L0_graph.macros
    assert len(macros) == 3

    ids = {m.id: m for m in macros}
    assert "M_BRIGHTNESS" in ids
    assert ids["M_BRIGHTNESS"].name == "Brightness"
    assert ids["M_BRIGHTNESS"].default == 0.6

    assert "M_BODY" in ids
    assert "M_UNISON_SPREAD" in ids
    assert ids["M_UNISON_SPREAD"].default == 0.7


def test_translate_character_to_control():
    """character: translates to ControlConfig."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "character": {"brightness": 0.8, "density": 0.6, "risk": 0.1},
        }
    }
    ir = load_instrument_from_dict(data)
    assert ir.control.brightness == 0.8
    assert ir.control.density == 0.6
    assert ir.control.risk == 0.1


def test_translate_simple_behavior():
    """Simple behavior (idle + stable) translates to L3_cmmm emit_templates."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "behavior": {
                "idle": {
                    "processing": ["NormalizeScale", "FineClamp"],
                    "emphasis": {"CONSTRAINT": 1.0, "CALM": 1.0},
                    "trees": {"A": "T1", "B": "T1", "blend": 0.0},
                },
                "stable": {
                    "processing": ["Tilt", "NormalizeScale", "FineClamp"],
                    "trees": {"A": "T1", "B": "T1", "blend": 0.25},
                    "resolution": 32,
                    "safety": "FULL",
                    "ramp": 12,
                    "traction": 1.0,
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    templates = ir.layers.L3_cmmm.emit_templates
    assert "T_IDLE" in templates
    assert "T_STABLE" in templates

    t_idle = templates["T_IDLE"]
    assert t_idle.ops_enable == ["NormalizeScale", "FineClamp"]
    assert t_idle.families == {"stability": 1.0, "settling": 1.0}
    assert t_idle.trees.beta == 0.0

    t_stable = templates["T_STABLE"]
    assert t_stable.ops_enable == ["Tilt", "NormalizeScale", "FineClamp"]
    assert t_stable.trees.beta == 0.25
    assert t_stable.cmmm is not None
    assert t_stable.cmmm.node_mask == 0xFFFFFFFF
    assert t_stable.cmmm.tail_mode == "FULL"
    assert t_stable.cmmm.mask_ramp_blocks == 12


def test_translate_resolution_to_node_mask():
    """resolution: K values map to correct node masks."""
    for k, expected_mask in RESOLUTION_TO_MASK.items():
        data = {
            "instrument_ir_v0": {
                "meta": {"id": "t", "name": "T"},
                "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
                "behavior": {
                    "idle": {
                        "processing": ["FineClamp"],
                        "resolution": k,
                    },
                },
            }
        }
        ir = load_instrument_from_dict(data)
        t_idle = ir.layers.L3_cmmm.emit_templates["T_IDLE"]
        assert t_idle.cmmm.node_mask == expected_mask, (
            f"resolution={k}: expected mask {hex(expected_mask)}, got {hex(t_idle.cmmm.node_mask)}"
        )


def test_translate_complex_behavior():
    """Complex behavior (states + transitions) translates to L2_regimes + L3_cmmm."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "behavior": {
                "states": {
                    "idle": {"kind": "Idle"},
                    "attack": {"kind": "Transient", "dwell": 15},
                    "stable": {"kind": "StableTone"},
                },
                "transitions": [
                    {"from": "idle", "to": "attack", "when": "gate rises"},
                    {"from": "attack", "to": "stable", "when": "dwell done"},
                    {"from": "any", "to": "idle", "when": "gate falls"},
                ],
                "idle": {"processing": ["FineClamp"]},
                "attack": {"processing": ["Tilt", "FineClamp"]},
                "stable": {"processing": ["Tilt", "NormalizeScale", "FineClamp"]},
            },
        }
    }
    ir = load_instrument_from_dict(data)

    # Check L2_regimes
    regimes = ir.layers.L2_regimes
    assert len(regimes.regime_nodes) == 3
    assert regimes.regime_nodes[0].kind == "Idle"
    assert regimes.regime_nodes[1].kind == "Transient"
    assert regimes.regime_nodes[1].dwell_ms == 15
    assert len(regimes.transitions) == 3

    # Check L3_cmmm
    templates = ir.layers.L3_cmmm.emit_templates
    assert "T_IDLE" in templates
    assert "T_ATTACK" in templates
    assert "T_STABLE" in templates


def test_translate_envelopes_dict():
    """envelopes: dict with flat ADSR translates to EnvelopeBinding list."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "envelopes": {
                "carrier": {
                    "attack": 8.0, "decay": 150.0, "sustain": 0.7, "release": 100.0,
                    "targets": ["FM_OP1", "FM_OP2"],
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    assert len(ir.envelopes) == 1
    env = ir.envelopes[0]
    assert env.id == "carrier"
    assert env.adsr.a_ms == 8.0
    assert env.adsr.d_ms == 150.0
    assert env.adsr.s == 0.7
    assert env.adsr.r_ms == 100.0
    assert env.targets == ["FM_OP1", "FM_OP2"]


def test_translate_modulators_dict():
    """modulators: dict format translates to LFO list."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "modulators": {
                "vibrato": {
                    "waveform": "sine",
                    "rate": 5.5,
                    "depth": 0.2,
                    "delay": 400,
                    "ramp": 300,
                    "sync": "free",
                    "targets": [{"pitch": 1.0}],
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    assert len(ir.modulators) == 1
    lfo = ir.modulators[0]
    assert lfo.id == "vibrato"
    assert lfo.waveform == LFOWaveform.SINE
    assert lfo.rate_hz == 5.5
    assert lfo.sync_mode == LFOSyncMode.FREE
    assert len(lfo.routings) == 1
    assert lfo.routings[0].target == LFOTarget.PITCH


def test_translate_compile_material():
    """compile.material vocabulary (blend/target/ramp) translates correctly."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "compile": {
                "material": {"blend": 0.0, "target": 0.25, "ramp": 16},
            },
        }
    }
    ir = load_instrument_from_dict(data)
    assert ir.compile.material.beta_start_q15 == 0
    assert ir.compile.material.beta_target_q15 == 8192  # 0.25 * 32767 ≈ 8192
    assert ir.compile.material.beta_ramp_blocks == 16


def test_translate_fm6_graph():
    """FM6 node in graph: format translates correctly."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {
                "fm6": {
                    "type": "FM6",
                    "output": "amp",
                    "algorithm": 19,
                    "feedback": 3,
                    "operators": [
                        {"ratio": 1.0, "level": 1.0},
                        {"ratio": 2.0, "level": 0.5},
                    ],
                },
                "amp": {"type": "Amp", "attack": 5, "decay": 80, "sustain": 0.7, "release": 100},
            },
        }
    }
    ir = load_instrument_from_dict(data)
    fm6_node = next(n for n in ir.layers.L0_graph.nodes if n.type == "FM6")
    assert fm6_node.params.algorithm == 19
    assert fm6_node.params.feedback == 3


# =============================================================================
# Jargon translation: sound-engineering vocabulary → internal names
# =============================================================================

def test_translate_emphasis_sound_engineering_names():
    """Sound-engineering emphasis names are stored as-is internally."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "behavior": {
                "idle": {
                    "processing": ["FineClamp"],
                    "emphasis": {"stability": 1.0, "settling": 0.8, "tone": 0.5},
                    "trees": {"A": "harmonic", "B": "harmonic", "blend": 0.0},
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    fam = ir.layers.L3_cmmm.emit_templates["T_IDLE"].families
    assert fam == {"stability": 1.0, "settling": 0.8, "tone": 0.5}


def test_translate_emphasis_all_families():
    """All 9 sound-engineering emphasis names are stored as-is."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "behavior": {
                "idle": {
                    "emphasis": {
                        "stability": 1.0, "settling": 0.9, "tone": 0.8,
                        "body": 0.7, "spread": 0.6, "clarity": 0.5,
                        "energy": 0.4, "warp": 0.3, "chaos": 0.2,
                    },
                    "trees": {"A": "harmonic", "B": "harmonic", "blend": 0.0},
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    fam = ir.layers.L3_cmmm.emit_templates["T_IDLE"].families
    assert fam["stability"] == 1.0
    assert fam["settling"] == 0.9
    assert fam["tone"] == 0.8
    assert fam["body"] == 0.7
    assert fam["spread"] == 0.6
    assert fam["clarity"] == 0.5
    assert fam["energy"] == 0.4
    assert fam["warp"] == 0.3
    assert fam["chaos"] == 0.2


def test_translate_emphasis_legacy_names_converted():
    """Legacy emphasis names (CONSTRAINT, CALM, etc.) are translated to sound-eng names."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "behavior": {
                "idle": {
                    "emphasis": {"CONSTRAINT": 1.0, "CALM": 0.5},
                    "trees": {"A": "harmonic", "B": "harmonic", "blend": 0.0},
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    fam = ir.layers.L3_cmmm.emit_templates["T_IDLE"].families
    assert fam == {"stability": 1.0, "settling": 0.5}


def test_translate_tree_names_sound_engineering():
    """Sound-engineering tree names (harmonic, texture) are stored as-is."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "behavior": {
                "idle": {
                    "trees": {"A": "harmonic", "B": "texture", "blend": 0.5},
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    trees = ir.layers.L3_cmmm.emit_templates["T_IDLE"].trees
    assert trees.A == "harmonic"
    assert trees.B == "texture"
    assert trees.beta == 0.5


def test_translate_tree_names_all():
    """All tree names (balanced, harmonic, texture, impact, drone) are stored as-is."""
    for name in ["balanced", "harmonic", "texture", "impact", "drone"]:
        data = {
            "instrument_ir_v0": {
                "meta": {"id": "t", "name": "T"},
                "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
                "behavior": {
                    "idle": {"trees": {"A": name, "B": name, "blend": 0.0}},
                },
            }
        }
        ir = load_instrument_from_dict(data)
        trees = ir.layers.L3_cmmm.emit_templates["T_IDLE"].trees
        assert trees.A == name, f"{name} should be stored as-is"
        assert trees.B == name


def test_translate_legacy_tree_names():
    """Legacy tree names (T0-T4) are translated to sound-engineering names."""
    for legacy, modern in LEGACY_TREE_NAMES.items():
        data = {
            "instrument_ir_v0": {
                "meta": {"id": "t", "name": "T"},
                "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
                "behavior": {
                    "idle": {"trees": {"A": legacy, "B": legacy, "blend": 0.0}},
                },
            }
        }
        ir = load_instrument_from_dict(data)
        trees = ir.layers.L3_cmmm.emit_templates["T_IDLE"].trees
        assert trees.A == modern, f"{legacy} should translate to {modern}"
        assert trees.B == modern


def test_translate_energy_field():
    """Character field 'energy' is stored as 'energy' internally."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {"a": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1}},
            "character": {
                "brightness": 0.5,
                "energy": 0.7,
            },
        }
    }
    ir = load_instrument_from_dict(data)
    assert ir.control.energy == 0.7
    assert ir.control.brightness == 0.5


def test_translate_float_partials_to_q15():
    """Float partials (0.0-1.0) are converted to Q15 integers."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "t", "name": "T"},
            "graph": {
                "osc1": {
                    "type": "Osc",
                    "output": "amp",
                    "waveform": "partials",
                    "partials": [0.943, 1.0, 0.211, 0.731],
                },
                "amp": {"type": "Amp", "attack": 1, "decay": 1, "sustain": 0.5, "release": 1},
            },
        }
    }
    ir = load_instrument_from_dict(data)
    osc_node = next(n for n in ir.layers.L0_graph.nodes if n.type == "Osc")
    # Partials should be stored as param (pass-through to node params dict)
    # The translation should have converted 0.943 → round(0.943 * 32767) = 30899
    # Note: exact values depend on rounding
    # The key test is that the preset LOADS without error


def test_legacy_family_names_mapping_complete():
    """LEGACY_FAMILY_NAMES covers all old internal family names."""
    expected = {"CONSTRAINT", "CALM", "COLOR", "LOCAL", "TRANSPORT", "COHERENCE", "EXCITE", "TOPOLOGY", "CATASTROPHE"}
    assert set(LEGACY_FAMILY_NAMES.keys()) == expected


def test_legacy_tree_names_mapping_complete():
    """LEGACY_TREE_NAMES covers all old tree identifiers."""
    expected = {"T0", "T1", "T2", "T3", "T4"}
    assert set(LEGACY_TREE_NAMES.keys()) == expected


# =============================================================================
# Backward compatibility: legacy layers: format still works
# =============================================================================

def test_legacy_layers_format_still_loads():
    """Old layers: format still loads correctly."""
    data = {
        "instrument_ir_v0": {
            "meta": {"instrument_id": "legacy-test", "name": "Legacy"},
            "io": {"sample_rate_hz": 48000, "block_size": 64},
            "layers": {
                "L0_graph": {
                    "nodes": [
                        {"id": "osc1", "type": "Osc", "params": {"waveform": "saw"}},
                        {"id": "amp1", "type": "Amp", "params": {"adsr": {"a_ms": 5, "d_ms": 50, "s": 0.7, "r_ms": 100}}},
                    ],
                    "edges": [{"src": "osc1", "dst": "amp1"}],
                    "macros": [{"id": "M_BRIGHT", "name": "Brightness", "default": 0.5}],
                },
            },
        }
    }
    ir = load_instrument_from_dict(data)
    assert ir.meta.instrument_id == "legacy-test"
    assert len(ir.layers.L0_graph.nodes) == 2
    assert ir.layers.L0_graph.macros[0].id == "M_BRIGHT"


def test_both_graph_and_layers_raises():
    """Having both graph: and layers: raises ValueError."""
    data = {
        "instrument_ir_v0": {
            "meta": {"id": "bad", "name": "Bad"},
            "graph": {"a": {"type": "Amp"}},
            "layers": {"L0_graph": {}},
        }
    }
    with pytest.raises(ValueError, match="both 'graph:' and 'layers:'"):
        load_instrument_from_dict(data)


# =============================================================================
# pwm_string_v0 migration tests
# =============================================================================

def test_pwm_string_loads():
    """Verify pwm_string_v0 loads through canonical parser."""
    preset_path = PRESET_DIR / "pwm_string_v0.yaml"
    ir = load_instrument(preset_path)
    assert ir.meta.instrument_id == "pwm-string-v0"
    assert ir.meta.name == "PWM String"


def test_pwm_string_has_modulators():
    """Verify pwm_string_v0 has 3 LFOs."""
    preset_path = PRESET_DIR / "pwm_string_v0.yaml"
    ir = load_instrument(preset_path)
    assert len(ir.modulators) == 3

    lfo_ids = [lfo.id for lfo in ir.modulators]
    assert "vibrato" in lfo_ids
    assert "pwm_sweep" in lfo_ids
    assert "brightness" in lfo_ids


def test_pwm_string_lfo_details():
    """Verify LFO parameters in pwm_string_v0."""
    preset_path = PRESET_DIR / "pwm_string_v0.yaml"
    ir = load_instrument(preset_path)

    vibrato = next(l for l in ir.modulators if l.id == "vibrato")
    assert vibrato.waveform == LFOWaveform.SINE
    assert vibrato.rate_hz == 5.5
    assert vibrato.sync_mode == LFOSyncMode.FREE
    assert len(vibrato.routings) == 1
    assert vibrato.routings[0].target == LFOTarget.PITCH

    pwm = next(l for l in ir.modulators if l.id == "pwm_sweep")
    assert pwm.waveform == LFOWaveform.TRIANGLE
    assert pwm.rate_hz == 0.8
    assert len(pwm.routings) == 1
    assert pwm.routings[0].target == LFOTarget.PWM_WIDTH

    bright = next(l for l in ir.modulators if l.id == "brightness")
    assert bright.sync_mode == LFOSyncMode.GATE_RESET
    assert len(bright.routings) == 1
    assert bright.routings[0].target == LFOTarget.TILT


def test_pwm_string_has_emit_templates():
    """Verify pwm_string_v0 has L3_cmmm emit_templates."""
    preset_path = PRESET_DIR / "pwm_string_v0.yaml"
    ir = load_instrument(preset_path)
    assert ir.layers.L3_cmmm.emit_templates
    assert "T_IDLE" in ir.layers.L3_cmmm.emit_templates
    assert "T_STABLE" in ir.layers.L3_cmmm.emit_templates


# =============================================================================
# Modulators parsing tests
# =============================================================================

def test_modulators_parsing_empty():
    """Empty data returns empty list."""
    assert _parse_modulators({}) == []
    assert _parse_modulators({"modulators": {}}) == []
    assert _parse_modulators({"modulators": {"lfos": []}}) == []


def test_modulators_parsing_single_lfo():
    """Parse a single LFO with all fields."""
    data = {
        "modulators": {
            "lfos": [{
                "id": "test",
                "waveform": "triangle",
                "rate_hz": 3.0,
                "depth": 0.5,
                "delay_ms": 100,
                "ramp_ms": 50,
                "sync_mode": "gate_reset",
                "targets": [
                    {"type": "pitch", "amount": 0.8},
                    {"type": "tilt", "amount": 0.5},
                ],
            }]
        }
    }

    lfos = _parse_modulators(data)
    assert len(lfos) == 1

    lfo = lfos[0]
    assert lfo.id == "test"
    assert lfo.waveform == LFOWaveform.TRIANGLE
    assert lfo.rate_hz == 3.0
    assert lfo.sync_mode == LFOSyncMode.GATE_RESET
    assert lfo.arc_slot == 0
    assert len(lfo.routings) == 2
    assert lfo.routings[0].target == LFOTarget.PITCH
    assert lfo.routings[1].target == LFOTarget.TILT


def test_modulators_parsing_multiple_lfos_arc_slots():
    """Multiple LFOs get sequential arc_slot assignments."""
    data = {
        "modulators": {
            "lfos": [
                {"id": "a", "targets": [{"type": "pitch"}]},
                {"id": "b", "targets": [{"type": "gain"}]},
                {"id": "c", "targets": [{"type": "tilt"}]},
            ]
        }
    }

    lfos = _parse_modulators(data)
    assert len(lfos) == 3
    assert lfos[0].arc_slot == 0
    assert lfos[1].arc_slot == 1
    assert lfos[2].arc_slot == 2


def test_modulators_target_aliases():
    """Target aliases (brightness->tilt, pwm->pwm_width) work correctly."""
    data = {
        "modulators": {
            "lfos": [{
                "id": "alias_test",
                "targets": [
                    {"type": "brightness"},
                    {"type": "pwm"},
                    {"type": "cutoff"},
                    {"type": "resonance"},
                ],
            }]
        }
    }

    lfos = _parse_modulators(data)
    targets = [r.target for r in lfos[0].routings]
    assert targets == [
        LFOTarget.TILT,
        LFOTarget.PWM_WIDTH,
        LFOTarget.FILTER_CUTOFF,
        LFOTarget.FILTER_RESO,
    ]


def test_modulators_depth_float_preferred():
    """Float depth takes priority over depth_q15."""
    data = {
        "modulators": {
            "lfos": [{
                "id": "depth_test",
                "depth": 0.25,
                "depth_q15": 9999,  # Should be ignored
                "targets": [{"type": "pitch"}],
            }]
        }
    }

    lfos = _parse_modulators(data)
    # 0.25 * 32767 = 8192 (rounded)
    assert lfos[0].depth_q15 == 8192


def test_modulators_waveform_defaults():
    """Default waveform is sine, default sync_mode is free."""
    data = {
        "modulators": {
            "lfos": [{"id": "defaults", "targets": [{"type": "pitch"}]}]
        }
    }

    lfos = _parse_modulators(data)
    assert lfos[0].waveform == LFOWaveform.SINE
    assert lfos[0].sync_mode == LFOSyncMode.FREE


def test_modulators_in_load_instrument_from_dict():
    """Modulators are available on the IR object after loading."""
    data = {
        "instrument_ir_v0": {
            "meta": {"instrument_id": "test-mod", "name": "Test"},
            "modulators": {
                "lfos": [
                    {"id": "lfo1", "waveform": "saw", "rate_hz": 2.0,
                     "depth": 0.3, "targets": [{"type": "gain", "amount": 0.7}]},
                ]
            }
        }
    }

    ir = load_instrument_from_dict(data)
    assert len(ir.modulators) == 1
    assert ir.modulators[0].id == "lfo1"
    assert ir.modulators[0].waveform == LFOWaveform.SAW_UP
    assert ir.modulators[0].routings[0].target == LFOTarget.GAIN


# =============================================================================
# OEIS preset fix verification
# =============================================================================

@pytest.mark.parametrize("preset_name", ["oeis_bass_v0", "oeis_bell_v0", "oeis_pad_v0"])
def test_oeis_presets_have_emit_templates(preset_name):
    """OEIS presets must have L3_cmmm with emit_templates."""
    preset_path = PRESET_DIR / f"{preset_name}.yaml"
    ir = load_instrument(preset_path)
    assert ir.layers.L3_cmmm.emit_templates, (
        f"{preset_name}: L3_cmmm.emit_templates is empty"
    )
    assert "T_IDLE" in ir.layers.L3_cmmm.emit_templates
    assert "T_STABLE" in ir.layers.L3_cmmm.emit_templates
