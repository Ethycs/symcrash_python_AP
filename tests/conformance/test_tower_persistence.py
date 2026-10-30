"""WSE-STD-070 save/load contracts and deterministic authored-layer tracking."""
from copy import deepcopy
import json
import os
import struct
import subprocess
import sys
import zlib

import pytest

from symcrash_python_ap.library.preset_model import (
    Preset, PresetBody, PresetHeader, TowerLevel as L,
    L0PatchLayer, L1FamilyLayer, L2InstrumentLayer, L3CockpitLayer, L4ForgeLayer, L5LabLayer,
    create_preset, save_preset, load_preset, complete_tower,
    set_param, delta_param, set_bound, set_budget, set_tree_id, add_regime, remove_regime,
    set_tower_level, set_operator_allowed, project_to_level_op, reset_to_safe_op,
    validate_determinism,
)
from symcrash_python_ap.library.tower_codec import BODY_MAGIC


def authored_layers():
    return [
        L0PatchLayer(midi_channel=3, gain=.25, pan=-.75, transpose=-12, velocity_curve='soft'),
        L1FamilyLayer('bowed', {'STABLE', 'BOW'}, 'BOW', {'COLOR': .75}),
        L2InstrumentLayer(params={'name': 'café 音', 'ratio': (3, 2), 'values': [None, True, -5],
                                 'tags': {'a', 'b'}, 'map': {7: 'int', '7': 'string'}},
                          bounds={'pressure': (0., 1.)}, allowed_operators={'Tilt', 'IntraMix'},
                          tree_adsr={'attack': .2, 'decay': .5, 'sustain': .7, 'release': .9}),
        L3CockpitLayer({'Tilt': .8}, {'attack': (.01, 2.)}, 4000, 5000,
                      {2: {'variant': 3, 'sequence': (1, 3, 5)}}),
        L4ForgeLayer({'BOW': ['STABLE'], 'STABLE': ['BOW']}, [{'op': 'add', 'at': 2}],
                    {('BOW', 'STABLE'): .7, ('STABLE', 'BOW'): .3}),
        L5LabLayer({'custom': {'coeffs': (1., -.5), 'bounded': True}},
                   {99: {'splits': [1, 2]}}, {'node': 'root', 'children': []}),
    ]


def envelope(data):
    return json.loads(data[16 + len(BODY_MAGIC):])


def encoded_levels(data):
    return [item['level'] for item in envelope(data)['layers']]


def with_body(body, level=0, crc=None):
    return struct.pack('>IHBBHHI', 123, 1, level, 0, 2, 0,
                       zlib.crc32(body) if crc is None else crc) + body


@pytest.mark.parametrize('level', list(L))
def test_each_single_authored_layer_roundtrips_without_serializing_completion(level):
    original = Preset(PresetHeader(123, tower_level=level, tree_id=42, flags=17))
    layer = authored_layers()[int(level)]
    original.body.set_layer(level, layer)
    original.complete(L.L5_LAB)
    data = save_preset(original)
    restored = load_preset(data)
    assert encoded_levels(data) == [int(level)]
    assert restored.header.tower_level == level
    assert restored.header.tree_id == 42 and restored.header.flags == 17
    assert restored.body.get_layer(level) == layer
    assert restored.is_valid()
    assert restored.body.highest_explicit_level() == level
    assert save_preset(restored) == data
    restored.complete(L.L5_LAB)
    assert save_preset(restored) == data


def test_all_layers_and_typed_dictionary_keys_survive_roundtrip():
    original = Preset(PresetHeader(321, preset_version=2, tower_level=L.L5_LAB))
    for level, layer in zip(L, authored_layers()):
        original.body.set_layer(level, layer)
    data = save_preset(original)
    restored = load_preset(data)
    assert restored.body == original.body
    assert restored.header.preset_version == 2
    assert restored.body.L2_instrument.params['map'] == {7: 'int', '7': 'string'}
    assert restored.body.L4_forge.transition_probs[('BOW', 'STABLE')] == .7
    assert 99 in restored.body.L5_lab.custom_trees
    assert save_preset(restored) == data


def test_known_body_fixture_keeps_existing_header_layout():
    body = (b'SYTW{"layers":[{"fields":{"items":[["gain",0.5],["midi_channel",2],'
            b'["pan",0.0],["transpose",12],["velocity_curve","linear"]],'
            b'"type":"dict"},"level":0}],"version":1}')
    expected = with_body(body)
    original = Preset(PresetHeader(123, tower_level=L.L0_PATCH, tree_id=2),
                      PresetBody(L0_patch=L0PatchLayer(2, .5, 0., 12)))
    assert save_preset(original) == expected
    assert load_preset(expected).body == original.body


def test_completed_defaults_stay_implicit_until_edited():
    p = create_preset(12, tower_level=L.L5_LAB)
    assert encoded_levels(save_preset(p)) == []
    p.complete(L.L5_LAB)
    p.apply_op(set_budget(1000))  # Equal to the generated default, still authored.
    assert encoded_levels(save_preset(p)) == [3]
    p.body.L2_instrument.params['nested'] = {'x': [1, 2]}
    assert encoded_levels(save_preset(p)) == [2, 3]
    assert load_preset(save_preset(p)).body.L2_instrument.params['nested'] == {'x': [1, 2]}


def test_replacing_a_generated_layer_with_an_equal_object_is_explicit():
    p = create_preset(12)
    p.body.L1_family = deepcopy(p.body.L1_family)
    assert encoded_levels(save_preset(p)) == [1]


def test_completion_updates_derived_defaults_without_overwriting_authored_layers():
    p = create_preset(1, tree_id=3, tower_level=L.L4_FORGE)
    p.apply_op(set_tree_id(42))
    assert p.body.L1_family.family_name == 'TREE_42'
    assert not p.body.is_explicit(L.L1_FAMILY)
    p.apply_op(add_regime('BOW'))
    assert p.body.L4_forge.regime_graph == {'BOW': ['BOW'], 'STABLE': ['STABLE']}
    p.body.L1_family.family_name = 'custom'
    p.apply_op(set_tree_id(4))
    assert p.body.L1_family.family_name == 'custom'
    assert encoded_levels(save_preset(p)) == [1]


def test_editor_operations_create_needed_layers_and_record_independent_values():
    p = create_preset(1, tower_level=L.L0_PATCH)
    value = {'array': [1, 2]}
    op = set_param('data', value)
    p.apply_op(op)
    value['array'].append(3)
    op.args['value']['array'].append(4)
    p.body.L2_instrument.params['data']['array'].append(5)
    assert p.edit_history[0].args['value'] == {'array': [1, 2]}
    p.apply_op(set_param('gain', .2))
    p.apply_op(delta_param('gain', .3))
    assert p.body.L2_instrument.params['gain'] == .5
    p.apply_op(set_bound('gain', 0., 1.))
    restored = load_preset(save_preset(p))
    assert restored.body.L2_instrument.bounds['gain'] == (0., 1.)


def test_projection_and_panic_do_not_resurrect_removed_authored_layers():
    p = Preset(PresetHeader(1, tower_level=L.L5_LAB))
    for level, layer in zip(L, authored_layers()):
        p.body.set_layer(level, layer)
    p.apply_op(project_to_level_op(L.L2_INSTRUMENT))
    assert encoded_levels(save_preset(p)) == [0, 1, 2]
    p.complete(L.L5_LAB)
    assert encoded_levels(save_preset(p)) == [0, 1, 2]
    p.apply_op(reset_to_safe_op())
    assert p.header.tower_level == L.L2_INSTRUMENT and p.is_valid()
    assert load_preset(save_preset(p)).body.L3_cockpit is None


def test_setting_tower_level_authors_that_layer_and_is_persisted():
    p = create_preset(1)
    p.apply_op(set_tower_level(L.L4_FORGE))
    assert encoded_levels(save_preset(p)) == [4]
    p.apply_op(set_tower_level(L.L1_FAMILY))
    assert encoded_levels(save_preset(p)) == [1]


def test_crc_is_mandatory_including_zero_and_truncation():
    p = Preset(PresetHeader(1), PresetBody(L0_patch=L0PatchLayer(gain=.5)))
    data = save_preset(p)
    assert p.header.crc32 == zlib.crc32(data[16:]) != 0
    for damaged in (data[:-1], data[:20] + bytes([data[20] ^ 1]) + data[21:],
                    data[:12] + b'\0' * 4 + data[16:]):
        with pytest.raises(ValueError, match='CRC mismatch'):
            load_preset(damaged)


def test_legacy_header_only_files_remain_readable():
    p = load_preset(PresetHeader(123, tower_level=L.L3_COCKPIT).to_bytes())
    assert p.is_valid() and p.body.L3_cockpit is not None
    assert not any(p.body.is_explicit(level) for level in L)


@pytest.mark.parametrize('body', [b'unknown', b'SYTW\xff', b'SYTW{', b'SYTW[]',
    b'SYTW{"version":2,"layers":[]}', b'SYTW{"version":true,"layers":[]}',
    b'SYTW{"version":1,"version":1,"layers":[]}',
    b'SYTW{"version":1,"layers":[],"extra":1}',
    b'SYTW{"version":1,"layers":[{"level":6,"fields":null}]}',
    b'SYTW{"version":1,"layers":[{"level":false,"fields":null}]}',
    b'SYTW{"version":1,"layers":[{"level":0,"fields":NaN}]}'])
def test_crc_valid_malformed_bodies_are_rejected(body):
    with pytest.raises(ValueError):
        load_preset(with_body(body))


def test_header_level_and_layer_schema_are_checked():
    data = save_preset(Preset(PresetHeader(1), PresetBody(L0_patch=L0PatchLayer())))
    with pytest.raises(ValueError, match='tower_level'):
        load_preset(data[:6] + bytes([3]) + data[7:])
    obj = envelope(data)
    obj['layers'][0]['fields']['items'][0][1] = 'not-a-number'  # gain
    with pytest.raises(ValueError, match='Invalid type'):
        load_preset(with_body(BODY_MAGIC + json.dumps(obj).encode()))
    obj = envelope(data)
    obj['layers'].append(deepcopy(obj['layers'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        load_preset(with_body(BODY_MAGIC + json.dumps(obj).encode()))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), object(), b'bytes'])
def test_unsupported_values_fail_save_without_replacing_crc(value):
    p = create_preset(1)
    p.body.L2_instrument.params['bad'] = value
    before = deepcopy(p.header)
    with pytest.raises(ValueError):
        save_preset(p)
    assert p.header == before


def test_serialization_is_stable_across_hash_seeds_and_container_order():
    source = '''from symcrash_python_ap.library.preset_model import *
import hashlib
p = create_preset(7, tower_level=TowerLevel.L5_LAB)
p.body.L2_instrument.allowed_operators = {'Tilt', 'DiffuseUp', 'IntraMix'}
p.body.L2_instrument.params = {key: key for key in {'gamma', 'alpha', 'beta'}}
print(hashlib.sha256(save_preset(p)).hexdigest())
'''
    hashes = [subprocess.check_output([sys.executable, '-c', source],
              env={**os.environ, 'PYTHONHASHSEED': seed}, text=True).strip() for seed in ('1', '57')]
    assert hashes[0] == hashes[1]


def test_determinism_validator_compares_values_and_preserves_inputs():
    a, b = create_preset(1), create_preset(1)
    before = deepcopy(a)
    assert not validate_determinism(a, b, [set_param('gain', .2)], [set_param('gain', .3)])
    assert validate_determinism(a, b, [set_param('gain', .2)], [set_param('gain', .2)])
    assert a == before


def test_editor_removal_preserves_playability():
    p = create_preset(1)
    with pytest.raises(ValueError, match='final regime'):
        p.apply_op(remove_regime('STABLE'))
    p.apply_op(add_regime('BOW'))
    p.apply_op(remove_regime('STABLE'))
    assert p.body.L1_family.initial_regime == 'BOW' and p.is_valid()
    p.body.L2_instrument.allowed_operators = {'Tilt'}
    with pytest.raises(ValueError, match='final allowed operator'):
        p.apply_op(set_operator_allowed('Tilt', False))


@pytest.mark.parametrize('replacement', [
    {'type': 'object', 'items': []},
    {'type': 'tuple', 'items': []},
    {'type': 'dict', 'items': [[{'type': 'list', 'items': []}, 1]]},
    {'type': 'dict', 'items': [['gain', .5], ['gain', .8]]},
])
def test_invalid_typed_containers_cannot_construct_a_layer(replacement):
    obj = {'version': 1, 'layers': [{'level': 0, 'fields': replacement}]}
    with pytest.raises(ValueError):
        load_preset(with_body(BODY_MAGIC + json.dumps(obj).encode()))


def test_codec_limits_and_cyclic_values_are_rejected(monkeypatch):
    from symcrash_python_ap.library import tower_codec

    p = create_preset(1)
    cycle = []
    cycle.append(cycle)
    p.body.L2_instrument.params['cycle'] = cycle
    with pytest.raises(ValueError, match='nesting'):
        save_preset(p)
    valid = save_preset(create_preset(2))
    monkeypatch.setattr(tower_codec, 'MAX_BODY_BYTES', 8)
    with pytest.raises(ValueError, match='size limit'):
        save_preset(create_preset(3))
    with pytest.raises(ValueError, match='size limit'):
        load_preset(valid)


@pytest.mark.parametrize('data', [b'', b'\0' * 15,
    struct.pack('>IHBBHHI', 1, 1, 0, 1, 0, 0, 0),
    struct.pack('>IHBBHHI', 1, 1, 6, 0, 0, 0, 0)])
def test_invalid_headers_are_rejected(data):
    with pytest.raises(ValueError):
        load_preset(data)
