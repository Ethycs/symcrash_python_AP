"""Structural OSC patchboard contracts and YAML round trips."""

import shutil

import pytest
from ruamel.yaml import YAML

from symcrash_python_ap.instrument_ir import load_instrument
from symcrash_python_ap.paths import PRESET_INSTRUMENTS_DIR
from symcrash_workbench.patchboard import MOD_TARGETS
from symcrash_workbench.session import WorkbenchSession


@pytest.fixture
def make_session(tmp_path):
    sessions = []

    def factory(name):
        path = tmp_path / f"{name}.yaml"
        shutil.copy(PRESET_INSTRUMENTS_DIR / f"{name}.yaml", path)
        session = WorkbenchSession(path, autoplay=False, log_dir=tmp_path)
        sessions.append(session)
        return session

    yield factory
    for session in sessions:
        session.close()


def _root(path):
    return YAML().load(path.read_bytes().decode("utf-8"))["instrument_ir_v0"]


def test_graph_patch_rejects_cycle_and_roundtrips(make_session):
    session = make_session("string_pad_v0")
    nodes = session.ir.layers.L0_graph.nodes
    assert len(nodes) >= 2
    first, second = nodes[:2]

    # Reverse the original route, then prove the cycle guard rejects adding
    # the original direction on top of it.
    assert session.handle_message("/editor/graph", "node1", "audio")
    assert session.handle_message("/editor/graph", "node2", "node1")
    assert session.handle_message("/editor/graph", "node1", "node2") is False
    assert "cycle" in session.patchboard.status.lower()

    assert session.save()
    root = _root(session.preset_path)
    assert root["graph"][second.id]["output"] == first.id
    assert "output" not in root["graph"][first.id]
    loaded = load_instrument(str(session.preset_path))
    assert [(edge.src, edge.dst) for edge in loaded.layers.L0_graph.edges] == [
        (second.id, first.id),
    ]


def test_fm_algorithm_updates_topology_and_yaml(make_session):
    session = make_session("dx_marimba_v0")
    feedback = []
    session.on_state_changed = lambda current: current.push_state(
        lambda address, value: feedback.append((address, value)))

    assert session.handle("/fm/algorithm", 5)
    assert session.save()
    assert _root(session.preset_path)["graph"]["fm6"]["algorithm"] == 5
    topology = [value for address, value in feedback
                if address == "/editor/fm/topology"]
    assert ["op2", "op1"] in topology
    assert ["op1", "audio"] in topology


def test_envelope_patchbay_many_targets_roundtrip(make_session):
    session = make_session("dx_marimba_v0")
    targets = ("FM_OP1", "FM_OP3", "FILTER_CUTOFF")
    assert set(targets) <= set(MOD_TARGETS)
    assert session.handle_message("/editor/mod", "env1", *targets)
    assert session.ir.envelopes[0].targets == list(targets)
    assert session.save()
    assert list(_root(session.preset_path)["envelopes"]["mod_env"]["targets"]) \
        == list(targets)
    assert load_instrument(str(session.preset_path)).envelopes[0].targets \
        == list(targets)


def test_behavior_patch_preserves_existing_guard_and_hysteresis(make_session):
    session = make_session("glitch_lead_v0")
    states = session.ir.layers.L2_regimes.regime_nodes
    stable = next(i for i, state in enumerate(states)
                  if state.emit_template == "T_STABLE")
    glitch = next(i for i, state in enumerate(states)
                  if state.emit_template == "T_GLITCH")

    assert session.handle_message(
        "/editor/behavior", f"state{stable + 1}", f"state{glitch + 1}")
    assert session.save()
    transitions = _root(session.preset_path)["behavior"]["transitions"]
    edge = next(item for item in transitions
                if item["from"] == "stable" and item["to"] == "glitch")
    assert edge["when"] == "risk > 0.7"
    assert dict(edge["hysteresis"]) == {"enter": 0.7, "exit": 0.3}


def test_materialize_default_behavior_roundtrip(make_session):
    session = make_session("string_pad_v0")
    assert not session.ir.layers.L2_regimes.regime_nodes
    assert session.handle_message("/editor/behavior/materialize", 1)
    assert len(session.ir.layers.L2_regimes.regime_nodes) == 4
    assert session.save()
    behavior = _root(session.preset_path)["behavior"]
    assert list(behavior["states"]) == ["idle", "attack", "stable", "recovery"]
    assert behavior["transitions"][0]["from"] == "idle"


def test_push_state_uses_stable_patchboard_slots(make_session):
    session = make_session("dx_marimba_v0")
    messages = []
    session.push_state(lambda address, value: messages.append((address, value)))
    graph = [value for address, value in messages
             if address == "/editor/graph"]
    mod = [value for address, value in messages if address == "/editor/mod"]
    assert ["node1", "node2"] in graph
    assert ["node2", "audio"] in graph
    assert any(value[0] == "env1" and "FM_OP2" in value for value in mod)


def test_full_source_apply_is_validated_atomic_and_versioned(make_session):
    session = make_session("dx_marimba_v0")
    before = session.preset_path.read_bytes()
    version = session.ir.meta.version
    source = before.decode("utf-8").replace(
        'name: "DX Marimba"', 'name: "DX Marimba Edited"')

    ok, message = session.apply_source(source)
    assert ok, message
    assert session.ir.meta.name == "DX Marimba Edited"
    assert session.ir.meta.version == version + 1
    written = session.preset_path.read_bytes().decode("utf-8")
    assert "# DX Marimba v0" in written  # round-trip comments survived

    accepted = session.preset_path.read_bytes()
    ok, _ = session.apply_source("instrument_ir_v0: [not, a, mapping]")
    assert not ok
    assert session.preset_path.read_bytes() == accepted

    assert session.handle("/amp/decay", 222.0)
    ok, message = session.apply_source(accepted.decode("utf-8"))
    assert not ok and "SAVE or RELOAD" in message
    assert session.preset_path.read_bytes() == accepted


def test_embedded_source_editor_serves_and_validates(make_session):
    import json
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from symcrash_workbench.source_editor import SourceEditorServer

    session = make_session("string_pad_v0")
    server = SourceEditorServer(session, port=0).start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urlopen(f"{base}/api/source", timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["ok"] is True
        assert "instrument_ir_v0:" in payload["source"]

        request = Request(f"{base}/api/validate", data=b"not: valid: yaml",
                          method="POST",
                          headers={"Content-Type": "text/plain"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
    finally:
        server.stop()


def test_save_write_failure_keeps_file_and_pending_edits(make_session,
                                                          monkeypatch):
    import os

    session = make_session("dx_marimba_v0")
    before = session.preset_path.read_bytes()
    version = session.ir.meta.version
    assert session.handle("/amp/decay", 321.0)

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    assert session.save() is False
    assert session.preset_path.read_bytes() == before
    assert session.ir.meta.version == version
    assert session.changed["/amp/decay"] == 321.0


def test_save_rejects_external_source_change(make_session):
    session = make_session("dx_marimba_v0")
    assert session.handle("/amp/decay", 654.0)
    externally_edited = (session.preset_path.read_bytes()
                         + b"\n# concurrent external edit\n")
    session.preset_path.write_bytes(externally_edited)

    assert session.save() is False
    assert session.preset_path.read_bytes() == externally_edited
    assert session.changed["/amp/decay"] == 654.0
    assert "changed on disk" in session.patchboard.status
