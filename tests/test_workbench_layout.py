"""Generated Open Stage Control layout contract tests."""

from symcrash_workbench.make_layout import build_layout
from symcrash_workbench.params import build_registry


def _walk(widget):
    yield widget
    for child in widget.get("widgets", []):
        yield from _walk(child)
    for child in widget.get("tabs", []):
        yield from _walk(child)


def test_layout_has_playable_keyboard_contract():
    widgets = {w.get("id"): w for w in _walk(build_layout()["content"])}
    keyboard = widgets["performance_keyboard"]

    assert keyboard["type"] == "keyboard"
    assert keyboard["address"] == "/stream/keyboard"
    assert keyboard["keys"] == 25
    assert keyboard["start"] == 48
    assert keyboard["split"] is False
    assert keyboard["on"] == 1
    assert keyboard["off"] == 0


def test_every_dial_has_an_explicit_visible_label():
    widgets = {w.get("id"): w for w in _walk(build_layout()["content"])}
    expected = {
        address.strip("/").replace("/", "_"): param.label
        for address, param in build_registry().items()
    }
    expected.update({"note": "NOTE", "velocity": "VEL", "duration": "DUR s"})

    for dial_id, label in expected.items():
        expected_type = "dropdown" if dial_id == "fm_algorithm" else "knob"
        assert widgets[dial_id]["type"] == expected_type
        caption = widgets[f"{dial_id}_label"]
        assert caption["type"] == "text"
        assert caption["value"] == label


def test_layout_is_tabbed_patchboard():
    layout = build_layout()
    assert [tab["label"] for tab in layout["content"]["tabs"]] == [
        "PLAY", "PATCH", "FM6", "MOD MAP", "BEHAVIOR", "SYSTEM", "SOURCE",
    ]
    widgets = {widget.get("id"): widget
               for widget in _walk(layout["content"])}
    assert widgets["graph_patchbay"]["address"] == "/editor/graph"
    assert widgets["mod_patchbay"]["address"] == "/editor/mod"
    assert widgets["behavior_patchbay"]["address"] == "/editor/behavior"
    assert widgets["fm_topology_patchbay"]["bypass"] is True
    assert widgets["fm_algorithm"]["address"] == "/fm/algorithm"
