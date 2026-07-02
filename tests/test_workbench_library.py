"""
Workbench library + live-engine-gap tests (OSC preset browsing, noise
exciter latch, unison/stereo streaming, new registry params).

Complements test_workbench_streaming.py. Audio rendered per test stays
under ~1.5 s.
"""

import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from symcrash_workbench.library import (  # noqa: E402
    resolve_preset, scan_preset_catalog, send_preset_list,
)
from symcrash_workbench.session import WorkbenchSession  # noqa: E402
from symcrash_workbench.streaming import (  # noqa: E402
    OSC_AVAILABLE, StreamingVoice,
)

PRESET_DIR = REPO_ROOT / "presets" / "instruments"

# Pre-change golden SHA256 of 0.5 s mono streams (note 60, fresh session).
# Captured on the legacy mono path BEFORE the noise/unison/stereo work:
# proves the mono path is still bit-exact after it.
GOLDEN_MONO = {
    "string_pad_v0":
        "7fb3844585025cea84e0ef3eb1c214a8f0dd8a33121100f9d258ccf52f099a5d",
    "dx_marimba_v0":
        "7a549167a0fb5bb7f461aa265ff624a375dd551921c3715c8b11d2b724b59eac",
}


@pytest.fixture
def make_session(tmp_path):
    """Factory: isolated session on a scratch copy of a named preset."""
    created = []

    def factory(name="string_pad_v0"):
        scratch = tmp_path / f"scratch_{name}.yaml"
        if not scratch.exists():
            shutil.copy(PRESET_DIR / f"{name}.yaml", scratch)
        s = WorkbenchSession(scratch, autoplay=False, log_dir=tmp_path)
        created.append(s)
        return s

    yield factory
    for s in created:
        s.close()


# ============================================================ mono bit-exact

@pytest.mark.parametrize("name", sorted(GOLDEN_MONO))
def test_mono_stream_bit_exact(make_session, name):
    """Mono presets (no noise, no unison) render bit-exactly as before
    the noise/unison work: 1-D output, golden hash match."""
    voice = StreamingVoice(make_session(name))
    voice.note_on(60)
    audio = voice.render_seconds(0.5)
    voice.stop()
    assert audio.ndim == 1, "mono preset must stay on the 1-D legacy path"
    assert hashlib.sha256(audio.tobytes()).hexdigest() == GOLDEN_MONO[name]


# ===================================================================== noise

def test_noise_live_latch_changes_audio(make_session):
    """/noise/level injected mid-stream reaches the audio (per-block
    latch), including on a preset that authors no noise at all."""
    control_voice = StreamingVoice(make_session())
    control_voice.note_on(60)
    control = [control_voice.render_block() for _ in range(24)]
    control_voice.stop()

    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    blocks = []
    for i in range(24):
        if i == 8:
            voice.inject("/noise/level", 0.6)
        blocks.append(voice.render_block())
    voice.stop()

    pre = max(int(np.abs(a.astype(np.int32) - c.astype(np.int32)).max())
              for a, c in zip(blocks[:8], control[:8]))
    post = max(int(np.abs(a.astype(np.int32) - c.astype(np.int32)).max())
               for a, c in zip(blocks[8:], control[8:]))
    assert pre == 0, "audio diverged before the noise injection"
    assert post > 50, "noise level injection never reached the audio"
    # tracked for SAVE
    assert abs(session.changed["/noise/level"] - 0.6) < 1e-6


def test_noise_params_roundtrip(make_session):
    """New /noise/* registry params: get/set on the IR + yaml_apply."""
    from ruamel.yaml import YAML

    session = make_session()
    reg, ir = session.registry, session.ir

    assert reg["/noise/level"].get(ir) == 0.0  # unauthored -> off
    for address, value in (("/noise/level", 0.4), ("/noise/attack_ms", 2.5),
                           ("/noise/decay_ms", 80.0), ("/noise/color", 1.0)):
        assert reg[address].set(ir, value)
        assert abs(reg[address].get(ir) - value) < 1e-6

    from symcrash_python_ap.instrument_ir import extract_noise_params
    np_ = extract_noise_params(ir)
    assert np_ is not None and np_["color"] == 1  # pink

    yaml = YAML()
    doc = yaml.load(session.preset_path.read_bytes().decode("utf-8"))
    root = doc["instrument_ir_v0"]
    for address, value in (("/noise/level", 0.4), ("/noise/attack_ms", 2.5),
                           ("/noise/decay_ms", 80.0), ("/noise/color", 1.0)):
        assert reg[address].yaml_apply(root, ir, value)
    ynoise = next(n.get("noise") for n in root["graph"].values()
                  if n.get("noise"))
    assert ynoise["level"] == 0.4
    assert ynoise["color"] == "pink"


# ==================================================================== unison

def test_unison_stream_is_stereo(make_session):
    """A unison preset streams (BLOCK, 2) stereo blocks with L != R."""
    voice = StreamingVoice(make_session("super_saw_v0"))
    voice.autoclamp = False  # deterministic: keep the authored 7 voices
    voice.note_on(60)
    audio = voice.render_seconds(0.5)
    voice.stop()
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert len(voice.copies) == 7
    left = audio[:, 0].astype(np.int32)
    right = audio[:, 1].astype(np.int32)
    assert np.abs(left).max() > 1000 and np.abs(right).max() > 1000
    assert np.abs(left - right).max() > 100, "stereo channels identical"


def test_unison_params_roundtrip(make_session):
    """New /unison/* registry params: get/set on the IR + yaml_apply."""
    from ruamel.yaml import YAML

    session = make_session()
    reg, ir = session.registry, session.ir

    assert reg["/unison/voices"].get(ir) == 1.0  # unauthored -> mono
    assert reg["/unison/voices"].set(ir, 4)
    assert reg["/unison/spread_cents"].set(ir, 18)
    assert reg["/unison/width"].set(ir, 0.7)
    assert reg["/unison/voices"].get(ir) == 4.0

    from symcrash_python_ap.instrument_ir import extract_unison_params
    u = extract_unison_params(ir)
    assert u == {"voices": 4, "spread_cents": 18, "width": 0.7}

    yaml = YAML()
    doc = yaml.load(session.preset_path.read_bytes().decode("utf-8"))
    root = doc["instrument_ir_v0"]
    for address, value in (("/unison/voices", 4),
                           ("/unison/spread_cents", 18),
                           ("/unison/width", 0.7)):
        assert reg[address].yaml_apply(root, ir, value)
    yunison = next(n.get("unison") for n in root["graph"].values()
                   if n.get("unison"))
    assert yunison["voices"] == 4
    assert yunison["spread_cents"] == 18
    assert abs(yunison["width"] - 0.7) < 1e-9


def test_unison_voices_1_stays_mono(make_session):
    """Setting /unison/voices back to 1 restores the mono stream path."""
    session = make_session("super_saw_v0")
    session.registry["/unison/voices"].set(session.ir, 1)
    voice = StreamingVoice(session)
    voice.note_on(60)
    audio = voice.render_seconds(0.2)
    voice.stop()
    assert audio.ndim == 1


# =================================================================== library

def test_catalog_scan_known_ids():
    catalog = scan_preset_catalog()
    stems = {e.stem for e in catalog}
    assert {"string_pad_v0", "dx_marimba_v0", "super_saw_v0",
            "acoustic_marimba_v0"} <= stems
    pad = next(e for e in catalog if e.stem == "string_pad_v0")
    assert pad.preset_id == "string-pad-v0"
    assert pad.name == "String Pad"
    assert pad.path.exists()


def test_catalog_capsule_ids_resolve():
    """Capsule ids from the library index map back to YAML sources."""
    catalog = scan_preset_catalog()
    with_capsules = [e for e in catalog if e.capsule_ids]
    assert with_capsules, "no capsule ids resolved from the library index"
    entry = with_capsules[0]
    resolved = resolve_preset(entry.capsule_ids[0], catalog)
    assert resolved is entry


def test_resolve_preset_forms():
    catalog = scan_preset_catalog()
    by_stem = resolve_preset("string_pad_v0", catalog)
    assert by_stem is not None
    assert resolve_preset("string-pad-v0", catalog) is by_stem   # meta id
    assert resolve_preset("String Pad", catalog) is by_stem      # name
    path = str(PRESET_DIR / "string_pad_v0.yaml")
    assert resolve_preset(path, catalog).path == by_stem.path    # path
    assert resolve_preset("no_such_preset_xyz", catalog) is None


def test_send_preset_list_reply_shape():
    msgs = []
    n = send_preset_list(lambda addr, args: msgs.append((addr, args)))
    items = [m for m in msgs if m[0] == "/preset/item"]
    assert len(items) == n and n >= 40
    stem, preset_id, name, capsules = items[0][1]
    assert isinstance(stem, str) and isinstance(capsules, str)
    assert msgs[-1] == ("/preset/count", n)


# ============================================================== /preset/load

def test_preset_load_swaps_session(make_session):
    """session.handle('/preset/load', id): IR swaps, callback fires."""
    session = make_session()
    fired = []
    session.on_preset_loaded = lambda s: fired.append(s.preset_path.stem)
    assert session.handle("/preset/load", "dx_marimba_v0") is True
    assert session.preset_path.stem == "dx_marimba_v0"
    assert session.ir.meta.instrument_id == "dx-marimba-v0"
    assert fired == ["dx_marimba_v0"]
    assert session.changed == {}
    assert session.handle("/preset/load", "no_such_preset_xyz") is False


def test_preset_load_mid_stream_survives(make_session):
    """A live /preset/load swap keeps the stream running: the old note
    fades, the new preset retriggers, knobs re-sync via push_state."""
    session = make_session()
    pushed = []
    session.on_preset_loaded = lambda s: s.push_state(
        lambda a, v: pushed.append((a, v)))
    voice = StreamingVoice(session)
    voice.note_on(60)
    for _ in range(10):
        voice.render_block()
    voice.inject("/preset/load", "dx_marimba_v0")
    post = [voice.render_block() for _ in range(80)]
    assert voice.active, "stream died across the preset swap"
    assert session.preset_path.stem == "dx_marimba_v0"
    assert session.ir.meta.instrument_id == "dx-marimba-v0"
    # push_state fired and mirrored the new preset to the surface
    assert ("/preset/load", "dx_marimba_v0") in pushed
    # the new preset is audible after the swap
    peak = max(int(np.abs(b.astype(np.int32)).max()) for b in post[5:])
    assert peak > 500, "no audio from the swapped-in preset"
    voice.stop()


def test_preset_load_unknown_mid_stream_is_harmless(make_session):
    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    voice.render_block()
    voice.inject("/preset/load", "no_such_preset_xyz")
    for _ in range(4):
        voice.render_block()
    assert voice.active
    assert session.preset_path.stem.startswith("scratch_string_pad")
    voice.stop()


# ================================================================ OSC bridge

@pytest.mark.skipif(not OSC_AVAILABLE, reason="python-osc not installed")
def test_osc_bridge_preset_list_and_load(make_session):
    """End-to-end over UDP: /preset/list replies on the feedback port;
    /preset/load swaps the streamed session."""
    import threading

    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient
    from symcrash_workbench.streaming import StreamingOSCBridge

    received = []
    surface_dispatcher = Dispatcher()
    surface_dispatcher.set_default_handler(
        lambda addr, *args: received.append((addr, args)))
    surface = ThreadingOSCUDPServer(("127.0.0.1", 0), surface_dispatcher)
    surface_thread = threading.Thread(target=surface.serve_forever,
                                      daemon=True)
    surface_thread.start()

    session = make_session()
    voice = StreamingVoice(session)
    voice.note_on(60)
    bridge = StreamingOSCBridge(
        voice, listen_host="127.0.0.1", listen_port=0,
        feedback_host="127.0.0.1",
        feedback_port=surface.server_address[1]).start()
    try:
        client = SimpleUDPClient("127.0.0.1", bridge.port)
        client.send_message("/preset/list", 1)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(a == "/preset/count" for a, _ in received):
                break
            time.sleep(0.02)
        items = [a for a, _ in received if a == "/preset/item"]
        assert len(items) >= 40, "no /preset/item replies on feedback port"

        client.send_message("/preset/load", "dx_marimba_v0")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            voice.render_block()  # drains the inject queue -> swap
            if session.preset_path.stem == "dx_marimba_v0":
                break
        assert session.preset_path.stem == "dx_marimba_v0"
        assert voice.active
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(a == "/preset/loaded" for a, _ in received):
                break
            time.sleep(0.02)
        assert any(a == "/preset/loaded" and args[0] == "dx_marimba_v0"
                   for a, args in received)
    finally:
        bridge.stop()
        voice.stop()
        surface.shutdown()
        surface.server_close()
