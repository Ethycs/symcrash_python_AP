"""
CLI entry point - the canonical live-synth surface.

    # one-time: generate the control surface layout
    pixi run python -m symcrash_workbench --write-layout

    # start a LIVE streaming session on a preset (name or path):
    # OSC serving + realtime audio, per-block param latch, live
    # /preset/load swaps from the surface
    pixi run python -m symcrash_workbench dx_marimba_v0

    # legacy offline mode (debounced full re-render on each edit)
    pixi run python -m symcrash_workbench dx_marimba_v0 --offline

    # then launch the surface (separate terminal, Windows):
    pixi run workbench-ui
"""

import argparse
import sys
from pathlib import Path

from .make_layout import DEFAULT_LAYOUT_PATH, write_layout

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_preset(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = REPO_ROOT / "presets" / "instruments" / f"{arg}.yaml"
    if candidate.exists():
        return candidate
    # id-or-name resolution through the catalog (meta id, display name,
    # capsule id from the library index)
    from .library import resolve_preset
    entry = resolve_preset(arg)
    if entry is not None:
        return entry.path
    sys.exit(f"preset not found: {arg} (tried {candidate})")


def main():
    ap = argparse.ArgumentParser(
        prog="symcrash_workbench",
        description="OSC live-synth workbench (Open Stage Control "
                    "front-end). Default: streaming mode - realtime "
                    "audio with per-block parameter latch.")
    ap.add_argument("preset", nargs="?",
                    help="preset name (presets/instruments/<name>.yaml), "
                         "meta id, capsule id, or path")
    ap.add_argument("--offline", action="store_true",
                    help="offline mode: debounced full re-render on each "
                         "edit instead of a live stream")
    ap.add_argument("--note", type=int, default=60,
                    help="MIDI note for the streamed voice (default 60)")
    ap.add_argument("--velocity", type=float, default=0.85,
                    help="velocity 0-1 for the streamed voice")
    ap.add_argument("--listen-port", type=int, default=7400,
                    help="UDP port to receive OSC on (default 7400)")
    ap.add_argument("--feedback", default="127.0.0.1:7500",
                    help="host:port for widget feedback to Open Stage Control "
                         "(its --osc-port); 'none' to disable")
    ap.add_argument("--source-port", type=int, default=7410,
                    help="loopback port for the YAML editor embedded in the "
                         "SOURCE tab; 0 disables it (default 7410)")
    ap.add_argument("--debounce", type=float, default=180.0,
                    help="[offline] ms of knob silence before auto re-render")
    ap.add_argument("--no-autoplay", action="store_true",
                    help="[offline] only render on PLAY, not on every change")
    ap.add_argument("--write-layout", action="store_true",
                    help=f"write {DEFAULT_LAYOUT_PATH.name} and exit")
    args = ap.parse_args()

    if args.write_layout:
        path = write_layout()
        print(f"wrote {path}")
        if not args.preset:
            return

    if not args.preset:
        ap.error("preset required (or use --write-layout)")

    from .session import WorkbenchSession

    preset_path = _resolve_preset(args.preset)

    if not DEFAULT_LAYOUT_PATH.exists():
        write_layout()
        print(f"[workbench] generated layout: {DEFAULT_LAYOUT_PATH}")
    print("[workbench] surface (second terminal): pixi run workbench-ui")

    feedback_host, feedback_port = None, None
    if args.feedback.lower() != "none":
        host, _, port = args.feedback.partition(":")
        feedback_host, feedback_port = host, int(port or 7500)

    if args.offline:
        from .osc_server import serve
        from .source_editor import SourceEditorServer

        session = WorkbenchSession(preset_path, debounce_ms=args.debounce,
                                   autoplay=not args.no_autoplay)
        source_editor = (SourceEditorServer(session, args.source_port).start()
                         if args.source_port else None)
        if source_editor is not None:
            print(f"[workbench] YAML editor http://127.0.0.1:"
                  f"{source_editor.port}/")
        print(f"[workbench] loaded {preset_path.name} "
              f"({session.ir.meta.name} v{session.ir.meta.version}) [offline]")
        try:
            serve(session, listen_port=args.listen_port,
                  feedback_host=feedback_host, feedback_port=feedback_port)
        finally:
            if source_editor is not None:
                source_editor.stop()
        return

    # ------------------------------------------------- streaming (default)
    from .streaming import OSC_AVAILABLE, StreamingOSCBridge, StreamingVoice

    session = WorkbenchSession(preset_path, autoplay=False)
    print(f"[workbench] loaded {preset_path.name} "
          f"({session.ir.meta.name} v{session.ir.meta.version}) [streaming]")
    voice = StreamingVoice(session, note=args.note, velocity=args.velocity)
    from .source_editor import SourceEditorServer
    source_editor = (SourceEditorServer(
        session, args.source_port,
        submit=lambda source: voice.inject("/editor/source/apply", source),
    ).start() if args.source_port else None)
    if source_editor is not None:
        print(f"[workbench] YAML editor http://127.0.0.1:"
              f"{source_editor.port}/")
    bridge = None
    if OSC_AVAILABLE:
        bridge = StreamingOSCBridge(
            voice, listen_port=args.listen_port,
            feedback_host=feedback_host, feedback_port=feedback_port).start()
        print(f"[workbench] OSC listening on udp/{bridge.port} "
              "(knobs, /preset/list, /preset/load, /stream/note_on, "
              "/stream/sweep)")
    else:
        print("[workbench] python-osc not installed; in-process only")
    voice.note_on(args.note)
    print(f"[workbench] sustaining note {args.note}; Ctrl+C to stop")
    try:
        if not voice.play_realtime(keep_open=True):
            while not voice.stopped:  # headless OSC service: free-run
                voice.render_block()
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()
        if bridge is not None:
            bridge.stop()
        if source_editor is not None:
            source_editor.stop()
        session.close()


if __name__ == "__main__":
    main()
