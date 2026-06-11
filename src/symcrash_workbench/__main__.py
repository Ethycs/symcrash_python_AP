"""
CLI entry point.

    # one-time: generate the control surface layout
    pixi run python -m symcrash_workbench --write-layout

    # start a tuning session on a preset (name or path)
    pixi run python -m symcrash_workbench dx_marimba_v0

    # then launch the surface (separate terminal / any machine):
    open-stage-control --load tools/workbench/symcrash-workbench.json ^
        --send 127.0.0.1:7400 --osc-port 7500
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
    sys.exit(f"preset not found: {arg} (tried {candidate})")


def main():
    ap = argparse.ArgumentParser(
        prog="symcrash_workbench",
        description="OSC tuning workbench (Open Stage Control front-end)")
    ap.add_argument("preset", nargs="?",
                    help="preset name (presets/instruments/<name>.yaml) or path")
    ap.add_argument("--listen-port", type=int, default=7400,
                    help="UDP port to receive OSC on (default 7400)")
    ap.add_argument("--feedback", default="127.0.0.1:7500",
                    help="host:port for widget feedback to Open Stage Control "
                         "(its --osc-port); 'none' to disable")
    ap.add_argument("--debounce", type=float, default=180.0,
                    help="ms of knob silence before auto re-render")
    ap.add_argument("--no-autoplay", action="store_true",
                    help="only render on PLAY, not on every change")
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

    from .osc_server import serve
    from .session import WorkbenchSession

    preset_path = _resolve_preset(args.preset)
    session = WorkbenchSession(preset_path, debounce_ms=args.debounce,
                               autoplay=not args.no_autoplay)
    print(f"[workbench] loaded {preset_path.name} "
          f"({session.ir.meta.name} v{session.ir.meta.version})")
    if not DEFAULT_LAYOUT_PATH.exists():
        write_layout()
        print(f"[workbench] generated layout: {DEFAULT_LAYOUT_PATH}")
    print("[workbench] surface: open-stage-control "
          f"--load {DEFAULT_LAYOUT_PATH.relative_to(REPO_ROOT)} "
          f"--send 127.0.0.1:{args.listen_port} --osc-port 7500")

    feedback_host, feedback_port = None, None
    if args.feedback.lower() != "none":
        host, _, port = args.feedback.partition(":")
        feedback_host, feedback_port = host, int(port or 7500)

    serve(session, listen_port=args.listen_port,
          feedback_host=feedback_host, feedback_port=feedback_port)


if __name__ == "__main__":
    main()
