"""
OSC server: receives widget changes from Open Stage Control, feeds the
session; pushes state back so the surface mirrors the loaded preset.

Wire-up (defaults):
    backend listens on  udp/7400   (O-S-C started with --send 127.0.0.1:7400)
    feedback sent to    udp/7500   (O-S-C started with --osc-port 7500)
"""

from typing import Optional

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

from .session import WorkbenchSession


def serve(session: WorkbenchSession,
          listen_host: str = "0.0.0.0", listen_port: int = 7400,
          feedback_host: Optional[str] = "127.0.0.1",
          feedback_port: Optional[int] = 7500):
    client = None
    if feedback_host and feedback_port:
        client = SimpleUDPClient(feedback_host, feedback_port)

    def on_message(address, *args):
        value = args[0] if args else None
        session.handle(address, value)
        # After save/reload, resync the surface
        if address in ("/reload",) and client is not None:
            session.push_state(client.send_message)

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(on_message)

    server = BlockingOSCUDPServer((listen_host, listen_port), dispatcher)
    if client is not None:
        session.push_state(client.send_message)
        print(f"[workbench] feedback -> {feedback_host}:{feedback_port}")
    print(f"[workbench] listening on {listen_host}:{listen_port} "
          f"(Ctrl+C to quit)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
