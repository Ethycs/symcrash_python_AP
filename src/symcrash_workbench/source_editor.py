"""Local YAML editor embedded in Open Stage Control's SOURCE tab."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

_MAX_SOURCE_BYTES = 1024 * 1024

_EDITOR_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Symcrash YAML</title>
<style>
:root { color-scheme: dark; --bg:#17191d; --panel:#22262c; --line:#404751;
        --text:#e8edf2; --muted:#9ba7b4; --accent:#65c9b3; --bad:#ff756f; }
* { box-sizing:border-box; }
body { margin:0; height:100vh; display:flex; flex-direction:column;
       background:var(--bg); color:var(--text); font:13px system-ui,sans-serif; }
header { display:flex; gap:8px; align-items:center; padding:8px;
         background:var(--panel); border-bottom:1px solid var(--line); }
button { border:1px solid var(--line); border-radius:4px; padding:7px 12px;
         color:var(--text); background:#303640; cursor:pointer; }
button.primary { background:#245f54; border-color:var(--accent); }
#path { color:var(--muted); overflow:hidden; text-overflow:ellipsis;
        white-space:nowrap; flex:1; }
#status { padding:6px 10px; color:var(--muted); border-bottom:1px solid var(--line); }
#status.bad { color:var(--bad); }
textarea { flex:1; width:100%; resize:none; border:0; outline:0; padding:14px;
           background:var(--bg); color:var(--text); tab-size:2; white-space:pre;
           font:13px/1.45 Consolas,"Cascadia Mono",monospace; }
</style>
</head>
<body>
<header>
  <button onclick="loadSource()">REFRESH</button>
  <button onclick="validateSource()">VALIDATE</button>
  <button class="primary" onclick="applySource()">APPLY + RELOAD</button>
  <span id="path">Connecting…</span>
</header>
<div id="status">Ctrl+S applies after full IR validation.</div>
<textarea id="source" spellcheck="false"></textarea>
<script>
const source = document.querySelector('#source');
const statusEl = document.querySelector('#status');
let baseline = '';
function status(text, bad=false) {
  statusEl.textContent = text; statusEl.classList.toggle('bad', bad);
}
async function loadSource() {
  try {
    const response = await fetch('/api/source', {cache:'no-store'});
    const data = await response.json();
    source.value = data.source; baseline = data.source;
    document.querySelector('#path').textContent = data.path;
    status(`Loaded ${data.name} · version ${data.version}`);
  } catch (error) { status(`Load failed: ${error}`, true); }
}
async function post(path) {
  const response = await fetch(path, {method:'POST', headers:{
    'Content-Type':'text/plain; charset=utf-8'}, body:source.value});
  const data = await response.json();
  status(data.message, !data.ok);
  if (data.ok && path === '/api/apply') setTimeout(loadSource, 180);
}
function validateSource() { post('/api/validate'); }
function applySource() { post('/api/apply'); }
source.addEventListener('input', () => {
  if (source.value !== baseline) status('Edited · not yet applied');
});
window.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
    event.preventDefault(); applySource();
  }
});
loadSource();
</script>
</body>
</html>"""


class SourceEditorServer:
    """A loopback-only HTTP surface for editing the session's YAML source."""

    def __init__(self, session, port: int = 7410,
                 submit: Optional[Callable[[str], None]] = None):
        self.session = session
        self.submit = submit
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format_, *args):  # quiet embedded requests
                return

            def _send(self, status, body, content_type):
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(encoded)

            def _json(self, status, payload):
                self._send(status, json.dumps(payload),
                           "application/json; charset=utf-8")

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/?"):
                    self._send(200, _EDITOR_HTML,
                               "text/html; charset=utf-8")
                    return
                if self.path == "/api/source":
                    session = owner.session
                    try:
                        source = session.preset_path.read_bytes().decode("utf-8")
                        self._json(200, {
                            "ok": True,
                            "source": source,
                            "path": str(session.preset_path),
                            "name": session.ir.meta.name,
                            "version": session.ir.meta.version,
                        })
                    except OSError as error:
                        self._json(500, {"ok": False, "message": str(error)})
                    return
                self._json(404, {"ok": False, "message": "Not found"})

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > _MAX_SOURCE_BYTES:
                    self._json(413, {"ok": False,
                                     "message": "Source is empty or too large"})
                    return
                try:
                    source = self.rfile.read(length).decode("utf-8")
                except UnicodeDecodeError as error:
                    self._json(400, {"ok": False, "message": str(error)})
                    return
                if self.path == "/api/validate":
                    ok, message, _ = owner.session.validate_source(source)
                    self._json(200 if ok else 400,
                               {"ok": ok, "message": message})
                    return
                if self.path == "/api/apply":
                    if (owner.session.changed
                            or owner.session.patchboard.dirty.any):
                        self._json(409, {
                            "ok": False,
                            "message": "SAVE or RELOAD pending knob/patchboard "
                                       "edits before applying source",
                        })
                        return
                    ok, message, _ = owner.session.validate_source(source)
                    if not ok:
                        self._json(400, {"ok": False, "message": message})
                        return
                    if owner.submit is None:
                        ok, message = owner.session.apply_source(source)
                    else:
                        owner.submit(source)
                        ok, message = True, "Validated; queued for next audio block"
                    self._json(200 if ok else 400,
                               {"ok": ok, "message": message})
                    return
                self._json(404, {"ok": False, "message": "Not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        self.server.daemon_threads = True
        self._thread = threading.Thread(target=self.server.serve_forever,
                                        daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=1.0)


__all__ = ["SourceEditorServer"]
