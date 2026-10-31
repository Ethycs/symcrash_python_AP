"""Prepare and validate dense exact vectors for mapped voice streaming."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.check_hdl_voice_stream import check


if __name__ == '__main__':
    print(check(headroom=2, latency=20, dense=True)['summary'], flush=True)
