"""Check renderer bounds, FIFO credit, and composed cache ownership."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import check_hdl_voice_stream_bounds as bounds
from tools import check_hdl_voice_stream_fifo as fifo
from tools import check_hdl_voice_stream_ownership as ownership


def check():
    return {'bounds': bounds.check(), 'fifo': fifo.check(), 'ownership': ownership.check()}


if __name__ == '__main__':
    check()
