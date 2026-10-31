"""Run 32 runtime-K profiles: every headroom/stall pair, full vectors at h=0/2/15."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import check_hdl_idwpt_runtime as checks
from tools.build_hdl_dma import digest, DIRECTORY


def main():
    sources = {n: digest(checks.HDL/n) for n in ('ccs_idwpt_runtime.v', 'tb_ccs_idwpt_runtime.v')}
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = DIRECTORY/f'runtime_matrix_{fingerprint[:12]}.json'
    report = {'sources': sources, 'runner_sha256': digest(Path(__file__)), 'configurations': []}
    for headroom in range(16):
        for stalls in (False, True):
            row = checks.check(headroom, stalls, headroom in (0, 2, 15))
            assert row['sources'] == sources
            report['configurations'].append(row)
            destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print(row['summary'], flush=True)
    print('Runtime matrix complete:', len(report['configurations']), 'configurations,',
          sum(row['exact_samples'] for row in report['configurations']), 'exact samples.', flush=True)


if __name__ == '__main__':
    main()
