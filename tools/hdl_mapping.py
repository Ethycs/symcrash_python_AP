"""Require clean, verified ABC9 mapping; never accept its pre-crash fallback."""
import hashlib
from pathlib import Path
import re

MAPPING = 'abc9-nomfs-verify'
NETLIST_TAG = 'abc9'
PRELUDE = 'scratchpad -set abc9.nomfs true\nscratchpad -set abc9.verify true\n'


def verification_count(output):
    # Yosys 0.33 can return zero after ABC aborts, retaining output written
    # before MFS. A process exit code alone therefore does not qualify mapping.
    failures = r'(?im)^ERROR:|Warning: ABC: execution|ABC:.*(?:Assertion.*failed|Aborted|not equivalent|verification failed)'
    if re.search(failures, output):
        raise RuntimeError('ABC9 mapping reported a subtool failure')
    if re.search(r'(?m)^ABC: \+ &mfs(?:\s|$)', output):
        raise RuntimeError('ABC9 mapping ran the disabled MFS pass')
    invoked = len(re.findall(r'(?m)^ABC: \+ &verify\s*$', output))
    passed = len(re.findall(r'(?m)^ABC: Networks are equivalent\.', output))
    if not invoked or passed != invoked:
        raise RuntimeError('ABC9 mapping lacks complete equivalence verification')
    return passed


def evidence(log):
    return {'mapping_equivalence_checks': verification_count(log.read_text(encoding='utf-8')),
            'synthesis_log': log.name,
            'synthesis_log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}


def matches(row, directory, expected_stem):
    log = Path(directory) / (expected_stem+'_ecp5.log')
    if row.get('synthesis_log') != log.name or not log.is_file():
        return False
    try:
        return all(row.get(key) == value for key, value in evidence(log).items())
    except RuntimeError:
        return False
