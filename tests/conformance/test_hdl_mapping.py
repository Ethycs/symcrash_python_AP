"""Reject the observed zero-exit ABC crash and missing/failed verification."""
import pytest
from tools.hdl_mapping import verification_count

PASS = 'ABC: + &verify \nABC: Networks are equivalent.  Time = 0.29 sec\n'


def test_clean_mapping_requires_one_success_per_invocation():
    assert verification_count(PASS) == 1
    assert verification_count(PASS+PASS) == 2


@pytest.mark.parametrize('output', [
    '',
    'ABC: Networks are equivalent.\n',
    PASS+'ABC: + &verify\n',
    PASS+'ABC: + &mfs\n',
    PASS+'ABC: Networks are NOT equivalent.\n',
    PASS+"ABC: yosys-abc: giaMfs.c:386: Assertion `iLitNew >= 0' failed.\nABC: Aborted (core dumped)\n",
    PASS+'Warning: ABC: execution of command failed: return code 134.\n',
])
def test_mapping_failure_cannot_be_accepted_as_verified(output):
    with pytest.raises(RuntimeError, match='ABC9 mapping'):
        verification_count(output)
