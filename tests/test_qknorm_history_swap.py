import numpy as np
import pytest
from bench.qknorm_history_swap import splice


def stream(rounds, offset=0):
    n=len(rounds)
    t=np.zeros((n,186),np.uint8)
    t[:,0]=1;t[:,158]=1
    t[:,4]=np.arange(n)+offset
    return t,np.array(rounds,np.int64),np.full(n,3,np.int64)


@pytest.mark.parametrize('fixed',[False,True])
def test_equal_length_splice_preserves_current_round_and_source(fixed):
    original=stream([0,0,1,1,2,2])
    donor=stream([0,0,0,1,1],10)
    before=tuple(a.copy() for a in original)
    changed=splice(original,donor,2,fixed)
    for a,b in zip(original,before):np.testing.assert_array_equal(a,b)
    for a,b in zip(original,changed):np.testing.assert_array_equal(a[4:],b[4:])
    np.testing.assert_array_equal(changed[0][:4],donor[0][-4:])
    assert changed[0].shape==original[0].shape
    if fixed:
        np.testing.assert_array_equal(changed[1],original[1])
        np.testing.assert_array_equal(changed[2],original[2])
    else:np.testing.assert_array_equal(changed[1][:4],donor[1][-4:])


def test_splice_rejects_current_or_future_donor_events_and_short_donor():
    original=stream([0,0,1])
    with pytest.raises(AssertionError):splice(original,stream([0,1]),1)
    with pytest.raises(AssertionError):splice(original,stream([0]),1)


def test_splice_rejects_private_exchange_bits():
    original=stream([0,1]);donor=stream([0]);donor[0][0,150]=1
    with pytest.raises(AssertionError):splice(original,donor,1)
