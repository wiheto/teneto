import numpy as np
from teneto.temporalcommunity import promiscuity
import pytest

def test_recruitment():
    temporalcommunities = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 2]])
    p = promiscuity(temporalcommunities)
    if not p[0] == 0. and p[1] == 1/2 and p[2] == 1:
        raise AssertionError()
