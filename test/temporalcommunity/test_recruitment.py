import numpy as np
from teneto.temporalcommunity import recruitment
import pytest

def test_recruitment():
    temporalcommunities = np.array([[0, 0, 0], [0, 0, 0], [2, 1, 1], [0, 1, 2], [2, 2, 2], [0, 2, 2]])
    staticcommunities = np.array([[0, 0, 1, 1, 2, 2]])
    r = recruitment(temporalcommunities, staticcommunities)
    if not r[0] == 1. and r[2] == 1/3 and r[4] == 2/3:
        raise AssertionError()
    # Check tests that input is right size
    with pytest.raises(ValueError):
        recruitment(temporalcommunities, temporalcommunities)
