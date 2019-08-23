import numpy as np
from teneto.communitymeasures import promiscuity

def test_promiscuity():
    temporalcommunities = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 2]])
    p = promiscuity(temporalcommunities)
    if not p[0] == 0. and p[1] == 1/2 and p[2] == 1:
        raise AssertionError()
