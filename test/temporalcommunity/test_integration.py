import numpy as np
from teneto.temporalcommunity import integration
import pytest

def test_integration():
    temporalcommunities = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 2], [1, 2, 2], [1, 1, 2]])
    staticcommunities = np.array([[0, 0, 1, 1, 2, 2]])
    icoeff = integration(temporalcommunities, staticcommunities)
    # All have the same number of edges outside of static community
    denominator = np.prod(temporalcommunities.shape)-(2*temporalcommunities.shape[1])
    # icoeff[0] = 0 has no nodes outside of community. 
    # icoeff[1] has 1 node outside of community
    # icoeff[3] has 5 nodes outside of community
    if not icoeff[0] == 0/denominator and np.round(icoeff[1],5) == np.round(1/denominator,5) and np.round(icoeff[3],5) == np.round(5/denominator,5):
        raise AssertionError()
    # Check tests that input is right size
    with pytest.raises(ValueError):
        integration(temporalcommunities, temporalcommunities)
    staticcommunities = np.array([[0, 0, 1, 1, 2]])
    with pytest.raises(ValueError):
        integration(temporalcommunities, staticcommunities)
