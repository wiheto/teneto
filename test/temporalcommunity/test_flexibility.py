import numpy as np 
from teneto.temporalcommunity import flexibility

# Tests could import results from community detection toolbox
def test_flexibility():
    # First community doesnt change, second community changes once (out of two), third also changes once, third one changes twice. 
    communities = np.array([[0,0,0],[0,0,1],[0,1,1],[0,1,2]])
    flex = flexibility(communities)
    if not (flex == [1., 0.5, 0.5, 0.]).all():
        raise AssertionError()