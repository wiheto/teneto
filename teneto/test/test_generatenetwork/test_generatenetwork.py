
import teneto 
import pytest 

def test_gennet_fail():
    with pytest.raises(ValueError): 
        teneto.generatenetwork.rand_binomial([3,2],[0,1,2])
    with pytest.raises(ValueError): 
        teneto.generatenetwork.rand_binomial([3,2,1],[0])

def test_gen_randbinomial(): 
    G = teneto.generatenetwork.rand_binomial([3,2],[1])   
    assert G.shape == (3,3,2) 
    G = teneto.utils.set_diagonal(G,1) 
    assert G[:,:,-1].min() == 1 
    G = teneto.generatenetwork.rand_binomial([3,2],[0,1])     
    assert G.shape == (3,3,2)
    assert G[:,:,-1].max() == 0 