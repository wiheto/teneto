import numpy as np 
import teneto 
np.random.seed(20)

def test_slidingwindow(): 
    X = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],20)
    R_sw = teneto.misc.corrcoef_matrix(X[:10,:].transpose())[0][0,1]
    TR_sw = teneto.derive.derive(X.transpose(),{'method': 'slidingwindow', 'windowsize': 10,'dimord':'node,time'})
    assert np.round(R_sw,12) == np.round(TR_sw[0,1,0],12)

def test_jc(): 
    X = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],20)
    R_jc = teneto.misc.corrcoef_matrix(X[1:,:].transpose())[0][0,1]
    TR_jc = teneto.derive.derive(X.transpose(),{'method': 'jackknife','dimord':'node,time'})
    assert np.round(R_jc,12) == np.round(TR_jc[0,1,0]*-1,12)

def test_mtd(): 
    X = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]],20)
    mtd = np.diff(X.transpose())
    R_mtd = (mtd[0]*mtd[1])/(np.std(mtd[0])*np.std(mtd[1]))
    TR_mtd = teneto.derive.derive(X.transpose(),{'method': 'mtd','windowsize':1,'dimord':'node,time'})
    assert (np.round(TR_mtd[0,1,:],12)==np.round(R_mtd,12)).all()



