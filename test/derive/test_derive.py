import numpy as np
import teneto
np.random.seed(20)


def test_slidingwindow():
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20)
    R_sw = teneto.misc.corrcoef_matrix(X[:10, :].transpose())[0][0, 1]
    TR_sw = teneto.timeseries.derive_temporalnetwork(X.transpose(
    ), {'method': 'slidingwindow', 'windowsize': 10, 'dimord': 'node,time'})
    if not np.round(R_sw, 12) == np.round(TR_sw[0, 1, 0], 12):
        raise AssertionError()


def test_slidingwindow_postpro():
    np.random.seed(2018)
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20)
    R_sw = np.arctanh(teneto.misc.corrcoef_matrix(
        X[:10, :].transpose())[0][0, 1])
    TR_sw_z = teneto.timeseries.derive_temporalnetwork(X.transpose(), {
        'method': 'slidingwindow', 'windowsize': 10, 'dimord': 'node,time', 'postpro': 'fisher'})
    if not np.round(R_sw, 12) == np.round(TR_sw_z[0, 1, 0], 12):
        raise AssertionError()
    TR_sw_box = teneto.timeseries.derive_temporalnetwork(X.transpose(), {
        'method': 'slidingwindow', 'windowsize': 10, 'dimord': 'node,time', 'postpro': 'fisher+boxcox+standardize', 'report': True})
    if not TR_sw_box[0, 1, :].std() == 1:
        raise AssertionError()


def test_taperedslidingwindow():
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20)
    TR_tsw = teneto.timeseries.derive_temporalnetwork(X, {'method': 'taperedslidingwindow', 'windowsize': 10,
                                                          'dimord': 'time,node', 'distribution': 'norm', 'distribution_params': [0, 5], 'report': True})
    # Perhaps make a better test
    if not TR_tsw.shape == (2, 2, 11):
        raise AssertionError()


def test_jc():
    np.random.seed(2018)
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20)
    R_jc = teneto.misc.corrcoef_matrix(X[1:, :].transpose())[0][0, 1]
    TR_jc = teneto.timeseries.derive_temporalnetwork(
        X.transpose(), {'method': 'jackknife', 'dimord': 'node,time'})
    if not np.round(R_jc, 12) == np.round(TR_jc[0, 1, 0]*-1, 12):
        raise AssertionError()


def test_mtd():
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20)
    mtd = np.diff(X.transpose())
    R_mtd = (mtd[0]*mtd[1])/(np.std(mtd[0])*np.std(mtd[1]))
    TR_mtd = teneto.timeseries.derive_temporalnetwork(
        X.transpose(), {'method': 'mtd', 'windowsize': 1, 'dimord': 'node,time'})
    if not (np.round(TR_mtd[0, 1, :], 12) == np.round(R_mtd, 12)).all():
        raise AssertionError()


def test_sd():
    X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 20)
    TR_sd = teneto.timeseries.derive_temporalnetwork(X.transpose(), {
        'method': 'spatialdistance', 'distance': 'euclidean', 'dimord': 'node,time'})
    # Perhaps make a better test
    if not TR_sd.shape == (2, 2, 20):
        raise AssertionError()
