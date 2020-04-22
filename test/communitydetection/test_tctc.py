import numpy as np
import pandas as pd
from teneto.communitydetection import tctc
import itertools

df = pd.DataFrame({'i': [0, 1, 0, 0, 0, 1, 1, 3, 0], 'j': [
                  2, 2, 1, 1, 2, 2, 3, 2, 1], 't': [0, 0, 0, 1, 2, 2, 2, 2, 4]})

data = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 1, 2, 1], [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 1], [
                1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0], [-1, 0, 1, 1, 0, -1, 0, -1, 0, 2, 1, 0, -1]], dtype=float)
data = data.transpose()
np.random.seed(2019)
data += np.random.uniform(-0.2, 0.2, data.shape)


def df_to_array(df, outshape):
    out = np.zeros(outshape)
    for _, r in df.iterrows():
        ind = list(
            zip(*list(itertools.combinations_with_replacement(r['community'], 2))))
        out[ind[0]+ind[1], ind[1]+ind[0], r['start']:r['end']] = 1
    return out


def test_output_simularity():
    array = tctc(data, 3, 0.5, 2, 1)
    df = tctc(data, 3, 0.5, 2, 1, output='df')
    array2 = df_to_array(df, (4, 4, 13))
    if np.sum(array == array2) != np.prod(array2.shape):
        raise AssertionError()

    array = tctc(data, 3, 0.5, 2, 0)
    df = tctc(data, 3, 0.5, 2, 0, output='df')
    array2 = df_to_array(df, (4, 4, 13))
    if np.sum(array == array2) != np.prod(array2.shape):
        raise AssertionError()


def test_epsilon():
    # Test goes through and makes sure that the epsilon parameter is working correctly by using input data
    epsilon = np.sort(np.abs(data[:, 0]-data[:, 1]))
    result = []
    for e in epsilon:
        array = tctc(data[:, :2], 1, e, 2, 0, output='array')
        result.append(np.sum(array[0, 1, :]))
    if not all(result == np.arange(1, 14)):
        raise AssertionError()


def test_sigma():
    df1 = tctc(data, 3, 0.5, 2, 1, output='df')
    df2 = tctc(data, 3, 0.5, 3, 1, output='df')
    if df1['size'].min() != 2:
        raise AssertionError()
    if df2['size'].min() != 3:
        raise AssertionError()


def test_tau():
    df1 = tctc(data, 3, 0.5, 2, 1, output='df')
    df2 = tctc(data, 5, 0.5, 2, 1, output='df')
    if df1['length'].min() < 3:
        raise AssertionError()
    if df2['length'].min() < 5:
        raise AssertionError()
    df3 = tctc(data, 13, 0.5, 2, 1, output='df')
    if df3.shape[0] != 1:
        raise AssertionError()


def test_kappa():
    df1 = tctc(data, 2, 0.5, 2, 1, output='df')
    df2 = tctc(data, 2, 0.5, 2, 0, output='df')
    r = 0
    for _, d in df1.iterrows():
        if [0, 1] == d['community']:
            r += 1
    r2 = 0
    for _, d in df2.iterrows():
        if [0, 1] == d['community']:
            r2 += 1
    if r != 1 or r2 != 2:
        raise AssertionError()
    rt = []
    for _, d in df1.iterrows():
        if set(d['community']).issuperset([0, 1]):
            rt += list(np.arange(d['start'], d['end']))
    rt = np.unique(rt)
    rt2 = []
    for _, d in df2.iterrows():
        if set(d['community']).issuperset([0, 1]):
            rt2 += list(np.arange(d['start'], d['end']))
    rt2 = np.unique(rt2)
    if len(rt) != 13:
        raise AssertionError()
    # The two time points which shouldnt be included if kappa = 0
    if 7 in rt2 or 10 in rt2:
        raise AssertionError()
