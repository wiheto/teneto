import numpy as np
import teneto
import pytest


def test_partcoef():

    communities = [1, 1, 1, 0]
    G = np.zeros([4, 4, 3])
    G[0, 1, [0, 2]] = 1
    G[2, 3, [0, 2]] = 1
    G[1, 2, [1, 2]] = 1
    G[0, 3, [1, 2]] = 1
    G += G.transpose([1, 0, 2])
    G[1:, 1:, 0] = 0.5
    part = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities))
    # Calculate value of node 1 at time 0.
    if not part[1, 0] == 1-(np.power(1.5/2, 2)+np.power(.5/2, 2)):
        raise AssertionError()
    # Hardcode known partcoeff
    if not np.all(part[:, 2] == [0.5, 0, 0.5, 0]):
        raise AssertionError()
    with pytest.raises(ValueError):
        teneto.networkmeasures.temporal_participation_coeff(G)
    tnet = teneto.TemporalNetwork(from_array=G)
    with pytest.raises(ValueError):
        teneto.networkmeasures.temporal_participation_coeff(G)


def test_partcoef_decay():

    communities = [1, 1, 1, 0]
    G = np.zeros([4, 4, 3])
    G[0, 1, [0, 2]] = 1
    G[2, 3, [0, 2]] = 1
    G[1, 2, [1, 2]] = 1
    G[0, 3, [1, 2]] = 1
    G += G.transpose([1, 0, 2])
    G[1:, 1:, 0] = 0.5
    part = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities), decay=1)
    part2 = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities))
    part3 = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities), decay=0.25)
    if not (np.sum(part2, axis=1) == part[:, -1]).all():
        raise AssertionError()
    if not ((0.25 * part2[:, 0] + part2[:, 1]) == part3[:, 1]).all():
        raise AssertionError()


def test_partcoef_tvc():

    communities = np.array([[0, 0], [0, 1], [1, 1], [1, 1]])
    G = np.zeros([4, 4, 2])
    G[:, :, 0] = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    G[:, :, 1] = np.array(
        [[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    part = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities))
    # Bit of a weird network as node 0 has only one connection with a community outside of its own (making it 0).
    # Hardcode answer. Node 1 is outside has 1 connection outside of its partition for both communities. Node 2 has 1 out of community edge for one partion.
    if not (part[:, 0] == [0, 0.5, 0.25, 0]).all():
        raise AssertionError()
    part2 = teneto.networkmeasures.temporal_participation_coeff(
        G, np.array(communities), decay=0.25)
    if not (part[:, 0] * 0.25 + part[:, 1] == part2[:, 1]).all():
        raise AssertionError()
