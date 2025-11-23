from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import SeedSequence
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from teneto.classes import TemporalNetwork
from teneto.communitydetection import temporal_louvain, temporal_spectral


def _adj_list_to_temporalnetwork_raw(adj_list: Sequence, nettype: str = "bu", forcesparse: bool = False) -> TemporalNetwork:
    dense = []
    for A in adj_list:
        if issparse(A):
            A = A.toarray()
        else:
            A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Each snapshot must be square 2D.")
        dense.append(A.astype(float, copy=False))
    arr = np.stack(dense, axis=2)
    tnet = TemporalNetwork(N=arr.shape[0], T=arr.shape[2], nettype=nettype, timetype="discrete")
    tnet.network_from_array(arr, forcesparse=forcesparse)
    return tnet


def sbm_dynamic_model_2(
    N: int = 150,
    k: int = 3,
    pin: Sequence[float] | float = (0.3, 0.3, 0.2),
    pout: float = 0.15,
    p_switch: float = 0.15,
    T: int = 10,
    Totalsims: int = 2,
    base_seed: Optional[int] = None,
    community_probs: Optional[np.ndarray] = None,
    verbose: bool = False,
    return_teneto: bool = False,
    teneto_nettype: str = "bu",
) -> Tuple[List[List[csr_matrix]], np.ndarray, Optional[List[TemporalNetwork]]]:
    ss = SeedSequence(base_seed)
    child_seeds = ss.spawn(Totalsims)

    adjacency_all: List[List[csr_matrix]] = [[] for _ in range(Totalsims)]
    true_labels_all = np.zeros((Totalsims, T, N), dtype=int)
    teneto_list: List[TemporalNetwork] = []

    for sims in range(Totalsims):
        rng = np.random.default_rng(child_seeds[sims])
        G = rng.binomial(1, pout, (N, N))
        G = np.diag(np.diag(G)) + np.tril(G, -1) + np.tril(G, -1).T
        if community_probs is None:
            labels = rng.integers(1, k + 1, size=N)
        else:
            labels = rng.choice(np.arange(1, k + 1), size=N, p=community_probs)
        true_labels_all[sims, 0, :] = labels
        clusters = {i: np.where(labels == i)[0] for i in range(1, k + 1)}

        for i in range(1, k + 1):
            Gk = rng.binomial(1, pin[i - 1], (len(clusters[i]), len(clusters[i])))
            Gk = np.diag(np.diag(Gk)) + np.tril(Gk, -1) + np.tril(Gk, -1).T
            G[np.ix_(clusters[i], clusters[i])] = Gk

        for t in range(T):
            if verbose:
                print("making graph for time step ", t)
            if t > 0:
                G = rng.binomial(1, pout, (N, N))
                G = np.diag(np.diag(G)) + np.tril(G, -1) + np.tril(G, -1).T
                for i in range(1, k + 1):
                    clusters[i] = np.where(labels == i)[0]
                    if clusters[i].size > 0:
                        permute_indices = rng.permutation(len(clusters[i]))
                        changing_members = permute_indices[: int(np.ceil(len(clusters[i]) * p_switch))]
                        if changing_members.size > 0:
                            z = rng.binomial(1, p_switch, len(changing_members))
                            labels_temp = rng.integers(1, k + 1, size=np.sum(z))
                            labels[changing_members[z == 1]] = labels_temp

                for i in range(1, k + 1):
                    clusters[i] = np.where(labels == i)[0]
                    Gk = rng.binomial(1, pin[i - 1], (len(clusters[i]), len(clusters[i])))
                    Gk = np.diag(np.diag(Gk)) + np.tril(Gk, -1) + np.tril(Gk, -1).T
                    G[np.ix_(clusters[i], clusters[i])] = Gk

            adjacency_all[sims].append(csr_matrix(G))
            true_labels_all[sims, t, :] = labels

        if return_teneto:
            teneto_list.append(_adj_list_to_temporalnetwork_raw(adjacency_all[sims], nettype=teneto_nettype, forcesparse=True))

    if return_teneto:
        return adjacency_all, true_labels_all, teneto_list
    return adjacency_all, true_labels_all


def test_temporal_spectral_recovers_sbm_partitions():
    T = 50
    k=2
    _, true_labels, tnet_list = sbm_dynamic_model_2(
        N=200,
        k=k,
        pin=(0.3, 0.3),
        pout=0.1,
        p_switch=0.05,
        T=T,
        Totalsims=1,
        base_seed=7,
        return_teneto=True,
    )

    labels = temporal_spectral(
        tnet_list[0], ke=k, kc=k, mode="simple-nsc", stable_communities=True, smoothing_filter="median", smoothing_parameter=T // 10 + 1, which_eig="smallest",
    )

    ari_scores = []
    nmi_scores = []
    ami_scores = []
    for t in range(labels.shape[1]):
        gt = true_labels[0, t, :]
        est = labels[:, t]
        ari_scores.append(adjusted_rand_score(gt, est))
        nmi_scores.append(normalized_mutual_info_score(gt, est))
        ami_scores.append(adjusted_mutual_info_score(gt, est))

    ari_scores = np.array(ari_scores)
    nmi_scores = np.array(nmi_scores)
    ami_scores = np.array(ami_scores)

    assert np.all(ari_scores >= 0.94)
    assert np.all(nmi_scores >= 0.88)
    assert np.all(ami_scores >= 0.88)


def test_temporal_spectral_aligns_with_temporal_louvain():
    _, _, tnet_list = sbm_dynamic_model_2(
        N=20,
        k=2,
        pin=(0.9, 0.9),
        pout=0.05,
        p_switch=0.1,
        T=4,
        Totalsims=1,
        base_seed=11,
        return_teneto=True,
    )

    labels_spectral = temporal_spectral(
        tnet_list[0], ke=2, kc=2, mode="simple-nsc", stable_communities=True
    )
    labels_louvain = temporal_louvain(tnet_list[0], intersliceweight=0.1, n_iter=5)

    ari_scores = []
    for t in range(labels_spectral.shape[1]):
        ari_scores.append(adjusted_rand_score(labels_spectral[:, t], labels_louvain[:, t]))

    ari_scores = np.array(ari_scores)
    assert np.all(ari_scores >= 0.75)
