
"""Temporal community detection by fitting a Grassmannian geodesic to spectral embeddings."""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import cosm, sinm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh, svds
from sklearn.cluster import KMeans

from teneto.classes import TemporalNetwork
from teneto.utils import process_input


__all__ = ["temporal_spectral"]

logger = logging.getLogger(__name__)


def _as_sparse_snapshots(tnet: TemporalNetwork) -> Tuple[list[csr_matrix], int, int]:
    """Return sparse adjacency snapshots and their dimensions."""
    arr = tnet.df_to_array(start_at="auto")
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array (nodes, nodes, time); got {arr.shape}")

    n_nodes, _, n_time = arr.shape
    snapshots: list[csr_matrix] = []
    for t in range(n_time):
        # Copy to avoid mutating the TemporalNetwork backing array.
        mat = np.array(arr[:, :, t], copy=True)
        np.fill_diagonal(mat, 0)
        mat = np.triu(mat, k=1)
        mat = mat + mat.T
        snapshots.append(csr_matrix(mat))
    return snapshots, n_nodes, n_time


def _labels_list_to_matrix(labels_list: Iterable[Iterable[int]]) -> np.ndarray:
    """Convert a list of per-time label vectors into a (node, time) array."""
    label_series = list(labels_list)
    if not label_series:
        raise ValueError("No community assignments were returned.")

    n_nodes = len(label_series[0])
    labels = np.empty((n_nodes, len(label_series)), dtype=int)
    for t_idx, lab in enumerate(label_series):
        lab = np.asarray(lab, dtype=int)
        if lab.shape[0] != n_nodes:
            raise ValueError("Community label lengths must match node count at each timepoint.")
        labels[:, t_idx] = lab
    return labels


def temporal_spectral(
    tnet,
    ke: int,
    kc: int | list[int] | str = "auto",
    mode: str = "simple-nsc",
    stable_communities: bool = False,
    smoothing_filter: str | None = None,
    smoothing_parameter: float | None = None,
    return_embeddings: bool = False,
    use_intermediate_iterations: bool = False,
    num_intermediate_iterations: int = 20,
    which_eig: str = "smallest",
    **mode_kwargs,
) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
    """
    Detect temporal communities by geodesically modeling spectral embeddings.

    Parameters
    ----------
    tnet : array | dict | TemporalNetwork
        Temporal network input accepted by :func:`teneto.utils.process_input`.
    ke : int
        The dimension of the learned spectral embeddings. Should be an upper bound on the expected number of communities at any time. (In classical static spectral clustering, ``ke`` is chosen to be equal to the target number of communities ``kc`` .)
    kc : int | list[int] | 'auto'
        Number of communities to detect per snapshot. Either a single int to detect a fixed given number of communities at every snapshot, a list of ints coindexed with the snapshots, or 'auto' to select per snapshot using a benefit curve.
    mode : str
        Algorithm mode, e.g. ``'simple-nsc'`` (normalized spectral clustering, default), ``'simple-usc'`` (unnormalized), ``'simple-smm'`` (spectral modularity maximization).
    stable_communities : bool
        If ``True`` and ``kc='auto'``, a single community count is enforced across time by maximizing the benefit curve aggregated over all snapshots.
    smoothing_filter : str | None
        ``'median'`` or ``'gaussian'`` to optionally smooth the benefit-vs-``k`` curve when ``kc='auto'`` toward (slightly) more stable community counts.
    smoothing_parameter : float | None
        Parameter for the selected smoothing filter (kernel size or sigma).
    return_embeddings : bool
        If ``True``, also return the per-time embeddings used for clustering.
    use_intermediate_iterations : bool
        If ``True``, keep intermediate geodesic-fitting iterations and concatenate their embeddings.
    num_intermediate_iterations : int
        Number of intermediate iterations retained if ``use_intermediate_iterations`` is set. The idea is that since MM algorithms monotonically improve their objective, intermediate fits may also be useful as 'softer regularization' than the final geodesic fit.
    which_eig : str
        Controls which eigensolver is used inside the base smoother (``'smallest'``, ``'largest'``, ``'svd'``).
    mode_kwargs :
        Additional keyword arguments forwarded to the smoother subclass.

    Returns
    -------
    communities : ndarray (nodes, time)
        node,time array of community assignment
    embeddings : list[np.ndarray], optional
        Returned when ``return_embeddings`` is ``True``.
        
        
    References
    ----------
    Hume, J. and Balzano, L. (2025).
    A Spectral Framework for Tracking Communities in Evolving Networks.
    In *Proceedings of the Third Learning on Graphs Conference (LoG 2024)*,
    Proceedings of Machine Learning Research, vol. 269, pp. 9:1–9:34.
    Available at https://proceedings.mlr.press/v269/hume25a.html.

    
    """
    tnet = process_input(tnet, ["C", "G", "TN"], "TN")
    snapshots, n_nodes, n_time = _as_sparse_snapshots(tnet)

    assignments, embeddings = _spectral_geodesic_smoothing(
        sadj_list=snapshots,
        T=n_time,
        num_nodes=n_nodes,
        ke=ke,
        kc=kc,
        stable_communities=stable_communities,
        mode=mode,
        smoothing_filter=smoothing_filter,
        smoothing_parameter=smoothing_parameter,
        return_geo_embeddings_only=False,
        use_intermediate_iterations=use_intermediate_iterations,
        num_intermediate_iterations=num_intermediate_iterations,
        which_eig=which_eig,
        **mode_kwargs,
    )

    communities = _labels_list_to_matrix(assignments)
    if return_embeddings:
        return communities, embeddings
    return communities


def _is_symmetric(matrix, tol=1e-8):
    if issparse(matrix):
        return (matrix != matrix.T).nnz == 0
    return np.allclose(matrix, matrix.T, atol=tol)

def _theta_majorizer(H, Y, X):
    T = len(X)
    k = H.shape[1]
    
    r = np.empty((T, k), dtype=np.float64)
    phi = np.empty((T, k), dtype=np.float64)
    bias = np.empty((T, k), dtype=np.float64)
    
    symmetric_flags = [_is_symmetric(Xi) for Xi in X]
    
    if not H.flags['C_CONTIGUOUS']:
        H = np.ascontiguousarray(H)
    if not Y.flags['C_CONTIGUOUS']:
        Y = np.ascontiguousarray(Y)
    
    for ii in range(T):
        Xi = X[ii]
        symmetric = symmetric_flags[ii]
        
        if symmetric:
            XiT_H = Xi @ H  
            XiT_Y = Xi @ Y 
        else:
            if issparse(Xi):
                XiT = Xi.transpose().tocsr() 
            else:
                XiT = Xi.T  
            XiT_H = XiT @ H  
            XiT_Y = XiT @ Y 
        

        a = np.sum(np.abs(XiT_H)**2, axis=0)  
        

        b = np.real(np.sum(XiT_Y.conj() * XiT_H, axis=0))
        
        c = np.sum(np.abs(XiT_Y)**2, axis=0) 
        
        a_minus_c_over_2 = (a - c) / 2.0
        two_b = 2.0 * b
        
        np.sqrt(a_minus_c_over_2**2 + b**2, out=r[ii, :])
        
        np.arctan2(two_b, a - c, out=phi[ii, :])
        
        bias[ii, :] = (a + c) / 2.0
    
    return r, phi, bias

def _estimate_theta(H, Y, X, t, niter=5, tH=0, Theta_init=None):
    r, phi, bias = _theta_majorizer(H, Y, X)
    
    t = np.asarray(t, dtype=np.float64)[:, np.newaxis] - tH 
    tt = 2.0 * t  
    
    n2tr = -tt * r  
    L = tt * n2tr  
    
    if Theta_init is None:
        Theta = np.ones(H.shape[1], dtype=np.float64)  # Shape: (k,)
    else:
        Theta = Theta_init.astype(np.float64).copy()  # Shape: (k,)
    
    for _ in range(niter):
        arg = tt * Theta  
        arg -= phi 
        
        gradf = n2tr * np.sin(arg)  
        
        denom = (arg + np.pi) % (2 * np.pi) - np.pi  
        
        with np.errstate(divide='ignore', invalid='ignore'):
            curvf = np.divide(tt * gradf, denom, out=np.zeros_like(tt * gradf), where=denom!=0)
            curvf += L * (denom == 0)
        
        gradf_sum = np.sum(gradf, axis=0)  
        curvf_sum = np.sum(curvf, axis=0) 
        step = np.divide(gradf_sum, curvf_sum, out=np.zeros_like(gradf_sum), where=curvf_sum!=0)  # Shape: (k,)
        
        Theta -= step  
    
    return Theta




def _estimate_point_tangent(H, Y, Theta, X, t, tH=0):
    M_AB = 0.0

    for ti, Xi in zip(t, X):
        arg = Theta * (ti - tH)  # (k,)
        cos_arg = np.cos(arg)  # (k,)
        sin_arg = np.sin(arg)  # (k,)
        Ui = H * cos_arg + Y * sin_arg  

        if _is_symmetric(Xi):
            G = Xi @ Ui  # (n_i, k)
            XG = G  # for symmetric matrices Xi @ G.T = G
        else:
            G = Ui.T @ Xi  # (k, n_i)
            XG = Xi @ G.T  # (d, k)

        XGcos = XG * cos_arg  
        XGsin = XG * sin_arg 

        M_AB_i = np.concatenate((XGcos, XGsin), axis=1)  #(d, 2k)
        M_AB += M_AB_i

    return M_AB

def _sfit_point_tangent_geodesic(data, k, max_iter, tol=1e-5, rel_tol=1e-2, return_intermediate_iterations=False, num_intermediate_iterations=20):
    X, t = data  # X is list of matrices, t is list of times
    init_start = time.time()
    # Extract M1 and MT
    M1 = X[0]
    MT = X[-1]

    # are M1 and MT symmetric?
    M1_symmetric = _is_symmetric(M1)
    MT_symmetric = _is_symmetric(MT)
    
    logger.debug("M1 symmetric: %s", M1_symmetric)
    logger.debug("MT symmetric: %s", MT_symmetric)

    # rank-k truncated SVD of M1
    if M1_symmetric:
        if issparse(M1):
            S1, U1 = eigsh(M1, k=k, which='LM')
        else:
            S1, U1 = np.linalg.eigh(M1)
            idx = np.argsort(-S1)[:k]
            U1 = U1[:, idx]
            S1 = S1[idx]
    else:
        U1, S1, _ = svds(M1, k=k)
        idx = np.argsort(-S1)
        U1 = U1[:, idx]
        S1 = S1[idx]
    H1 = U1  # (d, k)

    # ..... and for MT also
    if MT_symmetric:
        if issparse(MT):
            ST, UT = eigsh(MT, k=k, which='LM')
        else:
            ST, UT = np.linalg.eigh(MT)
            idx = np.argsort(-ST)[:k]
            UT = UT[:, idx]
            ST = ST[idx]
    else:
        UT, ST, _ = svds(MT, k=k)
        idx = np.argsort(-ST)
        UT = UT[:, idx]
        ST = ST[idx]
    H_T = UT  # (d, k)

    H1T_H_T = H1.T @ H_T  # (k, k)

    Z, S, Q_T = np.linalg.svd(H1T_H_T, full_matrices=False)
    Q = Q_T.T  # (k, k)
    
    
    if np.any((S < -1 - 1e-6) | (S > 1 + 1e-6)): # clip S 
        logger.warning("Singular values exceeded [-1,1] by more than 1e-6; clipping.")
    S = np.clip(S, -1, 1)

    H_TQ = H_T @ Q  # (d, k)

    # orthogonal component
    H1_H1T_HTQ = H1 @ (H1.T @ H_TQ)  # (d, k)
    Orth_comp = H_TQ - H1_H1T_HTQ  # (d, k)

    # SVD of orthocomplement
    F, D, G_T = np.linalg.svd(Orth_comp, full_matrices=False)

    # direction Y = F G_T
    Y = F @ G_T  # (d, k)



    # Theta init (see paper)
    Theta = np.arccos(S)  # (k,)

    
    
    #print("init time:", time.time() - init_start)
    
    #print("Starting iterations")
    iter_start = time.time()
    point_tangent_times = []
    theta_times = []
    conv_check_times = []

    # init H = H1
    H = H1.copy()

    # set the initial values for convergence checking
    H_old = H.copy()
    Y_old = Y.copy()
    Theta_old = Theta.copy()
    
    # Storage for intermediate iterations if requested
    if return_intermediate_iterations:
        intermediate_H = []
        intermediate_Y = []
        intermediate_Theta = []

    # fix tH=0, at least until there seems to be a reason to not do so
    tH = 0 

    for itr in range(max_iter):        
        if itr == max_iter - 1:
            logger.warning("Maximum point-tangent iterations reached without convergence.")
        
        #print(f"Iteration {itr}")
        
        # P-Update, P = [H, Y]
        pt_time = time.time()
        M_AB = _estimate_point_tangent(H, Y, Theta, X, t, tH)
        try: 
            U, _, Vh = np.linalg.svd(M_AB, full_matrices=False)
        except np.linalg.LinAlgError as err:
            logger.error("SVD failed for M_AB (shape=%s): %s", M_AB.shape, err)
            logger.debug("Matrix contains NaN? %s | Inf? %s", np.any(np.isnan(M_AB)), np.any(np.isinf(M_AB)))
            logger.info("Retrying SVD for M_AB after adding jitter.")
            M_AB += 1e-6 * np.random.randn(*M_AB.shape)
            U, _, Vh = np.linalg.svd(M_AB, full_matrices=False)
        C = U @ Vh  # (d, 2k)
        H = C[:, :k]
        Y = C[:, k:]
        point_tangent_times.append(time.time() - pt_time)

        # Theta-Update
        theta_time = time.time()
        Theta = _estimate_theta(H, Y, X, t, niter=5, tH=tH, Theta_init=Theta)
        theta_times.append(time.time() - theta_time)

        # Store intermediate iterations if requested (only last num_intermediate_iterations)
        if return_intermediate_iterations:
            intermediate_H.append(H.copy())
            intermediate_Y.append(Y.copy())
            intermediate_Theta.append(Theta.copy())
            
            # Keep only the last num_intermediate_iterations
            if len(intermediate_H) > num_intermediate_iterations:
                intermediate_H.pop(0)
                intermediate_Y.pop(0)
                intermediate_Theta.pop(0)

        # Check convergence?
        conv_check_time = time.time()
        delta_H = np.linalg.norm(H - H_old)
        delta_Y = np.linalg.norm(Y - Y_old)
        delta_Theta = np.linalg.norm(Theta - Theta_old)


        rel_delta_H = delta_H / (np.linalg.norm(H) + 1e-15)
        rel_delta_Y = delta_Y / (np.linalg.norm(Y) + 1e-15)
        rel_delta_Theta = delta_Theta / (np.linalg.norm(Theta) + 1e-15)

        if (delta_H < tol and delta_Y < tol and delta_Theta < tol) or (rel_delta_H < rel_tol and rel_delta_Y < rel_tol and rel_delta_Theta < rel_tol):
            logger.info("Point-tangent geodesic converged in %s iterations.", itr)
            break
            
        # update the old values
        H_old = H.copy()
        Y_old = Y.copy()
        Theta_old = Theta.copy()
        
        conv_check_times.append(time.time() - conv_check_time)
    
    #print("point tangent time", np.sum(np.array(point_tangent_times)))
    #print("theta time", np.sum(np.array(theta_times)))
    #print("conv check time", np.sum(np.array(conv_check_times)))
    
    if return_intermediate_iterations:
        return H, Y, Theta, intermediate_H, intermediate_Y, intermediate_Theta
    return H, Y, Theta



class _SpectralGeodesicSmoother(ABC):
    def __init__(self, *args, d, T, sadj_list=None, precomputed_embeddings=None, ke='auto', kc_list='auto', stable_communities=False, which_eig='smallest', t=None, benefit_fn=None, smoothing_filter=None, smoothing_parameter=None, max_iter=1000, use_intermediate_iterations=False, num_intermediate_iterations=20):
        if len(args) > 0:
            raise ValueError("This class does not accept positional arguments")
            
        if precomputed_embeddings is not None and sadj_list is not None:
            logger.warning("Both precomputed_embeddings and sadj_list are provided. The sadj_list will be ignored.")

        self.d = d
        self.max_iter = max_iter
        self.sadj_list = sadj_list
        self.precomputed_embeddings = precomputed_embeddings

        self.ke = ke if ke != 'auto' else self.choose_ke()
        if isinstance(kc_list, int):
            kc_list = [kc_list]*T
        self.kc_list = kc_list
        self.stable_communities = stable_communities
        self.which_eig = which_eig
        if t is not None:
            self.t = t
            
        elif sadj_list is not None:
            self.t = np.linspace(0.0, 1.0, len(sadj_list))
            
        elif precomputed_embeddings is not None:
            self.t = np.linspace(0.0, 1.0, len(precomputed_embeddings))
        
        else: 
            raise ValueError("Either sadj_list or precomputed_embeddings must be provided")
        
        
        self.t = t if t is not None else np.linspace(0.0, 1.0, len(sadj_list)) if sadj_list is not None else np.linspace(0.0, 1.0, len(precomputed_embeddings)) 
        self.benefit_fn = benefit_fn
        self.smoothing_filter = smoothing_filter
        self.smoothing_parameter = smoothing_parameter
        self.T = T
        self.use_intermediate_iterations = use_intermediate_iterations
        self.num_intermediate_iterations = num_intermediate_iterations
        
        if self.stable_communities and kc_list != 'auto' and len(set(kc_list)) > 1:
            raise ValueError("stable_communities=True requires a single kc value or 'auto'.")

    def benefit_fn_broadcast(self, sadj_list, labels_list):
        return [self.benefit_fn(sadj, labels) for sadj, labels in zip(sadj_list, labels_list)]
    
    
    @abstractmethod
    def make_clustering_matrix(self, sadj):
        pass
    
    def choose_ke(self):
        raise NotImplementedError("must implement choose_ke")
    
    def make_clustering_matrices(self):
        self.clustering_matrices = [self.make_clustering_matrix(sadj) for sadj in self.sadj_list]
    
    def make_modeled_clustering_matrices(self):
        # this is permitted to be overwritten by subclasses, but usually no need; the following default implementation usually works fine
        assert self.clustering_matrices is not None, "You must first run make_clustering_matrices"
        self.modeled_clustering_matrices = []
        for R in self.clustering_matrices:
            frobenius_norm = sparse.linalg.norm(R, 'fro')
            n,m= R.shape
            I = sparse.eye(n, m, format=R.format)
            if self.which_eig == 'smallest':
                self.modeled_clustering_matrices.append(frobenius_norm*I - R)
            elif self.which_eig == 'largest':
                self.modeled_clustering_matrices.append(frobenius_norm*I + R)
            elif self.which_eig == 'svd':
                self.modeled_clustering_matrices.append(R)
            else:
                raise ValueError("Invalid value for which_eig")
            

    # this never overwritten by subclasses
    def get_geodesic_embeddings(self):
        # This method now assumes self.Xs is already prepared.
        
        # Get geodesic parameters, optionally with intermediate iterations
        if self.use_intermediate_iterations:
            H, Y, Theta, intermediate_H, intermediate_Y, intermediate_Theta = _sfit_point_tangent_geodesic(
                (self.Xs, self.t), k=self.ke, max_iter=self.max_iter, 
                return_intermediate_iterations=True, 
                num_intermediate_iterations=self.num_intermediate_iterations
            )
            self.intermediate_H = intermediate_H
            self.intermediate_Y = intermediate_Y 
            self.intermediate_Theta = intermediate_Theta
        else:
            H, Y, Theta = _sfit_point_tangent_geodesic((self.Xs, self.t), k=self.ke, max_iter=self.max_iter)
            
        # Compute main geodesic embeddings
        self.Us = [H @ cosm(np.diag(Theta)*self.t[i]) + Y @ sinm(np.diag(Theta)*self.t[i]) for i in range(self.T)]
        
        # Optionally compute intermediate geodesic embeddings for positional encodings
        if self.use_intermediate_iterations:
            self.intermediate_Us = []
            for H_inter, Y_inter, Theta_inter in zip(intermediate_H, intermediate_Y, intermediate_Theta):
                Us_inter = [H_inter @ cosm(np.diag(Theta_inter)*self.t[i]) + Y_inter @ sinm(np.diag(Theta_inter)*self.t[i]) for i in range(self.T)]
                self.intermediate_Us.append(Us_inter)
            
            # Concatenate intermediate embeddings to main embeddings for enriched positional encodings
            self.enriched_Us = []
            for i in range(self.T):
                # Start with main embedding
                enriched_embedding = [self.Us[i]]
                # Add intermediate embeddings for this timestep
                for inter_Us in self.intermediate_Us:
                    enriched_embedding.append(inter_Us[i])
                # Concatenate along feature dimension
                self.enriched_Us.append(np.concatenate(enriched_embedding, axis=1))
        else:
            self.enriched_Us = self.Us
    
    # this overwritten sometimes by subclasses
    def clustering_Euclidean(self):
        # Use enriched embeddings if available, otherwise use standard embeddings
        embeddings_to_use = self.enriched_Us if hasattr(self, 'enriched_Us') else self.Us
        T = len(embeddings_to_use)

        if all(k == self.kc_list[0] for k in self.kc_list):  # Constant kc_list
            k_val = self.kc_list[0]
            kmeans = KMeans(n_clusters=k_val, n_init=10)
            labels = [kmeans.fit_predict(U_i) for U_i in embeddings_to_use]
            return labels
        elif all(isinstance(k, int) for k in self.kc_list):  # Non-constant, provided kc_list
            labels = []
            for U_i, k_val in zip(embeddings_to_use, self.kc_list):
                kmeans = KMeans(n_clusters=k_val, n_init=10) 
                labels.append(kmeans.fit_predict(U_i))
            return labels
        else:  # Auto kc_list
            logger.info("Auto community-count selection uses placeholder kmin=2. Might later consider implementing a heuristic to guess higher when appropriate.")
            kmin, kmax = 2, self.ke 
            k_vals = range(kmin, kmax + 1)
            benefit_vs_time_and_k = np.zeros((len(k_vals), T)) - np.inf
            labels_by_k = {k: [] for k in k_vals}

            start = time.time()
            for i, U_i in enumerate(embeddings_to_use):
                for j, k_val in enumerate(k_vals):
                    kmeans = KMeans(n_clusters=k_val, n_init=10)
                    labels_i = kmeans.fit_predict(U_i)
                    labels_by_k[k_val].append(labels_i)
                    benefit = self.benefit_fn_broadcast([self.sadj_list[i]], [labels_i])[0]
                    benefit_vs_time_and_k[j, i] = benefit


            if self.smoothing_filter == 'median':
                kernel_size = self.smoothing_parameter
                smoothed_benefit = np.array([medfilt(row, kernel_size) for row in benefit_vs_time_and_k])
            elif self.smoothing_filter == 'gaussian':
                sigma = self.smoothing_parameter
                smoothed_benefit = np.array([gaussian_filter1d(row, sigma) for row in benefit_vs_time_and_k])
            else:
                smoothed_benefit = benefit_vs_time_and_k

            if self.stable_communities:
                aggregated_benefit = smoothed_benefit.mean(axis=1)
                best_k = k_vals[int(np.argmax(aggregated_benefit))]
                best_labels_all = labels_by_k[best_k]
            else:
                k_maximizing_benefit = [k_vals[l] for l in np.argmax(smoothed_benefit, axis=0)]
                best_labels_all = [labels_by_k[k][i] for i, k in enumerate(k_maximizing_benefit)]

            return best_labels_all
        
    
    # this is never overwritten by subclasses 
    def run_dcd(self):
        self.run_geo_embeddings()
        time_start = time.time()
        assignments = self.clustering_Euclidean()
        return assignments 
    
    def run_geo_embeddings(self):
        time_start = time.time()
        
        # If custom embeddings are provided, use them directly for fitting.
        if self.precomputed_embeddings is not None:
            logger.info("Using pre-computed embeddings for geodesic fitting.")
            self.Xs = self.precomputed_embeddings
        # Otherwise, run the standard matrix processing pipeline.
        else:
            if self.sadj_list is None:
                raise ValueError("sadj_list must be provided if precomputed_embeddings is not used.")
            self.make_clustering_matrices()
            self.make_modeled_clustering_matrices()
            
            self.Xs = self.modeled_clustering_matrices

        # Now, call the fitting method, which assumes self.Xs is ready
        self.get_geodesic_embeddings()
        logger.debug("Time to get geodesic embeddings: %.3fs", time.time() - time_start)
        

def _spectral_geodesic_smoothing(sadj_list=None, T=None, num_nodes=None, ke=None, precomputed_embeddings=None, kc='auto', stable_communities=False, mode='simple-nsc', smoothing_filter=None, smoothing_parameter=None, return_geo_embeddings_only=False, use_intermediate_iterations=False, num_intermediate_iterations=20, **mode_kwargs):
    if T is None:
        if sadj_list is not None:
            T = len(sadj_list)
        elif precomputed_embeddings is not None:
            T = len(precomputed_embeddings)
        else:
            raise ValueError("sadj_list or precomputed_embeddings must be provided")
    if not isinstance(kc, list) and kc != 'auto':
        try:
            kc = [kc] * T
        except TypeError:
            raise ValueError("kc must be a list, a natural number, or the default 'auto'")

    algorithms = {
        'simple': ['nsc', 'smm', 'bhc', 'usc'],
        'signed': ['srsc', 'gmsc', 'spmsc'],
        'overlapping': ['osc', 'csc'],
        'directed': ['ddsc', 'bsc', 'rwsc'],
        'multiview': ['gmsc', 'pmlc'],
        'cocommunity': ['scc'],
        'hierarchical': ['hsc'],
    }

    simple_class_lookup = {
        'nsc': _NSC,
        'smm': _SMM,
        'bhc': _BHC,
        'usc': _USC,
    }
    simple_modes = {
        f"simple-{alg}": simple_class_lookup[alg]
        for alg in algorithms['simple']
    }
    disabled_modes = {
        f"{category}-{alg}"
        for category, algs in algorithms.items()
        if category != 'simple'
        for alg in algs
    }

    if mode in simple_modes:
        smoother_class = simple_modes[mode]
    elif mode in disabled_modes:
        raise NotImplementedError("Non-'simple-*' temporal spectral modes are currently disabled.")
    else:
        valid_modes = ', '.join(sorted(simple_modes.keys()))
        raise ValueError(f"Invalid simple mode '{mode}'. Valid simple modes are: {valid_modes}")

    smoother = smoother_class(T=T, d=num_nodes, sadj_list=sadj_list, precomputed_embeddings=precomputed_embeddings, ke=ke, kc_list=kc,
                              stable_communities=stable_communities,
                              smoothing_filter=smoothing_filter,
                              smoothing_parameter=smoothing_parameter,
                              use_intermediate_iterations=use_intermediate_iterations,
                              num_intermediate_iterations=num_intermediate_iterations,
                              **mode_kwargs)  

    if return_geo_embeddings_only:
        smoother.run_geo_embeddings()
        # Return enriched embeddings if available, otherwise standard embeddings
        return smoother.enriched_Us if hasattr(smoother, 'enriched_Us') else smoother.Us
    assignments = smoother.run_dcd()
    # Return enriched embeddings if available, otherwise standard embeddings  
    embeddings_to_return = smoother.enriched_Us if hasattr(smoother, 'enriched_Us') else smoother.Us
    return assignments, embeddings_to_return


## Begin subclass implementations
class _Simple(_SpectralGeodesicSmoother):
    @staticmethod
    def calculate_modularity(adj_matrix: csr_matrix, communities: list) -> float:
        """
        Calculate the modularity of a network given its adjacency matrix and community assignments.

        :param adj_matrix: Sparse adjacency matrix of the network (scipy.sparse.csr_matrix)
        :param communities: List of community assignments for each node
        :return: Modularity value
        """
        if not isinstance(adj_matrix, csr_matrix):
            raise ValueError("adj_matrix must be a scipy.sparse.csr_matrix")

        if len(communities) != adj_matrix.shape[0]:
            raise ValueError("Number of community assignments must match number of nodes")

        n_edges = adj_matrix.sum() / 2
        n_nodes = adj_matrix.shape[0]

        modularity = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if communities[i] == communities[j]:
                    a_ij = adj_matrix[i, j]
                    k_i = adj_matrix.getrow(i).sum()
                    k_j = adj_matrix.getrow(j).sum()
                    expected = (k_i * k_j) / (2 * n_edges)
                    modularity += a_ij - expected

        modularity /= (2 * n_edges)
        return modularity
    
    @staticmethod
    def calculate_modularity_vectorized(adj_matrix: csr_matrix, communities: list) -> float:
        """
        Calculate the modularity of a network given its adjacency matrix and community assignments
        using a fully vectorized approach.

        :param adj_matrix: Sparse adjacency matrix of the network (scipy.sparse.csr_matrix)
        :param communities: List of community assignments for each node
        :return: Modularity value
        """
        if not isinstance(adj_matrix, csr_matrix):
            raise ValueError("adj_matrix must be a scipy.sparse.csr_matrix")

        n_nodes = adj_matrix.shape[0]

        if len(communities) != n_nodes:
            raise ValueError("Number of community assignments must match number of nodes")

        m = adj_matrix.sum() / 2
        if m == 0:
            raise ValueError("The network has no edges.")

        communities = np.array(communities)
        unique_communities, inverse_indices = np.unique(communities, return_inverse=True)
        n_communities = unique_communities.size

        degrees = np.array(adj_matrix.sum(axis=1)).flatten()

        degree_sum_per_community = np.bincount(inverse_indices, weights=degrees)

        community_matrix = csr_matrix(
            (np.ones(n_nodes), (inverse_indices, np.arange(n_nodes))),
            shape=(n_communities, n_nodes)
        )

        connections_within = community_matrix.dot(adj_matrix).dot(community_matrix.transpose())

        internal_edges_per_community = connections_within.diagonal() / 2

        modularity = (internal_edges_per_community.sum() / m) - np.sum((degree_sum_per_community / (2 * m)) ** 2)

        return modularity
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benefit_fn = self.calculate_modularity_vectorized
    

class _Signed(_SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def power_mean_laplacian(self, A_pos, A_neg, p, epsilon=1e-6):
        pass
    
    def create_clustering_matrix(self, A_pos, A_neg, p, epsilon=1e-6):
        pass

class _Directed(_SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _Overlapping(_SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _Multiview(_SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _Cocommunity(_SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class _Hierarchical(_SpectralGeodesicSmoother):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

## end subclass implementations

## Begin subsubclass implementations

class _NSC(_Simple):
    def make_clustering_matrix(self, sadj):
        pass # we override the full make_modeled_clustering_matrices method using normalized_signless_laplacian (recall that normalized signless laplacian SVD is equivalent to spectral clustering with normalized graph laplacian)
    

    
    @staticmethod
    def normalized_signless_laplacian(A):
        degrees = A.sum(axis=1).A1 

        with np.errstate(divide='ignore', invalid='ignore'):
            D_inv_sqrt = 1.0 / np.sqrt(degrees)
        D_inv_sqrt[np.isinf(D_inv_sqrt) | np.isnan(D_inv_sqrt)] = 0
        D_inv_sqrt_matrix = sparse.diags(D_inv_sqrt)
        D_plus_A = sparse.diags(degrees) + A
        L_signless = D_inv_sqrt_matrix @ D_plus_A @ D_inv_sqrt_matrix
        return L_signless
    
    def make_modeled_clustering_matrices(self):
        self.modeled_clustering_matrices =  [_NSC.normalized_signless_laplacian(sadj) for sadj in self.sadj_list]
        
  


class _SMM(_Simple): 
    def __init__(self, *args, **kwargs):
        kwargs['which_eig'] = 'largest' 
        super().__init__(*args, **kwargs)
      
    def make_clustering_matrix(self, A): 
        degrees = A.sum(axis=1).A1  
        m = A.sum() / 2  
        expected = sparse.csr_matrix((np.outer(degrees, degrees) / (2 * m)).astype(A.dtype))  
        B = A - expected  
        return B

class _BHC(_Simple):
    def make_clustering_matrix(self, sadj):
        r=None
        if r is None:
            r = np.sqrt(sadj.mean() * sadj.shape[0]) 
    
        n = sadj.shape[0]  
        d = sadj.sum(axis=1).A1  
        H = (r**2 - 1) * sparse.eye(n) - r * sadj + sparse.diags(d)  
        return H


class _USC(_Simple):
    """Unnormalized Spectral Clustering using the unnormalized graph Laplacian L = D - A."""
    
    def make_clustering_matrix(self, sadj):
        pass  # we override the full make_modeled_clustering_matrices method using unnormalized_laplacian
    
    @staticmethod
    def unnormalized_laplacian(A):
        """
        Compute the unnormalized graph Laplacian L = D - A.
        
        Args:
            A: Adjacency matrix (sparse)
            
        Returns:
            L: Unnormalized Laplacian matrix (sparse)
        """
        degrees = A.sum(axis=1).A1
        D = sparse.diags(degrees)
        L = D - A
        return L
    
    def make_modeled_clustering_matrices(self):
        self.modeled_clustering_matrices = [_USC.unnormalized_laplacian(sadj) for sadj in self.sadj_list]
    
    
def _signed_power_mean_laplacian(A_pos, A_neg, p, epsilon=1e-6):
    pass
    
class _SRSC(_Signed):
    def make_clustering_matrix(self, sadj_plus, sadj_minus):
        pass
    
class _GMSC(_Signed):
    def make_clustering_matrix(self, sadj_plus, sadj_minus):
        pass
    
class _SPMSC(_Signed):
    def make_clustering_matrix(self, sadj_plus, sadj_minus):
        pass
    
class _OSC(_Overlapping):
    def make_clustering_matrix(self, sadj):
        pass
    
class _CSC(_Overlapping):
    def make_clustering_matrix(self, sadj):
        pass
    
class _DDSC(_Directed):
    def make_clustering_matrix(self, sadj):
        pass
    
class _BSC(_Directed):
    def make_clustering_matrix(self, sadj):
        pass
    
class _RWSC(_Directed):
    def make_clustering_matrix(self, sadj):
        pass
    
class _PMLC(_Multiview):
    def make_clustering_matrix(self, sadj):
        pass
    
class _SCC(_Cocommunity):
    def make_clustering_matrix(self, sadj):
        pass
    
class _HSC(_Hierarchical):
    def make_clustering_matrix(self, sadj):
        pass

## End subsubclass implementations
