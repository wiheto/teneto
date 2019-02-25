import numpy as np

def sig_perm_test(net,community,T):
    """
    Uses permutation test to calculate the significance of clusters in a community structure

    Parameters:
    -----------
        net : NxN weighted adjacency matrix
        community : 1xN partition (cluster) vector
        T : number of random permutations

    Returns:
        sig_matrix : the significance of all clusters
        Q : the modularity of the given partition/cluster
        Q_rand : the modularity of all random partitions

    """

    N = net.shape[0] # size(net,1)
    C = len(np.unique(community)) # numel(unique(community))
    Q = modularity(net,community)
    Q_rand = np.zeros((T,C,C)) # zeros(T, C,C)

    for i in range(T):
        community_r = community[np.random.permutation(range(N))] # % community(randperm(N))
        Q_rand[i,:,:] = modularity(net,community_r)

    Q_avg = np.zeros((C,C))
    Q_std = Q_avg

    for i in range(C):
        for j in range(C):
            temp = np.squeeze(Q_rand[:,i,j]) # % saqueeze(Qmatrix_r[:,i,j])
            Q_avg[i,j] = np.mean(temp)
            Q_std[i,j] = np.std(temp)

    sig_matrix = np.zeros(Q.shape)

    for i in range(T):
        q_random = np.squeeze(Q_rand[i,:,:])
        sig_matrix = sig_matrix + (Q < q_random)

    sig_matrix = sig_matrix / T

    return sig_matrix, Q, Q_rand


def modularity(net,community):
    """
    auxiliary function for 'sig_perm_test'.
    
    This definition says that Qmatrix(i,j) is the inner product of the ith row of net with the jth column of cl.   
    """
    N = np.shape(net)[0] # number of nodes

    cl_label = np.unique(community)
    C = len(cl_label) # possibly ravel to count all
    cl = np.zeros((N, C))

    for i in range(C):
        cl[:, i] = community==cl_label[i]

    tmp = np.zeros((N,C))
    Q = np.zeros((C,C))

    for i in range(N):
        for j in range(C):
            tmp[i,j] = np.dot(net[i,:],cl[:,j])

    ct = np.transpose(cl)

    for i in range(C):
        for j in range(C):
            Q[i,j] = np.dot(tmp[:,i],ct[j,:])

    return Q
