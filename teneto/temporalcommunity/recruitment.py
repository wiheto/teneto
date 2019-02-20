def recruitment(community,system):
    """
    Calculates recruitment coefficient for each node. Recruitment coefficient is the average probability of nodes from the
      same community being in the same communities at other time-points or during different tasks. 
      
    Parameters:
    ------------
    community :  	time,node community vector 
    system : array containing the system assignment for each node
              eg. nodes from the Power parcellation assigned to brain networks

    returns:
    -------
      R : recruitment coefficient for each node

    References: 
    ----------- 

    Danielle S. Bassett, Muzhi Yang, Nicholas F. Wymbs, Scott T. Grafton. 
    Learning-Induced Autonomy of Sensorimotor Systems. Nat Neurosci. 2015 May;18(5):744-51.

    Marcelo Mattar, Michael W. Cole, Sharon Thompson-Schill, Danielle S. Bassett. A Functional
    Cartography of Cognitive Systems. PLoS Comput Biol. 2015 Dec 2;11(12):e1004533. 
    """

    N = len(community)
    C = np.unique(community)

    MA = module_allegience_matrix(N,C,community)

    R = np.zeros(len(system))

    for i in range(len(system)):
      system_i = system[i]
      R[i] = np.mean(MA[i, system == system_i]) # nanmean(MA(i,strcmp(systemByNode,thisSystem)));

    return R
	
