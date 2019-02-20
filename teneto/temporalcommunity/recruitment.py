import numpy as np
from .allegiance import allegiance


def recruitment(temporalcommunities, staticcommunities):
    """
    Calculates recruitment coefficient for each node. Recruitment coefficient is the average probability of nodes from the
      same communities being in the same communities at other time-points or during different tasks.

    Parameters:
    ------------
    temporalcommunities :  array
        temporal communities vector (node,time)
    staticcommunities : array
        Static communities vector for each node

    Returns:
    -------
    Rcoeff : array
        recruitment coefficient for each node

    References:
    -----------

    Danielle S. Bassett, Muzhi Yang, Nicholas F. Wymbs, Scott T. Grafton.
    Learning-Induced Autonomy of Sensorimotor Systems. Nat Neurosci. 2015 May;18(5):744-51.

    Marcelo Mattar, Michael W. Cole, Sharon Thompson-Schill, Danielle S. Bassett. A Functional
    Cartography of Cognitive Systems. PLoS Comput Biol. 2015 Dec 2;11(12):e1004533.
    """

    # make sure the static and temporal communities have the same number of nodes
    if np.shape[0] != temporalcommunities.shape[0]:
        raise ValueError(
            'Temporal and static communities have different dimensions')

    alleg = allegiance(temporalcommunities)

    Rcoeff = np.zeros(len(staticcommunities))

    for i, statcom in enumerate(staticcommunities):
        Rcoeff[i] = np.mean(alleg[i, staticcommunities == statcom])

    return Rcoeff
