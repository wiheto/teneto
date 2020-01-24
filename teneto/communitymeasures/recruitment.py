import numpy as np
from .allegiance import allegiance


def recruitment(temporalcommunities, staticcommunities):
    """
    Calculates recruitment in relation to static communities.

    Calculates recruitment coefficient for each node.
    Recruitment coefficient is the average probability of nodes from the
    same static communities being in the same temporal communities at other time-points or during different tasks.

    Parameters:
    ------------
    temporalcommunities :  array
        temporal communities vector (node,time)
    staticcommunities : array
        Static communities vector for each node

    Returns:
    -------
    recruit : array
        recruitment coefficient for each node

    References:
    -----------

    .. [recruit-1]

        Danielle S. Bassett, Muzhi Yang, Nicholas F. Wymbs, Scott T. Grafton.
        Learning-Induced Autonomy of Sensorimotor Systems.
        Nat Neurosci. 2015 May;18(5):744-51.

    .. [recruit-2]

        Marcelo Mattar, Michael W. Cole, Sharon Thompson-Schill, Danielle S. Bassett. A Functional
        Cartography of Cognitive Systems.
        PLoS Comput Biol. 2015 Dec 2;11(12):e1004533.
    """
    # make sure the static and temporal communities have the same number of nodes
    staticcommunities = np.squeeze(staticcommunities)
    if staticcommunities.shape[0] != temporalcommunities.shape[0]:
        raise ValueError(
            'Temporal and static communities have different dimensions')
    if len(staticcommunities.shape) > 1:
        raise ValueError(
            'Incorrect static community shape')

    alleg = allegiance(temporalcommunities)

    recruit = np.zeros(len(staticcommunities))

    for i, statcom in enumerate(staticcommunities):
        recruit[i] = np.nanmean(alleg[i, staticcommunities == statcom])

    return recruit
