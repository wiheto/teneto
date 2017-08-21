import numpy as np

def shufflegroups(dat1, dat2, pnum=1000, exchange='subjects', tail=2):
    """Dat1, dat2 are of equal length.
    They get shuffled p number of times with permuted groups equalling the len of
    inpput data. exchangeblocks = subjects (then len(dat1)==len(dat2)) and shuffling
    only occurs by making sure that dat1(s1) and dat2(s1) end up in opposite permute groups
    Returns permutation distirbution of avg(perm(len(dat1))-avg(perm(len(dat2))
    """
    if exchange == 'subjects':
        if len(dat1) != len(dat2):
            raise ValueError("dat vectros must be of same length")
        permdist = np.zeros(pnum)
        for i in range(0, pnum):
            porder = np.random.randint(1, 3, len(dat1))
            permutation_group1 = np.concatenate(
                (dat1[np.where(porder == 1)], dat2[np.where(porder == 2)]))
            permutation_group2 = np.concatenate(
                (dat1[np.where(porder == 2)], dat2[np.where(porder == 1)]))
            if tail == 2:
                permdist[i] = abs(np.mean(permutation_group1) - np.mean(permutation_group2))
            elif tail == 1:
                permdist[i] = np.mean(permutation_group1) - np.mean(permutation_group2)

        permdist = np.sort(permdist)
        if tail == 2:
            empdiff = abs(dat1.mean() - dat2.mean())
        elif tail == 1:
            empdiff = dat1.mean() - dat2.mean()
        p_value = sum(empdiff < permdist) / (pnum + 1)
        return p_value, permdist

    if exchange == 'all':
        raise ValueError('Non-subject exchange blocks still have to be made')
