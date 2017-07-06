import numpy as np

'''
These functions are the possible distance functions called in  getDistanceFunction()
'''

def hamming_distance(x1, x2):
    """Requires numpy compabtble input with equal shape"""
    return np.sum(np.subtract(x1,x2)!=0)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(np.power(np.subtract(x1,x2),2)))

def taxicab_distance(x1, x2):
    return np.sum(np.abs(np.subtract(x1,x2)))

def derivative(x1, x2):
    """
    x1 (t), x2 (t+1)
    Average derivative of input
    """
    #Derivative
    nd = np.array(x2) - np.array(x1)
    return np.mean(nd)
