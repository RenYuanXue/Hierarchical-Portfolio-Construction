# Standard imports.
import numpy as np
from scipy import stats

# Supplied functions.
def NVariables(X):
    '''
    Find the number of variables in the input.

    @param X: List. A 2D array of samples. 
        Size of (number of periods, number of variables).

    '''
    X = np.array(X)
    return np.size(X, 1)


def findCorrMat(X):
    N = NVariables(X)
    corr_mat = np.ones((N,N))
    for i in range(N-1):
        for j in range(i+1, N):
            pho = stats.pearsonr(X[:,i], X[:,j])
            corr_mat[i,j], corr_mat[j,i] = pho, pho


class HierarchicalRiskParity():
    '''
    A class that contains implementation of 
        Hierarchical Risk Parity algorithm.

    Attributes:
    ----------

    Methods:
    -------

    '''
    def __init__(self):
        pass

    def stageOne(self):
        pass
    def stageTwo(self):
        pass
    def stageThree(self):
        pass
    
    def __init__(self):
        pass

if __name__ == "__main__":
    print(range(2,5))