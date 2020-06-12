# Standard imports.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Supplied functions.
def NVariables(x):
    '''
    Find the number of variables in the input.

    @param X: List. A 2D array of samples. 
        Size of (number of periods, number of variables).

    '''
    x = np.array(x)
    return np.size(x, 1)

''' 

###### For now doesn't need

def TPeriods(x):
    Find the number of samples in the input.

    @param X: List. A 2D array of samples. 
        Size of (number of periods, number of variables).

    return len(x)

'''

def findCovMat(x):
    pass

def findCorrMat(x):
    N = NVariables(x)
    x = np.array(x)
    corr_mat = np.ones((N,N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            rho, _ = stats.pearsonr(x[:, i], x[:, j])
            corr_mat[i,j], corr_mat[j,i] = rho, rho
    return corr_mat

def findDistMat(x):
    x = np.array(x)
    dist_mat = np.sqrt(1 / 2 * (1 - x))
    return dist_mat

'''
### May used later, Euclidean distance is one available metrics
### in the linkage function in scipy.cluster.hierarchy.

def findEuclideanDistMat(D):
    euclidean_dist_mat = D
    for i in range(len(D) - 1):
        for j in range(i + 1, len(D)):
            d_i_j = np.sqrt(sum(np.power(D[:, i] - D[:, j], 2)))
            euclidean_dist_mat[i,j], euclidean_dist_mat[j,i] = d_i,j, d_i,j
    return euclidean_dist_mat
'''

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
        self.link = None

    def getSerial(self, link, N, curr_idx):
        if curr_idx < N:
            return [curr_idx]
        else:
            left = int(link[curr_idx - N, 0]) 
            right = int(link[curr_idx - N, 1])
            return (self.getSerial(link, N, left) + self.getSerial(link, N, right))

    def stageOne(self, dist_mat, method='single'):
        # Dense the distance matrix, so won't raise an error.
        densed_dist_mat = squareform(dist_mat)
        link = linkage(densed_dist_mat, method = method)
        self.link = link

    def stageTwo(self, dist_mat):
        N = NVariables(dist_mat)
        order = self.getSerial(self.link, N, N + N - 2)
        serial_matrix = np.zeros((N, N))
        a, b = np.triu_indices(N, k = 1)
        serial_matrix[a, b] = dist_mat[[order[i] for i in a], 
                                            [order[j] for j in b]]
        serial_matrix[b, a] = serial_matrix[a, b]
        return serial_matrix, order

    def stageThree(self, covariance, order):
        covariance = np.array(covariance)
        weights = np.ones((1,len(order)))
        # weights = pd.Series(1, index=order)
        clustered_alphas = [order]

        while len(clustered_alphas) > 0:
            # Devide the current 
            clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2),
                                                (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]
                                
            for subcluster in range(0, len(clustered_alphas), 2):
                left_cluster = clustered_alphas[subcluster]
                right_cluster = clustered_alphas[subcluster + 1]

                left_subcovar = covariance[np.ix_(left_cluster, left_cluster)]
                inv_diag = 1 / np.diag(left_subcovar)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                left_cluster_var = np.dot(parity_w, np.dot(left_subcovar, parity_w))

                right_subcovar = covariance[np.ix_(right_cluster, right_cluster)]
                inv_diag = 1 / np.diag(right_subcovar)
                parity_w = inv_diag * (1 / np.sum(inv_diag))
                right_cluster_var = np.dot(parity_w, np.dot(right_subcovar, parity_w))

                alloc_factor = 1 - left_cluster_var / (left_cluster_var + right_cluster_var)

                weights[np.ix_([0],left_cluster)] *= alloc_factor
                weights[np.ix_([0],right_cluster)] *= 1 - alloc_factor

                # weights[left_cluster] *= alloc_factor
                # weights[right_cluster] *= 1 - alloc_factor
                
        return weights

    def plotLink(self):
        fig = plt.figure(figsize = (25,10))
        dn = dendrogram(self.link)
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()