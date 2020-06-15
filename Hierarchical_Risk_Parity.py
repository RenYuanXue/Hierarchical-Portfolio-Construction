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
    N = NVariables(x)

    Find the number of variables in the input.

    @param x: list/ndarray
        A 2D array of samples.
        Size of (number of periods, number of variables).

    @returns N: int
        The first dimension of the matrix.
    '''
    # Turn the variable into ndarray in case it's not.
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
    ''' 
    cov_mat = findCovMat(x)

    Calculate the covariance matrix for the given data.
    Notice that x has variables in columns.

    @param x: list/ndarray
        A 2D array of samples.
        Size of (number of periods, number of variables).

    @returns cov_mat: ndarray
        The covariance matrix for the given input x.
    '''
    # Turn the variable into ndarray in case it's not.
    x = np.array(x)
    return np.cov(x)

def findCorrMat(x):
    '''
    corr_mat = findCorrMat(x)

    Calculate the correlation matrix for the given data.
    Notice that x has variables in columns.

    @param x: list/ndarray
        A 2D array of samples.
        Size of (number of periods, number of variables).

    @returns corr_mat: ndarray
        The correlation matrix for the given input x.
    '''
    N = NVariables(x)
    # Turn the variable into ndarray in case it's not.
    x = np.array(x)
    corr_mat = np.ones((N,N))
    for i in range(N - 1):
        for j in range(i + 1, N):
            rho, _ = stats.pearsonr(x[:, i], x[:, j])
            corr_mat[i,j], corr_mat[j,i] = rho, rho
    return corr_mat

def findDistMat(x):
    '''
    dist_mat = findDistMat(x)

    Calculate the distance matrix for the given data.
    Notice that x has variables in columns.

    @param x: list/ndarray
        A 2D array of samples.
        Size of (number of periods, number of variables).

    @returns dist_mat: ndarray
        The distance matrix for the given input x.
    '''
    x = np.array(x)
    # Turn the variable into ndarray in case it's not.
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
    three process stages in Hierarchical Risk Parity algorithm.

    Attributes:
    ----------
    link: ndarray
        The hierarchical clustering encoded as a linkage matrix.

    Methods:
    -------
    getSerial(link, N, curr_idx)
        Find the order represented by the linkage matrix.
    stageOne(dist_mat, method='single'):
        Get the hierarchical clustering encoded as a linkage matrix.
    stageTwo(dist_mat):
        Find the reordered distance matrix and 
        the order represented by the linkage matrix.
    stageThree(covariance, order):
        Get the portfolio weights for hierarchical risk parity method.
    plotLink():
        Plot the dendrogram, which is a diagram that 
        shows the hierarchical relationship between variables.
    getWeights(x, method = 'single', cov_method = ''):
        An automated process of computing the portfolio weights 
        for hierarchical risk partity method for a given dataset.
    '''
    def __init__(self):
        self.link = None

    def getSerial(self, link, N, curr_idx):
        '''
        idx_list = getSerial(self, link, N, curr_idx)

        Find the order represented by the linkage matrix.

        @param link: ndarray
            The hierarchical clustering encoded as a linkage matrix.
        @param N: int
            Dimensionality in distance matrix that formed the linkage matrix.
            Note: distance matrix is a square matrix, therefore it's a (N x N) matrix. 
        @param curr_idx: int
            The row index in the linkage matrix.

        @return idx_list: list
            The list of order that represented by the linkage matrix. 
        '''
        # When curr_idx < N, it means it's a cluster with one item.
        if curr_idx < N:
            return [curr_idx]
        else:
            left = int(link[curr_idx - N, 0])   # The left cluster. 
            right = int(link[curr_idx - N, 1])  # The right cluster.
            return (self.getSerial(link, N, left) + self.getSerial(link, N, right))

    def stageOne(self, dist_mat, method = 'single'):
        '''
        None = stageOne(dist_mat, method = 'single')

        Get the hierarchical clustering encoded as a linkage matrix,
        the resulting matrix is stored as an attribute in the class.

        @param dist_mat: list/ndarray
            The distance matrix where entry (i,j) 
            represents the distance between variable i and j.
        @param method: str
            The method that used when finding the linkage metrix.
            Choice between 'single', 'average', 'weighted', 
            'centroid', 'median' and 'ward'.
            Default is 'single'.
        
        @return: None. 
        '''
        # Dense the distance matrix, so won't raise an error.
        densed_dist_mat = squareform(dist_mat)
        link = linkage(densed_dist_mat, method = method)
        self.link = link

    def stageTwo(self, dist_mat):
        '''
        serial_matrix, order = stageOne(dist_mat, method = 'single')

        Find the order represented by the linkage matrix 
        and the reordered distance matrix.

        @param dist_mat: list/ndarray
            The distance matrix where entry (i,j) 
            represents the distance between variable i and j.
        
        @return serial_matrix: ndarray
            The reordered distance matrix according to linkage matrix.
        @return order: list
            The order represented by the linkage matrix.
        '''
        N = NVariables(dist_mat)
        order = self.getSerial(self.link, N, N + N - 2)
        serial_matrix = np.zeros((N, N))
        # The upper triangular indices for serial matrix.
        a, b = np.triu_indices(N, k = 1)
        serial_matrix[a, b] = dist_mat[[order[i] for i in a], 
                                            [order[j] for j in b]]
        serial_matrix[b, a] = serial_matrix[a, b]
        return order, serial_matrix

    def stageThree(self, covariance, order):
        '''
        weights = stageThree(covariance, order)

        Get the portfolio weights for hierarachical risk parity method.

        @param covariance: list/ndarray
            The covariance matrix.
        @param order: list
            The order represented by the linkage matrix.

        @return weights: ndarray
            portfolio weights for hierarachical risk parity method.
        '''
        # Turn the variable into ndarray in case it's not.
        covariance = np.array(covariance)
        weights = np.ones((1,len(order)))
        # weights = pd.Series(1, index=order)
        clustered_alphas = [order]
        while len(clustered_alphas) > 0:
            # Devide the current clusters into left and right subclusters.
            clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                                for start, end in ((0, len(cluster) // 2),
                                                (len(cluster) // 2, len(cluster)))
                                if len(cluster) > 1]
            # Going through all subclusters.                    
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

    def getWeights(self, x, method = 'single', cov_method = ''):
        '''
        weights = getWeights(x, method = 'single', cov_method = '')

        An automated process of computing the portfolio weights 
        for hierarchical risk partity method for a given dataset.

        @param x: list/ndarry
            A 2D array of samples.
            Size of (number of periods, number of variables).

        @param method: str
            The method that used when finding the linkage metrix.
            Available choices are: 'single', 'average', 'weighted', 
            'centroid', 'median' and 'ward'.
            Default is 'single'.
        @param cov_method: str
            The method used when forecasting the covariance matrix.
            Available choices are: '', 'SMPL', 'EWMA', 'DCCGARCH'.
            Default is ''.

        @return weights: ndarray
            portfolio weights for hierarachical risk parity method.
        '''
        if cov_method == '':
            cov_mat = findCovMat(x)
        elif cov_method == 'SMPL':
            pass
        elif cov_method == 'EWMA':
            pass
        elif cov_method == 'DCCGARCH':
            pass
        corr_mat = findCorrMat(x)
        dist_mat = findDistMat(corr_mat)
        self.stageOne(dist_mat, method)
        order, _ = self.stageTwo(dist_mat)
        weights = self.stageThree(cov_mat, order)
        return weights

    def plotLink(self):
        '''
        None = plotLink()

        Plot the dendrogram, which is a diagram that 
        shows the hierarchical relationship between variables.

        @param: None.

        @return: None.
        '''
        fig = plt.figure(figsize = (25,10))
        dn = dendrogram(self.link)
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()