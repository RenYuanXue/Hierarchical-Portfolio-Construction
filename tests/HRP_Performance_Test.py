# Standard imports.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy.optimize import Bounds
from random import random
import random
from pandas_datareader import data
import yfinance as yf

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
    return np.cov(np.transpose(x))

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

#----------------------HRP------------------------------------------------------

def HRP(returns):
    hp = HierarchicalRiskParity()
    hrp_w = hp.getWeights(returns, 'single')
    return hrp_w

#----------------------Market Cap-----------------------------------------------
def MCWP(tickers):
    mkt_caps = data.get_quote_yahoo(tickers)['marketCap']
    weight = mkt_caps/sum(mkt_caps)
    return weight


########################## minimum variance portfolio###################################################################

def minimumVariance(covariance):
    '''
    weights = minimumVariance(covariance)

    Calculate the weights for Minimum Variance Portfolio with nonnegative constraints.
    Solve the problem min{ x' * covariance *x : sum(x)=1, x>=0 }.
    @param covariance: ndarray
        The covariance matrix.

    @return weights: 1darray
        The weights vector for Minimum Variance Portfolio.
    '''
    n = len(covariance)   # number of variables/assets
    cons = ({'type': 'eq', 'fun': budget_constraint})  # sum(x) = 1 constraint
    w0 = np.ones(n) / n  # initial guess for the algorithm
    bnd = Bounds(np.zeros(n),np.ones(n))  # bounds on each variable; enforcing 0<=x<=1
    # solve the problem using optimization package
    res = minimize(portfolio_var, w0, args=covariance, method='SLSQP', bounds = bnd, constraints=cons,options={'disp': False})
    weights = res.x
    return weights


#############################Budget Constraint##################################################
def budget_constraint(x):
    '''
     variance = portfolio_var(w,V)

     Serve as an equality constraint to the optimization problem, commonly know as budget constraint sum(x) =1.

     @param x: ndarray
         input array that needs to meet sum(x) =1.

     @return residual: scalar
         the residual at the point x.
     '''

    return np.sum(x)-1.0

########################### inverse volatility weighted portfolio ######################################################

def inverseVolatilityWeighted(covariance):
    '''
    weights = inverseVolatilityWeighted(covariance)

    Calculate the weights for Inverse Volatility Weighted Portfolio.

    @param covariance: ndarray
        The covariance matrix.

    @return weights: ndarray
        The weights for Inverse Volatility Weighted Portfolio.
    '''
    volatility = 1 / np.sqrt(np.diag(covariance))  # take the reciprocal of the standard deviation of each asset
    sum_volatility = sum(volatility)               # normalization factor
    weights = np.array([sigma / sum_volatility for sigma in volatility])  # normalize the weights
    return weights


############################ equal risk contribution portfolio #########################################################


def assetRiskContribution(w,covariance):
    '''
    RC = assetRiskContribution(w,V)

    Calculate the percentage risk of assets to total risk

    @param w: 1darray
        The weight vector.
    @param covariance: ndarray
        The covariance matrix.

     @return RC: 1darray
         the risk contribution with input w,V.
     '''
    sigma = w @ (covariance @ w)   # variance
    MRC = covariance @ w           # Marginal Risk Contribution
    RC = np.multiply(MRC,w)/sigma  # Risk Contribution
    return RC

def ercObjective(x,pars):
    '''
    ercObj = ercObjective(x,pars)

    Compute objective function for equal risk contribution portfolio

    @param x: ndarray
        The covariance matrix.
    @param pars: list
        pars[0]: covariance matrix.
        pars[1]: target weights for risk contribution.

     @return RC: scalar
        The objective function value for equal risk contribution portfolio.
     '''

    covariance = pars[0]             # covariance matrix
    target_percentage = pars[1]      # target risk in percent of portfolio risk
    asset_RC = assetRiskContribution(x,covariance)          # compute risk asset contribution
    ercObj = sum(np.square(asset_RC - target_percentage))   # sum of squared error, objective function for erc
    return ercObj


def equalRiskContribution(covariance):
    '''
    x = equalRiskContribution(covariance)

    Compute the equal risk contribution portfolio with equally weighted target.

    @param covariance: ndarray
        The covariance matrix.

     @return weights: 1darray
        The equal risk contribution portfolio given covariance matrix.
     '''
    n = covariance.shape[0]
    x_t = np.ones(n) / n        # target risk in percent of portfolio risk
    cons = ({'type': 'eq', 'fun': budget_constraint})    # constraint function
    w0 = x_t                    # initial guess
    bnd = bnd = Bounds(np.zeros(n), np.ones(n))          # bnd: lower and upper bound on each variable
    # solve the problem using optimization package
    res = minimize(ercObjective, w0, args=[covariance, x_t], method='SLSQP', bounds = bnd, constraints=cons,options={'disp': False})
    weights = res.x       # extract the solution

    return weights

############################# maximum diversification portfolio ########################################################



def diversificationRatio(w,covariance):
    '''
    DR = diversificationRatio(w,V)

    Calculate the negative of diversification ratio given weight w and covariance matrix

    @param w: 1darray
        The weight vector.
    @param covariance: ndarray
        The covariance matrix.

     @return DC: 1darray
         the negative of diversification ratio with input w,V.
     '''
    # This function computes the diversification ratio (objective function for MDP)

    w_vol = np.dot(np.sqrt(np.diag(covariance)), w.T)  # inner product between std and each weight; the numerator of DR
    port_vol = np.sqrt( w@(covariance@w) )             # portfolio volatility; the denominator of DR
    diversification_ratio = w_vol/port_vol    # obtain the diversification ratio
    return -diversification_ratio

def maximumDiversification(covariance):
    '''
    x = maximumDiversification(covariance)

    Calculate the maximum diversification portfolio given covariance matrix.

    @param covariance: ndarray
        The covariance matrix.

     @return weights: 1darray
        the maximum diversification portfolio.
     '''
    Sigma = covariance
    n = Sigma.shape[0]      # number of assets
    w0 = np.ones(n) / n     # initial guess
    bnd = bnd = Bounds(np.zeros(n),np.ones(n))           # bnd: lower and upper bound on each variable
    cons = ({'type': 'eq', 'fun': budget_constraint})    # constraint function
    # solve the problem using optimization package
    res = minimize(diversificationRatio, w0, args=Sigma, method='SLSQP', bounds=bnd, constraints=cons, options={'disp': False})
    weights = res.x      # extract the solution
    return weights


################################################################################################

def uniformlyWeighted(covariance):
    '''
    weights = uniformlyWeighted(covariance)

    Calculate the weights for 1 / N Portfolio.

    @param covariance: list/ndarray
        The covariance matrix.

    @return weights: ndarray
        The weights for 1 / N portfolio.
    '''
    covariance = np.array(covariance)
    N = len(covariance)
    weights = np.array([1 / N for i in range(N)])
    return weights


##Tickers
top_ten_mktcap = ['AAPL','MSFT','BRK-B','AMZN','GOOGL','WMT','UNH','INTC','PG','ADBE']
fin_serv = ['WFC','GS', 'JPM','MFC','SPGI','AMT','V','MA','TD', 'BAC']
energy_sec = ['XOM','CVX', 'RDS-A','NEE','PTR','TOT','BP','E','EPD','D']
random_one = ['NTES','JNJ','TSM','RY','BUD','OESX','PEP','ODFL','CSCO','INO']
random_two = ['HSBC','NVDA','MMM','VZ'  ,'CL','DIS','FISV', 'RIO','WMT','VOD']
all_of_them = top_ten_mktcap + fin_serv + energy_sec + random_one + random_two

#Data Collection
hist_data = yf.download(all_of_them,'2010-1-1','2015-12-31')['Close']   #Prices
future_data = yf.download(all_of_them,'2016-1-1','2017-1-1')['Close']
sp_data = np.array(yf.download('^GSPC,','2016-1-1','2017-1-1')['Close'])
hist_rtn = hist_data.pct_change().iloc[1:]                              #Return
future_rtn = np.array(future_data.pct_change().iloc[1:])


#Covariance:
cov_port = findCovMat(hist_rtn)

#---------------------Iterations------------------------------------------------
def performance_test(future_rtn, investment, covariance, method):

    if method == 'HRP':
        weights = HRP(hist_rtn)
    elif method == 'MCWP':
        weights = MCWP(all_of_them)
    elif method == 'uniform':
        weights = uniformlyWeighted(covariance)
    elif method ==  'diversification':
        weights = maximumDiversification(covariance)
    elif method == 'equalRiskContribution':
        weights = equalRiskContribution(covariance)
    elif method == 'inverseVol':
        weights = inverseVolatilityWeighted(covariance)
    else:
        print('No Method')
        return

    prev_w = np.array(weights)*investment
    port_val = np.array([investment])
    n = np.size(future_rtn, 0)
    for i in range(0,n,1):

       next_w = np.multiply((future_rtn[i]+1),prev_w)
       port_val = np.append(port_val, np.sum(next_w))
       prev_w = np.sum(next_w)*weights

    return port_val

inv = 1000
hrp_pv = performance_test(future_rtn, inv,cov_port,'HRP')
mcwp_pv = performance_test(future_rtn, inv,cov_port,'MCWP')
uni_pv = performance_test(future_rtn, inv,cov_port,'uniform')
div_pv = performance_test(future_rtn, inv,cov_port,'diversification')
erc_pv = performance_test(future_rtn, inv,cov_port,'equalRiskContribution')
inv_vol_pv = performance_test(future_rtn, inv,cov_port,'inverseVol')
sp_port = (sp_data/sp_data[0])*inv
n = np.size(hrp_pv)
index = np.array(future_data.pct_change().iloc[1:].index)
xaxis = np.arange(n)
plt.plot(xaxis, hrp_pv,label = "HRP")
plt.plot(xaxis, mcwp_pv,label = "MCWP")
plt.plot(xaxis, uni_pv,label = "Uniform")
plt.plot(xaxis, div_pv,label = "Diversification")
plt.plot(xaxis, erc_pv,label = "ERC")
plt.plot(xaxis, inv_vol_pv,label = "Inverse Vol")
plt.plot(xaxis, sp_port,label = "SP500")
plt.title('Portfolio Valuations in 2016')
plt.legend()
plt.show()

print('Final Portfolio Value:')
print('HRP:', hrp_pv[n-1])
print('MCWP:', mcwp_pv[n-1])
print('Uniform:', uni_pv[n-1])
print('Diversification:', div_pv[n-1])
print('ERC:', erc_pv[n-1])
print('Inverse Vol:', inv_vol_pv[n-1])
print('SP500:', sp_port[n-1])
