import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from random import random
import random
################################################################################################


def portfolio_var(w,V):
    '''
    variance = portfolio_var(w,V)

    Compute the variance given the covariance matrix and the weight vector.

    @param w: 1darray
        The weight vector.
    @param covariance: ndarray
        The covariance matrix.

    @return weights: scalar
        The variance determined by the input w,V.
    '''
    variance = w @( V @ w)
    return variance


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

########################### inverse volatility weighted portfolio ######################################################

def inverseVarianceWeighted(covariance):
    '''
    weights = inverseVarianceWeighted(covariance)

    Calculate the weights for Inverse Variance Weighted Portfolio.

    @param covariance: ndarray
        The covariance matrix.

    @return weights: ndarray
        The weights for Inverse Variance Weighted Portfolio.
    '''
    volatility = 1 / np.diag(covariance)  # take the reciprocal of the standard deviation of each asset
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

def marketCapitalizationWeighted(market_price):
    '''
     x = marketCapitalizationWeighted(market_price)

     Calculate the market capitalization weighted portfolio.

     @param market_price: 1darray
         The market price for asset

     @return weights: 1darray
         The market capitalization weighted portfolio.
    '''
    sum_price = sum(market_price)               # normalization factor
    weights = np.array([market_price / sum_price for price in market_price])  # normalize the weights
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

def main():
    n=5   # problem size
    #Sigma = np.eye(n)    # created an easy covariance matrix
    random.seed(9001)

    A = np.random.rand(n,n)
    Sigma = A @ A.T  # a random covariance matrix

    x_MVP = minimumVariance(Sigma)
    x_IVP = inverseVolatilityWeighted(Sigma)
    x_ERC = equalRiskContribution(Sigma)
    x_MDP = maximumDiversification(Sigma)
    x_MCWP = 0    # marketCapitalizationWeighted(market_price)
    x_1overN = uniformlyWeighted(Sigma)

    print('<Report>')
    print('x_MVP : ',x_MVP)
    print('x_IVP : ',x_IVP)
    print('x_ERC : ',x_ERC)
    print('x_MDP : ',x_MDP)
    print('x_MCWP : ',x_MCWP,'working on it')
    print('x_1overN : ',x_1overN)

if __name__ == "__main__":
    main()


