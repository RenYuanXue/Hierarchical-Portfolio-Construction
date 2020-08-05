import numpy as np
from random import random
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt

def PortfolioVariance(w,real_Sigma):
    # input :
    # w : portfolio obtained using the forecasted covariance matrix
    # Sigma : realized covariance matrix from CovarianceEstimator function
    variance =  w @ (real_Sigma @ w)
    return variance

def CVaR(w,Sigma):


    return 1000


def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma =  (w*V*w.T)[0,0]   # variance
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def HerfindahlIndex(w,Sigma):
    RC = calculate_risk_contribution(w,Sigma)
    N = len(RC)
    HI = ( np.sum( np.multiply(RC,RC) ) - 1/N ) /( 1-1/N)

    return HI



def DR(w, V):
    # This function computes the diversification ratio (objective function for MDP)
    # This is the same as the one in Traditional_Methods, but we need to be careful with the inputs
    # input :
    # w : portfolio obtained using the forecasted covariance matrix
    # Sigma : realized covariance matrix from CovarianceEstimator function
    w_vol = np.dot(np.sqrt(np.diag(V)), w.T)  # inner product between std and each weight # the numerator
    port_vol = np.sqrt( w@(V@w) )  # portfolio volatility   # the denominator
    diversification_ratio = w_vol/port_vol   # obtain the diversification ratio
    return -diversification_ratio

def main():

    # create artificial inputs for test
    n=5   # problem size
    w = np.ones(n) / n  # artifically made
    A = np.random.rand(n,n)
    Sigma = A @ A.T  # a random covariance matrix

    print('<REPORT>')
    print('Portfolio variance is ', PortfolioVariance(w,Sigma) )
    print('CVaR is ',CVaR(w,Sigma),'!!!!!!!!!!!!need to work on this')
    print('Herfindahl Index is ', HerfindahlIndex(w, Sigma))
    print('Diversificatio ratio is',DR(w, Sigma) )

if __name__ == '__main__':
    main()