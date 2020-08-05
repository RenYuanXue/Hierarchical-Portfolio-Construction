# Standard codes.
import sys
sys.path.append("../Supplied code") # for finding the source files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Traditional_Methods import *
from Hierarchical_Risk_Parity import *

def generateData(nObs,size0,size1,sigma1,seed_num): 
    # Time series of correlated variables 
    # #1) generating some uncorrelated data 
    np.random.seed(seed_num)
    x=np.random.normal(0.005,0.1,size=(nObs,size0)) 
    # each row is a variable 
    # 2) creating correlation between the variables 
    cols=[np.random.randint(0,size0-1) for i in range(size1)] 
    y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols))) 
    x=np.append(x,y,axis=1) 
    x=pd.DataFrame(x,columns=range(1,x.shape[1]+1)).to_numpy()
    return x

def main():
    cond_numbers = np.array([])
    count = 500
    result = np.zeros((3,count))
    for k in range(count):
        nObs,size0,size1,sigma1=45, 5, 5, .05 
        x = generateData(nObs,size0,size1,sigma1, k)
        cov = findCovMat(x)
        x_mod= x + np.random.normal(0, 1, size = (x.shape))
        cov_mod = findCovMat(x_mod)

        HRP = HierarchicalRiskParity()
        x_HRP = HRP.getWeights(x, cov)
        x_MVP = minimumVariance(cov)
        x_IVP = inverseVarianceWeighted(cov)

        x_HRP_mod = HRP.getWeights(x, cov_mod)
        x_MVP_mod = minimumVariance(cov_mod)
        x_IVP_mod = inverseVarianceWeighted(cov_mod)

        HRP_dist = x_HRP_mod @ cov @ x_HRP_mod.T
        MVP_dist = x_MVP_mod @ cov @ x_MVP_mod.T
        IVP_dist = x_IVP_mod @ cov @ x_IVP_mod.T

        result[:,k] = np.array([HRP_dist, MVP_dist, IVP_dist])
        cond_numbers = np.append(cond_numbers, np.linalg.cond(cov))

    plt.figure(0)
    plt.title('Euclidean distance of HRP and IVP for given Condition number of Covariance')
    plt.xlabel('Condition Number of Covariance matrix')
    plt.ylabel('Euclidean distance between two solutions')
    plt.scatter(cond_numbers, result[0], c='b')
    plt.scatter(cond_numbers, result[1], c='g')
    plt.scatter(cond_numbers, result[2], c='r')
    plt.legend(['HRP', 'MVP', 'IVP'])
    plt.show()

if __name__ == "__main__":
    main()