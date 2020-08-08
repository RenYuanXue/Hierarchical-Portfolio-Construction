# Standard codes.
import sys
sys.path.append("../Supplied code") # for finding the source files
import scipy.cluster.hierarchy as sch,random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Traditional_Methods import *
from Hierarchical_Risk_Parity import *

def generateData(nObs,sLength,size0,size1,mu0,sigma0,sigma1F):
    np.random.seed(4)
    #1) generate random uncorrelated data
    x=np.random.normal(mu0,sigma0,size=(nObs,size0)) # each row is a variable 
    #2) create correlation between the variables 
    cols=[random.randint(0,size0-1) for i in range(size1)] 
    y=x[:,cols]+np.random.normal(0,sigma0*sigma1F,size=(nObs,len(cols))) 
    x=np.append(x,y,axis=1) 
    #3) add common random shock
    point=np.random.randint(sLength,nObs-1,size=2) 
    x[np.ix_(point,[cols[0],size0])]=np.array([[-.5,-.5],[2,2]])
    #4) add specific random shock 
    point=np.random.randint(sLength,nObs-1,size=2) 
    x[point,cols[-1]]=np.array([-.5,2]) 
    return x

def IVarP(cov, **kargs):
    weights = inverseVarianceWeighted(cov)
    return weights

def HRP(cov, x):
    HRP = HierarchicalRiskParity()
    weights = HRP.getWeights(x, cov)
    return weights

def MVP(cov, **kargs):
    weights = minimumVariance(cov)
    return weights

def IVP(cov, **kargs):
    weights = inverseVolatilityWeighted(cov)
    return weights

def main(numIters=1000,nObs=520,size0=5,size1=5,mu0=0,sigma0=1e-2, sigma1F=.25,sLength=260,rebal=22): 
    # Monte Carlo experiment on HRP
    count = 0 
    methods=[HRP,MVP,IVP, IVarP]
    stats={i.__name__:pd.Series(dtype='float64') for i in methods}
    pointers=range(sLength,nObs,rebal) 
    while count<numIters:
        #1) Prepare data for one experiment
        x =generateData(nObs,sLength,size0,size1,mu0,sigma0,sigma1F) 
        r = {i.__name__:pd.Series(dtype='float64') for i in methods} 
        #2) Compute portfolios in-sample 
        for pointer in pointers:
            x_=x[pointer-sLength:pointer] 
            cov_=np.cov(x_,rowvar=0)
            #3) Compute performance out-of-sample 
            x_=x[pointer:pointer+rebal]
            for func in methods: 
                w_=func(cov=cov_,x = x) # callback
                w_ = np.transpose(w_)
                r_=pd.Series(np.dot(x_,w_).flatten()) 
                r[func.__name__]=r[func.__name__].append(r_) 
        #4) Evaluate and store results 
        for func in methods: 
            r_=r[func.__name__].reset_index(drop=True) 
            p_=(1+r_).cumprod() 
            stats[func.__name__].loc[count]=p_.iloc[-1]-1 # terminal return 
        count += 1
    #5) Report results 
    stats=pd.DataFrame.from_dict(stats,orient='columns')
    df0,df1=stats.std(),stats.var() 
    print(pd.concat([df0,df1,df1/df1['HRP']-1],axis=1))
    return 0

if __name__ == '__main__':
    main(numIters=10000,nObs=520,size0=5,size1=5,mu0=0,sigma0=1e-2, sigma1F=.25,sLength=260,rebal=22)