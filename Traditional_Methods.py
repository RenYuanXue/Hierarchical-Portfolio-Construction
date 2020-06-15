import numpy as np

def minimumVariance(covariance):
    '''
    weights = minimumVariance(covariance)

    Calculate the weights for Minimum Variance Portfolio. 

    @param covariance: list/ndarray
        The covariance matrix.

    @return weights: ndarray
        The weights for Minimum Variance Portfolio.
    '''
    covariance = np.array(covariance)
    N = len(covariance)   
    inv_cov = np.linalg.inv(covariance)
    l = np.ones((N, 1))
    return (np.transpose(l) @ inv_cov) / (np.transpose(l) @ inv_cov @ l)

def inverseVolatilityWeighted(covariance):
    '''
    weights = inverseVolatilityWeighted(covariance)

    Calculate the weights for Inverse Volatility Weighted Portfolio.

    @param covariance: list/ndarray
        The covariance matrix.

    @return weights: ndarray
        The weights for Inverse Volatility Weighted Portfolio.
    '''
    covariance = np.array(covariance)
    volatility = 1 / np.sqrt(np.diag(covariance))
    sum_volatility = sum(volatility)
    return np.array([sigma / sum_volatility for sigma in volatility])   

def equalRiskContribution(covariance):
    pass

def maximumDiversification(covariance):
    pass

def marketCapitalizationWeighted(covariance):
    pass

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
    return np.array([1 / N for i in range(N)])

def criticalLineAlgorithm(covariance):
    pass

def main():
    pass

if __name__ == "__main__":
    main()