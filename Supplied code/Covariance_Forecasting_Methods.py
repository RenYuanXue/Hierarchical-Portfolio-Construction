import numpy as np
import pandas as pd
import pkg_resources
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def sampleBasedCovariance(epsilon):
    '''
    SMPL = sampleBasedCovariance(epsilon)

    Compute the sample based forecasting covariance matrix

    @param epsilon: ndarray
        A 2D array where each row holds the log return over one period.
        Size of (number of periods, number of stocks).

    @returns SMPL: ndarray
        The sample based forcasting covariance matrix for the given input epsilon.
    '''
    T = len(epsilon)   # number of periods
    # create an array for holding the covariance matrix
    SMPL = np.zeros([epsilon.shape[1], epsilon.shape[1]])
    for i in range(1, T):
        e_t = epsilon[i-1]  # log return of the period (i-1)-th period
        S = np.transpose(np.asmatrix(e_t)) * np.asmatrix(e_t)
        SMPL = SMPL + S     # aggregate the log returns

    SMPL = SMPL / T  # averaging over the total number of period

    return  SMPL

def exponentiallyWeightedMovingAverage(epsilon):
    '''
    EWMA_T = exponentiallyWeightedMovingAverage(epsilon)

    Compute the exponentially weighted moving average forecasting covariance matrix

    @param epsilon: ndarray
        A 2D array where each row holds the log return over one period.
        Size of (number of periods, number of stocks).

    @returns EWMA_T: ndarray
        The exponentially weighted moving average forecasting covariance matrix of the last period for the given input epsilon.
    '''
    # create an array for holding the covariance matrix for each t
    T = epsilon.shape[0]           # number of periods
    stock_num = epsilon.shape[1]   # number of stocks

    # create an array for holding the covariance matrix over period T
    # each row of EWMA is a flattened covariance matrix for each period
    EWMA = np.full([T, stock_num**2], np.nan)
    lamb = 0.94    # a decay constant
    S = np.cov(epsilon[0])      # initial (t=0) covariance matrix
    EWMA[0, :] = S.flatten()    # save the flattened covariance for the first period
    for i in range(1, T):
        e_t = epsilon[i-1]      # log return of the period (i-1)-th period
        # compute the exponentially weighted covariance matrix with exponent lambda
        S = lamb * S + (1 - lamb) * np.transpose(np.asmatrix(e_t)) * np.asmatrix(e_t)
        EWMA[i, :] = S.flatten()   # save the flattened covariance for the i-th period
    EWMA_T = EWMA[-1].reshape((stock_num, stock_num))  # return the forecasting matrix for last period

    return EWMA_T

#  achieve it using R in Python.
def DynamicConditionalCorrelationGARCH(epsilon, n_days):
    epsilon = pd.DataFrame(data = epsilon)
    # compute DCC-Garch in R using rmgarch package
    pandas2ri.activate()
    # convert the daily returns from pandas dataframe in Python to dataframe in R
    # pd_rets - a pandas dataframe of daily returns, where the column names are the tickers of stocks and index is the trading days.
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_rets = robjects.conversion.py2rpy(epsilon)
    
    r_dccgarch_code = """
                    library('rmgarch')
                    function(r_rets, n_days){
                            univariate_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
                                                        variance.model = list(garchOrder = c(1,1),
                                                                            variance.targeting = FALSE, 
                                                                            model = "sGARCH"),
                                                        distribution.model = "norm")
                            n <- dim(r_rets)[2]
                            dcc_spec <- dccspec(uspec = multispec(replicate(n, univariate_spec)),
                                                dccOrder = c(1,1),
                                                distribution = "mvnorm")
                            dcc_fit <- dccfit(dcc_spec, data=r_rets)
                            forecasts <- dccforecast(dcc_fit, n.ahead = n_days)
                            list(dcc_fit, forecasts@mforecast$H)
                    }
                    """
    r_dccgarch = robjects.r(r_dccgarch_code)
    r_res = r_dccgarch(r_rets,n_days)
    pandas2ri.deactivate()
    # end of R

    r_dccgarch_model = r_res[0] # model parameters
    r_forecast_cov = r_res[1]   # forecasted covariance matrices for n_days

    r_forecast_cov = robjects.DataFrame({'a': r_forecast_cov})

    # Convert back to ndarray
    with localconverter(robjects.default_converter + pandas2ri.converter):
        py_forecast_cov = robjects.conversion.rpy2py(r_forecast_cov)
    
    N = len(py_forecast_cov)
    py_forecast_cov = np.reshape(py_forecast_cov.to_numpy(), (n_days,N,N))[-1]

    return py_forecast_cov, r_dccgarch_model

def main():
    # import the data
    data = np.loadtxt('cleanup_data.txt')   # it is a tiny example
    #p = pd.read_excel('cleanup_data.xlsx',header=None)
    y = data

    stock_price = data
    y = np.diff(np.log(stock_price), n=1, axis=0) * 100  # calculate the 'log' returns
    Epsilon = np.zeros(y.shape)
    for ii in range(y.shape[1]):
        r_t = y[:, ii]
        mu_t = np.mean(y[:, ii])
        Epsilon[:,ii] =  r_t - mu_t  # subtract mean from the return
    T = len(Epsilon)
    SMPL = sampleBasedCovariance(Epsilon)

    EWMA = exponentiallyWeightedMovingAverage(Epsilon)

    DCCGARCH, _ = DynamicConditionalCorrelationGARCH(Epsilon, T)
    #print(EWMA)
    #print(SMPL)
    print(DCCGARCH.shape)
    #DCCGARCH.to_excel('test.xlsx', engine='xlsxwriter')

if __name__ == '__main__':
    main()



