import numpy as np
from random import random
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt


#####################
#       STEP 1      #   import the data
#####################

# uncomment for a toy example
#n=5   # problem size
#A = 10*np.random.rand(n,n+3) # n stocks with n+3 different prices
#A = np.abs(A)


# load the data: it must be pre-processed - 9:00AM-3:30PM prices - about 62 obs.
A = np.transpose( np.loadtxt('data.txt') )   # it is a tiny example
print('There are ', A.shape[0], ' stocks and ',A.shape[1],'observations.')
stock_number = A.shape[0]
# Important : the rows represent each stock and the columns represent prices at certain moment


#####################
#       STEP 2      #   interpolate
#####################

### toy example 1
#data = np.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100], [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
#xx = np.linspace(0,100,200)
#curve = pchip_interpolate(data[0], data[1],xx)
#plt.plot(xx, curve, "x"); #plt.plot(data[0],data[1],"o"); #plt.show()


### toy example 2
#x = np.arange(A.shape[1])
#m = 200  # number of the final data points
#xx = np.linspace(0,x[-1],200)
#curve1 = pchip_interpolate(x , A[0,:], xx)
#print(curve1.shape, type(curve1)) ; #plt.plot(xx, curve1, "x"); #plt.plot(x,A[0,:],"o") ;#plt.show()

m = 200
curve_save = np.zeros((stock_number,m))  # array created to save the interpolated data
for ii in range(stock_number):   # loop through each stock
    x = np.arange(A.shape[1])   # the prices
    m = 200  # number of the final data points
    xx = np.linspace(0, x[-1], 200)    # filling the mappings of these points via interpolation
    curve = pchip_interpolate(x, A[ii, :], xx)   # interpolate
    curve_save[ii,:] = curve   # saving the interpolated points

A = curve_save   # this is now the NEW data

#####################
#       STEP 3      #   get the scaling factor c  - this needs history of opening and closing prices
#####################



# getting c -- these need the correct data
open_price = A[:,0]    # open price of each stock  for the entire period
close_price = A[:,-1]  # closing price of each stock for the entire period
#### must get the right variance
var_oc = np.abs(np.random.rand( stock_number))   # artifically created
var_co = np.abs(np.random.rand( stock_number))   # artifically created
c = 1 + np.divide(var_oc,var_co)   # compute the scaling factor c



#####################
#       STEP 4      #   obtain the log return
#####################


# obtain the log return
P_pre = A[:,0:A.shape[1]-1]  # denominator, truncate the last price
P_next = A[:,1:A.shape[1]] # numerator, truncate the first price
r_tilde = np.log( np.divide(P_next,P_pre)  )



#####################
#       STEP 5      #   obtain the daily convariance from the intra-day return
#####################

# obtain the daily convariance from the intra-day return
Sigmasum = np.zeros(A.shape[0])
for ii in range(r_tilde.shape[1]):
    r_hat = np.multiply( np.sqrt(c), r_tilde[:,ii])
    S = np.transpose(np.asmatrix(r_hat)) * np.asmatrix(r_hat)
    Sigmasum = Sigmasum + S


#print(Sigmasum)











