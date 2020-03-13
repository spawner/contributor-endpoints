# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:50:27 2020

@author: jkcle
"""

def kstest(x, y):
    '''
    This test can peform a one-sample or two-sample Kolmogorov–Smirnov test,
    which tests the goodness of fit of CONTINUOUS data. The null hypothesis is
    x and y are from the same distribution.
    
    Returns the test statistic and p-value.
    
    One-sample K-S test:
        pass your data and a large enough sample from the distribution your 
        null hypothesis assumes your data is from to the function.
        
        Ex. 
            H_null: the standardized returns of a stock, x, are distributed Normal(0, 1)
            H_alt: the standardized returns of a stock, x, are not distributed Normal(0, 1)
            
            y =  scipy.stats.norm(loc=0, scale=1).rvs(size=10000), if using Python
            kstest(x, y)
            
    Two-sample K-S test:
        Just pass x and y to the function.
        
        H_null: the distribution breaking pressure of steel from plants 1 and 2 are the same
        H_alt: the distribution of breaking pressures are not the same
        
        kstest(x, y)
    '''
    # import the 2-sample K-S test from scipy.stats
    from scipy.stats import ks_2samp
    # return the test statistic and the p-value in a tuple
    return ks_2samp(x, y)[0], ks_2samp(x, y)[1]

###
# Unit testing
###
import scipy.stats

# generate data from 2 different distributions with the same location and scale
x = scipy.stats.gamma(5, 10).rvs(size=100)
y = scipy.stats.norm(5, 10).rvs(size=100)

# generate data from the same distribution
a = scipy.stats.laplace(5, 10).rvs(size=100)
b = scipy.stats.laplace(5, 10).rvs(size=100)

# generate data from very similar, but different distributions
c = scipy.stats.expon(7).rvs(size=100)
d = scipy.stats.expon(7.1).rvs(size=100)

assert kstest(x, y)[1] < 0.05 # H_null should be rejected at alpha = 0.05
assert kstest(a, b)[1] > 0.05 # H_null should NOT be rejected at alpha = 0.05
assert kstest(x, y)[1] < 0.05 # H_null should not be rejected at alpha = 0.05