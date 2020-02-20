# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:23:40 2020

@author: jkcle
"""



def one_sample_t_test(x, 
                      null_mu = 0, 
                      axis = 0,
                      greater = None):
    '''
    Tests if the mean of x is equal to null_mu using a t-test.
    
    An example query is "Is the average of x equal to 0"
    
    Parameters (some options are copied from scipy documentation, which this
    function uses to perform the tests)
        x: array_like
            An array of numeric data
        null_mu: float, int, optional (default = 0)
            We are testing if it is reasonable to say the population mean that
            the sample x comes from equals this  
        axis: int or None, optional (default = 0)
            Axis can equal None (ravel array first), or an integer (the axis 
            over which to operate on x).
        greater: {None, True, False}, optional (default = None)
            None: Performs two-tailed t-test
            True: Performs upper-tail t-test
            False: Performs lower-tail t-test
    '''
    # import the necessary functions from scipy.stats and numpy
    from scipy.stats import ttest_1samp
    from numpy import count_nonzero, isnan
    
    test_result = ttest_1samp(x, null_mu, axis=axis)
    tstat = test_result[0]
    pvalue = test_result[1]
    degrees_freedom = count_nonzero(~isnan(x)) - 1
    
    # if this is a two-tail test, our job is done
    if greater == None:
        # return the test result    
        return tstat, pvalue, degrees_freedom 
    
    # if this is an upper tail test, we need to adjust the p-value
    elif greater:
        if tstat > 0:
            pvalue = pvalue / 2
        else:
            pvalue = 1 - pvalue / 2 
    
    # if this is an lower tail test, we need to adjust the p-value
    else:
        if tstat < 0:
            pvalue = pvalue / 2
        else:
            pvalue = 1 - pvalue / 2
    # return the test result    
    return tstat, pvalue, degrees_freedom



def difference_in_means(x,                  
                        y,                  
                        axis = 0,              
                        equal_var = False,    
                        nan_policy = 'omit',
                        greater = None):
    '''
    Performs a difference in means t-test, which tests if the sample means are
    different from one another, mean x is greater than mean y, or mean x is 
    less than mean y. 
    
    An example query is "Does average x equal average y" for the vanilla test. 
    
    An example query is "Is average 'x' bigger than average 'y'" for the 
    upper-tail test.
    
    An example query is "Is average 'x' smaller than average 'y'" for the 
    lower-tail test.
    
    Parameters (some options are copied from scipy documentation, which this
    function uses to perform the tests)
        x, y: array_like
            The arrays must have the same shape, except in the dimension 
            corresponding to axis (the first, by default).
        axis: int or None, optional (default = 0)
            Axis along which to compute test. If None, compute over the whole 
            arrays, x, and y.
        equal_var: bool, optional (default = False)
            If True (default), perform a standard independent 2 sample test 
            that assumes equal population variances [1]. If False, 
            perform Welch’s t-test, which does not assume equal population 
            variance [2].
        nan_policy: {‘propagate’, ‘raise’, ‘omit’}, optional (default = 'omit')
            Defines how to handle when input contains nan. The following 
            options are available (default is ‘propagate’):
                ‘propagate’: returns nan
                ‘raise’: throws an error
                ‘omit’: performs the calculations ignoring nan values
        greater: {None, True, False}, optional (default = None)
            None: Performs two-tailed difference of means t-test
            True: Performs upper-tail difference of means t-test
            False: Performs lower-tail difference of means t-test
    '''
    # import the necessary functions from scipy.stats and also numpy
    from scipy.stats import ttest_ind
    from numpy import count_nonzero, isnan
    
    # Compute the sample size of each input
    samp_size_x = count_nonzero(~isnan(x))
    samp_size_y = count_nonzero(~isnan(y))
    
    ###
    # Calculate the degrees of freedom based on if equal variance is assumed
    # or not
    ###
    # calculate if sample variances are assumed to be equal
    if equal_var:
        degrees_freedom = samp_size_x + samp_size_y - 2
    # calculate if sample variances are NOT assumed to be equal using the 
    # Welch–Satterthwaite equation
    else:
        from scipy.stats import tvar
        degrees_freedom = (((tvar(x)/samp_size_x + tvar(y)/samp_size_y)**2) / 
              ((tvar(x)/samp_size_x)**2 / (samp_size_x - 1) + 
               (tvar(y)/samp_size_y)**2 / (samp_size_y - 1)))
        
    # perform the test for mean x = mean y as the null hypothesis 
    # and mean x != mean y as the alternative hypothesis(two-tail)
    if greater == None:
        # calculate the two-tailed t-test
        test_result = ttest_ind(x, y, axis=axis, equal_var=equal_var, nan_policy=nan_policy)
        # grab the t-statistic and p-value
        tstat = test_result[0]
        pvalue = test_result[1]
    
    # perform the test for mean x <= mean y as the null hypothesis 
    # and mean x > mean y as the alternative hypothesis (upper-tail)    
    elif greater:
        # calculate the two-tailed t-test
        test_result = ttest_ind(x, y, axis=axis, equal_var=equal_var, nan_policy=nan_policy)
        # grab the t-statistic
        tstat = test_result[0]
        # calculate the appropriate p-value for an upper-tail t-test
        if tstat > 0:
            pvalue = test_result[1] / 2
        else:
            pvalue = 1 - test_result[1] / 2
        
    # perform the test for mean x >= mean y as the null hypothesis 
    # and mean x < mean y as the alternative hypothesis (lower-tail)    
    else:
        # calculate the two-tailed t-test
        test_result = ttest_ind(x, y, axis=axis, equal_var=equal_var, nan_policy=nan_policy)      
        # grab the t-statistic
        tstat = test_result[0]
        # calculate the appropriate p-value for an lower-tail t-test
        if tstat < 0:
            pvalue = test_result[1] / 2
        else:
            pvalue = 1 - test_result[1] / 2 
            
    # return the test result    
    return tstat, pvalue, degrees_freedom