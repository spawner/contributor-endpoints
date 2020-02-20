# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:42:04 2020

@author: jkcle
"""

def minimum(x, 
            axis = None):
    '''
    Returns the minimum value in an array.
    
    An example query is "What is the smallest value in x?"
    
    Parameters (copied from numpy documentation)  
    x : array_like
        Input data.
        
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes, 
        instead of a single axis or all the axes as before.
    '''
    # import the necessary function from numpy
    from numpy import amin
    # return the answer
    return amin(x, axis=axis)



def maximum(x, 
            axis = None):
    '''
    Returns the maximum value in an array.
    
    An example query is "What is the largest value in x?"
    
    Parameters (copied from numpy documentation)  
    x : array_like
        Input data.
        
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        If this is a tuple of ints, the minimum is selected over multiple axes, 
        instead of a single axis or all the axes as before.
    '''
    # import the necessary function from numpy
    from numpy import amax
    # return the answer
    return amax(x, axis=axis)



def range(x, 
          axis = None):
    '''
    Returns the range of the input data; a measure of spread of the data.
    
    An example query is "what is the range of x?"
    
    See minimum or maximum for more info.
    '''
    return maximum(x, axis) - minimum(x, axis)



def skewness(x, 
             axis=0, 
             bias=True, 
             nan_policy='omit'):
    '''
    Returns the sample skewness (sample third central moment / sample standard
    deviation^3). Negative skewness indicates a heavier left-tail. Positive skewness
    indicates a heavier right-tail. Close to 0 suggests symmetric data.
    
    An example query is "what is the skewness of x?"
    
    Parameters (copied from scipy.stats documentation)
    x: ndarray
        Input array.
    axis: int or None, optional
        Axis along which skewness is calculated. Default is 0. If None, compute 
        over the whole array a.
    bias: bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy: {‘propagate’, ‘raise’, ‘omit’}, optional
        Defines how to handle when input contains nan. The following options are 
        available (default is ‘propagate’):
                ‘propagate’: returns nan
                ‘raise’: throws an error
                ‘omit’: performs the calculations ignoring nan values
    '''
    # import the necessary function from scipy.stats
    from scipy.stats import skew
    # return the skewness
    return skew(x, axis=axis, bias=bias, nan_policy=nan_policy)



def kurtosis(x, 
             axis=0, 
             fisher=False, 
             bias=True, 
             nan_policy='omit'):
    '''
    Returns the sample kurtosis (sample fourth central moment / sample variance^2). 
    Pearson's statistic is the default, ie kurtosis of a normal is 3.0. If fisher
    is True, 3 is subtracted from the statistic, so it can be thought of kurtosis
    in excess of a normal distribution.
    
    Kurtosis larger than the appropriate metric than the normal means the tails are
    heavier than the normal. Kurtosis smaller than the normal value means lighter
    tails than the normal distribution.
    
    An example query is "what is the kurtosis of x?"
    
    Parameters (copied from scipy.stats documentation)
    x: ndarray
        Input array.
    axis: int or None, optional
        Axis along which skewness is calculated. Default is 0. If None, compute 
        over the whole array a.
    fisher: bool, optional (default = False)
        If True, Fisher’s definition is used (normal ==> 0.0). If False, 
        Pearson’s definition is used (normal ==> 3.0).
    bias: bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy: {‘propagate’, ‘raise’, ‘omit’}, optional
        Defines how to handle when input contains nan. The following options are 
        available (default is ‘propagate’):
                ‘propagate’: returns nan
                ‘raise’: throws an error
                ‘omit’: performs the calculations ignoring nan values
    '''
    # import the necessary function from scipy.stats
    from scipy.stats import kurtosis as kurt
    # return the kurtosis
    return kurt(x, axis=axis, fisher= fisher, bias=bias, nan_policy=nan_policy)



def quantiles(x, q, axis=None, out=None, overwrite_input=False, 
              interpolation='linear', keepdims=False):
    '''
    Returns quantiles.
    
    An example query is "What is the 0.25 and 0.75 quantiles?"
    
    Parameters (copied from scipy.stats)	

    x : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between 0 
        and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is to
        compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the 
        same shape and buffer length as the expected output, but the type (of 
        the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array a to be modified by intermediate 
        calculations, to save memory. In this case, the contents of the input 
        a after this function completes is undefined.
    interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points i < j:
            linear: i + (j - i) * fraction, where fraction is the fractional
                part of the index surrounded by i and j.
            lower: i.
            higher: j.
            nearest: i or j, whichever is nearest.
            midpoint: (i + j) / 2.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the 
        result as dimensions with size one. With this option, the result will broadcast correctly against the original array a.
    '''
    # import the necessary function from numpy
    from numpy import quantile
    # return the quantiles
    return quantile(x, q, axis=axis, out=out, overwrite_input=overwrite_input, 
                    interpolation=interpolation, keepdims=keepdims)



def iqr(x, axis=None, out=None, overwrite_input=False, 
              interpolation='linear', keepdims=False):
    '''
    Returns the interquartile range, a measure of spread. It is the 75th 
    minus the 25th quantiles
    
    An example query is "what is the interquartile range of x?"
    
    See quantiles for more info.
    '''
    # calculate the 25th and 75th quantiles
    both_quantiles = quantiles(x, q=(0.25, 0.75), axis=None, out=None, 
                               overwrite_input=False, interpolation='linear', 
                               keepdims=False)
    # return the interquartile range
    return both_quantiles[1] - both_quantiles[0]