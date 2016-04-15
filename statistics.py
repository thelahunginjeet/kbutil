"""
@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2013, Kevin S. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.

3. Neither the name of the University of Connecticut  nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from numpy import arange,correlate,newaxis,dot,sort,int,floor
from numpy import ceil,interp,isnan,ones,asarray,argsort,zeros,linspace
from numpy import hanning,hamming,bartlett,blackman,r_,convolve
from numpy.random import randint
from scipy.stats import pearsonr,spearmanr,kendalltau

def spearman_footrule_distance(s,t):
    """
    Computes the Spearman footrule distance between two full lists of ranks:

        F(s,t) = (2/|S|^2)*sum[ |s(i) - t(i)| ],

    the normalized sum over all elements in a set of the absolute difference between
    the rank according to s and t.  As defined, 0 <= F(s,t) <= 1.

    If s,t are *not* full, this function should not be used. s,t should be array-like
    (lists are OK).
    """
    # check that size of intersection = size of s,t?
    assert len(s) == len(t)
    return (2.0/len(s)**2)*sum(abs(asarray(s) - asarray(t)))


def standardize(X,stype='row'):
    """
    Standardizes (mean subtraction + conversion to unit variance) the array X,
    according to either rows or columns.
    """
    if stype == 'row':
        return (X - X.mean(axis=1)[:,newaxis])/X.std(axis=1)[:,newaxis]
    return (X - X.mean(axis=0)[newaxis,:])/X.std(axis=0)[newaxis,:]


def autocovariance(X,norm=False):
    """
    Computes the autocovariance function of X:
        phi(T) = 1/(N-T) sum_i X'(t_i)*X'(t_i - T)
    As T gets closer and closer to N, the autocovariance becomes less and less well
    estimated, since the number of valid samples goes down.

    This version computes phi(T) at all T (0,...N-1).

    If norm = True, the autocorrelation (phi(T)/phi(0)), rather than the
    bare autocovariance is returned.
    """
    Xp = X - X.mean()
    phi = (1.0/(len(X) - arange(0,len(X))))*correlate(Xp,Xp,"full")[len(X)-1:]
    if norm:
        return phi/phi[0]
    return phi


def corrmatrix(X,Y):
    """
    For two data matrices X (M x T) and Y (N x T), corrmatrix(X,Y) computes the M x N set
    of pearson correlation coefficients between all the rows of X and the rows of Y.  X and
    Y must have the same column dimension for this to work.
    """
    assert X.shape[1] == Y.shape[1]
    # row standardize X and Y
    X = (X - X.mean(axis=1)[:,newaxis])/X.std(axis=1)[:,newaxis]
    Y = (Y - Y.mean(axis=1)[:,newaxis])/Y.std(axis=1)[:,newaxis]
    return dot(X,Y.T)/X.shape[1]


def empirical_ci(x,alpha=0.05):
    """
    Computes an empirial (alpha/2, 1-alpha/2) confidence interval for the distributional data
    in x.  Returns a tuple (lb,ub) which are the lower and upper bounds for the coverage
    interval.
    """
    assert alpha > 0.0
    xtilde = sort(x)
    xl = (alpha/2)*len(x)
    xu = (1.0 - alpha/2)*len(x)
    l1 = int(floor(xl))
    l2 = int(ceil(xl))
    u1 = int(floor(xu))
    u2 = int(ceil(xu))
    lb = interp(xl,[l1,l2],[xtilde[l1],xtilde[l2]])
    up = interp(xu,[u1,u2],[xtilde[u1],xtilde[u2]])
    return lb,ub


def bootstrap_correlation(x,y,cType='pearson',p=0.05,N=5000):
    """
    Computes a simple bootstrap CI on simple correlation coefficients.  The CI is
    computed using the interval method.  This won't work properly if x or y are time-series
    with significant autocorrelation.  You need to downsample or use a more complicated
    bootstrap that preserves that structure.

    Parameters:
    ------------
    x,y   : (1D) lists or arrays of data
    cType : string, optional
            should be 'pearson', 'spearman', or 'kendall'
    p     : float, optional
            the coverage interval will cover 1-p of the cumulative distribution
    N     : integer, optional
            number of bootstrap replications

    Returns:
    ------------
    rho,rL,rU : r(x,y) and lower, upper bounds for 1-p CI

    """
    corrTable = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}
    try:
        corr = corrTable[cType]
    except KeyError:
        # default to pearson
        print 'WARNING: Correlation type not supported. Defaulting to Pearson.'
        corr = pearsonr
    rho = corr(x,y)[0]
    rhobs = list()
    nSamp = len(x)
    iB = 0
    while iB < N:
        randx = randint(nSamp,size=(nSamp,))
        val = corr(x[randx],y[randx])[0]
        if not isnan(val):
            rhobs.append(val)
            iB = iB + 1
    # obtain the coverage interval
    rL,rU = empirical_ci(rhobs,alpha=p)
    return rho,rL,rU


def smooth(x,wlen=11,window='flat'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters:
    ------------
     x      : array type, required
              the input signal to smoothed
     wlen   : integer, odd, optional
              the size of the smoothing window; should be an odd integer
     window : string, optional
              the type of window. allowed choices are:
                 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
              'flat' produces a moving average.

    Returns:
    ------------
    y : array type
        the smoothed signal

    Example:
    ------------
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    Stolen from the scipy cookbook and modified.
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < wlen:
        raise ValueError, "Input vector needs to be bigger than window size."
    if wlen < 3:
        return x
    # dictionary of allowed windows
    winTable = {'flat' : ones, 'hanning' : hanning, 'hamming' : hamming, 'bartlett' : bartlett, 'blackman' : blackman}
    try:
        w = winTable[window]
    except KeyError:
        # default to flat
        print 'WARINING: Unsupported window type. Defaulting to \'flat\'.'
        w = winTable['flat']
    w = w(wlen)
    # wrap ends around and convolve with the scaled window
    s=r_[2*x[0]-x[wlen-1::-1],x,2*x[-1]-x[-1:-wlen:-1]]
    y=convolve(w/w.sum(),s,mode='same')
    # return vector with bogus ends trimmed
    return y[wlen:-wlen+1]


def discrete_frequency_calculator(intList):
    '''
    Accepts a list of integer values, returning two numpy arrays of values and frequencies.
    The output frequencies are normalized so that sum(Pn) = 1.0.  Input values are requried
    to be integer; no binning is performed for floating point values.

    INPUT
    ------
    intList: list of integers, required
             input list of data of integer type

    OUTPUT
    ------
    n,Pn : two lists of values and freq(values)
           frequencies are normalized so that sum(Pn) = 1.0, and the two returned
           arrays are in 1-1 correspondence, sorted in order of increasing n

    '''
    assert all([type(x) == int for x in intList])
    freq = {}
    n = len(intList)
    for k in intList:
        pkinc = 1.0/n
        if not freq.has_key(k):
            freq[k] = pkinc
        else:
            freq[k] += pkinc
    # sorting
    indx = argsort(freq.keys())
    return asarray([freq.keys()[x] for x in indx]),asarray([freq.values()[x] for x in indx])


def cdist_sparse(data):
    '''
    Computes the (empirical) cumulative distribution F(x) of data, defined as:

                F(x) = int_a^b p(x) dx

    or as a sum.  This function only computes F(x) at the data values; to
    get a "stairstep" plot of the cdf use cdist_dense.

    INPUT
    ------
    data  : array-like, required
            input data

    OUTPUT
    ------
    dsort : array
            sorted data (increasing)

    cdf : array
          cdf, evaluated at the values in dsort
    '''
    # sort the data
    data_sorted = sort(data)
    # calculate the proportional values of samples
    p = 1. * arange(len(data)) / (len(data) - 1)
    return data_sorted,p


def cdist_dense(data,limits,npts=1024):
    '''
    Computes the (empirical) cumulative distribution F(x) of samples in data, over
    a specified range and number of support points. F(x) is defined as:

                F(x) = int_a^b p(x) dx

    or as a sum.

    INPUT
    ------
    data   : array-like, required
             input data

    limits : array-like, required
             cdf is computed for npts values between limits[0] and limits[1]

    npts   : int, optional
             number of support points to evaluate cdf

    OUTPUT
    ------
    x : array
        support for the cdf

    cdf : array
          cdf, evaluated at the values x
    '''
    # sort the data
    data_sorted = sort(data)
    x = linspace(limits[0],limits[1],npts)
    Fofx = zeros(len(data))
    for i in xrange(0,len(x)):
        Fofx[i] = sum(data_sorted <= x)
    return x,1.0*Fofx/len(x)
