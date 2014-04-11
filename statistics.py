from numpy import arange,correlate,newaxis,dot,sort,int,floor,ceil,interp,isnan,ones
from numpy import hanning,hamming,bartlett,blackman,r_,convolve
from numpy.random import randint
from scipy.stats import pearsonr,spearmanr,kendalltau

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




