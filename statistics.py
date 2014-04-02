from numpy import arange,correlate,newaxis,dot

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





