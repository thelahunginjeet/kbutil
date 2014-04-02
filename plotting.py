import pylab
from numpy import ceil,log2,histogram,min,max,abs,linspace
from scipy.stats import gaussian_kde

def pylab_pretty_plot(lines=10,width=4,size=8,labelsize=20,markersize=10,fontsize=18,usetex=True):
    """Changes pylab plot defaults to get nicer plots - frame size, marker size, etc.
    
    Parameters:
    ------------    
    lines      : linewidth
    width      : width of framelines and tickmarks
    size       : tick mark length
    labelsize  : size of ticklabels
    markersize : size of plotting markers
    fontsize   : size of font for labels and legend
    usetex     : use latex for labels/text?
    """
    pylab.rc("lines",linewidth=lines)
    pylab.rc("lines",markeredgewidth=size/3)
    pylab.rc("lines",markersize=markersize)
    pylab.rc("ytick",labelsize=labelsize)
    pylab.rc("ytick.major",pad=size)
    pylab.rc("ytick.minor",pad=size)
    pylab.rc("ytick.major",size=size*1.8)
    pylab.rc("ytick.minor",size=size)
    pylab.rc("xtick",labelsize=labelsize)
    pylab.rc("xtick.major",pad=size)
    pylab.rc("xtick.minor",pad=size)
    pylab.rc("xtick.major",size=size*1.8)
    pylab.rc("xtick.minor",size=size)
    pylab.rc("axes",linewidth=width)
    pylab.rc("legend",fontsize=18)
    pylab.rc("text",usetex=usetex)
    pylab.rc("font",size=18)


def pylab_hist_plus_kde(x,barcolor='k',linecolor='r',nbins=None):
    """
    Plots a histogram (bar plot) with an overlaid kernel density estimate of the distribution.
    Returns the barplot for further manipulation (label modification, limits, etc.)
    
    Parameters:
    -------------
    x         : input data (numpy array or list)
    barcolor  : color for barplot
    linecolor : kernel estimate color
    nbins     : number of histogram bins; if nbins=None, nbins defaults to 1 + ceil(log2(len(x)))
    """
    if nbins is None:
        nbins = 1 + ceil(log2(len(x)))

    ax = pylab.gca(frameon=False)

    # make the bar plot
    dens,bin_edges = pylab.histogram(x,bins=nbins,density=True)
    bin_edges = bin_edges[0:-1]
    # prevents overlapping bars
    barwidth = 0.9*(bin_edges[1] - bin_edges[0])
    barplot = ax.bar(bin_edges,dens,color=barcolor,width=barwidth)

    # kde on top
    kde = gaussian_kde(x)
    lpoint = min(x) - 0.025*abs(min(x))
    rpoint = max(x) + 0.025*abs(max(x))
    support = linspace(lpoint,rpoint,256)
    mPDF = kde(support)
    ax.plot(support,mPDF,color=linecolor,lw=4)

    # draw the x-axis
    ax.plot([lpoint,rpoint],[0.0,0.0],color='k',lw=4)

    # pretty things up
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().set_visible(False)
    ax.set_xlim([lpoint,rpoint])
    ax.set_ylim([-0.01,1.15*max(dens)])

    return ax

