from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time
from collections.abc import Iterable     # abstract base classes for containers :)

import sys
sys.path.append("./clustering/lib/python3.7/site-packages")



def __plot(axis, *args, **kwargs):
    assert "plottype" in kwargs
    plottype = None
    if "plottype" in kwargs:
        plottype = kwargs["plottype"]        

    if "alphas" in kwargs:
        alphas  = kwargs["alphas"]
        assert isinstance(alphas, Iterable)
    else:                 
        alphas = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    if "sizes" in kwargs: 
        sizes  = kwargs["sizes"]
        assert isinstance(sizes, Iterable)
    else:                
        sizes = [40, 40, 40,  40, 40, 40, 40, 40]
    # https://matplotlib.org/3.1.1/api/markers_api.html
    if "markers" in kwargs: 
        markers  = kwargs["markers"]
        assert isinstance(markers, Iterable)
    else:       # MORE: 1, 2, 3, 4, p pentagon, P plus, D and d for diamonds           
        markers = [".", "+", "x", "*", "v", "^", "<", ">" ] 
    if "colors" in kwargs: 
        colors  = kwargs["colors"]
        assert isinstance(colors, Iterable)
    else:                  
        colors  = ["g", "r", "b", "m", "y", "t", "p"]
    bins = None    # 10 default of m
    if "bins" in kwargs:
        assert plottype == "histogram", "histogram needs bins argument, and nothing else"
        plottype = kwargs["plottype"]        
    if "vertexlabels" in kwargs:
        vertexlabels = kwargs["vertexlabels"] # array for x, y
        assert isinstance(vertexlabels, Iterable)
    else:
        vertexlabels = None

    for i, arg in enumerate(args):    # (graph,graph,...) = ((x,y),(x,y),...) (2D case)

        if plottype == "versus_2d":
            axis.scatter(arg[:,0],arg[:,1],
                         marker=markers[i], c=colors[i], s=sizes[i], alpha=alphas[i])
            # vertex label
            if vertexlabels != None:
                for vertex,vertexlabel in enumerate(vertexlabels[i]):
                    axis.text(arg[vertex,0], arg[vertex,1], vertexlabel, size=20)

        elif plottype == "versus_3d": # versus_3d
            axis.scatter(arg[:,0], arg[:,1], arg[:,2],
                         marker=markers[i], c=colors[i], s=sizes[i], alpha=alphas[i])
            # vertex label
            if vertexlabels != None:
                for vertex,vertexlabel in enumerate(vertexlabels[i]):
                    axis.text(arg[vertex,0], arg[vertex,1], arg[vertex,2], vertexlabel, size=20, zorder=1)

        elif plottype == "histogram":
            axis.hist(arg, color=colors[i], bins=bins)

        else:
            assert False, "Provide a valid named argument plottype to me!!"

    if "labels" in kwargs:
        labels = kwargs["labels"] # array for x, y
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        if plottype == "versus_3d":
            axis.set_zlabel(labels[2])
    else: # no labels for axis
        axis.set_xlabel("")
        axis.set_ylabel("")
        if plottype == "versus_3d":
            axis.set_zlabel("")
        
    if "legend" in kwargs:
        legend = kwargs["legend"]

        if len(args)>1:     # multiple graphs / plots?
            assert isinstance(legend, Iterable)
        axis.legend(legend, loc='best',fontsize=8)
    if "xlim" in kwargs:  #  limit for axis?
        minMax = kwargs["xlim"]      # must be [min,max]
        isinstance(minMax, Iterable)
        axis.set_xlim(minMax)
    if "ylim" in kwargs:  #  limit for axis?
        minMax = kwargs["ylim"]      # must be [min,max]
        isinstance(minMax, Iterable)
        axis.set_ylim(minMax)


def plotBegin(nx, ny, filename=""):
    global g_nplot
    global g_fig      # matplot figure
    global g_pdf      # pdf page object to add plots
    global g_filename # current file name if exported to file
    global g_nx
    global g_ny

    if ("g_filename" not in globals()) or (g_filename == None):
        if ".pdf" in filename.lower():    
            from matplotlib.backends.backend_pdf import PdfPages
            g_filename = filename
            g_pdf = PdfPages(filename)
            print("plotBegin(): starting PDF file", g_filename)
        elif ".png" in filename.lower():    
            g_filename = filename
            print("plotBegin(): starting PNG file", g_filename)
        else:
            print("plotBegin(): no file defined ...")
            g_filename = None
    else:
        if filename != "":
            if filename != g_filename:
                print("plotBegin(): filename is", filename,"!= g_filename was", g_filename)
                raise ValueError('You must close the file before new output')
            else:
                print("plotBegin(): New plot in file", g_filename)
        else:
            print("plotBegin(): file already defined:", g_filename)
            raise ValueError('You must close the file before new output')

    g_nplot = 1 # reset  for the first subplot / only plot
    g_nx    = nx
    g_ny    = ny # nx,ny is for 
    g_fig = plt.figure()


def addPlot(*args, **kwargs):
    global g_ax
    global g_nplot   # count the subplot, 1,..
    global g_axis
    global g_fig
    global g_nx
    global g_ny

    plt.style.use('ggplot')
    plt.tight_layout()# let's make good plots
    g_fig.set_size_inches(10,8) 
    
    plottype = "versus_2d"
    if "plottype" in kwargs:
        plottype = kwargs["plottype"]

    samecanvas=False
    if "samecanvas" in kwargs:
        samecanvas = kwargs["samecanvas"]

    if samecanvas:
        assert not g_ax is None
    else: # new window / subplot canvas
        if plottype == "versus_3d":     
            projection="3d"
            g_ax = g_fig.add_subplot(g_nx, g_ny, g_nplot, projection=projection)
        else:
            g_ax = g_fig.add_subplot(g_nx, g_ny, g_nplot )
    
    __plot(g_ax, *args, **kwargs)
    
    if not samecanvas:
        assert( g_nplot <= g_nx*g_ny )
        g_nplot += 1

def plotInSubPlot(*args, **kwargs):
    __plot(g_ax, *args, **kwargs)

def plotShow(): 
    global g_filename
    global g_pdf
    
    if "g_filename" in globals():          
        if ".png" in g_filename.lower():      # png export
            plt.savefig(g_filename) # single file export
            print("plotShow(): Exporting PNG file", g_filename)
        elif ".pdf" in g_filename.lower():    # PDF export
            #CHECK WHETHER PDF EXPORT. THEN, add the plot!
            assert(g_pdf)
            g_pdf.savefig( g_fig )
            print("plotShow(): Extending PDF file", g_filename)
        else:
            raise Exception("plotShow() no png or pdf defined.")

    else:                 # inline canvas
        plt.show()
        print("plotShow(): no filename -> plt.show()")
        
def plotAddText(txt): 
    global g_fig

    plt.text(0.05, 0.95, txt, transform=g_fig.transFigure, size=24)

def plotSavePdf():
    global g_pdf
    global g_filename

    assert(g_pdf)
    assert(g_filename)

    g_pdf.close()
    g_filename = None


