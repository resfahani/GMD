import numpy as np
import scipy as sp

from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata


def findfreq(freq, freq_value):
    
    i = (np.abs(freq - freq_value)).argmin()

    return freq[i], i 

def Gen_bins(amin, amax, n):
    # Generate linear bins
    
    return np.linspace(amin, amax, n )

def binning(err, a, offset, binnum, freqindx):
    
    amin = a.min() - offset
    amax = a.max() + offset
    
    d = Gen_bins(amin, amax, binnum)
    
    stdv = np.zeros(len(d)-1)
    
    meanv = np.zeros(len(d)-1)
    bins = np.zeros(len(d)-1)
    
    for j in range(len(d)-1):
        indxbin =  np.where((a>d[j])& (a<d[j+1]))[0]
        
        if freqindx != None:
            values = err[indxbin, freqindx]
        else:
            values = err[indxbin]
        
        stdv[j]  = np.std(values)
        meanv[j] = np.mean(values)
        bins[j] = (d[j] + d[j+1])/2
        
    return bins, stdv, meanv


def Manifold_Interpolation(bt, md):
    
    #Generate Grid in the bottleneck between min and max values
    bt1 = np.linspace(bt[:,0].min(), bt[:,0].max(), 200)
    bt2 = np.linspace(bt[:,1].min(), bt[:,1].max(), 200)

    xi, yi = np.meshgrid(bt1,bt2)

    #interpolate the manifold on the liniear sample grids
    c1M = griddata((bt[:,0], bt[:,1]), md[:,0], (xi, yi), method = 'cubic')
    c2R = griddata((bt[:,0], bt[:,1]), md[:,1], (xi, yi), method = 'cubic')

    c1M = np.round(c1M,2)
    c2R = np.round(c2R,2)
    
    c1M = sp.ndimage.gaussian_filter(c1M, 0.1)
    c2R = sp.ndimage.gaussian_filter(c2R, 0.1)

    return c1M, c2R, bt1, bt2

def Select2D(a, var, interval, mode="linear"):
    
    #Select linearly in 2D
    if mode =="linear":
        return  ((np.where(a < var+interval, 1, 0)) & (np.where(a > var-interval, 1, 0)))

    elif mode =="exp":
    
        return  ((np.where(a < var*interval, 1, 0)) & (np.where(a > var/interval, 1, 0)))


def Manifoldsampling(mfM, mfR, bt1, bt2, model_dec,  M, Minterval, R, Rinterval):
    
    indMbt = Select2D(mfM, M, Minterval)
    indRbt = Select2D(mfR, R, Rinterval, mode = 'exp')
    
    indx1 = indMbt * indRbt
    indx = np.where(indx1 ==1)
    
    nodm1 = np.zeros([len(indx[0]),2])
    nodm1[:,0] = bt1[indx[1]]
    nodm1[:,1] = bt2[indx[0]]

    if len(nodm1[:,0]) ==1:
        cm1 = model_dec.predict(nodm1[0,:][:,np.newaxis].T)
    else:
        cm1 = model_dec.predict(nodm1)
    
    return indx1, indMbt + indRbt, cm1

def Dataselection2D(d, md, M, Minterval, R, Rinterval):
    
    indMdata = Select2D(md[:,0], M, Minterval)
    indRdata = Select2D(md[:,1], R, Rinterval)
    
    indx = indMdata * indRdata
    indx = np.where(indx ==1)[0]
    
    labels = md[indx,:]
    
    y = d[indx,:]
    
    return labels, y

def residualanalysis(data, md, manifoldmaping, Minterval = 0.1, Rinterval = 1 ):
    
    resmatrix = np.empty_like(data)
    
    n = len(md)

    for i in range(n):
        m = np.round(md[i, 0], 1)
        R = np.round(md[i, 1], 1)

        #l, yy = Dataselection2D(data_train, md_train, m, Minterval, R, Rinterval )

        sm1, sm2, cm1 = manifoldmaping(m,R)
        
        obs = data[i,:]
        #dm = np.mean(yy,0)
        gensecnario = np.mean(cm1,0)
        
        resmatrix[i,:] = obs - gensecnario
        
    return resmatrix


