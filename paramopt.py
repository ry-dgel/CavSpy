import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import scipy.optimize as opt
import os

dbl_array = ndpointer(ctypes.c_double)
dbl = ctypes.c_double
intg = ctypes.c_int

if os.name == "nt":
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),"minPar.dll"))
else:
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__),"minPar.so"))

lib.minDist.argtypes = [dbl, dbl,             #x, y
                        dbl_array, dbl_array, #Es, Ts
                        dbl, dbl,             #SigmaX, SigmaY
                        intg]                 #Len(Es)
lib.minDist.restype = dbl

lib.minDists.argtypes = [dbl_array, dbl_array, #Xs, Ys
                         dbl_array, dbl_array, #Es, Ts
                         dbl_array,            #Output
                         dbl,dbl,              #SigmaX, SigmaY
                         intg,intg]            #Len(Xs), #Len(Es)
lib.minDists.restype = intg

lib.minLength.argtypes = [dbl, dbl,             #x, y
                          dbl_array, dbl_array, #Es, Ts
                          dbl_array,            #Dls
                          dbl, dbl,             #SigmaX, SigmaY
                          intg]                 #Len(Es)
lib.minLength.restype = dbl


lib.minLengths.argtypes = [dbl_array, dbl_array, #Xs, Ys
                           dbl_array, dbl_array, #Es, Ts
                           dbl_array, dbl_array, #Dls, Output
                           dbl, dbl,             #SigmaX, SigmaY
                           intg, intg]           #Len(Xs), #Len(Es)
lib.minLengths.restype = intg

lib.minLengthHist.argtypes = [dbl, dbl,             #x, y
                              dbl_array, dbl_array, #Es, Ts
                              dbl_array,            #Dls
                              dbl, dbl,             #SigmaX, SigmaY
                              intg,                 #Len(Es)
                              dbl, dbl]             #prev, limit    
lib.minLengthHist.restype = dbl

lib.minLengthsHist.argtypes = [dbl_array, dbl_array, #Xs, Ys
                               dbl_array, dbl_array, #Es, Ts
                               dbl_array, dbl_array, #Dls, Output
                               dbl, dbl,             #SigmaX, SigmaY
                               intg, intg,           #Len(Xs), #Len(Es)
                               dbl]                  #limit
lib.minLengthsHist.restype = intg

def make_func(pairs, es, ts, sigmae, sigmat):
    xs = np.copy(pairs[0])
    ys = np.copy(pairs[1])
    es = np.copy(es)
    ts = np.copy(ts)
    N = len(xs)
    M = len(es)

    def func(p):
        output = np.zeros(N)
        lib.minDists(xs, ys, p[0] * es + p[1], p[2] * ts,
                     output, sigmae, sigmat, N, M)
        return np.sum(np.power(output,2))/len(xs)

    return func

def param_opt(func, ps0, callback):
    print("Starting Optimization...")
    result = opt.minimize(func, ps0, method='BFGS', options={'disp':True, 'maxiter':1000}, 
                          callback=callback)
    if not result.success:
        print("Something has gone Awry")
        print(result.message)
    return result

def get_lengths(points, es, ts, sigmae, sigmat, ls, p, hist=None):
    new_es = p[0] * es + p[1]
    new_ts = p[2] * ts
    if hist is None:
        return min_lengths(points, new_es, new_ts, sigmae, sigmat, ls)
    else:
        return min_lengths_hist(points, new_es, new_ts, sigmae, sigmat, ls, hist)
    
######################
# C-Library Wrappers #
######################
def min_dists(points,es,ts,sigmae,sigmat):
    xs = points[0]
    ys = points[1]
    N = len(xs)
    M = len(es)

    output = np.zeros(N,dtype=np.float64)
    res = lib.minDists(xs,ys,es,ts,output,sigmae,sigmat,N,M)
    return output


def min_lengths(points, es, ts, sigmae, sigmat, dLs):
    xs = points[0]
    ys = points[1]
    N = len(xs)
    M = len(es)

    output = np.zeros(N,dtype=np.float64)
    res = lib.minLengths(xs,ys,es,ts,dLs,output,sigmae,sigmat,N,M)
    return output

def min_lengths_hist(points, es, ts, sigmae, sigmat, dLs, limit):
    xs = points[0]
    ys = points[1]
    N = len(xs)
    M = len(es)

    output = np.zeros(N,dtype=np.float64)
    res = lib.minLengthsHist(xs,ys,es,ts,dLs,output,sigmae,sigmat,N,M,limit)
    return output
