# -*- coding: utf-8 -*-
# @Author  : 作者
# @Time    : 2020/10/15 10:58
# @Function: 
import numpy as np
from math import *
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from tqdm import tqdm
# from DynamicProGrammingQ import *
# from numba import jit, vectorize, int64, float64, njit
from numba import jit, vectorize, int64, float64, njit, cuda
# from tools.utils import visual
from time import time
from ctypes import *
import ctypes
import os
NNBRS = 23
Nbrs = [
	[1, 1],
	[1, 2 ],
	[ 2, 1 ],
	[ 2, 3 ],
	[ 3, 2 ],
	[ 1, 3 ],
	[ 3, 1 ],
	[ 1, 4 ],
	[ 3, 4 ],
	[ 4, 3 ],
	[ 4, 1 ],
	[ 1, 5 ],
	[ 2, 5 ],
	[ 3, 5 ],
	[ 4, 5 ],
	[ 5, 4 ],
	[ 5, 3 ],
	[ 5, 2 ],
	[ 5, 1 ],
	[ 1, 6 ],
	[ 5, 6 ],
	[ 6, 5 ],
	[ 6, 1 ]]

def Reshape(p):
    q = np.append(p[:, 0], p[:, 1])
    return q


def DP_Resampling_C(p1, p2, lam, Disp):
    c_func = r"./FaceReg.dll"
    FaceReg = ctypes.cdll.LoadLibrary(c_func)
    rows, cols = p1.shape


    p1_ptr = p1.reshape((1, -1))
    p2_ptr = p2.reshape((1, -1))

    # p1_ctypes_ptr = p1_ptr.ctypes.data_as(POINTER(c_double))
    # p2_ctypes_ptr = p2_ptr.ctypes.data_as(POINTER(c_double))
    p1_ctypes_ptr = cast(p1_ptr.ctypes.data, POINTER(c_double))
    p2_ctypes_ptr = cast(p2_ptr.ctypes.data, POINTER(c_double))


    FaceReg.DP_Resampling_C.argtypes = POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int, c_int, c_int, c_int
    FaceReg.DP_Resampling_C.restypes = None

    ret = (ctypes.c_double*rows)()
    FaceReg.DP_Resampling_C(ret, p1_ctypes_ptr, p2_ctypes_ptr, cols, rows, lam, Disp)

    return np.array(ret)


def InnerProd_Q(q1, q2):
    n, T = q1.shape
    val = np.trapz(np.sum(q1*q2, axis=1), np.linspace(0, 1, n))
    return val
def curve_to_q_open(p):
    '''
    :param p: nx2 or nx3 point set
    :return:
    '''
    [N, dim] = p.shape
    v = np.zeros((N, dim))
    for i in range(dim):
        v[:, i] = np.gradient(p[:, i], 1 / N)
    len = np.sum(np.sqrt(np.sum(v*v, axis=1))) / N
    v = v / len
    q = np.zeros((N, dim))
    for i in range(N):
        L = np.sqrt(np.linalg.norm(v[i, :]))
        if L > 0.00001:
            q[i, :] = v[i,:] / L
        else:
            q[i, :] = v[i,:] * 0.0001
    return q
def ReSampleCurve(X, N):
    '''
    :param X:nx2 or nx3 pointset
    :param N: number of points
    :return: new point set of size of Nx2 or Nx3
    '''
    n, dim = X.shape
    Del = [0]
    for r in range(n):
        if(r==0):
            continue
        norm = np.linalg.norm((X[r, :] - X[r-1,:]))
        Del.append(norm)
    cumdel = np.cumsum(Del) / np.sum(Del)
    newdel = np.linspace(0, 1, N)
    Xn = np.zeros((N, dim))
    for j in range(dim):
        f = interp1d(cumdel, X[:, j], kind='linear',fill_value="extrapolate")
        Xn[:, j] = f(newdel)
    return Xn
def invertGamma(gam):
    N = gam.shape[0]
    gam = gam.reshape((-1, ))
    x = np.linspace(1,N, N) / N
    f = interp1d(gam,x,  kind='linear', fill_value="extrapolate")
    gamI = f(x)

    return gamI
def Distance_of_two_curve(X1, X2, N=100):
    lam = 0
    X1 = ReSampleCurve(X1,N)
    X2 = ReSampleCurve(X2,N)

    X1 = X1 - np.mean(X1, axis=0)
    X2 = X2 - np.mean(X2, axis=0)

    q1 = curve_to_q_open(X1)
    q2 = curve_to_q_open(X2)

    A = np.dot(q1.T,q2)
    # print(A)
    u,s,v = np.linalg.svd(A)
    detA = np.linalg.det(A)
    # print(detA)
    if detA > 0:
        Ot = np.dot(u.T,v)
    else:
        if(X1.shape[1] == 2):
            Ot = np.dot(u.T,np.array([v[:, 0],-v[:, 1]]))
        else:
            Ot = np.dot(u.T,np.array([v[:, 0],v[:, 1],- v[:, 2]]))
    # print(q2, Ot)
    X2 = np.dot(X2, Ot.T)
    q2 = np.dot(q2, Ot.T)
    # print(q1)
    p1 = q1/np.sqrt(InnerProd_Q(q1,q1))
    p2 = q2/np.sqrt(InnerProd_Q(q2,q2))
    # plt.scatter(p1[:, 0], p1[:, 1], c = 'r')
    # plt.scatter(p2[:, 0], p2[:, 1], c = 'g')
    # plt.show()
    starttime = time()
    # print(p1)
    gam = DP_Resampling_C(p1, p2, 0, 0)
    endtime = time()
    # print("jit:%f"%(endtime - starttime))

    gamI = invertGamma(gam)

    gamI = (gamI - [gamI[0]]) / (gamI[-1] - gamI[0])

    X2n = Group_Action_by_Gamma_Coord(X2, gamI)

    q2n = curve_to_q_open(X2n)
    A = np.dot(q1.T,q2n)
    u,s,v = np.linalg.svd(A)
    detA = np.linalg.det(A)

    if detA > 0:
        Ot = np.dot(u,v.T)
        # print(u, v, Ot)
    else:
        if(X1.shape[1] == 2):
            Ot = np.dot(u,np.array([v[:, 0],-v[:, 1]]).T)
        else:
            Ot = np.dot(u,np.array([v[:, 0],v[:, 1],- v[:, 2]]).T)

    X2n = np.dot(X2n, Ot.T)
    q2n = np.dot(q2n, Ot.T)

    dist = np.arccos(np.sum(np.sum(q1*q2n, axis=0), axis=0) / N)
    # print((q2n*q1).shape)

    return dist
def Group_Action_by_Gamma_Coord(f, gamI):
    N, dim = f.shape
    fn = np.zeros((f.shape))
    for j in range(dim):
        func = interp1d(np.linspace(0, 1, N), f[:, j], kind= 'linear')
        fn[:, j] = func(gamI)
    return fn
def Distance_of_two_face(feature1, feature2,numoflines, N = 100):
    sum = 0
    for i in range(numoflines):
        line1 = feature1[i]
        line2 = feature2[i]
        dist = Distance_of_two_curve(line1, line2, N)
        sum += dist
    return sum

