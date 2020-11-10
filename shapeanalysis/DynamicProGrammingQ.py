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

# @cuda.jit()
def thomas(x, a, b, c, n):
    c[0] = c[0] / b[0]
    x[0] = x[0] / b[0]
    for i in range(1, n):
        tmp = 1/(b[i] - c[i-1] * a[i])
        c[i] *= tmp
        x[i] = (x[i] - x[i-1] * a[i])*tmp
    for i in range((n-2), -1, -1):
        x[i] -= c[i]*x[i+1]
    return x
# @njit()
# @cuda.jit()
def lookupspline(t, k, dist, len, n):

    t = (n-1) * dist / len
    k = int(np.floor(t))
    k = (k > 0) * k
    k = k + (k > (n-2)) * (n-2-k)

    t = t - k

    return t, k
# @njit()
# @cuda.jit()
def evalspline(t, D, y):
    c = np.zeros((4, 1))
    c[0] = y[0]
    c[1] = D[0]
    c[2] = 3*(y[1]-y[0])-2*D[0]-D[1]
    c[3] = 2*(y[0]-y[1])+D[0]+D[1]

    return t*(t*(t*c[3] + c[2]) + c[1]) + c[0]
# @njit()
# @cuda.jit()
def spline(D, y, n):
    a = np.zeros((n, 1))
    b = np.zeros((n, 1))
    c = np.zeros((n, 1))

    if(n<4):
        a[0] = 0
        b[0] = 2
        c[0] = 1
        D[0] = 3*(y[1]-y[0])

        a[n-1] = 1
        b[n-1] = 2
        c[n-1] = 0
        D[n-1] = 3*(y[n-1]-y[n-2])
    else:
        a[0] = 0
        b[0] = 2
        c[0] = 4
        D[0] = -5*y[0] + 4*y[1] + y[2]

        a[n-1] = 4
        b[n-1] = 2
        c[n-1] = 0
        D[n-1] = 5*y[n-1] - 4*y[n-2] - y[n-3]

    for i in range(1,(n-1)):
        a[i] = 1
        b[i] = 4
        c[i] = 1
        D[i] = 3*(y[i+1]-y[i-1])

    D = thomas(D, a, b, c, n)
    return D


# @njit()
# @jit()
# @cuda.jit()
def CostFn2(q1, q2, q2L, k, l, i, j, n, N, M, lam):
    m = (j - l) / (i - k)

    sqrtm = np.sqrt(m)
    idx = 0
    E = 0
    ip = 0
    for x in range(k, i+1):


        y = (x-k)*m + l + 1
        # fp, ip = modf(y*M/N)
        ip = int(y*M/N)
        fp = y*M/N - ip

        idx = (int)(ip + (fp >= 0.5)) - 1

        for d in range(n):
            tmp = q1[n*x + d] - sqrtm*q2L[n*idx + d]
            E = E + tmp*tmp
    return E / N
# @njit()

# @jit()
def Reshape(p):
    # p = p.T
    q = np.append(p[:, 0], p[:, 1])
    return q
# @jit(nopython = True)
# @cuda.jit()

def DP_Resampling_C(p1, p2, lam, Disp):
    os.chdir(r'C:\Users\HIT\Desktop\facerecognition\shapeanalysic')
    c_func = r"./FaceReg.dll"
    FaceReg = ctypes.cdll.LoadLibrary("FaceReg.dll")
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



def DP_Resampling(p1, p2, lam, Disp):
    N, n = p1.shape
    assert n==2 or n==3, "shape must be nx2 or nx3"
    M = 5 * N
    q2L     = np.zeros((n* M,1))
    D       = np.zeros((2*N,1))
    yy      = np.zeros((N,1))
    tmp     = np.zeros((N,1))
    # print(p1.shape)
    # q1      = p1.reshape((-1, 1), order = 'F')
    q1      = Reshape(p1)
    # print(q1, q1_test)
    # q2      = p2.reshape((-1, 1), order = 'F')
    q2      = Reshape(p2)
    a       = 1.0 / N
    b       = 1.0
    t       = 0
    k       = 0
    for i in range(n):
        for j in range(N):
            tmp[j] = q2[n*j+i]
        D = spline(D, tmp, N)

        for j in range(M):
            t, k = lookupspline(t, k, ((j+1.0) / M) - a, b - a, N)

            q2L[n*j+i] = evalspline(t, D[k:], tmp[k:])
    E = np.zeros((N*N, 1))
    Path = np.zeros((2*N*N, 1))

    E = np.zeros((N*N, 1))
    Path = np.zeros((2*N*N, 1))
    for i in range(N):
        E[N * i + 0] = 1
        E[N * 0 + i] = 1
        Path[N * (N * 0 + i) + 0] = -1
        Path[N * (N * 0 + 0) + i] = -1
        Path[N * (N * 1 + i) + 0] = -1
        Path[N * (N * 1 + 0) + i] = -1
    E[N * 0 + 0] = 0
    for j in range(1, N):
        for i in range(1, N):
            Emin = 100000
            Eidx = 0

            for Num in range(NNBRS):
                k = i - Nbrs[Num][0]
                l = j - Nbrs[Num][1]

                if(k >= 0 and l >= 0):
                    Etmp = E[N * l + k] + CostFn2(q1, q2, q2L, k, l, i, j, n, N, M, lam)
                    # print('test')
                    if(Num == 0 or Etmp < Emin ):
                        Emin = Etmp
                        Eidx = Num

            E[N * j + i] = Emin
            Path[N * (N * 0 + j) + i] = i - Nbrs[Eidx][0]
            Path[N * (N * 1 + j) + i] = j - Nbrs[Eidx][1]


    x = np.zeros((N, 1))
    y = np.zeros((N, 1))
    # y = x[N:-1]
    x[0] = N - 1
    y[0] = N - 1

    cnt = 1
    while(x[cnt - 1] > 0):

        y[cnt] = Path[int(N * (N * 0 + x[cnt - 1]) + y[cnt - 1])]
        x[cnt] = Path[int(N * (N * 1 + x[cnt - 1]) + y[cnt - 1])]
        cnt = cnt + 1


    i = 0
    j = cnt - 1

    x = list(x)
    y = list(y)
    while(i < j):
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]
        i +=1
        j -=1
    x = np.array(x)
    # print()
    y = np.array(y)


    for i in range(N):
        Fmin = 100000
        Fidx = 0
        for j in range(cnt):
            Ftmp = int(np.fabs(i - x[j]))
            if(j == 0 or Ftmp < Fmin):
                Fmin = Ftmp
                Fidx = j
        # print(Fidx)
        if(x[Fidx] == i):
            yy[i] = (y[Fidx] + 1)
        else:
            if(x[Fidx] > i):
                a = x[Fidx] - i
                b = i - x[Fidx - 1]
                yy[i] = (a * (y[Fidx - 1] + 1) + b * (y[Fidx] + 1)) / (a + b)
            else:
                a = i - x[Fidx]
                b = x[Fidx + 1] - i
                yy[i] = (a * (y[Fidx + 1] + 1) + b * (y[Fidx] + 1)) / (a + b)
        yy[i] /= N
    return x, y, yy

# @cuda.jit()
def InnerProd_Q(q1, q2):
    # print(q1.shape)
    n, T = q1.shape

    val = np.trapz(np.sum(q1*q2, axis=1), np.linspace(0, 1, n))
    # plt.plot(np.linspace(0, 1, n), np.sum(q1*q2, axis=1) )
    # plt.show()
    return val
# @cuda.jit()
def curve_to_q_open(p):
    '''
    :param p: nx2 or nx3 point set
    :return:
    '''
    [N, dim] = p.shape

    # diff = p[:,1] - p[:, 0]
    # v = diff * dim
    #
    # v = np.column_stack((v, v))

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
# @cuda.jit(nopython = True)
def ReSampleCurve(X, N):
    '''
    :param X:nx2 or nx3 pointset
    :param N: number of points
    :return: new point set of size of Nx2 or Nx3
    '''
    n, dim = X.shape
    Del = [0]
    # norm = np.linalg.norm((X[1, :] - X[0,:]))
    # print(norm)
    for r in range(n):
        if(r==0):
            continue
        # print(r)
        norm = np.linalg.norm((X[r, :] - X[r-1,:]))
        Del.append(norm)
    cumdel = np.cumsum(Del) / np.sum(Del)
    newdel = np.linspace(0, 1, N)
    Xn = np.zeros((N, dim))
    for j in range(dim):
        f = interp1d(cumdel, X[:, j], kind='linear',fill_value="extrapolate")
        Xn[:, j] = f(newdel)
    return Xn
# @cuda.jit()
def invertGamma(gam):
    N = gam.shape[0]
    # print(N)
    gam = gam.reshape((-1, ))
    x = np.linspace(1,N, N) / N
    # print(x.shape, gam.shape)
    f = interp1d(gam,x,  kind='linear', fill_value="extrapolate")
    gamI = f(x)

    return gamI
# @cuda.jit()
def Distance_of_two_curve(X1, X2, N=100):
    lam = 0
    X1 = ReSampleCurve(X1,N)
    X2 = ReSampleCurve(X2,N)
    # print(np.mean(X1, axis=0))
    # print(X1)
    # visual(X1, X2)
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
        # print(u, v, Ot)
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
# @cuda.jit()
def Group_Action_by_Gamma_Coord(f, gamI):
    N, dim = f.shape
    fn = np.zeros((f.shape))
    for j in range(dim):
        func = interp1d(np.linspace(0, 1, N), f[:, j], kind= 'linear')
        fn[:, j] = func(gamI)
    return fn
# @cuda.jit()
def Distance_of_two_face(feature1, feature2,numoflines, N = 100):
    sum = 0
    for i in range(numoflines):
        line1 = feature1[i]
        line2 = feature2[i]
        dist = Distance_of_two_curve(line1, line2, N)
        sum += dist
    return sum

