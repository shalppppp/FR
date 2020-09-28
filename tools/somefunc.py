import numpy as np
import sys,os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, vectorize, int64, float64, njit
from scipy.spatial import Delaunay
import re
from sklearn.neighbors import KDTree
from sympy import *
import os, sys
# 平滑，已经不用
def Smooth_KNearest(AllPointList, k, Ratio):
    for i in range(len(AllPointList)):
        point = AllPointList[i]
        SortedDistance = GetSortedDistancesList(AllPointList, point)
        KNearestPointList = AllPointList[SortedDistance[1:k+1,1].astype('int64')]
        AverDistanceOfCurrentPoint = GetAverDistance(KNearestPointList, point)
        KNearestPointListOfNextPoint = AllPointList[SortedDistance[2:k+1,1].astype('int64')]
        NextPoint = KNearestPointList[1]
        AverDistanceOfNextPoint = GetAverDistance(KNearestPointListOfNextPoint, NextPoint)
        Ratio = AverDistanceOfCurrentPoint / AverDistanceOfNextPoint
        if (Ratio > 3.5):
            AllPointList[i] = CoordUpdate(KNearestPointList)
    return AllPointList

# @njit
# 计算欧式距离
def Euclidean(p1, p2):
    # assert len(p1) == 3
    # assert len(p2) == 3
    p1 = np.reshape(p1, (1,3))
    p2 = np.reshape(p2, (1,3))
    # p2 = np.array(p2)
    d = 0.0
    # d = np.linalg.norm(p1, p2)
    # return d
    return np.sqrt(np.sum(np.square(p1 - p2)))
    # for i in range(len(p1)):
    #     d += np.square(p1[i] - p2[i])
    # return np.sqrt(d)


@njit
def GetSortedDistancesList(data, point):
    distancelist = []
    for i in range(len(data)):
        distancelist.append((Euclidean(data[i], point), i))
    distancelist.sort()
    return np.array(distancelist)


@njit
def GetAverDistance(PointList, point):
    SumOfDistance = 0.0
    for i in range(len(PointList)):
        SumOfDistance += Euclidean(PointList[i], point)
    AverDistance = SumOfDistance / len(PointList)
    return AverDistance


@njit
def CoordUpdate(PointList):
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    for i in range(len(PointList)):
        sum_x += PointList[0]
        sum_y += PointList[1]
        sum_z += PointList[2]

    x = sum_x / len(PointList)
    y = sum_y / len(PointList)
    z = sum_z / len(PointList)

    return np.array((x, y, z))


def NosetiptoCoord(points, nosetip):
    nosetip_x = nosetip[0] / 0.006
    nosetip_y = nosetip[1] / 0.006
    max_z = np.max(points[:, 2])
    min_z = np.min(points[:, 2])
    move_z = (max_z + min_z) / 2

    nosetip_z = nosetip[2] / 0.006 + move_z
    return [nosetip_x, nosetip_y, nosetip_z]


def CropFace1(points, nosetip):
    x_range = [nosetip[0] - 100, nosetip[0] + 100]
    y_range = [nosetip[1] - 100, nosetip[1] + 100]
    z_range = [nosetip[2] - 100, nosetip[2] + 100]

    new_ind = np.where((points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1]) & \
                       (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1]) & \
                       (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1]))
    new_points = points[new_ind]
    cropface = []
    for i, point in enumerate(new_points):
        if np.linalg.norm(point - nosetip) < 100:
            # new_points = np.delete(new_points, i, axis=0)
            cropface.append(point)
    cropface = np.array(cropface)
    # print(cropface.shape)
    assert cropface.shape[1] == 3
    # assert new_nosetip.shape[1] == 3
    return cropface


def DelaunayTri(data, stdTol=0.6):
    assert data.shape[1] == 3
    u = data[:, 0]
    v = data[:, 1]

    x = u
    y = v

    z = data[:, 2]
    tri = Delaunay(np.array([u, v]).T)
    edgeLength = [np.sqrt(np.square(np.sum(data[tri.simplices[:, 0], :] - data[tri.simplices[:, 1], :], 1))), \
                  np.sqrt(np.square(np.sum(data[tri.simplices[:, 1], :] - data[tri.simplices[:, 2], :], 1))), \
                  np.sqrt(np.square(np.sum(data[tri.simplices[:, 2], :] - data[tri.simplices[:, 0], :], 1)))]
    triangles = np.array(tri.simplices)
    '''

    if stdTol > 0:
        resolution = np.mean(edgeLength)
        stdeviation = np.std(edgeLength)
        filtLimit = resolution + stdTol*stdeviation
        print(filtLimit)
        bigTriangles = np.where(edgeLength > filtLimit)
        triangles = np.delete(triangles, bigTriangles, axis=0)
        print(triangles.shape)

    refNormal = [0, 0, 1]
    triangleNormals = []
    for i, index in enumerate(triangles):

        pointIndex0 = index[0]
        pointIndex1 = index[1]
        pointIndex2 = index[2]

        point1 = triangles[pointIndex0, :]
        point2 = triangles[pointIndex1, :]
        point3 = triangles[pointIndex2, :]

        vector1 = point2 - point1
        vector2 = point3 - point2

        normal = np.cross(vector1, vector2)
        normal = normal / np.linalg.norm(normal)

        theta = np.arccos(np.dot(refNormal, normal))
        if theta > np.pi / 2:
            normal = normal * (-1)
            a = triangles[i, 2]
            triangles[i, 1] = triangles[1, 0]
            triangles[i, 0] = a

        triangleNormals.append(normal)
    '''
    return triangles


# @njit
def Dijkstra(V, F, nosetip):
    # V = np.row_stack((V,nosetip))
    distance = np.zeros((int(V.shape[0])))
    distance[:] = np.inf
    open = np.zeros((int(V.shape[0])))
    far = np.ones((int(V.shape[0])))
    dead = np.zeros((int(V.shape[0])))

    distance[-1] = 0  # 起始点距离设置为0，其他店设置为inf
    open[-1] = 1  # 起始点状态设置为open
    far[-1] = 0  # 其他点状态设置为far
    while (dead.sum() < V.shape[0]):
        open_ind = np.where(open == 1)  # 将open中的点全部取出来，是下标
        for ind in open_ind[0]:
            # ind = open_ind[i]
            # print("ind :" + str(ind))
            x = V[ind]  # x表示open中的某一个点
            # 找相邻的点

            connect_points_ind = np.where(F == ind)  # 如果当前点在三角面片中的一个顶点， 则取出该行下标
            # 将相邻点的状态改为open
            # print(connect_points_ind)
            # print(F[connect_points_ind[0]])
            # pass
            connect_points = F[connect_points_ind[0]]
            connect_points = np.unique(connect_points)
            # print("connect_points:" + str(connect_points))
            open[connect_points] = 1

            # print(open.sum())
            # print(connect_points)
            # for row_ind in connect_points:
            # 邻域的点
            for conn_ind in connect_points:
                # print("conn_ind" + str(conn_ind))
                # print("x:" + str(x) + "  " + "con:" +str(np.array(V[conn_ind])))
                # print(conn_ind)
                # open[conn_ind] = 1
                x_ = np.reshape(V[conn_ind], (3, 1))
                # print(x_.shape)
                dx_ = distance[ind] + Euclidean(x, x_)
                # print("dx_:"+str(dx_) + "distance:" + str(distance[conn_ind]))
                if (dx_ < distance[conn_ind]):
                    distance[conn_ind] = dx_
        open[open_ind] = 0
        dead[open_ind] = 1
        print("%d / %d" % (dead.sum(0), V.shape[0]))
        # if j== 2:
        #     break
    print("end of dijkstra")
    return distance

# @njit
def compute_curve(res, point):


    x = Symbol("x")
    y = Symbol("y")
    a = res[0]
    b = res[1]
    c = res[2]
    d = res[3]
    e = res[4]
    f = res[5]
    # print("res :"+ str(res))
    sf = a*x*x + b*x*y + c*y*y + d*x + e*y + f
    r_xy = np.array([x, y, sf])
    sfx = diff(r_xy, x)
    sfy = diff(r_xy, y)
    round_x = diff(r_xy, x).subs({x:point[0], y:point[1]})
    # round_y = diff(r_xy, y).subs({x:point[0], y:point[1]})
    round_y = diff(r_xy, y).subs({x:point[0], y:point[1]})
    round_xx = diff(sfx, x).subs({x:point[0], y:point[1]})
    round_yy = diff(sfy, y).subs({x:point[0], y:point[1]})
    round_xy = diff(sfx, y).subs({x:point[0], y:point[1]})
    round_yx = diff(sfy, x).subs({x:point[0], y:point[1]})
    # print("rx:"+ str(round_x))
    # print("ry:" + str(round_y))
    # print("rxx:"+ str(round_xx))
    # print("ryy:" + str(round_yy))
    # print("rxy:" + str(round_xy))
    # print("ryx:" + str(round_yx))
    rx_ry = np.cross(round_x, round_y)
    rx_ry = rx_ry.astype(np.float)
    rx_ry_norm = np.linalg.norm(rx_ry)
    n = rx_ry/ rx_ry_norm
    # print("n:" + str(n))
    # L = np.dot(round_x, n)
    # N = np.dot(round_yy, n)
    # M = np.dot(round_xy, n)
    # E = np.dot(round_x, round_x)
    # F = np.dot(round_x, round_y)
    # G = np.dot(round_y, round_y)
    L = np.dot(round_xx, n)
    M = np.dot(round_xy, n)
    N = np.dot(round_yy, n)
    E = np.dot(round_x, round_x)
    F = np.dot(round_x, round_y)
    G = np.dot(round_y, round_y)


    # print("L:"+ str(L))
    # print("N:" + str(N))
    # print("M:" + str(M))
    # print("E:" + str(E))
    # print("F:" + str(F))
    # print("G:" + str(G))
    GUSS_K = (L*N - M*M) / (E*G - F*F)
    MEAN_H = (E*N - 2*F*M + G*L) / 2*(E*G - F*F)
    # print("GUSS_K:" + str(GUSS_K))
    # print("MEAN_H:" + str(MEAN_H))
    A = MEAN_H
    B = MEAN_H * MEAN_H - GUSS_K
    B = np.float(B)
    # print(type(B))
    # print("A:" + str(A))
    # print("B:" + str(B))
    K_MAX = A + np.sqrt(B)
    K_MIN = A - np.sqrt(B)
    # K_MAX = MEAN_H + np.sqrt(MEAN_H*MEAN_H - GUSS_K)
    # K_MIN = MEAN_H - np.sqrt(MEAN_H*MEAN_H - GUSS_K)
    # print("K_MAX:" + str(K_MAX))
    # print("K_MIN:" + str(K_MIN))
    return GUSS_K, MEAN_H, K_MAX, K_MIN
@njit
def fit_surface(data):
    X = data[:, 0]
    # print("X:" + str(X))
    Y = data[:, 1]
    # print("Y:" + str(Y))
    Z = data[:, 2]
    # print("Z:" + str(Z))
    r = data.shape[0]

    sigma_x = np.sum(X)
    sigma_y = np.sum(Y)
    sigma_z = np.sum(Z)
    sigma_x2 = np.sum(X*X)
    sigma_y2 = np.sum(Y*Y)
    sigma_x3 = np.sum(X*X*X)
    sigma_y3 = np.sum(Y*Y*Y)
    sigma_x4 = np.sum(X*X*X*X)
    sigma_y4 = np.sum(Y*Y*Y*Y)

    sigma_xy = np.sum(X*Y)
    sigma_zx = np.sum(Z*X)
    sigma_zy = np.sum(Z*Y)

    sigma_x2y = np.sum(X*X*Y)
    sigma_xy2 = np.sum(X*Y*Y)


    sigma_x3y = np.sum(X*X*X*Y)
    sigma_x2y2 = np.sum(X*X*Y*Y)
    sigma_x2z = np.sum(X*X*Z)
    sigma_xy3 = np.sum(X*Y*Y*Y)
    sigma_xzy = np.sum(X*Z*Y)
    sigma_y2z = np.sum(Y*Y*Z)
    sigma_zx2 = np.sum(Z*X*X)
    sigma_zy2 = np.sum(Z*Y*Y)
    sigma_zxy = np.sum(Z*X*Y)
    sigma_zx = np.sum(Z*X)
    sigma_zy = np.sum(Z*Y)


    # a = np.array([[sigma_x4, sigma_x2y2, sigma_x3y, sigma_x3, sigma_x2y, sigma_x2],
    #               [sigma_x2y2, sigma_y4, sigma_xy3, sigma_xy2, sigma_y3, sigma_y2],
    #               [sigma_x3y, sigma_xy3, sigma_x2y2, sigma_x2y, sigma_xy2, sigma_xy],
    #               [sigma_x3, sigma_xy2, sigma_x2y, sigma_x2, sigma_xy, sigma_x],
    #               [sigma_x2y, sigma_y3, sigma_xy2, sigma_xy, sigma_y2, sigma_y],
    #               [sigma_x2, sigma_y2, sigma_xy, sigma_x, sigma_y, r]])

    # b = np.array([sigma_zx2, sigma_zy2, sigma_zxy, sigma_zx, sigma_zy, sigma_z])
    a=np.array([[sigma_x4,sigma_x3y,sigma_x2y2,sigma_x3,sigma_x2y,sigma_x2],
               [sigma_x3y,sigma_x2y2,sigma_xy3,sigma_x2y,sigma_xy2,sigma_xy],
               [sigma_x2y2,sigma_xy3,sigma_y4,sigma_xy2,sigma_y3,sigma_y2],
               [sigma_x3,sigma_x2y,sigma_xy2,sigma_x2,sigma_xy,sigma_x],
               [sigma_x2y,sigma_xy2,sigma_y3,sigma_xy,sigma_y2,sigma_y],
               [sigma_x2,sigma_xy,sigma_y2,sigma_x,sigma_y,r]])
    b=np.array([sigma_zx2,sigma_zxy,sigma_zy2,sigma_zx,sigma_zy,sigma_z])

    res= np.linalg.solve(a,b)
    return res
