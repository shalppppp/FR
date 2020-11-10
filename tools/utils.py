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
'''
def PointCloudVisualization(PointCloud, color = 'r', marker = '.'):
    PointCloud = PointCloud.reshape((-1, 3))
    x=PointCloud[:,0]
    y=PointCloud[:,1]
    z=PointCloud[:,2]
    # print(x)
    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    plt.title('point cloud')
    ax.scatter(x,y,z,c=color,marker=marker,s=2,linewidth=0,alpha=1,cmap='spectral')

    #ax.set_facecolor((0,0,0))
    # ax.axis('scaled')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
'''
# 用matplot进行可视化

'''
def CurveVisualization(PointCloud, color = 'r', marker = '.'):
    PointCloud = PointCloud.reshape((-1, 3))
    x=PointCloud[:,0]
    y=PointCloud[:,1]
    z=PointCloud[:,2]
    # print(x)
    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    plt.title('point cloud')
    ax.plot(x,y,z,c=color)

    #ax.set_facecolor((0,0,0))
    # ax.axis('scaled')
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    ax.set_xlim(-50, 50)
    ax.set_ylabel('')
    ax.set_ylim(-50, 50)
    ax.set_zlabel('')
    ax.set_zlim(-50, 50)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.axis("off")
    plt.show()
'''


def CropFace(points, N = 80):
    '''
    :param points:鼻尖点为坐标系原点， 已normal
    :return: -80<x, y, z <80, default N = 80
    '''
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ind = np.where((x > (0- N)) & (x < N) \
                  &(y > (0- N)) & (y <  N) \
                  &(z > (0- N *2 / 3)) & (z < N))
    points = points[ind]

    return points

def knn_kdtree(point_cloud, nsample):

    kdt = KDTree(point_cloud, leaf_size=30)
    n_points = point_cloud.shape[0]
    indices = np.zeros((n_points, nsample))
    _, indices = kdt.query(point_cloud, k = nsample)
    return indices

import re
class WRL:
    def __init__(self, WRLfile):
        self.filename = WRLfile
        file = open(WRLfile, "r+")
        self.data = file
        self.WRLLineData = file.readlines()
    def get_NumOfVertices(self):

        i = 0
        for line in self.WRLLineData:
            if(re.search("vertices", line)):
                NumOfVertices = int(line.split(' ')[1])
                NumOfTriangles = int(line.split(' ')[3])
            if (re.search("Coordinate", line)):
                StartOfCoordinate = i
            i = i + 1
        data_np = []
        return NumOfVertices
    def get_Vertices(self):
        i = 0
        for line in self.WRLLineData:
            if (re.search("Coordinate", line)):
                StartOfCoordinate = i
                # break
            i = i + 1
        data_np = []
        for i in range(StartOfCoordinate + 1, StartOfCoordinate + self.get_NumOfVertices() + 1):
            line = self.WRLLineData[i].split(' ')
            if (re.search("point", self.WRLLineData[i])):

                line_np = [line[3], line[4], line[5].split(',')[0]]
                data_np.append(line_np)
                continue
            x = float(line[1])
            y = float(line[2])
            z = float(line[3].split(',')[0])
            line_np = [x, y, z]
            data_np.append(line_np)
        data_np = np.array(data_np, dtype=float)
        return data_np
    def get_Normal(self):
        i = 0
        for line in self.WRLLineData:
            if (re.search("Normal", line)):
                StartOfNormal = i
                # break
            i = i + 1
        data_np = []
        for i in range(StartOfNormal + 1, StartOfNormal + self.get_NumOfVertices() + 1):
            line = self.WRLLineData[i].split(' ')
            if (re.search("vector", self.WRLLineData[i])):

                line_np = [line[3], line[4], line[5].split(',')[0]]
                data_np.append(line_np)
                continue
            x = float(line[1])
            y = float(line[2])
            z = float(line[3].split(',')[0])
            line_np = [x, y, z]
            data_np.append(line_np)
        data_np = np.array(data_np, dtype=float)
        return data_np

def RandomDownSample(point_cloud, pointnumber):
    length = point_cloud.shape[0]
    index = np.random.choice(length, pointnumber)
    return point_cloud[index]

import vispy
from vispy.scene import visuals

'''
def PointCloudVisualbyVispy(data, size = 1):
    # def vis_show(data, size = 1):
    data = data.reshape((-1, 3))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor= 'white')
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    scatter.set_data(data, edge_color='black', face_color=(1, 1, 1, .5), size=size)

    view.add(scatter)

    view.camera = 'turntable'  # or try 'arcball'

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()
'''

def NormalizePointcloud(point_cloud, nosetip):
    x = point_cloud[:, 0] - nosetip[0]
    y = point_cloud[:, 1] - nosetip[1]
    z = point_cloud[:, 2] - nosetip[2]

    return np.array([x, y, z]).T
'''
def visual(data1, data2):
    import vispy
    from vispy.scene import visuals
    data1 = data1.reshape((-1, 3))
    data2 = data2.reshape((-1, 3))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor= 'white')
    view = canvas.central_widget.add_view()

    org_points = visuals.Markers()
    org_points.set_data(data1, edge_color='black', face_color=(1, 1, 1, .5), size=1)

    inter_points = visuals.Markers()
    inter_points.set_data(data2, edge_color='red', face_color=(1, 1, 1, .5), size=3)
    view.add(org_points)
    view.add(inter_points)
    # utils.PointCloudVisualization(data)
    view.camera = 'arcball'
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()
'''

def Smooth(data, k = 20, d = 8):
    kdt = KDTree(data, leaf_size=30)
    for i,point in enumerate(data):
        point = point.reshape(-1, 1)
        dist,  indices = kdt.query(point.T, 20)

        dist_i = np.where(dist < d)
        num = len(dist_i[1])
        if(num < k):
            near_points = data[indices]
            data[i] = np.mean(near_points, axis=1)

    return data

def smooth1(V, k = 0.95):
    kdt = KDTree(V, leaf_size=30)
    for i,point in enumerate(V):
        point = point.reshape(1, -1)
        dist, indices = kdt.query(point, 20)
        mean_d = np.mean(dist)
        var_d = np.var(dist)
        for d in dist[0]:
            if(d > mean_d + k * var_d):
                V[i] = np.mean(V[indices[0]], axis=0)
                break
    return V

def GetNosetip(data):
    # data = Smooth(data, 30, 8)
    nosetip_i = np.where(data[:, 2] == np.max(data[:, 2]))
    nosetip = data[nosetip_i]

    return nosetip

from tools.PCS import *
def PreProcessing(data, nosetip):
    mask = np.loadtxt(r'../mask002-005.txt')
    normallizedata = data - np.array(nosetip)
    newnosetip = [[0,0,0]]
    smoothdata = smooth(normallizedata)
    transformdata = transform_icp(smoothdata, mask, newnosetip, 1000)
    cropdata = CropFace(transformdata, 100)
    return cropdata

import struct
def ReadBntfile(filename):
    fp = open(filename, 'rb')
    nrows = struct.unpack('@h',fp.read(2))
    ncols = struct.unpack('@h',fp.read(2))
    zmin = struct.unpack('@d', fp.read(8))
    len_of_filename = struct.unpack('@h', fp.read(2))
    imfile = ''
    for i in range(len_of_filename[0]):
        c = struct.unpack('@s', fp.read(1))
        c = str(c).split("'")[1]
        imfile = imfile + str(c[0])

    len_of_data = struct.unpack('@i', fp.read(4))

    len = int(len_of_data[0] / 5)
    data = np.empty((len, 0))
    for i in range(5):
        col = []
        for j in range(len):
            c = struct.unpack('@d', fp.read(8))
            col.append(c[0])
        data = np.column_stack((data, col))

    idata = np.where(data[:, 2] > zmin)
    data = data[idata]
    fp.close()

    return nrows, ncols, imfile, data
