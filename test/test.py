# import torch.utils.data as data
import os
import sys
import numpy as np
import copy
from tqdm import tqdm
import struct
from plyfile import PlyData,PlyElement
import pyntcloud
import pandas as pd
from pyntcloud import PyntCloud

def readbcn(file):
    """
    Args:
        file: filepath

    Returns:
        pointcloud:Nx3
    """
    npoints = os.path.getsize(file) // 4
    with open(file,'rb') as f:
        raw_data = struct.unpack('f'*npoints,f.read(npoints*4))
        data = np.asarray(raw_data,dtype=np.float32)       
    data = data.reshape(3, len(data)//3)
    return data.T
def write_ply(save_path,points,text=True):

    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def compute_normal_and_curvature(points, k = 30):
    """
    Args:
        points: pointcloud-Nx3
        k: number of pointset

    Returns:
        data = [x, y, z, nx, ny, nz, curvature]
    """
    assert points.shape[1] == 3
    points = pd.DataFrame(points)
    points.columns = ['x', 'y', 'z']
    cloud = PyntCloud(points)

    k_neighbors = cloud.get_neighbors(k=k)
    cloud_normal = copy.copy(cloud)
    cloud_normal = cloud_normal.add_scalar_field("normals", k_neighbors=k_neighbors)
    ev = copy.copy(cloud)
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    cloud.add_scalar_field("curvature", ev=ev)
    points = cloud.points
    nx = "nx(" + (k + 1).__str__() + ")"
    ny = "ny(" + (k + 1).__str__() + ")"
    nz = "nz(" + (k + 1).__str__() + ")"
    curvature = "curvature(" + (k + 1).__str__() + ")"

    filter_columns = ['x', 'y', 'z', nx, ny, nz, curvature]
    data = points.reindex(columns=filter_columns)
    return data

from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, CurveVisualization2D
data = np.loadtxt('./pointset.txt')

print(data.shape)

# data = readbcn("./017.bc")
# data = compute_normal_and_curvature(data, 30)
curvature = data[:, 6]
#
data1 = data[curvature > 1.2 * curvature.mean()]
# data1 = data[curvature > 1.2]
point =  data1[:, 0:3]
PointCloudVisualbyVispy(data[:, 0:3])
# visualtwodata(data[:, 0:3], point, size2=5)
# print(data["z"])
# nosetip = data[max(data["z"])]

# data = np.asarray(data)
# data = data[:, 0:3]
# nosetip = data[np.where(data[:, 2] == np.max(data[:, 2]))]

# print(nosetip)

# # PointCloudVisualbyVispy(data)
#
# # PointCloudVisualization(data)
# # visualtwodata(data, nosetip, size2=10)
#
# #二维插值，求出水平侧影线
# y = np.linspace(0, 0, 100)
# x = np.linspace(-80, 80, 100)
# # y = np.linspace(0, MAX_SIZE , N)
# # x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
# # x = np.linspace(0, MAX_SIZE, N)
# xy = np.linspace(0, 100, 100)
# # print(MAX_SIZE)
# point_grid = np.column_stack((x, y))
# # print(point_grid)
# from scipy.interpolate import griddata, interp1d
#
# grid_z = griddata(data[:, 0:2], data[:, 2], point_grid, method='nearest')
# print(grid_z)
# print(grid_z.shape)
# line = np.column_stack((point_grid, grid_z))
# line = np.column_stack((x, grid_z))
# # print(x)
# # visual(points, line)
# # inan = np.where(np.isnan(grid_z))
# # inan = np.where(np.isnan(line))
# # x = x[inan]
# # line = np.column_stack((x, grid_z[inan]))
#
# print(np.nan_to_num(line))
# CurveVisualization2D(line)

# visualtwodata(data, line)


# grid_z = griddata(points[:, 0:2], points[:, 2], point_grid, method='cubic')

# print(data.shape)
# k = 30
#
# points = pd.DataFrame(data)
# points.columns = ['x','y','z']
# cloud = PyntCloud(points)
#
# k_neighbors = cloud.get_neighbors(k=k)
# print(k_neighbors.shape)
# cloud_normal = copy.copy(cloud)
# cloud_normal = cloud_normal.add_scalar_field("normals", k_neighbors=k_neighbors)
# ev = copy.copy(cloud)
# ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
# cloud.add_scalar_field("curvature", ev = ev)
# print(cloud.points.columns)
#
#
#
# import pylab
# import scipy
# from mpl_toolkits.mplot3d import Axes3D
# fig = pylab.figure()
# ax = Axes3D(fig)
#
# from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy
# # curvature = cloud.points["curvature(31)"]
# # cloud.points = cloud.points[curvature > 5 *  curvature.mean()]
# points = cloud.points
# # print(points)
# nx = "nx(" + (k + 1).__str__() + ")"
# ny = "ny(" + (k + 1).__str__() + ")"
# nz = "nz(" + (k + 1).__str__() + ")"
# curvature = "curvature(" + (k + 1).__str__() + ")"
#
# filter_columns = ['x', 'y', 'z', nx, ny, nz, curvature]
#
#
# # filter_columns = ['x', 'y', 'z', 'nx(31)', 'ny(31)', 'nz(31)', 'curvature(31)']
# # filter_columns = [0, 1, 2, 3, 4, 5, -1]
# data = points.reindex(columns = filter_columns)
# print(data)


