import os
import time
import scipy.io as sio
from tools.PCS import *
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
from tqdm import tqdm
from ExtractFeature.RayLikeCurveMakePictureSet import *

from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, \
    CurveVisualization2D
from tools.utils import *

#   @author: shalp
#   @version: 1.0.0
#   @license: Apache Licence
#   @file: datapreprocess.py
#   @time: 2020/10/30 21:57
#   @Function:数据预处理部分，只有平滑和和裁剪
import struct
import pandas as pd
import copy

from pyntcloud import PyntCloud

import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

def NormalizePointcloud(point_cloud, nosetip):
    """
    将点云数据归一化到鼻尖点处
    Args:
        point_cloud: 点云数据
        nosetip:鼻尖点坐标
    Returns:
    """
    x = point_cloud[:, 0] - nosetip[0]
    y = point_cloud[:, 1] - nosetip[1]
    z = point_cloud[:, 2] - nosetip[2]

    return np.array([x, y, z]).T

def smooth(data, k = 20, d = 8):
    """
    Args:
        data: pointcloud
        k: min number of pointset
        d: distance

    Returns:
        new pointcloud
    """
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

def ReadBntfile(filename):
    """
    读取bnt文件，将二进制文件中的数据转换成三维空间坐标
    Args:
        filename: 文件路径

    Returns:
        行， 列，二维图片签名，三维坐标
    """
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
def readfile_Bosphorus(filepath, k = 50):
    _, _, _, data = ReadBntfile(filepath)
    data = data[:, 0:3]
    data = compute_normal_and_curvature(data, k)
    data = np.asarray(data)
    curvature = data[:, -1]
    data = data[curvature < 15 * curvature.mean()]
    data[:, 0:3] = smooth(data[:, 0:3], 30, 8)
    nosetip = data[np.where(data[:, 2] == np.max(data[:, 2])), 0:3]
    data[:, 0:3] = data[:, 0:3] - nosetip
    data = CropFace(data, N = 100)

    assert data.shape[1] == 7
    return data

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



def PreProcessing(data, nosetip):
    mask = np.loadtxt(r'../mask002-005.txt')
    normallizedata = data - np.array(nosetip)
    newnosetip = [[0,0,0]]
    smoothdata = smooth(normallizedata)
    transformdata = transform_icp(smoothdata, mask, newnosetip, 1000)
    # cropdata = CropFace(transformdata, 100)
    return transformdata

def GetRayLikePointSet(PointCloud, param, MAX_SIZE, N, numoflines):
    grid = np.empty((0, 2))
    for i in range(numoflines):
        theta = i * np.pi / (numoflines / 2)
        # y = np.linspace(0, MAX_SIZE * np.sin(theta), N)
        # y = np.linspace(0, MAX_SIZE , N)
        # x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
        # x = np.linspace(0, MAX_SIZE, N)
        # xy = np.linspace(0, MAX_SIZE , N)
        # print(theta)
        x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
        y = np.linspace(0, MAX_SIZE * np.sin(theta), N)
        grid_xy = np.column_stack((x,y))
        # print(grid_xy)
        grid = np.row_stack((grid, grid_xy))
    # print(grid)
    print(grid.shape)
    points = PointCloud
    grid_z = griddata(points[:, 0:2], points[:, 2], grid, method='cubic', fill_value= 1 )
    grid_z = grid_z.reshape((-1, 1))
    pre = grid_z[0]


    print(grid.shape, grid_z.shape)
    grid_xyz = np.column_stack((grid, grid_z))
    for i in range(1, grid_z.shape[0]):
        if(grid_xyz[i, 2] == 1 and i % 40 != 0):
            grid_xyz[i] = pre
            # print(i)
        pre = grid_xyz[i]
    return grid_xyz

def processing(filename):
    _, _, _, data = ReadBntfile(filename)
    Vertices = data
    Vertices = Vertices[:, 0:3]
    Vertices = transform_icp(Vertices, mask, [[0, 0, 0]], 1000)
    Vertices = smooth(Vertices)
    # Vertice = WRL(filename).get_Vertices(
    # nosetip = np.array(nosetip
    # data.loc[ind])
    nosetipi = np.where(Vertices[:, 2] == max(Vertices[:, 2]))
    nosetip = Vertices[nosetipi[0][0], :].reshape((-1, 3))
    data = Vertices - nosetip
    return data
from time import time
starttime = time()
mask = np.loadtxt(r'C:\Users\12164\Desktop\FR\test\mask002-005.txt')


filename1 = r'C:\Users\12164\Desktop\FR\data\bs048\bs048_N_N_0.bnt'
filename2 = r'C:\Users\12164\Desktop\FR\data\bs048\bs048_E_HAPPY_0.bnt'
# filename3 = r'../data/BosphorousDB/bs000/bs000_E_ANGER_0.bnt'
data1 = readfile_Bosphorus(filename1)
data1 = data1[:, 0:3]
data2 = readfile_Bosphorus(filename2)
data2 = data2[:, 0:3]

# data3 = processing(filename3)

NUM_OF_POINT    = 200
NUM_OF_LINE     = 200
starttime = time()
feature1 = GetRayLikePointSet(data1, '3', 80, NUM_OF_POINT, NUM_OF_LINE)
feature2 = GetRayLikePointSet(data2, '3', 80, NUM_OF_POINT, NUM_OF_LINE)
# feature3 = GetRayLikePointSet(data3, '3', 80, NUM_OF_POINT, NUM_OF_LINE)

print("get feature time: %f"%(time() - starttime))

# print(feature)
# visual(data1, feature1)

starttime = time()
iso = np.zeros((10, NUM_OF_POINT, 3))


def getisocurve(feature):
    i = 0
    feature_copy = feature.copy()
    while(i < feature_copy.shape[0]):
        line = np.copy(feature_copy[i:i+NUM_OF_POINT, :])
        for j in range(3):
            line[:, j] = np.gradient(line[:, j])
        dist = line[:,0]**2 + line[:,1]**2 + line[:,2]**2
        dist = np.sqrt(dist)
        disline = np.cumsum(dist)

        # print(disline)
        for dis in range(10, 110, 10):
            idx = (np.abs(disline - dis)).argmin()
            # print(idx)
            # print(dis / 10 - 1,i / 50, i+idx)
            linenum = int(dis / 10 - 1)
            pointnum = int(i / NUM_OF_POINT)
            iso[linenum, pointnum, :] = feature[i+idx,:]

        i = i + NUM_OF_POINT
    return iso

iso1 = getisocurve(feature1)
iso2 = getisocurve(feature2)
# iso3 = getisocurve(feature3)
print(iso1.shape)
print("iso time: %f"%(time() - starttime))
# print(iso.shape)
iso1 = iso1[3].reshape((-1, 3))
iso2 = iso2[3].reshape((-1, 3))
print(iso.shape)
visualtwodata(data1, iso1,size2 = 5)
visualtwodata(data2, iso2,size2 = 5)
# CurveVisualization(iso[5])



from shapeanalysis.DynamicProGrammingQ_C import *


# print(iso1.shape, iso2.shape, iso3.shape)

# d1 = Distance_of_two_curve(iso1, iso2, NUM_OF_LINE)
# d2 = Distance_of_two_face(iso1, iso3, NUM_OF_LINE)
# print(d1, d2)





