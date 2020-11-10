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
def readfile_Bosphorus(file, k = 50):
    _, _, _, data = ReadBntfile(filepath)
    data = data[:, 0:3]
    data = compute_normal_and_curvature(data, k)
    data = np.asarray(data)
    curvature = data[:, -1]
    data = data[curvature < 15 * curvature.mean()]
    data[:, 0:3] = smooth(data[:, 0:3], 30, 8)
    nosetip = data[np.where(data[:, 2] == np.max(data[:, 2])), 0:3]
    data[:, 0:3] = data[:, 0:3] - nosetip
    data = CropFace(data, N = 80)

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


if __name__ == "__main__":

    filepath = r"C:\Users\12164\Desktop\FR\test\bs013_N_N_0.bnt"
    _, _, _, pc = ReadBntfile(filepath)
    data = readfile_Bosphorus(filepath)
    print(data.shape)
    from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, \
        CurveVisualization2D
    nosetip = np.zeros((1, 3))
    PointCloudVisualbyVispy(pc[:, 0:3] ,size=3)
    PointCloudVisualbyVispy(data[:, 0:3] ,size=3)

    # visualtwodata(data[:, 0:3], nosetip, size2=20)


# _, _, _, data = ReadBntfile(filepath)
# data = np.asarray(data)
#
# data = data[:, 0:3]
#
# # data = smooth(data, 20, 10)
# data = compute_normal_and_curvature(data, 50)
#
# curvature = data["curvature(51)"]
# data = data[curvature < 10 * curvature.mean()]
#
# data = np.asarray(data)
#
# data = data[:, 0:3]
# data = smooth(data, 25, 8)
#
# print(data.shape)
#

#
# data = CropFace(data)
# # print(nosetip)
# print(data.shape)
#
# PointCloudVisualbyVispy(data)

# PointCloudVisualization(data)
# visualtwodata(data, nosetip, size2=10)

# PointCloudVisualbyVispy(data)
