#   @author: shalp
#   @version: 1.0.0
#   @license: Apache Licence
#   @file: nosetip_test.py
#   @time: 2020/11/5 1:11
#   @Function:

from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, \
    CurveVisualization2D
import numpy as np
from scipy.interpolate import griddata, interp1d

import struct
# from test.datapreprocess import *
import struct
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud

# import numpy as np
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
    # data = CropFace(data, N = 100)

    assert data.shape[1] == 7
    return data

def visualtwodata(data1, data2, size1 = 1, size2 = 3):
    data1 = data1.reshape((-1, 3))
    data2 = data2.reshape((-1, 3))
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()

    org_points = visuals.Markers()
    org_points.set_data(data1, edge_color='black', face_color=(1, 1, 1, .5), size=size1)

    inter_points = visuals.Markers()
    inter_points.set_data(data2, edge_color='red', face_color=(1, 1, 1, .5), size=size2)
    view.add(org_points)
    view.add(inter_points)
    # utils.PointCloudVisualization(data)
    view.camera = 'arcball'
    # axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()




if __name__ == "__main__":

    filepath = r"C:\Users\12164\Desktop\FR\test\bs013_N_N_0.bnt"
    # _, _, _, pc = ReadBntfile(filepath)
    # data = pc[:, 0:3]
    data = readfile_Bosphorus(filepath)
    data = data[:, 0:3]
    PointCloud = data
    x=PointCloud[:,0]
    y=PointCloud[:,1]
    z=PointCloud[:,2]
    # print(point_grid)
    # fig = plt.figure(dpi=120)
    # ax = fig.add_subplot(111, projection='3d')
    # plt.title('point cloud')
    # # plt.axis('equal')
    #
    # ax.axis('off')
    # # ax.axis('equal')
    # ax.scatter(x, y, z, c='black', marker='.', s=1, linewidth=.5, alpha=1, cmap='spectral')
    #
    # for y_value in range(-100, 100, 5):
    #     # 二维插值，求出水平侧影线
    #     y = np.linspace(y_value, y_value, 100)
    #     x = np.linspace(-80, 80, 100)
    #     # y = np.linspace(0, MAX_SIZE , N)
    #     # x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
    #     # x = np.linspace(0, MAX_SIZE, N)
    #     xy = np.linspace(0, 100, 100)
    #     # print(MAX_SIZE)
    #     point_grid = np.column_stack((x, y))
    #     grid_z = griddata(data[:, 0:2], data[:, 2], point_grid, method='cubic')
    #     # print(grid_z)
    #     print(grid_z.shape)
    #     line = np.column_stack((point_grid, grid_z))
    #     ax.plot(line[:, 0], line[:, 1], line[:, 2], c='r')
    #
    #     # line = np.column_stack((line, grid_z))
    #     # print(line)
    #     ######
    #     # PointCloudVisualization(line)
    # plt.show()
#####################
    import vispy
    from vispy.scene import visuals
    size = 1
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    scatter = visuals.Markers()
    scatter.set_data(data, edge_color='black', face_color=(1, 1, 1, .5), size=.5)

    view.add(scatter)

    view.camera = 'turntable'  # or try 'arcball'
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111)

    for y_value in range(-100, 100, 5):
        # 二维插值，求出水平侧影线
        y = np.linspace(y_value, y_value, 100)
        x = np.linspace(-80, 80, 100)
        # y = np.linspace(0, MAX_SIZE , N)
        # x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
        # x = np.linspace(0, MAX_SIZE, N)
        xy = np.linspace(0, 100, 100)
        # print(MAX_SIZE)
        point_grid = np.column_stack((x, y))
        grid_z = griddata(data[:, 0:2], data[:, 2], point_grid, method='cubic')
        # print(grid_z)
        print(grid_z.shape)
        line = np.column_stack((point_grid, grid_z))
        # ax.plot(line[:, 0], line[:, 1], line[:, 2], c='r')
        Line = visuals.Line(color=(1, 0.5, 0.5, 1), width=1,
                 connect='strip', method='gl', antialias=False)
        Line.set_data(line, width = 5)


        # plt.title('point cloud')
        ax.plot(x, grid_z, c='r')
        ax.axis('off')
        ax.axis('scaled')



        # line = np.column_stack((line, grid_z))
        # print(line)
        ######
        # PointCloudVisualization(line)
        view.add(Line)
    plt.show()
    # vispy.app.run()


