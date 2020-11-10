#   @author: shalp
#   @version: 1.0.0
#   @license: Apache Licence
#   @file: datacheck.py
#   @time: 2020/10/30 15:02
#   @Function:BFM人脸模型查看

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
from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, CurveVisualization2D
import matplotlib.pyplot as plt

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
import open3d
data = readbcn("./400000000/010.bc")
point_cloud = open3d.PointCloud()
point_cloud.points = open3d.Vector3dVector(data)
# open3d.visualization.draw_geometries([point_cloud])

# points = pd.DataFrame(data)
# points.columns = ['x', 'y', 'z']
# cloud = PyntCloud(points)

# k_neighbors = cloud.get_neighbors(k=30)
# points = copy.copy(cloud)
# points = points.add_scalar_field("mesh", k_neighbors=k_neighbors)
# print(points)
# PointCloudVisualization(data)

# import numpy as np
# import pyvista as pv
#
# # points is a 3D numpy array (n_points, 3) coordinates of a sphere
# cloud = pv.PolyData(data)
# # cloud.plot()
#
# volume = cloud.delaunay_3d(alpha=1)
# # print(volume.shape)
# shell = volume.extract_geometry()
# shell.plot()



PointCloudVisualbyVispy(data)
