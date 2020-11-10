#   @author: shalp
#   @version: 1.0.0
#   @license: Apache Licence
#   @file: smooth_test.py
#   @time: 2020/11/4 23:30
#   @Function:

import struct
import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame
# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸

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

import open3d as o3d

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Removal Outlier', width=1920,
                                      height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

filepath = r"C:\Users\12164\Desktop\FR\test\bs001_N_N_0.bnt"
_, _, _, pcd = ReadBntfile(filepath)
pcd = o3d.Geometry(pcd[:, 0:3])

o3d.visualization.draw_geometries([pcd])
# 下采样
# voxel_down_sample（把点云分配在三维的网格中，取平均值）
# uniform_down_sample (可以通过收集每第n个点来对点云进行下采样)
# select_down_sample (使用带二进制掩码的select_down_sample仅输出所选点。选定的点和未选定的点并可视化。）
print("Downsample the point cloud with a voxel of 0.003")
downpcd = pcd.voxel_down_sample(voxel_size=0.003)
print(downpcd)
o3d.visualization.draw_geometries([downpcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                  point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

# 重新计算平面法线
# 顶点法线估计【Vertex normal estimation】
# 点云的另一个基本操作是点法线估计。按n查看法线。键-和键+可用于控制法线的长度。
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd], window_name='Open3D downSample Normals', width=1920, height=1080, left=50,
                                  top=50, point_show_normal=True, mesh_show_wireframe=False, mesh_show_back_face=False)

# 离群点去除 【outlier removal】
# 点云离群值去除 从扫描设备收集数据时，可能会出现点云包含要消除的噪声和伪影的情况。本教程介绍了Open3D的异常消除功能。 准备输入数据，使用降采样后的点云数据。
# statistical_outlier_removal 【统计离群值移除】 删除与点云的平均值相比更远离其邻居的点。
#          它带有两个输入参数：nb_neighbors 允许指定要考虑多少个邻居，以便计算给定点的平均距离
#                           std_ratio 允许基于跨点云的平均距离的标准偏差来设置阈值级别。此数字越低，过滤器将越具有攻击性
#
# radius_outlier_removal 【半径离群值去除】  删除在给定球体中周围几乎没有邻居的点。
#          两个参数可以调整以适应数据：nb_points 选择球体应包含的最小点数
#                                  radius 定义将用于计算邻居的球体的半径
print("Statistical oulier removal")
cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
display_inlier_outlier(downpcd, ind)
downpcd_inlier_cloud = downpcd.select_by_index(ind)
print(downpcd_inlier_cloud)

# 平面分割 【Plane Segmentation】
# Open3D还包含使用RANSAC从点云中分割几何图元的支持。要在点云中找到具有最大支持的平面，我们可以使用segement_plane。该方法具有三个参数。
# distance_threshold定义一个点到一个估计平面的最大距离，该点可被视为一个不规则点； ransac_n定义随机采样的点数以估计一个平面； num_iterations定义对随机平面进行采样和验证的频率。
# 函数然后将平面返回为（a，b，c，d） 这样，对于平面上的每个点（x，y，z），我们都有ax + by + cz + d = 0。该功能进一步调整内部点的索引列表。
plane_model, inliers = downpcd_inlier_cloud.segment_plane(distance_threshold=0.01,
                                             ransac_n=5,
                                             num_iterations=10000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = downpcd_inlier_cloud.select_by_index(inliers)
print('----inlier_cloud: ', inlier_cloud.points)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = downpcd_inlier_cloud.select_by_index(inliers, invert=True)
print('----outlier_cloud: ', outlier_cloud.points)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Plane Model', width=1920,
                                  height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                  mesh_show_back_face=False)
o3d.io.write_point_cloud("D:/pcd/1001140020191217_las2pcd_cx_g.pcd", inlier_cloud)
# help(o3d.visualization.draw_geometries)
