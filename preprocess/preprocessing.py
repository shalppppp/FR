# -*- coding: utf-8 -*-
# @Author  : shalp
# @Time    : 2020/10/13 9:37
# @Function: 

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