import os
import time

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
from tqdm import tqdm

import tools.utils
# from tools.PCS import PCSTransform

# starttime = time.time()
# BASE_DATA_DIR = r'D:\facerecognition\data\CASIA'
# root = BASE_DATA_DIR


# nosetipdata = pd.read_csv(r'D:\facerecognition\preprocess\nosetip.csv', index_col=0)
import matplotlib.pyplot as plt


def GetRayLikePointSet(PointCloud, param, MAX_SIZE, N, numoflines):
    # print(MAX_SIZE)

    if(param == '2'):

        allline = []
        # print(allpoints)
        for i in range(numoflines):

            x = PointCloud[:, 0]
            y = PointCloud[:, 1]
            theta = i * np.pi / (numoflines / 2)
            d = np.fabs(x * np.sin(theta) - y * np.cos(theta))
            itheta = np.where(d < 2)
            point_theta = PointCloud[itheta]
            inter_line = interpolation(point_theta, theta, MAX_SIZE, N)

            allline.append(inter_line)

        return np.array(allline)

    elif(param == '3'):

        allpoints = []
        # print(allpoints)
        for i in range(numoflines):

            x = PointCloud[:, 0]
            y = PointCloud[:, 1]
            theta = i * np.pi / (numoflines / 2)
            d = np.fabs(x * np.sin(theta) - y * np.cos(theta))
            # print(d)
            itheta = np.where(d < 2)
            point_theta = PointCloud[itheta]
            # inter_theta = interpolation(point_theta, theta, MAX_SIZE, N)
            inter_theta = interpolation(PointCloud, theta, MAX_SIZE, N)

            MAX_XY = inter_theta[-1, 0]

            x = np.linspace(0, MAX_XY * np.cos(theta), N)
            y = np.linspace(0, MAX_XY * np.sin(theta), N)

            grid_xy = np.column_stack((x,y))
            pointset = np.column_stack((grid_xy, inter_theta[:, 1]))
            # visual(point_theta, pointset)
            # allpoints = np.row_stack((allpoints, inter_theta))
            allpoints.append(pointset)
        return np.array(allpoints)
def interpolation(points, theta, MAX_SIZE, N):
    y = np.linspace(0, MAX_SIZE * np.sin(theta), N)
    # y = np.linspace(0, MAX_SIZE , N)
    x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
    # x = np.linspace(0, MAX_SIZE, N)
    xy = np.linspace(0, MAX_SIZE , N)
    # print(MAX_SIZE)
    point_grid = np.column_stack((x, y))
    # print(point_grid)
    grid_z = griddata(points[:, 0:2], points[:, 2], point_grid, method='cubic')
    # print(grid_z)
    # print(grid_z.shape)
    line = np.column_stack((xy, grid_z))
    # visual(points, line)
    inan = np.where(np.isnan(grid_z))
    # print(inan)
    if(len(inan[0])  > 0):
        imax = inan[0][0]
        x = line[(imax-1), 0]
        newx = np.linspace(0, x, N)
        line = line[0:(inan[0][0]), :]
        f = interp1d(line[:, 0], line[:, 1], kind='cubic',fill_value="extrapolate")
        grid_z = f(newx)
        line = np.column_stack((newx, grid_z))
    return line


def interpolation1(points, theta, MAX_SIZE, N):
    y = np.linspace(0, MAX_SIZE * np.sin(theta), N)
    x = np.linspace(0, MAX_SIZE * np.cos(theta), N)
    xy = np.linspace(0, MAX_SIZE , N)

    point_grid = np.column_stack((x, y))
    grid_z = griddata(points[:, 0:2], points[:, 2], point_grid, method='cubic')
    # print(grid_z)
    # print(grid_z.shape)
    line = np.column_stack((xy, grid_z))
    # visual(points, line)
    inan = np.where(np.isnan(grid_z))
    # print(inan)
    if(len(inan[0])  > 0):
        imax = inan[0][0]
        x = line[(imax-1), 0]
        newx = np.linspace(0, x, N)
        line = line[0:(inan[0][0]), :]
        f = interp1d(line[:, 0], line[:, 1], kind='cubic',fill_value="extrapolate")
        grid_z = f(newx)
        line = np.column_stack((newx, grid_z))
    return line

'''
with h5py.File('RayLikeCurve.hdf5','w') as f:
    for (dirpath, dirnames, filenames) in tqdm(os.walk(root)):
        # print(dirpath.split("\\")[-1])
        # print(filenames)
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.wrl':
                fileid = filename.split('.')[0]
                label = filename.split('-')[0]
                nosetip = nosetipdata.loc[fileid]
                filename = os.path.join(dirpath, filename)
                print("filename: %s \n fileid: %s \n label: %s"%(filename, fileid, label))

                data = utils.WRL(filename).get_Vertices()
                nosetip = nosetipdata.loc[fileid]
                data = utils.CropFace(data, nosetip)
                data = utils.NormalizePointcloud(data, nosetip)
                data = PCSTransform(data)
                z_value = GetRayLikePointSet(data, 'z')
                try:
                    f.create_dataset(label, data = z_value)
                except:
                    print("file %s save error"%(fileid))
                break

f.close()
endtime = time.time()

print("time: %f"%(endtime - starttime))
'''
'''
filename = r"D:\facerecognition\data\CASIA\WRL1-30\002\002-001.wrl"

data = utils.WRL(filename).get_Vertices()
nosetipdata = pd.read_csv(r'D:\facerecognition\preprocess\nosetip.csv', index_col=0)
nosetip = nosetipdata.loc['002-001']
data = utils.CropFace(data, nosetip)
data = utils.NormalizePointcloud(data, nosetip)

point0 = GetRayLikePointSet(data, 'point')
# point0 = np.array(point0).reshape((-1, 3))
# point0 = point0.reshape((-1, , 3))
print(point0.shape)
print(point0)
# utils.PointCloudVisualbyVispy(point0)
visual(data, point0)
'''
