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

from tools.utils import *
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
    # Vertice = WRL(filename).get_Vertices()
    # nosetip = np.array(nosetip
    # data.loc[ind])
    nosetipi = np.where(Vertices[:, 2] == max(Vertices[:, 2]))
    nosetip = Vertices[nosetipi[0][0], :].reshape((-1, 3))
    data = Vertices - nosetip
    return data
from time import time
starttime = time()
mask = np.loadtxt(r'../mask002-005.txt')


filename1 = r'../data/BosphorousDB/bs000/bs000_N_N_0.bnt'
filename2 = r'../data/BosphorousDB/bs000/bs000_N_N_1.bnt'
filename3 = r'../data/BosphorousDB/bs000/bs000_E_ANGER_0.bnt'
data1 = processing(filename1)
data2 = processing(filename2)
data3 = processing(filename3)

NUM_OF_POINT    = 200
NUM_OF_LINE     = 200
starttime = time()
feature1 = GetRayLikePointSet(data1, '3', 80, NUM_OF_POINT, NUM_OF_LINE)
feature2 = GetRayLikePointSet(data2, '3', 80, NUM_OF_POINT, NUM_OF_LINE)
feature3 = GetRayLikePointSet(data3, '3', 80, NUM_OF_POINT, NUM_OF_LINE)

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
iso3 = getisocurve(feature3)
print(iso1.shape)
print("iso time: %f"%(time() - starttime))
# print(iso.shape)
# iso = iso.reshape((-1, 3))
print(iso.shape)
# visual(data1, iso)
# CurveVisualization(iso[5])



from shapeanalysic.DynamicProGrammingQ_C import *


print(iso1.shape, iso2.shape, iso3.shape)

# d1 = Distance_of_two_curve(iso1, iso2, NUM_OF_LINE)
d2 = Distance_of_two_face(iso1, iso3, NUM_OF_LINE)
# print(d1, d2)





