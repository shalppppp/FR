
from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, \
    CurveVisualization2D
import os
import numpy as np
import struct
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
def nogrid(data):
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    print(np.max(x), np.max(y), np.max(z))

    choice = np.where(np.where(y > 50))
    data = data[choice,:]
    return data



filepath = r"C:\Users\12164\Desktop\FR\test\bs001_N_N_0.bnt"

data = readbcn(filepath)
data = nogrid(data)




PointCloudVisualbyVispy(data)

