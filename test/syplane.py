#   @author: shalp
#   @version: 1.0.0
#   @license: Apache Licence
#   @file: syplane.py
#   @time: 2020/11/5 22:27
#   @Function:

import os
import struct
from scipy.interpolate import griddata, interp1d

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, \
    CurveVisualization2D
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






fig = plt.figure(figsize=(12, 8),
                 facecolor='lightyellow'
                )

# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',
             projection='3d'
            )
ax.axis("off")
# 二元函数定义域平面
x = np.linspace(0, 0, 100)
y = np.linspace(-100, 100, 100)
z = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)
Z,_ = np.meshgrid(y, z)

# print(X, Y, Z)

filepath = r"C:\Users\12164\Desktop\FR\test\017.bc"

data = readbcn(filepath)

point_grid = np.column_stack((x, y))
grid_z = griddata(data[:, 0:2], data[:, 2], point_grid, method='cubic')
print(grid_z.shape)

xyz = np.column_stack((point_grid, grid_z))


ax.scatter(data[:, 0],data[:, 1],data[:, 2], c = 'black', s = 1,alpha = .5, marker = '.')
ax.plot_surface(X,
                Y,
                Z,
                color='r',
                alpha=0.2
               )
point_grid = np.column_stack((x, y))
grid_z = griddata(data[:, 0:2], data[:, 2], point_grid, method='cubic')
print(grid_z.shape)

xyz = np.column_stack((point_grid, grid_z))
# ax.plot_surface(X, Y, grid_z.reshape((100 , 2)), alpha = .3)
ax.plot(xyz[:, 0], xyz[:, 1],xyz[:, 2],c = 'b')

# print(data)
# PointCloudVisualization(data)
plt.show()








# import vispy
# from vispy.scene import visuals
#
# size = 1
#
# canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
# view = canvas.central_widget.add_view()
# scatter = visuals.Markers()
# scatter.set_data(data, edge_color='black', face_color=(1, 1, 1, .5), size=size)
#
# view.add(scatter)
# view.camera = 'turntable'  # or try 'arcball'
# Line = visuals.Line(color=(1, 0.5, 0.5, 1), width=2,
#                     connect='strip', method='gl', antialias=False)
# Line.set_data(xyz, width=5)
# view.add(Line)
#
# Plane = visuals.plane.PlaneVisual()
#
# Plane.set_data()
#
#
#
# vispy.app.run()


# add a colored 3D axis for orientation
# axis = visuals.XYZAxis(parent=view.scene)











#

