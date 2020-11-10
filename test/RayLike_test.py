import numpy as np
import matplotlib.pyplot as plt
from tools.utils import *
from shapeanalysis.DynamicProGrammingQ import *
from ExtractFeature.RayLikeCurveMakePictureSet import *

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



filepath1 = r"C:\Users\12164\Desktop\FR\test\017.bc"
filepath2 = r"C:\Users\12164\Desktop\FR\test\400000000\050.bc"
data1 = readbcn(filepath1)
data2 = readbcn(filepath2)




# visual(data1, data2)
# print(data1.shape, data2.shape)
num = 24

feature1 = GetRayLikePointSet(data1, '3', 100, 100, num)
feature2 = GetRayLikePointSet(data2, '3', 100, 100, num)
feature1 = feature1.reshape((-1, 3))
feature2 = feature2.reshape((-1, 3))


from tools.visualization import PointCloudVisualization, PointCloudVisualbyVispy, visualtwodata, \
    CurveVisualization2D

PointCloudVisualbyVispy(data1)
PointCloudVisualbyVispy(data2)
visualtwodata(data1, feature1)
visualtwodata(data2, feature2)


# feature2 = GetRayLikePointSet(data2, '3', 100, 100, num)
# feature3 = GetRayLikePointSet(data3, '3', 100, 100, num)
# sum1 = 0
# sum2 = 0
# for i in range(num):
#     d1 = Distance_of_two_curve(feature1[i], feature2[i])
#     d2 = Distance_of_two_curve(feature1[i], feature3[i])
#     if i in [4, 8, 16, 17, 18, 19]:
#         d1 *= 0.5
#         d2 *= 0.5
#     sum1 += d1
#     sum2 += d2
#     print("line %d:%f --- %f"%(i, d1, d2))
# print("all distance:%f --- %f"%(sum1,sum2))

# print(feature1.shape, feature2.shape)
# PointCloudVisualbyVispy(data1)
# PointCloudVisualbyVispy(data2)
# PointCloudVisualbyVispy(data3)
# visual(data1, data2)
# visual(data1, data3)
# # feature3 = feature3.reshape((-1, 3))
# # PointCloudVisualbyVispy(data3)
# print(data3.shape, nosetip3.shape)
# for i in range(0):
#     visual(data3, feature3[i])
