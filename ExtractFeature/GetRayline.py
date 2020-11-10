# -*- coding: utf-8 -*-
# @Author  : 作者
# @Time    : 2020/10/18 14:04
# @Function:
########################################################################################################################
from tqdm import tqdm
import h5py
import  re
from tools.utils import *
from ExtractFeature.RayLikeCurveMakePictureSet import GetRayLikePointSet
########################################################################################################################
# dataset =

mask = np.loadtxt('mask002-005.txt')

with h5py.File('data_neutral.hdf5','w') as f, h5py.File('../preprocess/data_BosDB.hdf5','r') as dataset:
    dataset = dataset["data"]
    for filename in tqdm(dataset.keys()):
        # print(filename)
        if(re.search("N_N", filename)):
            # nosetip = np.array(nosetipdata.loc[ind])
            print(dataset[filename].shape)
            pointcloud = dataset[filename][:, 0:3]
            nosetip = GetNosetip(pointcloud)
            # point = np.array(test_smile_dataset[ind])
            # nosetip = Nosetip(point)

            nosetip = nosetip.reshape((-1, 3))

            data = pointcloud - nosetip
            nosetip_new = [[0, 0, 0]]

            data = transform_icp(data, mask, nosetip_new, 500)

            data = CropFace(data, 100)
            # print(data.shape)
            # data = Smooth(data, 20, 5)
            data = data.reshape((-1, 3))
            print(data)

            print(data.shape, filename)
            feature = np.array(GetRayLikePointSet(data, '2',100, 40))
            print(feature)
            sys.exit(-1)
            # test_smile.append(feature)
            # f.create_dataset(ind, data = feature)