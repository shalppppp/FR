# -*- coding: utf-8 -*-
# @Author  : 作者
# @Time    : 2020/10/18 13:39
# @Function: 
import h5py

f = h5py.File(r"C:\Users\HIT\Desktop\FR\preprocess\data_BosDB.hdf5", "r")
# print(f.filename, ":")
print([key for key in f.keys()], "\n")
d = f["data"]
for key in d.keys():
    print(key)
    # print("\n")