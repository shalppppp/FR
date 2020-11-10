import struct
import numpy as np
import sys, os
import re
import h5py
from tqdm import tqdm
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

def GetListPahts(originfilepath, savefd,  filetype = "all", fileclass = "all"):
    for (dirpath, dirnames, filenames) in os.walk(originfilepath):
        # print(dirpath.split("\\")[-1])
        # print(filenames)
        for filename in filenames:
            if filetype == "all" and fileclass == "all":
                # print(os.path.join(dirpath, filename))
                savefd += [os.path.join(dirpath, filename)]

            elif filetype == "all":
                if re.search(fileclass, filenames):
                    # print(os.path.join(dirpath, filename))
                    savefd += [os.path.join(dirpath, filename)]

            elif fileclass == "all":
                if os.path.splitext(filename)[1] == filetype:
                    # print(os.path.join(dirpath, filename))
                    savefd += [os.path.join(dirpath, filename)]

            else:
                if os.path.splitext(filename)[1] == filetype and re.search(fileclass, filename):
                    # print(os.path.join(dirpath, filename))
                    savefd += [os.path.join(dirpath, filename)]
# print(sys.path[1])
def GetLabel(filename):
    file = filename.split("\\")[-1]
    label = file.split(".")[0]
    return label


def create_h5(paths):
    with h5py.File('data_BosDB.hdf5', 'w') as f:
        data = f.create_group('data')
        # label_group = f.create_group('label')
        label = []
        for path in tqdm(paths):
            nrows, ncols, imfile, pcdata = ReadBntfile(path)
            lb = GetLabel(path)
            label.append(lb)
            d = data.create_dataset(lb, data = pcdata)
            d.attrs["size"] = [nrows, ncols]
            try:
                d.attrs[imfile] = imfile
            except:
                print(path)
        # label_group.create_dataset('label' ,data = label)
        f.close()
# print(nrows, ncols, imfile, data)


path = r"C:\Users\HIT\Desktop\facerecognition\data\BosphorousDB"
list_paths = []
GetListPahts(path, list_paths, ".bnt")
# print(GetLabel(list_paths[9]))
create_h5(list_paths)
