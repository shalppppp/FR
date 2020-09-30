import struct
import numpy as np
import sys, os
import re


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

def getfilename(originfilepath, savefd,  filetype = "all", fileclass = "all"):
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
path = r"C:\Users\HIT\Desktop\facerecognition\data\BosphorousDB"
fd = []

getfilename(path, fd, ".bnt", "N_N_1")

nrows, ncols, imfile, data = ReadBntfile(fd[0])
print(nrows, ncols, imfile, data)