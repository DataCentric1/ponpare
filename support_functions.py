#########################################################################################################
#  Description: Collection of support functions that'll be used often
#
#########################################################################################################
import numpy as np

#########################################################################################################
__author__ = 'DataCentric1'
__pass__ = 1
__fail__ = 0


#  Returns number of lines in a file in a memory / time efficient way
def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i


#  Save numpy array from .npy file to txt file
def save_npy_array_to_txt(npy_fname, txt_fname):

    np.savetxt(txt_fname, np.load(npy_fname), fmt='%s')

    return __pass__


#  Save numpy array from .npy file to csv file. TODO - Doublce check fn
def save_npy_array_to_csv(npy_fname, csv_fname):

    temp_array = np.load(npy_fname)

    index_row, index_col = temp_array.shape

    print index_row
    print index_col

    f = open(csv_fname, 'w')

    for i in range(index_row):
        f.write(temp_array[i, 0])
        f.write(",")
        f.write(temp_array[i, 1])
        f.write("\n")

    f.close()

    return __pass__
