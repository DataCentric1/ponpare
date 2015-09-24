#########################################################################################################
# import
import os
import numpy as np
import logging
import logging.config
import support_functions as sf
#########################################################################################################
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0
#########################################################################################################
# Setup logging
logging.config.fileConfig('logging.conf')

logger = logging.getLogger("debug")


class MmlDataFormat:
    def __init__(self):
        #  Initialize variables
        self.cwd = os.getcwd()
        self.data_dir = "/home/harsha/kaggle/ponpare/data"
        self.train_data_dir = "/home/harsha/kaggle/ponpare/data/train"
        self.test_data_dir = "/home/harsha/kaggle/ponpare/data/train"
        self.inputtrainfile = 'valid_coupon_visit_train.csv'
        self.mmltrainfile = 'mml_train.csv'
        self.inputtestfile = 'user_coupon_ids.csv'
        self.mmltestfile = 'mml_test.csv'

        pass

    def mml_train_data(self):

        os.chdir(self.train_data_dir)

        f = open(self.inputtrainfile, 'r')
        fw = open(self.mmltrainfile, 'w')

        for line in f:
            fw.write(line.split(',')[5] + "," + line.split(',')[4] + "," + line.split(',')[0] + "\n")

        f.close()
        fw.close()

        os.chdir(self.cwd)

    def mml_test_data(self):

        os.chdir(self.train_data_dir)

        f = open(self.inputtestfile, 'r')

        numlines = sf.file_len(f.name)
        linenum = 0
        temp_array_userid = np.empty(numlines, dtype=object)
        temp_array_couponid = np.empty(numlines, dtype=object)

        for line in f:
            temp_array_userid[linenum] = line.split(',')[0]
            if line.split(',')[1] not in ["None", "", " ", "\n", "'\\n'"]:
                temp_array_couponid[linenum] = line.split(',')[1]
            linenum += 1

        # np.unique sorts the array, here's the unsorted version
        _, idx = np.unique(temp_array_userid, return_index=True)
        array_userid = temp_array_userid[np.sort(idx)]

        _, idx = np.unique(temp_array_couponid, return_index=True)
        array_couponid = temp_array_couponid[np.sort(idx)]

        logger.debug(array_userid.shape[0])
        logger.debug(array_couponid.shape[0])

        f.close()

        fw = open(self.mmltestfile, 'w')

        for i in range(array_userid.shape[0]):
            for j in range(array_couponid.shape[0]):
                if array_couponid[j]:
                    fw.write(array_userid[i] + "," + array_couponid[j] + "\n")

        fw.close()

        os.chdir(self.cwd)

if __name__ == "__main__":

    mdf = MmlDataFormat()

    mdf.mml_train_data()

    # mdf.mml_test_data()