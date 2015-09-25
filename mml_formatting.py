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
        self.test_data_dir = "/home/harsha/kaggle/ponpare/data/test"
        self.inputtrainfile = 'valid_coupon_visit_train.csv'
        self.mmltrainfile = 'mml_train.csv'
        self.inputtestfile = 'user_coupon_ids.csv'
        self.mmltestfile = 'mml_test.csv'

        pass

    # Convert train data into mml friendly format
    def mml_train_data(self):

        os.chdir(self.train_data_dir)

        f = open(self.inputtrainfile, 'r')
        fw = open(self.mmltrainfile, 'w')

        for line in f:
            fw.write(line.split(',')[5] + "," + line.split(',')[4] + "," + line.split(',')[0] + "\n")

        f.close()
        fw.close()

        os.chdir(self.cwd)

    # Eliminate multiple coupon views per user as it might confuse the algorithm. Instead provide num of visits data
    # as input into user and / or coupon attributes
    def mml_train_unique_data(self):

        os.chdir(self.train_data_dir)

        f = open('mml.data', 'r')
        fw = open('mml_unique.data', 'w')

        prevcouponid = ""
        prevuserid = ""
        prevpurchaseflag = 0
        linenum = 0

        for line in f:

            if linenum:
                # Preserve view where a coupon was purchased
                if line.split(',')[2] == 1:
                    fw.write(line.split(',')[0] + "," + line.split(',')[1] + "," + line.split(',')[2])
                else:
                    if prevuserid != line.split(',')[0] or prevcouponid != line.split(',')[1]:
                        fw.write(prevuserid + "," + prevcouponid + "," + prevpurchaseflag)

            prevuserid = line.split(',')[0]
            prevcouponid = line.split(',')[1]
            prevpurchaseflag = line.split(',')[2]

            linenum += 1

        f.close()
        fw.close()

        os.chdir(self.cwd)

    # Convert test data into mml friendly format
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

    # Converts userid and coupon_id hash into user and coupon numbers
    def mml_uid_cid(self, phase="train"):

        if phase == "train":
            os.chdir(self.train_data_dir)

            fr1 = open('usercou_dec_train.csv', 'r')
        elif phase == "test":
            os.chdir(self.test_data_dir)

            fr1 = open('usercou_dec_train.csv', 'r')
        else:
            raise ValueError("Only valid phases are train and test")

        linenum = 0
        numusers = 22873

        if phase == "train":
            numcoupons = 19413
        else:
            numcoupons = 310

        userid = np.zeros(numusers, int)
        couponid = np.zeros(numcoupons, int)

        userid_hash = np.empty(numusers, dtype=object)
        couponid_hash = np.empty(numcoupons, dtype=object)

        for line in fr1:

            if linenum < numcoupons:
                userid_hash[linenum] = line.split(',')[0]
                userid[linenum] = int(line.split(',')[1])
                couponid_hash[linenum] = line.split(',')[2]
                couponid[linenum] = int(line.split(',')[3])
            elif linenum < numusers:
                userid_hash[linenum] = line.split(',')[0]
                userid[linenum] = int(line.split(',')[1])

            linenum += 1

        logger.debug(userid)
        logger.debug(couponid)

        logger.debug(userid_hash)
        logger.debug(couponid_hash)

        fr1.close()

        fr2 = open('mml_unique.data', 'r')
        fw = open('mmlu_uid_cid.data', 'w')

        linenum = 0
        idxuser = 0
        idxcoupon = 0

        for line in fr2:

            arrayindexuser = np.where(userid_hash == line.split(',')[0])
            arrayindexcoupon = np.where(couponid_hash == line.split(',')[1])

            # userid / couponid values start from 1, so should get correct values even though idxuser
            # and idxcoupon will start from 0
            if arrayindexuser[0]:
                idxuser = int(userid[arrayindexuser[0]])

            if arrayindexcoupon[0]:
                idxcoupon = int(couponid[arrayindexcoupon[0]])

            # Uncomment for introducing fake 1s...for trying equal prob of 1s and 0s
            # if np.random.rand() > 0.5:
            fw.write("%d" % idxuser + "," + "%d" % idxcoupon + "," + line.split(',')[2])
            #    ones += 1
            # else:
            #    fw.write(line.split(',')[0] + "," + "%d" % idx + ",0\n")

            linenum += 1

        logger.debug(linenum)
        # logger.debug(ones)
        logger.debug(numusers)
        logger.debug(numcoupons)

        fw.close()
        fr2.close()

        os.chdir(self.cwd)

    # Adds real purchase data back in to files that were modified to add fake 1s (to see how algorithm performs with
    # equal probability of 1s and 0s.
    def add_real_purchase_flag(self):

        os.chdir(self.train_data_dir)

        fr1_fname = 'mml1_unique.base'

        purchaseflag = np.zeros(sf.file_len(fr1_fname))
        linenum = 0

        fr1 = open(fr1_fname, 'r')

        for line in fr1:
            purchaseflag[linenum] = line.split(',')[2]
            linenum += 1

        fr1.close()

        logger.debug(purchaseflag)

        fr2 = open('mml1_uid.base', 'r')
        fw = open('r1s_uid_cid.base', 'w')

        linenum = 0

        for line in fr2:

            fw.write(line.split(',')[0] + "," + line.split(',')[1] + "," + "%d" % purchaseflag[linenum] + "\n")

            linenum += 1

        logger.debug(linenum)

        fw.close()
        fr2.close()

        os.chdir(self.cwd)

if __name__ == "__main__":

    mdf = MmlDataFormat()

    # mdf.mml_train_unique_data()

    mdf.mml_uid_cid()

    # mdf.mml_test_data()
