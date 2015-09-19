#########################################################################################################
#  Description: Coupon visit train csv file has information on user visits. Need to pre process
#  to obtain data that's useful for training. Key information to be extracted are
#  1. Number of unique sessions per user per coupon
#  2. Number of views per session
#  3. Purchases / view / coupon / user
#  4. Referrer info - Number of referrers / site visit / user, % of referrals by referrer / user.
#     Purchase info by referrer - % of purchases / referrer / user.
#
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

logger = logging.getLogger("")


class PreProcess:
    def __init__(self):
        #  Initialize variables
        self.cwd = os.getcwd()
        self.train_data_dir = "/home/harsha/kaggle/ponpare/data/train"
        self.test_data_dir = "/home/harsha/kaggle/ponpare/data/train"
        pass

    #  Compare coupon information between the visit and list files. They should closely match
    def compare_coupons_visit_list(self):

        os.chdir(self.train_data_dir)

        # Get userid / couponid info from valid coupon visit file
        # valid_coupon_visit_train.csv was created from coupon_visit_train, has only coupon information
        # for coupons in coupon_list_train.csv. All other coupons are in file rogue_coupon_visit_train.csv
        f = open('valid_coupon_visit_train.csv', 'r')

        numlines = sf.file_len(f.name)
        linenum = 0
        temp_array_userid = np.empty(numlines, dtype=object)
        temp_array_couponid = np.empty(numlines, dtype=object)

        for line in f:
            temp_array_userid[linenum] = line.split(',')[5]
            temp_array_couponid[linenum] = line.split(',')[4]
            linenum += 1

        array_userid = np.unique(temp_array_userid, return_index=False)
        array_couponid = np.unique(temp_array_couponid, return_index=False)

        logger.debug(array_userid.shape)
        logger.debug(array_couponid.shape)

        f.close()

        # Get coupon info from coupon list file. These are the valid coupons for the train data set
        f = open('coupon_list_train.csv', 'r')
        #  f = open('/home/harsha/kaggle/ponpare/data/user_list.csv', 'r')

        numlines = sf.file_len(f.name)
        linenum = 0
        temp_array_couponid = np.empty(numlines, dtype=object)

        for line in f:
            temp_array_couponid[linenum] = line.split(',')[23]
            linenum += 1

        array_couponid_list = np.unique(temp_array_couponid, return_index=False)

        logger.debug(array_couponid_list.shape)

        # Compare coupon id between visit and list files. symmetric difference shows elements
        # present in one set but not the other.
        array_couponid_common = tuple(set(array_couponid).symmetric_difference(array_couponid_list))

        logger.debug(array_couponid_common)

        logger.debug(len(array_couponid_common))

        #  for i in range(0,len(array_couponid_common)):
        #      print array_couponid_common[i]

        f.close()

        os.chdir(self.cwd)

        return __pass__

if __name__ == "__main__":

    pp = PreProcess()

    pp.compare_coupons_visit_list()
