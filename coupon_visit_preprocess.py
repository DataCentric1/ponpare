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

logger = logging.getLogger("debug")


class PreProcess:
    def __init__(self):
        #  Initialize variables
        self.cwd = os.getcwd()
        self.data_dir = "/home/harsha/kaggle/ponpare/data"
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

    # More details on coupon purchase info by user. Will feed into attributes
    def purchase_stats(self):

        os.chdir(self.data_dir)

        f = open('user_list.csv', 'r')

        numlines = sf.file_len(f.name)
        linenum = 0

        logger.debug("user_list.csv %d lines", numlines)

        # Create 2D array that can store number of purchases per userid
        array_userid_purchase = np.zeros((numlines-1, 2), dtype=object)

        for line in f:
            if linenum:  # Ignore first line as it's the title
                array_userid_purchase[linenum-1][0] = line.split(',')[5]
            else:
                print line.split(',')[5]
            linenum += 1

        logger.debug("array_userid_purchase size is %s\n", array_userid_purchase.shape)

        f.close()

        os.chdir(self.train_data_dir)

        # Get coupon purchase info from coupon detail file. Calculate # coupons purchased vs. user
        f = open('coupon_detail_train.csv', 'r')

        datavalid = 0

        for line in f:
            if datavalid:  # Ignore first line as it's the title

                # Find index for each userid in the array populated from user_list.csv
                index_row, index_col = np.where(array_userid_purchase == line.split(',')[4])

                # Sum of total coupons purchased for each user
                array_userid_purchase[index_row, index_col + 1] += int(line.split(',')[0])

            datavalid = 1

        f.close()

        np.save('purchase_total_by_user', array_userid_purchase)

        np.savetxt('purchase_total_by_user.log', array_userid_purchase, fmt='%s')

        os.chdir(self.cwd)

        return __pass__

    def temp(self):
        os.chdir(self.train_data_dir)

        sf.save_npy_array_to_csv("purchase_total_by_user.npy", "purchase_total_by_user.csv")

        os.chdir(self.cwd)

if __name__ == "__main__":

    pp = PreProcess()

    # pp.purchase_stats()
