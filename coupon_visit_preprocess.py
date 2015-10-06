#########################################################################################################
#  Description: Coupon visit train csv file has information on user visits. Need to pre process
#  to obtain data that's useful for training. Key information to be extracted are
#  1. Compare coupon information between different train files
#  2. Purchases statistics
#
#########################################################################################################
# import
import os
import numpy as np
import logging
import logging.config
import support_functions as sf

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

__numusers__ = 22873
__numtraincoupons__ = 19413
__numtestcoupons__ = 310

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
    # TODO - take a look again, seems like a complex implementation. Maybe replace with total_purchases_by_cid
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

    # Total purchases by couponid (some coupons might have been purchased multiple times),
    # That's difference between this value and one computed in function purchase_view_attributes
    def purchase_stats_by_coupon(self):

        os.chdir(self.data_dir + "/npy_arrays")

        couponid_hash = np.load("couponid_hash_train.npy")

        os.chdir(self.train_data_dir)

        # Get coupon purchase info from coupon detail file. Calculate # coupons purchased vs. user
        fr1 = open('coupon_detail_train.csv', 'r')

        linenum = 0
        purchasetotalbycoupon = np.zeros(__numtraincoupons__, int)

        for line in fr1:
            if linenum:  # Ignore first line as it's the title

                arrayindexcoupon = np.where(couponid_hash == line.split(',')[5])

                # Sum of total coupons purchased for each user
                purchasetotalbycoupon[arrayindexcoupon] += int(line.split(',')[0])

            linenum += 1

        fr1.close()

        logger.debug(purchasetotalbycoupon)
        logger.debug(purchasetotalbycoupon.shape)

        np.save('purchase_total_by_coupon', purchasetotalbycoupon)

        np.savetxt('purchase_total_by_coupon.log', purchasetotalbycoupon, fmt='%s')

        os.chdir(self.cwd)

        return __pass__

    # Total views / purchases by users and coupons (from unique train data). Will be new features into the model
    def purchase_view_attributes(self):

        os.chdir(self.train_data_dir)

        # Total coupons viewed by user (number of coupons viewed)
        # and coupon (number of times each coupon was viewed)
        viewedbyuserid = np.zeros(__numusers__, int)
        viewedbycouponid = np.zeros(__numtraincoupons__, int)

        # Total coupons purchased by user (number of coupons purchased)
        # and coupon (number of times each coupon was purchased)
        purchasedbyuserid = np.zeros(__numusers__, int)
        purchasedbycouponid = np.zeros(__numtraincoupons__, int)

        fr1_fname = 'mmlu.data'

        fr1 = open(fr1_fname, 'r')

        linenum = 0
        for line in fr1:
            if int(line.split(',')[2]) == 1:  # Coupon was purchased
                if int(line.split(',')[0]) == 0:  # UID of 0 was a mistake, it should have been 1
                    purchasedbyuserid[0] += 1
                else:
                    purchasedbyuserid[int(line.split(',')[0]) - 1] += 1

                if int(line.split(',')[1]) == 0:  # CID of 0 was a mistake, it should have been 1
                    purchasedbycouponid[0] += 1
                else:
                    purchasedbycouponid[int(line.split(',')[1]) - 1] += 1

            elif int(line.split(',')[2]) == 0:  # Coupon was viewed
                if int(line.split(',')[0]) == 0:  # UID of 0 was a mistake, it should have been 1
                    viewedbyuserid[0] += 1
                else:
                    viewedbyuserid[int(line.split(',')[0]) - 1] += 1

                if int(line.split(',')[1]) == 0:  # CID of 0 was a mistake, it should have been 1
                    viewedbycouponid[0] += 1
                else:
                    viewedbycouponid[int(line.split(',')[1]) - 1] += 1

            else:
                raise ValueError("Invalid purchase flag info. Should only be 0 or 1")
        linenum += 1

        fr1.close()

        # np.save("total_coupons_purchased_by_user", purchasedbyuserid)
        # np.save("total_purchases_by_coupon", purchasedbycouponid)
        # np.save("total_coupons_viewed_by_coupon", viewedbyuserid)
        # np.save("total_views_by_coupon", viewedbycouponid)

        os.chdir(self.cwd)

    # Wrapper function to call plotting functions
    def plotdata(self):
        os.chdir(self.data_dir)

        fname = "user_list_mod.csv"

        nummembershipdays = np.zeros(sf.file_len(fname), int)
        linenum = 0

        # Get number of membership days data from input file
        f = open(fname, 'r')

        for line in f:
            nummembershipdays[linenum] = int(line.split(',')[4])
            linenum += 1

        f.close()

        # Still needs to be debugged
        # vis.cumprobplot(nummembershipdays)

    def save_np_arrays_to_txtcsv(self):
        os.chdir(self.data_dir + "/npy_arrays")

        sf.save_npy_array_to_txt("purchase_total_by_coupon.npy", "purchase_total_by_coupon.log")

        sf.save_npy_array_to_txt("total_purchase_instances_by_coupon.npy", "total_purchase_instances_by_coupon.log")

        sf.save_npy_array_to_txt("total_purchase_instances_by_user.npy", "total_purchase_instances_by_user.log")

        sf.save_npy_array_to_txt("total_views_by_coupon.npy", "total_views_by_coupon.log")

        sf.save_npy_array_to_txt("total_views_by_user.npy", "total_views_by_user.log")

        os.chdir(self.cwd)

if __name__ == "__main__":

    pp = PreProcess()

    # pp.purchase_stats()

    # pp.purchase_stats_by_coupon()

    # pp.plotdata()

    # pp.purchase_view_attributes()

    pp.save_np_arrays_to_txtcsv()
