#########################################################################################################
#  Description: Format data to be compatible with MyMultimediaLite recommender system
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

    # Split ~310 coupons from mml_unique data for cross-validation set
    def mml_cv_set_coupons(self):

        os.chdir(self.data_dir)

        fr1_name = 'mmlu.data'
        fw1_name = 'mmlu_train.data'
        fw2_fname = 'mmlu_cv.data'

        numofcouponsintrain = 19413
        swapcouponlist = np.empty(sf.file_len(fr1_name), dtype=object)
        cvcouponlist = np.empty(int(numofcouponsintrain / 60) + 1, dtype=object)

        fr1 = open(fr1_name, 'r')

        linenum = 0
        for line in fr1:
            swapcouponlist[linenum] = int(line.split(',')[1])
            linenum += 1

        fr1.close()

        couponlist = np.unique(swapcouponlist, return_index=False)

        logger.debug(swapcouponlist.shape)
        logger.debug(swapcouponlist)
        logger.debug(couponlist.shape[0])
        logger.debug(couponlist)

        couponnum = 0
        for i in range(int(couponlist.shape[0])):
            if not (i % 60):  # Every 60th coupon goes to CV set
                cvcouponlist[couponnum] = couponlist[couponnum * 60]
                couponnum += 1

        logger.debug(cvcouponlist.shape)
        logger.debug(cvcouponlist)

        np.save("mmu_cvcouponlist", cvcouponlist)

        fr1 = open(fr1_name, 'r')
        fw1 = open(fw1_name, 'w')
        fw2 = open(fw2_fname, 'w')

        for line in fr1:
            if int(line.split(',')[1]) in cvcouponlist:  # Coupons in cvcouponlist go to the cv set
                fw2.write(line)
            else:
                fw1.write(line)

        fr1.close()
        fw1.close()
        fw2.close()

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

    # Create train userid / couponid data. Fill in combinations from valid_coupon_visited_unique file
    # to 1s (mmlu.data is translation of that file into uid / cid)
    def train_viewed_data_uid_cid(self):

        os.chdir(self.train_data_dir)

        numusers = 22873
        numtraincoupons = 19413

        viewed = np.zeros((numusers + 1, numtraincoupons + 1), int)

        print viewed.shape
        print viewed

        fr1 = open('mmlu.data', 'r')

        for line in fr1:
            viewed[int(line.split(',')[0])][int(line.split(',')[1])] = 1
        fr1.close()

        fw = open('mmlu_train_viewed.data', 'w')

        for i in range(0, numusers):
            for j in range(0, numtraincoupons):
                if viewed[i][j]:
                    fw.write("%d" % i + "," + "%d" % j + ",1\n")
                else:
                    fw.write("%d" % i + "," + "%d" % j + ",0\n")

        fw.close()

        os.chdir(self.cwd)

    # Create test userid / couponid data.
    def test_data_uid_cid(self):

        os.chdir(self.test_data_dir)
        fw = open('mmlu_par_test.data', 'w')
        # numusers = 22873
        # numcoupons = 310

        for z in range(0, 10):
            for i in range(1, 2001):
                for j in range(20001, 20030):
                    fw.write("%d" % (z * 2000 + i) + "," + "%d" % (z * 30 + j) + "\n")

        fw.close()

    def coupon_area_cid(self):

        os.chdir("%s" % self.data_dir + "/npy_arrays")

        couponid = np.load("coupon_cid_combined.npy")
        couponid_hash = np.load("coupon_hash_combined.npy")

        logger.debug(couponid_hash.shape)
        logger.debug(couponid_hash)
        logger.debug(couponid.shape)
        logger.debug(couponid)

        testcouponstartindex = 19413
        testcouponendindex = 19722

        os.chdir(self.data_dir)

        logger.debug(couponid[0])
        logger.debug(couponid[testcouponendindex])

        fr1_fname = "coupon_area.csv"

        couponid_index = np.zeros(sf.file_len(fr1_fname), int)

        fr1 = open(fr1_fname, 'r')

        fw_fname = "coupon_area_cid.csv"
        fw = open(fw_fname, 'w')
        fw.write("SMALL_AREA_NAME,PREF_NAME,COUPONID,COUPONID_HASH\n")

        linenum = 0
        i = 0
        for line in fr1:
            if linenum:
                if np.where(line.split(',')[2] == couponid_hash)[0] <= testcouponstartindex - 1:
                    couponid_index[linenum] = np.where(line.split(',')[2] == couponid_hash)[0] + 1
                else:  # test coupons starts from 20001 - 20310
                    couponid_index[linenum] = np.where(line.split(',')[2] == couponid_hash)[0] + 20001 - 19413
                    if i < 10:
                        logger.debug(couponid_hash[np.where(line.split(',')[2] == couponid_hash)[0]])
                        logger.debug(couponid_index[linenum])
                    i += 1
                fw.write("%s," % line.split(',')[0] + "%s," % line.split(',')[1] + "%d," % couponid_index[linenum] +
                         "%s," % line.split(',')[2] + "\n")
            linenum += 1

        fr1.close()
        fw.close()

        logger.debug(np.unique(couponid_index, return_counts=True))
        logger.debug(couponid_index.shape)
        logger.debug(couponid_index)

        return None

    # Randomly split tets file of 7M lines (22k users x 310 coupons) into multiple small files
    # Doesn't end up as an equal split at all. Needs better implementation
    def random_split_test_file(self):

        os.chdir(self.test_data_dir)

        fw0_fname = 'mmlu_t0.base'
        fw1_fname = 'mmlu_t1.base'
        fw2_fname = 'mmlu_t2.base'
        fw3_fname = 'mmlu_t3.base'
        fw4_fname = 'mmlu_t4.base'
        fw5_fname = 'mmlu_t5.base'
        fw6_fname = 'mmlu_t6.base'
        fw7_fname = 'mmlu_t7.base'
        fw8_fname = 'mmlu_t8.base'
        fw9_fname = 'mmlu_t9.base'

        fr1 = open('mmlu_test.data', 'r')

        fw0 = open(fw0_fname, 'w')
        fw1 = open(fw1_fname, 'w')
        fw2 = open(fw2_fname, 'w')
        fw3 = open(fw3_fname, 'w')
        fw4 = open(fw4_fname, 'w')
        fw5 = open(fw5_fname, 'w')
        fw6 = open(fw6_fname, 'w')
        fw7 = open(fw7_fname, 'w')
        fw8 = open(fw8_fname, 'w')
        fw9 = open(fw9_fname, 'w')

        linenum = 0
        for line in fr1:
            if not linenum % 7:  # choose prime numbers
                fw0.write(line)
            elif not linenum % 11:
                fw1.write(line)
            elif not linenum % 13:
                fw2.write(line)
            elif not linenum % 19:
                fw3.write(line)
            elif not linenum % 23:
                fw4.write(line)
            elif not linenum % 9:  # Switch to more frequent #s to even out file sizes
                fw5.write(line)
            elif not linenum % 10:
                fw6.write(line)
            elif not linenum % 17:
                fw7.write(line)
            elif not linenum % 21:
                fw8.write(line)
            else:
                fw9.write(line)
            linenum += 1

        fr1.close()

        fw0.close()
        fw1.close()
        fw2.close()
        fw3.close()
        fw4.close()
        fw5.close()
        fw6.close()
        fw7.close()
        fw8.close()
        fw9.close()

        os.chdir(self.cwd)

    # Converts userid and coupon_id hash into user and coupon numbers
    # TODO - Found user#1 and coupon#0,1 were not there in output. Check what happened, very odd...
    # TODO - Doesn't work with test phase, needs further debug
    def mml_uid_cid(self, phase="train"):

        if phase == "train":
            os.chdir(self.train_data_dir)

            fr1 = open('usercou_dec_train.csv', 'r')
        elif phase == "test":
            os.chdir(self.test_data_dir)

            fr1 = open('usercou_dec_test.csv', 'r')
        elif phase == "common":
            os.chdir(self.test_data_dir)

            fr1 = open('usercou_dec_test.csv', 'r')
        else:
            raise ValueError("Only valid phases are train, test and common")

        linenum = 0
        numusers = 22873

        if phase == "train":
            numcoupons = 19413
        elif phase == "test":
            numcoupons = 310
        else:
            numcoupons = 0

        userid = np.zeros(numusers, int)
        couponid = np.zeros(numcoupons, int)

        userid_hash = np.empty(numusers, dtype=object)
        couponid_hash = np.empty(numcoupons, dtype=object)

        for line in fr1:

            if linenum < numcoupons:
                userid_hash[linenum] = line.split(',')[0]
                userid[linenum] = int(line.split(',')[1])
                couponid_hash[linenum] = line.split(',')[2]
                if phase == "train":
                    couponid[linenum] = int(line.split(',')[3])
                # Adding 20000 as we need new couponIDs for test that are different than train coupon IDs or
                # model will confuse these coupons to be from train and give wrong results.
                else:
                    couponid[linenum] = int(line.split(',')[3]) + 20000
            elif linenum < numusers:
                userid_hash[linenum] = line.split(',')[0]
                userid[linenum] = int(line.split(',')[1])

            linenum += 1

        fr1.close()

        print("#######")
        logger.debug(userid)
        logger.debug(couponid)

        logger.debug(userid_hash)
        logger.debug(couponid_hash)

        # Uncomment to save the arrays
        # np.save("userid", userid)
        # np.save("userid_hash", userid_hash)
        # np.save("couponid_train", couponid)
        # np.save("couponid_hash_train", couponid_hash)

        if phase == "common":
            logger.debug("Phase common, only has userid info")
            os.chdir(self.data_dir)
            fr2 = open('user_list_mod.csv', 'r')
            fw = open('user_list_mod_uid.data', 'w')
        elif phase == "test":
            fr2 = open('mml_test_user_coupon', 'r')
            fw = open('mml_uid_cid.test', 'w')
        else:
            fr2 = open('mml.data', 'r')
            fw = open('mml_uid_cid.data', 'w')

        linenum = 0
        idxuser = 0
        idxcoupon = 0

        for line in fr2:

            if phase != "common":
                arrayindexuser = np.where(userid_hash == line.split(',')[0])
                arrayindexcoupon = np.where(couponid_hash == line.split(',')[1])
                if arrayindexcoupon[0]:
                    idxcoupon = int(couponid[arrayindexcoupon[0]])
            else:
                arrayindexuser = np.where(userid_hash == line.split(',')[3])

            # userid / couponid values (above) start from 1 (test cid starts from 20001).
            # Should get correct values even though idxuser and idxcoupon will start from 0
            if arrayindexuser[0]:
                idxuser = int(userid[arrayindexuser[0]])

            # if linenum < 10:
            #     print idxuser
            #     print idxcoupon
            #     print arrayindexuser
            #     print repr(line.split(',')[0])
            #     print line.split(',')[1]
            # elif 1000 < linenum < 1020:
            #     print idxuser
            #     print idxcoupon
            # else:
            #     raise ValueError("Just stop!")

            if phase == "train":
                fw.write("%d" % idxuser + "," + "%d" % idxcoupon + "," + line.split(',')[2])
            elif phase == "test":  # Test data doesn't have any purchase flags
                fw.write("%d" % idxuser + "," + "%d" % idxcoupon + "\n")
            elif phase == "common":  # Test data doesn't have any purchase flags
                fw.write(line.split(',')[0] + "," + line.split(',')[1] + "," + line.split(',')[2] + "," +
                         "%d" % idxuser + "," + line.split(',')[4])

            linenum += 1

        logger.debug(linenum)
        logger.debug(numusers)
        logger.debug(numcoupons)

        fw.close()
        fr2.close()

        os.chdir(self.cwd)

    # Adds real purchase data back in to files that were modified to add fake 1s (to see how algorithm performs with
    # equal probability of 1s and 0s).
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

    # Create and save numpy arrays to be used by coupon_attributes
    def user_coupon_attributes_preprocess(self):

        os.chdir(self.data_dir)

        # # Save user pref names
        # fr1_fname = 'prefecture_locations.csv'
        #
        # userprefname = np.empty(sf.file_len(fr1_fname)-1, dtype=object)
        # linenum = 0
        #
        # fr1 = open(fr1_fname, 'r')
        #
        # for line in fr1:
        #     if linenum:
        #         userprefname[linenum-1] = line.split(',')[0]
        #     linenum += 1
        #
        # # print prefecture_locations
        # logger.debug(userprefname.shape)
        # logger.debug(userprefname)
        #
        # fr1.close()
        # # Save as numpy array
        # np.save('user_prefecture_names', userprefname)

        # Save Capsule text and genre name
        fr1_fname = 'coupon_list_mod.csv'

        swapcapsuletext = np.empty(sf.file_len(fr1_fname) - 1, dtype=object)
        swapgenrename = np.empty(sf.file_len(fr1_fname) - 1, dtype=object)

        fr1 = open(fr1_fname, 'r')
        linenum = 0

        for line in fr1:
            if linenum:
                swapcapsuletext[linenum - 1] = line.split(',')[0]
                swapgenrename[linenum - 1] = line.split(',')[1]
            linenum += 1

        capsuletext = np.unique(swapcapsuletext, return_index=False)
        genrename = np.unique(swapgenrename, return_index=False)

        # print prefecture_locations
        logger.debug(capsuletext.shape)
        logger.debug(capsuletext)
        logger.debug(genrename.shape)
        logger.debug(genrename)

        fr1.close()
        # Save as numpy array
        np.save('capsule_text', capsuletext)
        np.save('genre_name', genrename)

        # # Save coupon pref and small area names
        # fr1_fname = 'coupon_area.csv'
        #
        # swapcouponprefname = np.empty(sf.file_len(fr1_fname)-1, dtype=object)
        # swapcouponsmallareaname = np.empty(sf.file_len(fr1_fname)-1, dtype=object)
        #
        # linenum = 0
        #
        # fr1 = open(fr1_fname, 'r')
        #
        # for line in fr1:
        #     if linenum:
        #         swapcouponsmallareaname[linenum-1] = line.split(',')[0]
        #         swapcouponprefname[linenum-1] = line.split(',')[1]
        #     linenum += 1
        #
        # couponsmallareaname = np.unique(swapcouponsmallareaname, return_index=False)
        # couponprefname = np.unique(swapcouponprefname, return_index=False)
        #
        # # print prefecture_locations
        # logger.debug(couponprefname.shape)
        # logger.debug(couponprefname)
        # logger.debug(couponsmallareaname.shape)
        # logger.debug(couponsmallareaname)
        # logger.debug(linenum)
        #
        # fr1.close()
        #
        # # Save as numpy array
        # np.save('coupon_prefecture_names', couponprefname)
        # np.save('coupon_small_area_names', couponsmallareaname)

        os.chdir(self.cwd)

    # Create user attributes file in mml format as below
    # [UID1, Attrib] repeat for all attrib that are presnet and ignore others.
    # Loop through all user IDs
    def user_attributes(self):

        # Load npy arrays
        os.chdir(self.data_dir + "/npy_arrays")

        prefnames = np.load("user_prefecture_names.npy")

        total_views_by_user = np.load("total_views_by_user.npy")

        total_purchase_instances_by_user = np.load("total_purchase_instances_by_user.npy")

        temp_purchase_total_by_user = np.load("purchase_total_by_user.npy")

        purchase_total_by_user = temp_purchase_total_by_user[:, 1]

        logger.debug(total_views_by_user.shape)
        logger.debug(total_views_by_user)

        logger.debug(total_purchase_instances_by_user.shape)
        logger.debug(total_purchase_instances_by_user)

        logger.debug(purchase_total_by_user.shape)
        logger.debug(purchase_total_by_user)

        os.chdir(self.data_dir)

        fr1_fname = 'user_list_mod_uid.data'
        fw_fname = 'user_attributes.csv'

        # logger.debug(prefnames)
        # logger.debug(prefnames.shape)

        if logger.getEffectiveLevel() == 10:  # Level 10 is debug
            fr1 = open(fr1_fname, 'r')
            linenum = 0
            for line in fr1:
                if linenum < 20 and line.split(',')[2] in prefnames:
                    print int(np.where(prefnames == line.split(',')[2])[0])
                linenum += 1

            fr1.close()

        fr1 = open(fr1_fname, 'r')
        fw = open(fw_fname, 'w')

        for line in fr1:

            if int(line.split(',')[3]) == 0:  # Due to bug in input file that puts user 1 as 0
                idxuserid = 0
            else:
                idxuserid = int(line.split(',')[3]) - 1

            if line.split(',')[0] == 'm':  # Attrib 1 = Male, 2 = Female
                fw.write("%d" % int(line.split(',')[3]) + ",1" + "\n")
            elif line.split(',')[0] == 'f':  # Attrib 1 = Male, 2 = Female
                fw.write("%d" % int(line.split(',')[3]) + ",2" + "\n")

            # Attrib 3-9 for different age groups based on default marketing age groups
            if int(line.split(',')[1]) <= 17:
                fw.write("%d" % int(line.split(',')[3]) + ",3" + "\n")
            elif int(line.split(',')[1]) < 25:
                fw.write("%d" % int(line.split(',')[3]) + ",4" + "\n")
            elif int(line.split(',')[1]) < 35:
                fw.write("%d" % int(line.split(',')[3]) + ",5" + "\n")
            elif int(line.split(',')[1]) < 45:
                fw.write("%d" % int(line.split(',')[3]) + ",6" + "\n")
            elif int(line.split(',')[1]) < 55:
                fw.write("%d" % int(line.split(',')[3]) + ",7" + "\n")
            elif int(line.split(',')[1]) < 65:
                fw.write("%d" % int(line.split(',')[3]) + ",8" + "\n")
            else:
                fw.write("%d" % int(line.split(',')[3]) + ",9" + "\n")

            # Attrib 10-15 based on number of days user has been a member
            # Membership days > 200 based on quartile statistics
            # Lower membership days has more finer range as this might impact buying behavior
            # versus longer term members who are probably more predictable
            if int(line.split(',')[4]) < 50:
                fw.write("%d" % int(line.split(',')[3]) + ",10" + "\n")
            elif int(line.split(',')[4]) < 100:
                fw.write("%d" % int(line.split(',')[3]) + ",11" + "\n")
            elif int(line.split(',')[4]) < 200:  # Lower quartile
                fw.write("%d" % int(line.split(',')[3]) + ",12" + "\n")
            elif int(line.split(',')[4]) < 350:  # Median
                fw.write("%d" % int(line.split(',')[3]) + ",13" + "\n")
            elif int(line.split(',')[4]) < 500:  # Upper quartile
                fw.write("%d" % int(line.split(',')[3]) + ",14" + "\n")
            else:
                fw.write("%d" % int(line.split(',')[3]) + ",15" + "\n")

            # Attrib 16-63 are for the different pref names
            if line.split(',')[2]:
                fw.write("%d" % int(line.split(',')[3]) + "," + "%d" %
                         (16 + int(np.where(prefnames == line.split(',')[2])[0])) + "\n")

            # Attrib 64-79 for total coupon views by user
            if total_views_by_user[idxuserid] == 0:
                fw.write("%d" % int(line.split(',')[3]) + ",64" + "\n")
            elif total_views_by_user[idxuserid] < 5:
                fw.write("%d" % int(line.split(',')[3]) + ",65" + "\n")
            elif total_views_by_user[idxuserid] < 10:
                fw.write("%d" % int(line.split(',')[3]) + ",66" + "\n")
            elif total_views_by_user[idxuserid] < 15:
                fw.write("%d" % int(line.split(',')[3]) + ",67" + "\n")
            elif total_views_by_user[idxuserid] < 25:
                fw.write("%d" % int(line.split(',')[3]) + ",68" + "\n")
            elif total_views_by_user[idxuserid] < 40:
                fw.write("%d" % int(line.split(',')[3]) + ",69" + "\n")
            elif total_views_by_user[idxuserid] < 60:
                fw.write("%d" % int(line.split(',')[3]) + ",70" + "\n")
            elif total_views_by_user[idxuserid] < 90:
                fw.write("%d" % int(line.split(',')[3]) + ",71" + "\n")
            elif total_views_by_user[idxuserid] < 140:
                fw.write("%d" % int(line.split(',')[3]) + ",72" + "\n")
            elif total_views_by_user[idxuserid] < 210:
                fw.write("%d" % int(line.split(',')[3]) + ",73" + "\n")
            elif total_views_by_user[idxuserid] < 300:
                fw.write("%d" % int(line.split(',')[3]) + ",74" + "\n")
            elif total_views_by_user[idxuserid] < 500:
                fw.write("%d" % int(line.split(',')[3]) + ",75" + "\n")
            elif total_views_by_user[idxuserid] < 750:
                fw.write("%d" % int(line.split(',')[3]) + ",76" + "\n")
            elif total_views_by_user[idxuserid] < 1000:
                fw.write("%d" % int(line.split(',')[3]) + ",77" + "\n")
            elif total_views_by_user[idxuserid] < 1500:
                fw.write("%d" % int(line.split(',')[3]) + ",78" + "\n")
            else:
                fw.write("%d" % int(line.split(',')[3]) + ",79" + "\n")

            # Attrib 80-92 for coupon purchase total by user
            if purchase_total_by_user[idxuserid] == 0:
                fw.write("%d" % int(line.split(',')[3]) + ",80" + "\n")
            elif purchase_total_by_user[idxuserid] < 3:
                fw.write("%d" % int(line.split(',')[3]) + ",81" + "\n")
            elif purchase_total_by_user[idxuserid] < 5:
                fw.write("%d" % int(line.split(',')[3]) + ",82" + "\n")
            elif purchase_total_by_user[idxuserid] < 10:
                fw.write("%d" % int(line.split(',')[3]) + ",83" + "\n")
            elif purchase_total_by_user[idxuserid] < 15:
                fw.write("%d" % int(line.split(',')[3]) + ",84" + "\n")
            elif purchase_total_by_user[idxuserid] < 20:
                fw.write("%d" % int(line.split(',')[3]) + ",85" + "\n")
            elif purchase_total_by_user[idxuserid] < 30:
                fw.write("%d" % int(line.split(',')[3]) + ",86" + "\n")
            elif purchase_total_by_user[idxuserid] < 50:
                fw.write("%d" % int(line.split(',')[3]) + ",87" + "\n")
            elif purchase_total_by_user[idxuserid] < 75:
                fw.write("%d" % int(line.split(',')[3]) + ",88" + "\n")
            elif purchase_total_by_user[idxuserid] < 100:
                fw.write("%d" % int(line.split(',')[3]) + ",89" + "\n")
            elif purchase_total_by_user[idxuserid] < 150:
                fw.write("%d" % int(line.split(',')[3]) + ",90" + "\n")
            elif purchase_total_by_user[idxuserid] < 200:
                fw.write("%d" % int(line.split(',')[3]) + ",91" + "\n")
            else:
                fw.write("%d" % int(line.split(',')[3]) + ",92" + "\n")

            # Attrib 93-101 for coupon purchase total by user
            if total_purchase_instances_by_user[idxuserid] == 0:
                fw.write("%d" % int(line.split(',')[3]) + ",93" + "\n")
            elif total_purchase_instances_by_user[idxuserid] == 1:
                fw.write("%d" % int(line.split(',')[3]) + ",94" + "\n")
            elif total_purchase_instances_by_user[idxuserid] == 2:
                fw.write("%d" % int(line.split(',')[3]) + ",95" + "\n")
            elif total_purchase_instances_by_user[idxuserid] < 4:
                fw.write("%d" % int(line.split(',')[3]) + ",96" + "\n")
            elif total_purchase_instances_by_user[idxuserid] < 10:
                fw.write("%d" % int(line.split(',')[3]) + ",97" + "\n")
            elif total_purchase_instances_by_user[idxuserid] < 20:
                fw.write("%d" % int(line.split(',')[3]) + ",98" + "\n")
            elif total_purchase_instances_by_user[idxuserid] < 40:
                fw.write("%d" % int(line.split(',')[3]) + ",99" + "\n")
            elif total_purchase_instances_by_user[idxuserid] < 60:
                fw.write("%d" % int(line.split(',')[3]) + ",100" + "\n")
            else:
                fw.write("%d" % int(line.split(',')[3]) + ",101" + "\n")

        fr1.close()
        fw.close()

        os.chdir(self.cwd)

    # Create coupon attributes file in mml format as below
    # [CID1, Attrib] repeat for all attrib that are present and ignore others.
    # Loop through all coupon IDs
    def coupon_attributes(self):

        # Load npy arrays
        os.chdir(self.data_dir + "/npy_arrays")

        prefnames = np.load("coupon_prefecture_names.npy")
        smallareanames = np.load("coupon_small_area_names.npy")
        capsuletext = np.load("capsule_text.npy")
        genrename = np.load("genre_name.npy")

        os.chdir(self.data_dir)

        # Each coupon is listed in multiple pref / small area names.
        # This data for both train and test coupons is in coupon_area.csv, using that for attributes
        couponarea_fname = "coupon_area_cid.csv"

        prefnamesbycoupon = dict()

        smallareanamesbycoupon = dict()

        couponarea_file = open(couponarea_fname, 'r')

        couponarea_file_linenum = 0
        for couponarea_file_line in couponarea_file:
            # Check if we are looking at right coupon_id
            if couponarea_file_linenum:
                if int(couponarea_file_line.split(',')[2]) in prefnamesbycoupon.keys():
                    prefnamesbycoupon[int(couponarea_file_line.split(',')[2])].append(
                        couponarea_file_line.split(',')[1])
                else:  # Initialize dict[key] with first value in list and then append as above
                    prefnamesbycoupon[int(couponarea_file_line.split(',')[2])] = [couponarea_file_line.split(',')[1]]

                if int(couponarea_file_line.split(',')[2]) in smallareanamesbycoupon.keys():
                    smallareanamesbycoupon[int(couponarea_file_line.split(',')[2])].append(
                        couponarea_file_line.split(',')[0])
                else:  # Initialize dict[key] with first value in list and then append as above
                    smallareanamesbycoupon[int(couponarea_file_line.split(',')[2])] = [
                        couponarea_file_line.split(',')[0]]

                if couponarea_file_linenum == 1:
                    logger.debug(couponarea_file_line.split(',')[0])
                    logger.debug(couponarea_file_line.split(',')[1])
                    logger.debug(int(couponarea_file_line.split(',')[2]))
                    logger.debug(couponarea_file_line.split(',')[3])
            couponarea_file_linenum += 1

        couponarea_file.close()

        fr1_fname = 'coupon_list_mod.csv'
        fw_fname = 'coupon_attributes_1011.csv'

        # Both arrays' first elements starts with WEB which we want to remove
        logger.debug("Removing characters WEB from string")
        capsuletext[0] = capsuletext[0][3:]

        # logger.debug(prefnames)
        # logger.debug(prefnames.shape)
        # logger.debug(smallareanames)
        # logger.debug(smallareanames.shape)
        # logger.debug(capsuletext)
        # logger.debug(capsuletext.shape)
        # logger.debug(genrename)
        # logger.debug(genrename.shape)

        # Number of attributes needed for each feature
        numofprefnamesattributes = int(prefnames.shape[0])
        numofsmallareanamesattributes = int(smallareanames.shape[0])
        numofcapsuletextattributes = int(capsuletext.shape[0])
        numofgenrenameattributes = int(genrename.shape[0])

        logger.debug(numofsmallareanamesattributes)
        logger.debug(numofprefnamesattributes)
        logger.debug(numofgenrenameattributes)
        logger.debug(numofcapsuletextattributes)

        if logger.getEffectiveLevel() == 10:  # Level 10 is debug
            fr1 = open(fr1_fname, 'r')
            linenum = 0
            for line in fr1:
                if linenum < 10:
                    if line.split(',')[0] in capsuletext:
                        print int(np.where(capsuletext == line.split(',')[0])[0])
                    if line.split(',')[1] in genrename:
                        print int(np.where(genrename == line.split(',')[1])[0])
                    if line.split(',')[22] in prefnames:
                        print int(np.where(prefnames == line.split(',')[22])[0])
                    if line.split(',')[23] in smallareanames:
                        print int(np.where(smallareanames == line.split(',')[23])[0])
                    print("####")
                linenum += 1
            print linenum
            fr1.close()

        fr1 = open(fr1_fname, 'r')
        fw = open(fw_fname, 'w')
        linenum = 0

        for line in fr1:
            if linenum:

                # TODO - Look for better cleaner implementation
                if int(line.split(',')[2]) <= 5:  # Attrib discount rate %
                    fw.write("%d" % int(line.split(',')[25]) + ",1" + "\n")
                elif int(line.split(',')[2]) <= 10:
                    fw.write("%d" % int(line.split(',')[25]) + ",2" + "\n")
                elif int(line.split(',')[2]) <= 20:
                    fw.write("%d" % int(line.split(',')[25]) + ",3" + "\n")
                elif int(line.split(',')[2]) <= 30:
                    fw.write("%d" % int(line.split(',')[25]) + ",4" + "\n")
                elif int(line.split(',')[2]) <= 40:
                    fw.write("%d" % int(line.split(',')[25]) + ",5" + "\n")
                elif int(line.split(',')[2]) <= 50:
                    fw.write("%d" % int(line.split(',')[25]) + ",6" + "\n")
                elif int(line.split(',')[2]) <= 60:
                    fw.write("%d" % int(line.split(',')[25]) + ",7" + "\n")
                elif int(line.split(',')[2]) <= 70:
                    fw.write("%d" % int(line.split(',')[25]) + ",8" + "\n")
                elif int(line.split(',')[2]) <= 80:
                    fw.write("%d" % int(line.split(',')[25]) + ",9" + "\n")
                elif int(line.split(',')[2]) <= 90:
                    fw.write("%d" % int(line.split(',')[25]) + ",10" + "\n")
                else:
                    fw.write("%d" % int(line.split(',')[25]) + ",11" + "\n")

                # Attrib pre sale / regular price
                if int(line.split(',')[3]) <= 500:  # Attrib discount rate %
                    fw.write("%d" % int(line.split(',')[25]) + ",12" + "\n")
                elif int(line.split(',')[3]) <= 1000:
                    fw.write("%d" % int(line.split(',')[25]) + ",13" + "\n")
                elif int(line.split(',')[3]) <= 1500:
                    fw.write("%d" % int(line.split(',')[25]) + ",14" + "\n")
                elif int(line.split(',')[3]) <= 2000:
                    fw.write("%d" % int(line.split(',')[25]) + ",15" + "\n")
                elif int(line.split(',')[3]) <= 2500:
                    fw.write("%d" % int(line.split(',')[25]) + ",16" + "\n")
                elif int(line.split(',')[3]) <= 3000:
                    fw.write("%d" % int(line.split(',')[25]) + ",17" + "\n")
                elif int(line.split(',')[3]) <= 3750:
                    fw.write("%d" % int(line.split(',')[25]) + ",18" + "\n")
                elif int(line.split(',')[3]) <= 4750:
                    fw.write("%d" % int(line.split(',')[25]) + ",19" + "\n")
                elif int(line.split(',')[3]) <= 5750:
                    fw.write("%d" % int(line.split(',')[25]) + ",20" + "\n")
                elif int(line.split(',')[3]) <= 8000:
                    fw.write("%d" % int(line.split(',')[25]) + ",21" + "\n")
                elif int(line.split(',')[3]) <= 10000:
                    fw.write("%d" % int(line.split(',')[25]) + ",22" + "\n")
                elif int(line.split(',')[3]) <= 13000:
                    fw.write("%d" % int(line.split(',')[25]) + ",23" + "\n")
                elif int(line.split(',')[3]) <= 20000:
                    fw.write("%d" % int(line.split(',')[25]) + ",24" + "\n")
                else:
                    fw.write("%d" % int(line.split(',')[25]) + ",25" + "\n")

                # Coupon display period (days)
                if int(line.split(',')[7]) <= 1:
                    fw.write("%d" % int(line.split(',')[25]) + ",26" + "\n")
                elif int(line.split(',')[7]) <= 2:
                    fw.write("%d" % int(line.split(',')[25]) + ",27" + "\n")
                elif int(line.split(',')[7]) <= 3:
                    fw.write("%d" % int(line.split(',')[25]) + ",28" + "\n")
                elif int(line.split(',')[7]) <= 4:
                    fw.write("%d" % int(line.split(',')[25]) + ",29" + "\n")
                elif int(line.split(',')[7]) <= 7:
                    fw.write("%d" % int(line.split(',')[25]) + ",30" + "\n")
                elif int(line.split(',')[7]) <= 11:
                    fw.write("%d" % int(line.split(',')[25]) + ",31" + "\n")
                elif int(line.split(',')[7]) <= 15:
                    fw.write("%d" % int(line.split(',')[25]) + ",32" + "\n")
                else:
                    fw.write("%d" % int(line.split(',')[25]) + ",33" + "\n")

                # Coupon display to valid date - New feature we came up with
                if int(line.split(',')[8]) <= 1:
                    fw.write("%d" % int(line.split(',')[25]) + ",34" + "\n")
                elif int(line.split(',')[8]) <= 2:
                    fw.write("%d" % int(line.split(',')[25]) + ",35" + "\n")
                elif int(line.split(',')[8]) <= 3:
                    fw.write("%d" % int(line.split(',')[25]) + ",36" + "\n")
                elif int(line.split(',')[8]) <= 4:
                    fw.write("%d" % int(line.split(',')[25]) + ",37" + "\n")
                elif int(line.split(',')[8]) <= 5:
                    fw.write("%d" % int(line.split(',')[25]) + ",38" + "\n")
                elif int(line.split(',')[8]) <= 6:
                    fw.write("%d" % int(line.split(',')[25]) + ",39" + "\n")
                elif int(line.split(',')[8]) <= 9:
                    fw.write("%d" % int(line.split(',')[25]) + ",40" + "\n")
                elif int(line.split(',')[8]) <= 15:
                    fw.write("%d" % int(line.split(',')[25]) + ",41" + "\n")
                elif int(line.split(',')[8]) <= 30:
                    fw.write("%d" % int(line.split(',')[25]) + ",42" + "\n")
                elif int(line.split(',')[8]) <= 45:
                    fw.write("%d" % int(line.split(',')[25]) + ",43" + "\n")
                elif int(line.split(',')[8]) == 999:
                    fw.write("%d" % int(line.split(',')[25]) + ",44" + "\n")
                else:
                    fw.write("%d" % int(line.split(',')[25]) + ",45" + "\n")

                # Coupon validity period
                if int(line.split(',')[11]) <= 5:
                    fw.write("%d" % int(line.split(',')[25]) + ",46" + "\n")
                elif int(line.split(',')[11]) <= 10:
                    fw.write("%d" % int(line.split(',')[25]) + ",47" + "\n")
                elif int(line.split(',')[11]) <= 30:
                    fw.write("%d" % int(line.split(',')[25]) + ",48" + "\n")
                elif int(line.split(',')[11]) <= 60:
                    fw.write("%d" % int(line.split(',')[25]) + ",49" + "\n")
                elif int(line.split(',')[11]) <= 90:
                    fw.write("%d" % int(line.split(',')[25]) + ",50" + "\n")
                elif int(line.split(',')[11]) <= 130:
                    fw.write("%d" % int(line.split(',')[25]) + ",51" + "\n")
                elif int(line.split(',')[11]) <= 180:
                    fw.write("%d" % int(line.split(',')[25]) + ",52" + "\n")
                else:
                    fw.write("%d" % int(line.split(',')[25]) + ",53" + "\n")

                # Usable days Mon - Sun. Have no idea what a value of 2 means
                if line.split(',')[12] != "NA":
                    if int(line.split(',')[12]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",54" + "\n")
                    elif int(line.split(',')[12]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",55" + "\n")

                if line.split(',')[13] != "NA":
                    if int(line.split(',')[13]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",56" + "\n")
                    elif int(line.split(',')[13]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",57" + "\n")

                if line.split(',')[14] != "NA":
                    if int(line.split(',')[14]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",58" + "\n")
                    elif int(line.split(',')[14]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",59" + "\n")

                if line.split(',')[15] != "NA":
                    if int(line.split(',')[15]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",60" + "\n")
                    elif int(line.split(',')[15]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",61" + "\n")

                if line.split(',')[16] != "NA":
                    if int(line.split(',')[16]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",62" + "\n")
                    elif int(line.split(',')[16]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",63" + "\n")

                if line.split(',')[17] != "NA":
                    if int(line.split(',')[17]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",64" + "\n")
                    elif int(line.split(',')[17]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",65" + "\n")

                if line.split(',')[18] != "NA":
                    if int(line.split(',')[18]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",66" + "\n")
                    elif int(line.split(',')[18]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",67" + "\n")

                # Usable date holiday
                if line.split(',')[19] != "NA":
                    if int(line.split(',')[19]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",68" + "\n")
                    elif int(line.split(',')[19]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",69" + "\n")

                # Usable date before holiday
                if line.split(',')[20] != "NA":
                    if int(line.split(',')[20]) == 1:
                        fw.write("%d" % int(line.split(',')[25]) + ",70" + "\n")
                    elif int(line.split(',')[20]) == 2:
                        fw.write("%d" % int(line.split(',')[25]) + ",71" + "\n")

                # Capsule text
                startattributenum = 72
                if line.split(',')[0] in capsuletext:
                    fw.write("%d" % int(line.split(',')[25]) + ",%d" % (startattributenum +
                                                                        int(np.where(capsuletext == line.split(',')[0])[
                                                                                0])) + "\n")

                startattributenum += numofcapsuletextattributes
                if linenum == 1:
                    print startattributenum
                if line.split(',')[1] in genrename:
                    fw.write("%d" % int(line.split(',')[25]) + ",%d" % (startattributenum +
                                                                        int(np.where(genrename == line.split(',')[1])[
                                                                                0])) + "\n")

                startattributenum += numofgenrenameattributes

                # Iterate over all the prefnames listed for a give couponid, if that couponid has any pref info
                if int(line.split(',')[25]) in prefnamesbycoupon.keys():
                    for pref in prefnamesbycoupon[int(line.split(',')[25])]:
                        if pref in prefnames:
                            fw.write("%d" % int(line.split(',')[25]) + ",%d" % (
                                startattributenum + int(np.where(prefnames == pref)[0])) + "\n")

                startattributenum += numofprefnamesattributes

                # Iterate over all the smallareanames listed for a give couponid, if that couponid has any sa info
                if int(line.split(',')[25]) in smallareanamesbycoupon.keys():
                    for smallarea in smallareanamesbycoupon[int(line.split(',')[25])]:
                        if smallarea in smallareanames:
                            fw.write("%d" % int(line.split(',')[25]) + ",%d" % (
                                startattributenum + int(np.where(smallareanames == smallarea)[0])) + "\n")

                # if couponarea_file_line.split(',')[0] in smallareanames:
                #     fw.write("%d" % int(line.split(',')[25]) + ",%d" % (startattributenum +
                #                                                         int(np.where(smallareanames ==
                #                                                                      couponarea_file_line.split(
                #                                                                          ',')[0])[0])) + "\n")

                if linenum == 1:  # Print just once
                    logger.debug("Total number of attributes = %d", (startattributenum +
                                                                     numofsmallareanamesattributes))

            linenum += 1

        fr1.close()
        fw.close()

        os.chdir(self.cwd)

    def output_rating_pred_to_kaggle_format(self, modelpurchasethreshold=0.1, couponsperuserlimit=5):

        # modelpurchasethreshold - cutoff to decide if a coupon was purchased or not

        os.chdir(self.data_dir)

        numusers = 22873
        numtestcoupons = 310

        # Uncomment to create new user / coupon id/ hash files. Or load those arrays from saved npy files

        # numcoupons = 19723  # train + test coupons

        # userid = np.zeros(numusers, int)
        # couponid = np.zeros(numcoupons, int)
        # userid_hash = np.empty(numusers, dtype=object)
        # couponid_hash = np.empty(numcoupons, dtype=object)

        # fr1_fname = "user_cou_decode.csv"
        #
        # linenum = 0
        # numusers = 22873
        # numcoupons = 19723  # train + test coupons
        # numtestcoupons = 310
        #
        # fr1 = open(fr1_fname, 'r')
        #
        # for line in fr1:
        #
        #     if linenum < numcoupons:
        #         userid_hash[linenum] = line.split(',')[0]
        #         userid[linenum] = int(line.split(',')[1])
        #         couponid_hash[linenum] = line.split(',')[2]
        #         couponid[linenum] = int(line.split(',')[3])
        #     elif linenum < numusers:
        #         userid_hash[linenum] = line.split(',')[0]
        #         userid[linenum] = int(line.split(',')[1])
        #
        #     linenum += 1
        #
        # logger.debug(userid)
        # logger.debug(couponid)
        # logger.debug(userid_hash.shape)
        # logger.debug(userid_hash)
        #
        # fr1.close()
        #
        # os.chdir("%s" % self.data_dir + "/output_files")
        #
        # np.save("user_uid_combined", userid)
        # np.save("user_hash_combined", userid_hash)
        # np.save("coupon_cid_combined", couponid)
        # np.save("coupon_hash_combined", couponid_hash)
        #
        # os.chdir(self.data_dir)
        #
        # return __pass__

        os.chdir("%s" % self.data_dir + "/npy_arrays")

        userid = np.load("user_uid_combined.npy")
        userid_hash = np.load("user_hash_combined.npy")
        couponid = np.load("coupon_cid_combined.npy")
        couponid_hash = np.load("coupon_hash_combined.npy")

        print(couponid_hash.shape)
        print(couponid_hash)

        linenum = 0
        testcouponstartindex = 19413
        testcouponendindex = 19722
        numcouponspurchased = 0

        # purchased coupons by userid / test couponid
        purchasedcoupons = np.empty((numusers, numtestcoupons), dtype=object)

        print purchasedcoupons.shape

        print purchasedcoupons

        os.chdir("%s" % self.data_dir + "/output_files")

        fr2_fname = "mml_pred"

        fr2 = open(fr2_fname, 'r')

        logger.debug(couponid[testcouponstartindex])
        logger.debug(couponid[testcouponendindex])
        userindex = 0
        overweighteduser = 0
        newuserlinenum = 0

        for line in fr2:

            if userindex != int(line.split('\t')[0]) - 1:
                overweighteduser = 0
                newuserlinenum = 0

            # test coupons start from index 19413 bu have values starting from 20001
            userindex = int(line.split('\t')[0]) - 1
            couponidtestcouponindex = testcouponstartindex + (int(line.split('\t')[1]) - 20001)
            testcouponindex = int(line.split('\t')[1]) - 20001

            if logger.getEffectiveLevel() == 10 and linenum < 10:  # logger level 10 is debug
                if int(line.split('\t')[0]) in userid:
                    print int(line.split('\t')[0])
                    print userindex
                    print userid[userindex]
                    print userid_hash[userindex]
                    print linenum
                if int(line.split('\t')[1]) in couponid:
                    print int(line.split('\t')[1])
                    print testcouponindex
                    print couponid[testcouponindex]
                    print couponid_hash[testcouponindex]
                print int(line.split('\t')[0])

            if float(line.split('\t')[2]) > modelpurchasethreshold and newuserlinenum < 5 \
                    and testcouponstartindex <= couponidtestcouponindex < testcouponstartindex + 5:
                overweighteduser += 1
                # print "####"
                # print couponidtestcouponindex
                # print testcouponstartindex
                # return __pass__

            if float(line.split('\t')[2]) > modelpurchasethreshold and overweighteduser < 5:
                numcouponspurchased += 1
                purchasedcoupons[userindex, testcouponindex] = couponid_hash[couponidtestcouponindex]

            linenum += 1
            newuserlinenum += 1
        fr2.close()

        logger.debug("Total coupons purchased uncut %d", numcouponspurchased)

        os.chdir("%s" % self.data_dir + "/output_files")

        fw_fname = "gsvdpp_30i_attr_sat.csv"
        # fw_fname = "gsvdpp_second_input.csv"
        fw = open(fw_fname, 'w')
        fw.write("USER_ID_hash,PURCHASED_COUPONS\n")

        numcouponspurchased = 0

        for i in range(numusers):
            couponsperuser = 1
            fw.write(userid_hash[i] + ",")
            for j in range(numtestcoupons):
                if couponsperuser <= couponsperuserlimit and purchasedcoupons[i, j]:
                    # print purchasedcoupons[i, j]
                    fw.write(purchasedcoupons[i, j] + " ")
                    # fw.write("%d" % (i + 1) + "," + "%d" % (j + testcouponstartindex + 1) + "\n")
                    numcouponspurchased += 1
                    couponsperuser += 1
            fw.write("\n")

        fw.close()

        logger.debug("Total coupons purchased %d", numcouponspurchased)

        os.chdir(self.cwd)

        return None

    def output_item_rec_to_kaggle_format(self, modelpurchasethreshold=1.0):

        # modelpurchasethreshold - cutoff to decide if a coupon was purchased or not

        os.chdir(self.data_dir)

        # Uncomment to create new user / coupon id/ hash files. Or load those arrays from save npy files

        # numusers = 22873
        # numtestcoupons = 310
        # numcoupons = 19723  # train + test coupons

        # userid = np.zeros(numusers, int)
        # couponid = np.zeros(numcoupons, int)
        # userid_hash = np.empty(numusers, dtype=object)
        # couponid_hash = np.empty(numcoupons, dtype=object)

        # fr1_fname = "user_cou_decode.csv"
        #
        # linenum = 0
        # numusers = 22873
        # numcoupons = 19723  # train + test coupons
        # numtestcoupons = 310
        #
        # fr1 = open(fr1_fname, 'r')
        #
        # for line in fr1:
        #
        #     if linenum < numcoupons:
        #         userid_hash[linenum] = line.split(',')[0]
        #         userid[linenum] = int(line.split(',')[1])
        #         couponid_hash[linenum] = line.split(',')[2]
        #         couponid[linenum] = int(line.split(',')[3])
        #     elif linenum < numusers:
        #         userid_hash[linenum] = line.split(',')[0]
        #         userid[linenum] = int(line.split(',')[1])
        #
        #     linenum += 1
        #
        # logger.debug(userid)
        # logger.debug(couponid)
        # logger.debug(userid_hash.shape)
        # logger.debug(userid_hash)
        #
        # fr1.close()
        #
        # os.chdir("%s" % self.data_dir + "/output_files")
        #
        # np.save("user_uid_combined", userid)
        # np.save("user_hash_combined", userid_hash)
        # np.save("coupon_cid_combined", couponid)
        # np.save("coupon_hash_combined", couponid_hash)
        #
        # os.chdir(self.data_dir)
        #
        # return __pass__

        os.chdir("%s" % self.data_dir + "/npy_arrays")

        userid_hash = np.load("user_hash_combined.npy")
        couponid = np.load("coupon_cid_combined.npy")
        couponid_hash = np.load("coupon_hash_combined.npy")

        logger.debug(couponid_hash.shape)
        logger.debug(couponid_hash)

        linenum = 0
        testcouponstartindex = 19413
        testcouponendindex = 19722
        userindex = 0
        numcouponspurchased = 0

        os.chdir("%s" % self.data_dir + "/output_files")

        fr2_fname = "mml_itemr.pred"

        fr2 = open(fr2_fname, 'r')

        fw_fname = "itemattrknn_itemr.csv"
        fw = open(fw_fname, 'w')
        fw.write("USER_ID_hash,PURCHASED_COUPONS\n")

        logger.debug(couponid[testcouponstartindex])
        logger.debug(couponid[testcouponendindex])

        for line in fr2:
            # If model run without --test-users option or the option without all users, input unique training data
            # won't have info for some users who have never viewed a coupon or missing in the userlist.
            # They would have been skipped and user num sequence broken
            prevuserindex = userindex

            userindex = int(line.split('[')[0])

            if userindex == 0:
                fw.write("%s" % userid_hash[userindex] + ", ")
            else:
                # Uncomment if uncommenting prevuserindex above
                if userindex != prevuserindex + 1 and prevuserindex:
                    fw.write("%s" % userid_hash[prevuserindex] + ",\n")

                fw.write("%s" % userid_hash[userindex - 1] + ", ")

            for i in range(9):
                if line.split('[')[1] != ']\n':  # Check if user has any coupon recommendation
                    couponidfrominputfile = int(line.split('[')[1].split(',')[i].split(':')[0])
                else:
                    continue  # skip this user

                if couponidfrominputfile >= 20001:
                    couponidtestcouponindex = testcouponstartindex + couponidfrominputfile - 20001
                else:
                    logger.error(line)
                    logger.error("CouponId: %d", couponidfrominputfile)
                    raise ValueError("Invalid couponId, has to be >= 20001")

                if float(line.split('[')[1].split(',')[i].split(':')[1]) > modelpurchasethreshold:
                    fw.write("%s" % couponid_hash[couponidtestcouponindex] + " ")
                    numcouponspurchased += 1

            fw.write("\n")

            linenum += 1

        fr2.close()
        fw.close()

        logger.debug("Total coupons purchased uncut %d", numcouponspurchased)

        os.chdir(self.cwd)

    def avg_output_rating_pred_to_kaggle_format(self, modelpurchasethreshold=0.1, couponsperuserlimit=5):

        # modelpurchasethreshold - cutoff to decide if a coupon was purchased or not

        os.chdir("%s" % self.data_dir + "/output_files")

        userid = np.load("user_uid_combined.npy")
        userid_hash = np.load("user_hash_combined.npy")
        couponid = np.load("coupon_cid_combined.npy")
        couponid_hash = np.load("coupon_hash_combined.npy")

        numusers = 22873
        numtestcoupons = 310
        testcouponstartindex = 19413
        testcouponendindex = 19722
        numcouponspurchased = 0

        # purchased coupons by userid / test couponid
        purchasedcoupons = np.empty((numusers, numtestcoupons), dtype=object)

        print purchasedcoupons.shape

        print purchasedcoupons

        avg1_fname = "mml_pred"
        avg2_fname = "mml_pred2"

        logger.debug(couponid[testcouponstartindex])
        logger.debug(couponid[testcouponendindex])

        avg2_file = open(avg2_fname, 'r')

        file2modeloutput = np.zeros(sf.file_len(avg2_fname), float)

        linenum = 0

        for line in avg2_file:

            # test coupons start from index 19413 bu have values starting from 20001
            userindex = int(line.split('\t')[0]) - 1
            testcouponindex = int(line.split('\t')[1]) - 20001

            if logger.getEffectiveLevel() == 10 and linenum < 10:  # logger level 10 is debug
                if int(line.split('\t')[0]) in userid:
                    print int(line.split('\t')[0])
                    print userindex
                    print userid[userindex]
                    print userid_hash[userindex]
                    print linenum
                if int(line.split('\t')[1]) in couponid:
                    print int(line.split('\t')[1])
                    print testcouponindex
                    print couponid[testcouponindex]
                    print couponid_hash[testcouponindex]
                print int(line.split('\t')[0])

            file2modeloutput[linenum] = float(line.split('\t')[2])

            linenum += 1
        avg2_file.close()

        print file2modeloutput.shape
        print file2modeloutput

        avg1_file = open(avg1_fname, 'r')

        linenum = 0
        for line in avg1_file:

            # test coupons start from index 19413 bu have values starting from 20001
            userindex = int(line.split('\t')[0]) - 1
            couponidtestcouponindex = testcouponstartindex + (int(line.split('\t')[1]) - 20001)
            testcouponindex = int(line.split('\t')[1]) - 20001

            if logger.getEffectiveLevel() == 10 and linenum < 10:  # logger level 10 is debug
                if int(line.split('\t')[0]) in userid:
                    print int(line.split('\t')[0])
                    print userindex
                    print userid[userindex]
                    print userid_hash[userindex]
                    print linenum
                if int(line.split('\t')[1]) in couponid:
                    print int(line.split('\t')[1])
                    print testcouponindex
                    print couponid[testcouponindex]
                    print couponid_hash[testcouponindex]
                print int(line.split('\t')[0])

            avgmodeloutput = (file2modeloutput[linenum] + float(line.split('\t')[2])) / 2

            if avgmodeloutput > modelpurchasethreshold:
                numcouponspurchased += 1
                purchasedcoupons[userindex, testcouponindex] = couponid_hash[couponidtestcouponindex]

            linenum += 1
        avg1_file.close()

        logger.debug("Total coupons purchased uncut %d", numcouponspurchased)

        os.chdir("%s" % self.data_dir + "/output_files")

        fw_fname = "gsvdpp_output_avg.csv"
        fw = open(fw_fname, 'w')
        fw.write("USER_ID_hash,PURCHASED_COUPONS\n")

        numcouponspurchased = 0

        for i in range(numusers):
            couponsperuser = 1
            fw.write(userid_hash[i] + ",")
            for j in range(numtestcoupons):
                if couponsperuser <= couponsperuserlimit and purchasedcoupons[i, j]:
                    # print purchasedcoupons[i, j]
                    fw.write(purchasedcoupons[i, j] + " ")
                    numcouponspurchased += 1
                    couponsperuser += 1
            fw.write("\n")

        logger.debug("Total coupons purchased %d", numcouponspurchased)
        os.chdir(self.cwd)
        return None


if __name__ == "__main__":
    mdf = MmlDataFormat()

    # List of functions available. Go to implementation to see more comments on what each one does

    # mdf.mml_train_unique_data()

    # mdf.mml_uid_cid()

    # mdf.coupon_area_cid()

    # mdf.test_data_uid_cid()

    # mdf.train_viewed_data_uid_cid()

    # mdf.mml_cv_set_coupons()

    # mdf.user_coupon_attributes_preprocess()

    # mdf.user_attributes()

    # mdf.coupon_attributes()

    # mdf.mml_test_data()

    # mdf.output_rating_pred_to_kaggle_format(modelpurchasethreshold=0.07, couponsperuserlimit=10)

    mdf.output_item_rec_to_kaggle_format(modelpurchasethreshold=0.1)

    # mdf.avg_output_rating_pred_to_kaggle_format(modelpurchasethreshold=0.2, couponsperuserlimit=5)

    # mdf.random_split_test_file()
