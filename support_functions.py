__author__ = 'DataCentric1'
#########################################################################################################
#  Description: Collection of support functions that'll be used often
#
#########################################################################################################


#  Returns number of lines in a file in a memory / time efficient way
def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f, 1):
            pass
    return i
