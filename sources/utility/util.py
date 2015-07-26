################################################################################
#                                                                              #
#                            Utility Module:                                   #
#                                                                              #
################################################################################
#                                                                              #
# This module defines different utility data structures and implements         #
# different utility functions                                                  #
#                                                                              #
################################################################################





#------------------------------------------------------------------------------#
# import required python modules here                                          #
#------------------------------------------------------------------------------#
import os
import sys
import numpy as np
from scipy.spatial import distance as sci_dist
from enum import Enum
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# enum class which distinguishes different recommender systems
#------------------------------------------------------------------------------#
class LKind(Enum):
    linear_regression = 1
    logistic_regression = 2
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def compute_mean_squared_error(o_vec, p_vec):
    io_vec = [int(i) for i in o_vec]
    ip_vec = [int(i) for i in p_vec]
    d_vec = np.absolute(np.subtract(io_vec, ip_vec))
    sqd_vec = np.square(d_vec)
    sum_sqd_vec = np.sum(sqd_vec)
    n = len(o_vec)
    mse = sum_sqd_vec / n
    return mse
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def compute_sum_squared_error(i_mat, cluster, final_centroids):
    sse = 0.0
    for key, val_list in cluster.items():
        centroid = final_centroids[key]
        for index in val_list:
            sse += sci_dist.euclidean(i_mat[index], centroid)
    return sse
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def compute_error_and_accuracy(o_vec, p_vec):
    false_positive = 0
    false_negative = 0
    io_vec = [int(i) for i in o_vec]
    ip_vec = [int(i) for i in p_vec]
    N = len(o_vec)
    for r in range(0, N):
        if io_vec[r] == 1 and ip_vec[r] == 0:
           false_negative += 1
        elif io_vec[r] == 0 and ip_vec[r] == 1:
           false_positive += 1
    error = (false_positive + false_negative) / N
    accuracy = 1 - error
    return error, accuracy
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def median(row):
    srow = np.sort(row)
    ln = len(srow)
    if ln % 2 == 1:
       mid = ln // 2
       return srow[mid]
    else:
       mid_min_1 = (ln // 2) - 1
       mid = ln // 2
       return (srow[mid_min_1] +  srow[mid]) / 2
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def average(row):
    av = np.sum(row) / len(row)
    return av
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def standard_deviation(row):
    av = average(row)
    sb = np.subtract(row, av)
    sq = np.square(sb)
    ssq = np.sum(sq)
    vr = ssq / (len(row) - 1)
    sd = abs(np.sqrt(vr))
    return sd
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def modified_standardization(i_mat, skip_first_col=False):
    c = 0
    if skip_first_col == True:
       c = 1

    i_mat_t = np.matrix.transpose(i_mat)
    for r in range(c, len(i_mat_t)):
        md = median(i_mat_t[r])
        sd = standard_deviation(i_mat_t[r])
        numer = np.subtract(i_mat_t[r], md)
        i_mat_t[r] = np.divide(numer, sd)

    return np.matrix.transpose(i_mat_t)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def standardization(i_mat, skip_first_col=False):
    c = 0
    if skip_first_col == True:
       c = 1

    i_mat_t = np.matrix.transpose(i_mat)
    for r in range(c, len(i_mat_t)):
        av = average(i_mat_t[r])
        sd = standard_deviation(i_mat_t[r])
        numer = np.subtract(i_mat_t[r], av)
        i_mat_t[r] = np.divide(numer, sd)

    return np.matrix.transpose(i_mat_t)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def logarithmic_transformation(i_mat, skip_first_col=False):
    c = 0
    if skip_first_col == True:
       c = 1

    i_mat_t = np.matrix.transpose(i_mat)
    for r in range(c,len(i_mat_t)):
        i_mat_t[r] = np.log(i_mat_t[r] + 0.1)
    return np.matrix.transpose(i_mat_t)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def get_learner_kind():
    os.system("clear")

    print("\n")
    print("enter one of the following choices for learning methods.")
    print("\n")
    print("--------------------------------------------------------")
    print("*  enter 1 for linear regression")
    print("*  enter 2 for logistic regression")
    print("--------------------------------------------------------")
    print("\n")

    choice = int(input())

    if choice == 1:
       return LKind.linear_regression
    elif choice == 2:
       return LKind.logistic_regression
    else:
       print("\n\n")
       print(choice, "is invalid entry for the choice of learner." + \
             " exiting gracefully.")
       sys.exit()
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def parse_command_line_arguments(argv):
    if len(argv) != 4:
       print("\n")
       print("Error: while parsing commnad line arguments.")
       print("\n")
       print("-------------------------------------------------------------" +
             "-----------")
       print("Usage:         python main.py Kind TrainingData.txt TestData.txt")
       print("-------------------------------------------------------------" +
             "-----------")
       print("Kind:             Represents type of learner as follows.")
       print("                  Enter 1 for linear regression")
       print("                  Enter 2 for logistic regression")
       print("                  Enter 3 for k-mean clustering")
       print("TrainingData.txt: Choose it based on 'Kind' from './data_set' " +
             "directory")
       print("TestData.txt:     Choose it based on 'Kind' from './data_set' " +
             "directory")
       print("-------------------------------------------------------------" +
             "-----------")
       print("\n")
       print("Note:  For unsupervised learners like k-mean clustering, the " +
             "'TestData.txt'\n       should be empty 'NA.txt' as test data " +
             "is not applicable for these learners")
       #print("-------------------------------------------------------------" +
       #      "-----------")
       print("\n")
       sys.exit()

    kind = int(argv[1])
    d_file = argv[2]
    t_file = argv[3]

    return kind, d_file, t_file
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def print_linear_regression_output(o_vec, p_vec, mse):
    os.system("clear")
    print("\n\n")
    print("-------------------------------------------------------------------")
    print("Linear Regression Predictions For The Data Set:")
    print("       https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD")
    print("-------------------------------------------------------------------")
    #print("\n")
    #print("Actual Year In Test Data Set:")
    #print([int(i) for i in o_vec])
    #print("\n")
    #print("Predicted Year For Test Data Set:")
    #print([int(i) for i in p_vec])
    print("\n")
    print("Mean Squared Error:  ", mse)
    print("\n")
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def print_logistic_regression_output(o_vec, p_vec, error, accuracy):
    os.system("clear")
    print("\n\n")
    print("------------------------------------------------------------")
    print("Logistic Regression Predictions For The Data Set:")
    print("         https://archive.ics.uci.edu/ml/datasets/Spambase")
    print("------------------------------------------------------------")
    #print("\n")
    #print("Actual Class Of Test Emails:")
    #print([int(i) for i in o_vec])
    #print("\n")
    #print("Predicted Class Of Test Emails:")
    #print([int(i) for i in p_vec])
    print("\n")
    print("Error:     ", error)
    print("Accuracy:  ", accuracy)
    print("\n")
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
def print_k_mean_clustering_output(cluster, sse):
    os.system("clear")
    print("\n\n")
    print("------------------------------------------------------------")
    print("K Mean Clustering Output For The Data Set:")
    print("        http://guidetodatamining.com/guide/data/mpg.txt")
    print("------------------------------------------------------------")
    print("\n")
    print("K Clusters where K is :", len(cluster))
    print("\n")
    for key, val_list in cluster.items():
        print(key, ":   ", val_list)
        print("\n")
    print("Sum Squared Error:     ", sse)
    print("\n")
#------------------------------------------------------------------------------#
