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
       print("Usage: python main.py Kind TrainingData.txt TestData.txt")
       print("\n")
       print("Kind:             Represents type of learner as follows.")
       print("                   Enter 1 for linear regression")
       print("                   Enter 2 for logistic regression")
       print("TrainingData.txt: Choose it based on 'Kind' from './data_set' " +
             "directory")
       print("TestData.txt:     Choose it based on 'Kind' from './data_set' " +
             "directory")

       print("-------------------------------------------------------------" +
             "-----------")
       print("\n")
       sys.exit()

    kind = int(argv[1])
    d_file = argv[2]
    t_file = argv[3]

    return kind, d_file, t_file
#------------------------------------------------------------------------------#
