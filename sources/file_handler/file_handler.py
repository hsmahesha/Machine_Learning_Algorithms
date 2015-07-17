################################################################################
#                                                                              #
#                  Data Set File Handling Module:                              #
#                                                                              #
################################################################################
#                                                                              #
# This module defines a class which implements all data set file handling      #
# functions.                                                                   #
#                                                                              #
################################################################################



#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import sys
import csv
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# class FileHandler                                                            #
#------------------------------------------------------------------------------#
class FileHandler:
      # data members
      d_file = None
      t_file = None

      # init method 
      def __init__(self):
          pass

      # open training data set file
      def open_training_data_file(self, d_file):
          try:
              self.d_file = open(d_file, 'r')
          except IOError:
              print("\nError: The training data file " + d_file +
                    "does not exist. exiting gracefully.\n")
              sys.exit()

      # open test data set file
      def open_test_data_file(self, t_file):
          try:
              self.t_file = open(t_file, 'r')
          except IOError:
              print("\nError: The test data file " + t_file +
                    "does not exist. exiting gracefully.\n")
              sys.exit()

      # close training data set file
      def close_training_data_file(self):
          self.d_file.close()

      # close test data set file
      def close_test_data_file(self):
          self.t_file.close()

      # read training data set file
      def read_training_data_file(self):
          try:
              train_data = list(csv.reader(self.d_file, delimiter=','))
          except csv.Error:
              print("\nError: in reading trainig data file. " +
                    "exiting gracefully.\n")
              sys.exit()

          return train_data

      # read test data set file
      def read_test_data_file(self):
          try:
              test_data = list(csv.reader(self.t_file, delimiter=','))
          except csv.Error:
              print("\nError: in reading test data file. " +
                    "exiting gracefully.\n")
              sys.exit()

          return test_data
#------------------------------------------------------------------------------#
