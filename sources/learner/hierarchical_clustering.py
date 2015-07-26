################################################################################
#                                                                              #
#                          Hierarchical Clustering:                            #
#                                                                              #
################################################################################
#                                                                              #
# This module implements hierarchical clustering                               #
#                                                                              #
################################################################################



#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import sys
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as sci_dist
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# import package modules here                                                  #
#------------------------------------------------------------------------------#
import sources.utility.util as util
#------------------------------------------------------------------------------#




#------------------------------------------------------------------------------#
# Class HierarchicalCluster: which implements k-mean clustering                                #
#------------------------------------------------------------------------------#
class HierarchicalCluster:
      # normalize input matrix
      def normalize_input_matrix(self, i_mat):
          return util.modified_standardization(i_mat, False)

      # construct input matrix from data
      def construct_input_matrix(self, data):
          rows = len(data) - 1
          cols = len(data[0]) - 1
          i_mat = np.zeros(shape=(rows, cols))

          r = 0
          for line in data[1:]:
              c = 0
              for ele in line[1:]:
                  i_mat[r][c] = ele
                  c += 1
              r += 1
          return self.normalize_input_matrix(i_mat)

      # ask hierarchical clusterer to cluster the data 
      def cluster(self, data):
          # construct input matrix from data
          i_mat = self.construct_input_matrix(data)
#------------------------------------------------------------------------------#
