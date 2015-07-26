################################################################################
#                                                                              #
#                      Linear Regression Based Learner:                        #
#                                                                              #
################################################################################
#                                                                              #
# This module implements k-mean clustering                                     #
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
# compute initial cluster centroids
#------------------------------------------------------------------------------#
def get_initial_cluster_centroids(i_mat, k, rows, cols):
    # compute overall mean of the sample
    om_vec = np.zeros(cols)
    for r in range(0, rows):
        om_vec = np.add(om_vec, i_mat[r])
    om_vec = np.divide(om_vec, rows)

    # compute euclidian distance of each sample from overall mean
    uc_dict = {}
    for r in range(0, rows):
        dist = sci_dist.euclidean(i_mat[r], om_vec)
        uc_dict[r] = dist

    # sort samples based on euclidian distance of each sample from overall mean
    uc_dict = OrderedDict(sorted(uc_dict.items(), key=lambda x: x[1]))

    # get temporary sorted samples based on their euclidian distance of each
    # sample from overall mean
    t_i_mat = np.zeros(shape=(rows, cols))
    r = 0
    for key, val in uc_dict.items():
        t_i_mat[r] = i_mat[key]
        r += 1

    # get initial centroids
    centroids = {}
    for i in range(1,k+1):
        n_arr = np.zeros(cols)
        c = 1 + (i-1) * int(rows / k)
        n_arr = t_i_mat[c-1]
        centroids[i-1] = n_arr

    return centroids
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# classic Lloyd's k-mean clustering                                            #
#------------------------------------------------------------------------------#
class LloydKMC:
      # find the absolute difference between new and old centroids 
      def absolute_difference(self, new_centroids, old_centroids, cols):
          if len(new_centroids) != len(old_centroids):
             print("internal error")
             sys.exit()

          abs_d_mat = np.zeros(shape=(len(new_centroids), cols))

          r = 0
          for key, val_list in new_centroids.items():
              if key not in old_centroids:
                 print("internal error")
                 sys.exit()
              new_arr = new_centroids[key]
              old_arr = old_centroids[key]
              d_arr = np.subtract(old_arr, new_arr)
              a_d_arr = np.absolute(d_arr)
              abs_d_mat[r] = a_d_arr
              r += 1

          return abs_d_mat

      # update centroids
      def update_centroids(self, i_mat, k, rows, cols, cluster):
          new_centroids = {}

          for key, val_list in cluster.items():
              sn = len(val_list)
              n_arr = np.zeros(cols)
              for index in val_list:
                  n_arr = np.add(n_arr, i_mat[index])
              new_centroids[key] = np.divide(n_arr, sn)

          return new_centroids

      # adjust k value and centroids array based on the result of pruning
      def adjust(self, cluster, del_list, k, centroids):
          if len(cluster) < k:
             k = len(cluster)
             for val in del_list:
                 del centroids[val]
          return k, centroids

      # remove empty cluster if any
      def prune_cluster(self, cluster):
          del_list = []
          for key, val_list in cluster.items():
              if len(val_list) == 0:
                 del_list.append(key)

          for key in del_list:
              del cluster[key]

          return cluster, del_list

      # assign samples to cluster
      def assign_samples_to_clusters(self, i_mat, k, rows, cols, centroids):
          # contruct empty cluster groups
          cluster = {i: [] for i in range(k)}

          # assign samples to clusters
          for r in range(0, rows):
              dist = float("inf")
              cn = -1
              for key, val_list in centroids.items():
                  n_dist = sci_dist.euclidean(i_mat[r], val_list)
                  if n_dist < dist:
                     dist = n_dist
                     cn = key
              cluster[cn].append(r)

          return cluster

      # iteratively update centroids till centroids get converges
      def iterative_centroid_update(self, i_mat, k, rows, cols, centroids):
          #iterate till centroid values converge
          while True:
             # assign samples to clusters
             cluster = self.assign_samples_to_clusters(
                                                  i_mat, k, rows, cols, centroids)

             # prune cluster by eliminating if there exists any empty cluster
             cluster, del_list = self.prune_cluster(cluster)

             # adjust k value and centroids array based on the result of pruning
             k, centroids = self.adjust(cluster, del_list, k, centroids)

             # update centroids
             new_centroids = self.update_centroids(
                                                  i_mat, k, rows, cols, cluster)

             # compute the absolute difference between new centroids and old
             # centroids
             abs_d_mat = self.absolute_difference(
                                                 new_centroids, centroids, cols)

             # check if the centroids are converged, if so, stop the clustering
             # process 
             bool_mat = np.less(abs_d_mat, 0.00001)
             if bool_mat.all() == True:
                break;
             else:
                centroids = new_centroids

          return cluster, new_centroids

      # cluster data into k meangingful groups
      def k_mean_cluster(self, i_mat, k):
          # get rows and cols
          rows = len(i_mat)
          cols = len(i_mat[0])

          # compute initial centroids of samples
          centroids = get_initial_cluster_centroids(i_mat, k, rows, cols)

          # iteratively update centroids till centroids get converges 
          cluster, final_centroids = self.iterative_centroid_update(
                                            i_mat, k, rows, cols, centroids)

          # return final cluster
          return cluster, final_centroids
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Class KMC: which implements k-mean clustering                                #
#------------------------------------------------------------------------------#
class KMeanCluster:
      # cluster data into k meangingful groups
      def k_mean_cluster(self, i_mat, k):
          kmc = LloydKMC()
          cluster, final_centroids = kmc.k_mean_cluster(i_mat, k)
          return cluster, final_centroids

      # normalize input matrix
      def normalize_input_matrix(self, i_mat):
          return util.standardization(i_mat, False)

      # construct input matrix from data
      def construct_input_matrix(self, data):
          rows = len(data)
          cols = len(data[0]) - 1
          i_mat = np.zeros(shape=(rows, cols))

          r = 0
          for line in data:
              c = 0
              for ele in line[1:]:
                  i_mat[r][c] = ele
                  c += 1
              r += 1
          return self.normalize_input_matrix(i_mat)

      # ask k-mean clusterer to cluster the data into k meaningful groups 
      def cluster(self, data):
          # construct input matrix from data
          i_mat = self.construct_input_matrix(data)

          # cluster data into k meangingful groups
          k = 10
          cluster, final_centroids = self.k_mean_cluster(i_mat, k)

          return i_mat, cluster, final_centroids
#------------------------------------------------------------------------------#
