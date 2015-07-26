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
import copy
import numpy as np
from collections import OrderedDict
from queue import PriorityQueue
from scipy.spatial import distance as sci_dist
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# import package modules here                                                  #
#------------------------------------------------------------------------------#
import sources.utility.util as util
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# single linkage clustering                                                    #
#------------------------------------------------------------------------------#
class SingleLinkageCluster:
      # find two closest clusters
      def find_two_closest_clusters(self, i_mat, rows, cols, cluster,
                                    distance):
         dist = float("inf")
         c1 = -1
         c2 = -1
         for key, qu in distance.items():
             tu = qu.get()
             if tu[0] <  dist:
                dist = tu[0]
                c1 = key
                c2 = tu[1]
         return c1, c2

      # merge two clusters which are closer to each other
      def merge_two_closest_clusters(self, cluster, c1, c2):
         # construct a new merged cluster from old cluster by
         # merging c2 into c1
         n_cluster = {}
         for key, v_list in cluster.items():
             nv_list = copy.deepcopy(v_list)
             n_cluster[key] = nv_list
         v1_list = n_cluster[c1]
         v2_list = n_cluster[c2]
         v1_list.extend(v2_list)
         n_cluster[c1] = v1_list
         del n_cluster[c2]
         return n_cluster

      # merge clusters 
      def merge_clusters(self, i_mat, rows, cols, cluster, distance):
         # find two closest clusters
         c1, c2 = self.find_two_closest_clusters(i_mat, rows, cols, cluster,
                                                 distance)
         # merge c1 and c2
         cluster = self.merge_two_closest_clusters(cluster, c1, c2)
         return cluster

      # compute shortest distance between two given clusters
      def find_distance_between_two_clusters(self, v_list1, v_list2, i_mat,
                                             rows, cols, cluster):
          dist = float("inf")
          for v1 in v_list1:
              for v2 in v_list2:
                  n_dist = sci_dist.euclidean(i_mat[v1], i_mat[v2])
                  if n_dist < dist:
                     dist = n_dist
          return dist

      # compute distance between all clusters
      def compute_distance(self, i_mat, rows, cols, cluster):
          distance = {}
          for key1, v_list1 in cluster.items():
              distance[key1] = PriorityQueue()
              for key2, v_list2 in cluster.items():
                  if key1 != key2:
                     dist = self.find_distance_between_two_clusters(
                                   v_list1, v_list2, i_mat, rows, cols, cluster)
                     distance[key1].put((dist, key2))
          return distance

      # construct initial singleton clusters
      def get_initial_singleton_cluster(self, i_mat, rows, cols):
          cluster = {i: [] for i in range(rows)}
          for i in range(0, rows):
              cluster[i].append(i)

          return cluster

      # perform hierarchical clustering using single linkage technique
      def single_linkage_clustering(self, i_mat):
          # get rows and cols
          rows = len(i_mat)
          cols = len(i_mat[0])

          # construct initial singleton clusters
          cluster = self.get_initial_singleton_cluster(i_mat, rows, cols)

          # initialize empty cluster list
          cluster_list = []
          cluster_list.append(cluster)

          # iterate till all clusters merges into a single cluster
          while len(cluster) > 1:
              # compute (shortest) distances among clusters
              distance = self.compute_distance(i_mat, rows, cols, cluster)

              # merge two clusters which are closer to each other
              cluster = self.merge_clusters(i_mat, rows, cols, cluster,
                                            distance)

              # append the current cluster to cluster list
              cluster_list.append(cluster)

          return cluster_list
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Class HierarchicalCluster: which implements hierarchical clustering          #
#------------------------------------------------------------------------------#
class HierarchicalCluster:
      # cluster data in hierarchical form
      def hierarchical_cluster(self, i_mat):
          slc = SingleLinkageCluster()
          return slc.single_linkage_clustering(i_mat)

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

          # get cluster hierarchy
          cluster_list = self.hierarchical_cluster(i_mat)

          return cluster_list
#------------------------------------------------------------------------------#
