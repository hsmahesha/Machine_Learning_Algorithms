################################################################################
#                                                                              #
#                      Linear Regression Based Learner:                        #
#                                                                              #
################################################################################
#                                                                              #
# This module defines a class LinearRegression which implements linear regres- #
# -sion based learner.                                                         #
#                                                                              #
################################################################################



#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import sys
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# LinearRegression Class                                                       #
#------------------------------------------------------------------------------#
class LinearRegression:
      # compute regression coefficients 
      def compute_regression_coefficients(self, i_mat, o_vec):
          i_mat_t = np.matrix.transpose(i_mat)
          x_mat = np.dot(i_mat_t, i_mat)
          x_mat_inv = np.linalg.inv(x_mat)
          y_mat = np.dot(x_mat_inv, i_mat_t)
          r_vec = np.dot(y_mat, o_vec)
          return r_vec

      # construct output vector from data
      def construct_output_vector(self, data):
          rows = len(data)
          o_vec = np.zeros(rows)

          r = 0
          for line in data:
              o_vec[r] = line[0]
              r += 1
          return o_vec

      # construct input matrix from data
      def construct_input_matrix(self, data):
          rows = len(data)
          cols = len(data[0][1:]) + 1
          i_mat = np.zeros(shape=(rows, cols))

          for r in range(0, len(i_mat)):
              i_mat[r][0] = 1.0

          r = 0
          for line in data:
              c = 1
              for ele in line[1:]:
                  i_mat[r][c] = ele
                  c += 1
              r += 1

          return i_mat

      # predict test data
      def predict(self, test_data, r_vec):
          # construct input matrix from test data
          i_mat = self.construct_input_matrix(test_data)

          # construct output vector from test data
          o_vec = self.construct_output_vector(test_data)

          # for each test data row from i_mat predict the output
          p_vec = []
          for row in i_mat:
              p_vec.append(np.inner(r_vec, row))

          # return predicted output
          return o_vec, p_vec

      # ask linear regression learner to learn from training data
      def learn(self, training_data):
          # construct input matrix from train data
          i_mat = self.construct_input_matrix(training_data)

          # construct output vector from train data
          o_vec = self.construct_output_vector(training_data)

          # compute regression coefficients
          r_vec = self.compute_regression_coefficients(i_mat, o_vec)

          # return regression coefficients
          return r_vec
#------------------------------------------------------------------------------#
