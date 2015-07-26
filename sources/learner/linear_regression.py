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
# import package modules here                                                  #
#------------------------------------------------------------------------------#
import sources.utility.util as util
#------------------------------------------------------------------------------#



#------------------------------------------------------------------------------#
# LSLR Class: which implements the multi-variate least square linear           
# regression                                                                   
#                                                                              
# simple multi-variate least square linear regression.                         
#                                                                              
# Let [y1, [x11, x12, ..., x1N]], [y2, [x21, x22, ..., x2N]], ...              
# [yM, [xM1, xM2, ..., xMN]] where M > N are the given training data.          
# Then, the regression coefficients [b1, b2, ... bM] can be computed           
# through following equations.                                                 
# Let                                                                          
# 
#               1 x11 x12 ... x1N
#               1 x21 x22 ... x2N
#                     ...
#     X  =            ...
#                     ...
#               1 xM1 xM2 ... xMN
#    
#               y1
#               y2
#               .
#     Y  =      .
#               .
#               yM
#
#               b1
#               b2
#               .
#     B  =      .
#               .
#               bM
# Then
#         X^XB = X^Y             (1)
#
# where X^ is the transpose of X, and X^X is hessian matrix. More details can
# be found in any machine learning book. 
#------------------------------------------------------------------------------#
class LSLR:
      def least_square_linear_regression(self, phi_mat, t_vec):
          # compute equation (1) above
          phi_mat_t = np.matrix.transpose(phi_mat)
          hessian_mat = np.dot(phi_mat_t, phi_mat)
          hessian_mat_inv = np.linalg.inv(hessian_mat)
          phi_t_t_vec = np.dot(phi_mat_t, t_vec)
          r_vec = np.dot(hessian_mat_inv, phi_t_t_vec)
          return r_vec
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# LinearRegression Class                                                       #
#------------------------------------------------------------------------------#
class LinearRegression:
      # compute regression coefficients 
      def compute_regression_coefficients(self, i_mat, o_vec):
          lslr = LSLR()
          return lslr.least_square_linear_regression(i_mat, o_vec)

      # construct output vector from data
      def construct_output_vector(self, data):
          rows = len(data)
          o_vec = np.zeros(rows)

          r = 0
          for line in data:
              o_vec[r] = line[0]
              r += 1
          return o_vec

      # normalize input matrix
      def normalize_input_matrix(self, i_mat):
          return util.standardization(i_mat, True)

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

          return self.normalize_input_matrix(i_mat)

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
