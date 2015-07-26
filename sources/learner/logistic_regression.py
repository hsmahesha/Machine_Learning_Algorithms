################################################################################
#                                                                              #
#                      Logistic Regression Based Learner:                      #
#                                                                              #
################################################################################
#                                                                              #
# This module defines a class LogisticRegression which implements a binary     #
# logistic regression based learner.                                           #
#                                                                              #
################################################################################



#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# import package modules here                                                  #
#------------------------------------------------------------------------------#
import sources.utility.util as util
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# IRLS Class: which implements iteratively reweighted least squares.
#
# iterative reweighted least squares
#
# Let [t1, [x11, x12, ..., x1N]], [t2, [x21, x22, ..., x2N]], ...              
# [tM, [xM1, xM2, ..., xMN]] where M > N are the given training data.
#
# Let us assume that the above training set represents a binary classification,
# that is, ti for i = 1, 2, ... M belongs set [0,1].
# 
# Let [y1, y2, ... yM] represents the respective probabilities that 
# [t1, t2, ... tM] takes the value 1 
#
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
#               t1
#               t2
#               .
#     T  =      .
#               .
#               tM
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
#
# regression paramters can be iteratively computed as below.
#
#     B(new)  =  B(old) - H_inv * G
#
# where G, the gradient vector is given as
#
#     G  =  X^ * (Y - T)            (1) 
#
# and, H, the hessian matrix is given as
#
#     H  =  X^RX                     (2) 
#
# where R is an N x N diagonal weight matrix with elements Rnn = yn * (1 - yn).
#
# More details can be found in ML book by C M Bishop. 
#------------------------------------------------------------------------------#
class IRLS:
      # compute the new regression parameters 
      def compute_regression_parameters(self, h_mat, phi_t_rz_vec):
          h_mat_inv = np.linalg.inv(h_mat)
          return np.dot(h_mat_inv, phi_t_rz_vec)

      # compute ((i_mat_trans) * rz_vec) vector
      def compute_phi_t_rz_vector(self, phi_mat, rz_vec):
          phi_mat_t = np.matrix.transpose(phi_mat)
          return np.dot(phi_mat_t, rz_vec)

      # compute rz vector
      def compute_rz_vector(self, r_mat, z_vec):
          return np.dot(r_mat, z_vec)

      # compute z vector
      def compute_z_vector(self, i_mat, o_vec, b_vec, p_vec, r_mat):
          phi_w = np.dot(i_mat, b_vec)
          r_inv = np.linalg.inv(r_mat)
          y_t =  np.subtract(p_vec, o_vec)
          r_inv_y_t = np.dot(r_inv, y_t)
          z_vec = np.subtract(phi_w, r_inv_y_t)
          return z_vec

      # compute hessian matrix
      def compute_hessian_matrix(self, phi_mat, r_mat):
          phi_mat_t = np.matrix.transpose(phi_mat)
          a_mat = np.dot(phi_mat_t, r_mat)
          h_mat  = np.dot(a_mat, phi_mat)
          return h_mat

      # get weighting matrix 
      def get_weighting_matrix(self, p_vec):
          rows = len(p_vec)
          cols = len(p_vec)
          r_mat = np.zeros(shape=(rows, cols))
          for r in range(0, rows):
              r_mat[r][r] = p_vec[r] * (1 - p_vec[r])
          return r_mat

      # compute probability vector
      def compute_probability_vector(self, b_vec, i_mat):
          p_vec = np.zeros(len(i_mat))
          r = 0
          for row in i_mat:
              p = np.dot(row, b_vec)
              ex = np.exp(p)
              p_vec[r] = ex / (1 + ex)
              r += 1
          return p_vec

      # compute regression parameters for kth iteration
      def get_regression_parameters(self, i_mat, o_vec, b_vec):
          # compute probability vector using logistic function
          p_vec = self.compute_probability_vector(b_vec, i_mat)

          # get weighting matrix 
          r_mat = self.get_weighting_matrix(p_vec)

          # compute hessian matrix
          h_mat = self.compute_hessian_matrix(i_mat, r_mat)

          # compute z vector
          z_vec = self.compute_z_vector(i_mat, o_vec, b_vec, p_vec, r_mat)

          # compute rz vector
          rz_vec = self.compute_rz_vector(r_mat, z_vec)

          # compute (i_mat_trans * rz_vec) vector
          phi_t_rz_vec = self.compute_phi_t_rz_vector(i_mat, rz_vec)

          # compute the new regression parameters 
          nb_vec = self.compute_regression_parameters(h_mat, phi_t_rz_vec)

          # return new regression parameters vector
          return nb_vec

      # find the absolute difference between new and old regression
      # parameters vector 
      def absolute_difference(self, nb_vec, b_vec):
          d_vec = np.subtract(b_vec, nb_vec)
          abs_d_vec = np.absolute(d_vec)
          return abs_d_vec

      # iteratively reweighted least squares
      def iteratively_reweighted_least_squares(self, i_mat, o_vec):
          # initialize the regression coefficient vector to 0 
          b_vec = np.zeros(len(i_mat[0]))

          # iterate till the regression parameters get converges
          while True:
              nb_vec = self.get_regression_parameters(i_mat, o_vec, b_vec)
              abs_d_vec = self.absolute_difference(nb_vec, b_vec)
              bool_vec = np.less(abs_d_vec, 0.00001)
              if bool_vec.all() == True:
                 break
              else:
                 b_vec = nb_vec

          # return egression parameters vector
          return nb_vec
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# LogisticRegression Class                                                     #
#------------------------------------------------------------------------------#
class LogisticRegression:
      # compute regression coefficients 
      def compute_regression_coefficients(self, i_mat, o_vec):
          irls = IRLS()
          return irls.iteratively_reweighted_least_squares(i_mat, o_vec)

      # construct output vector from data
      def construct_output_vector(self, data):
          rows = len(data)
          o_vec = np.zeros(rows)

          r = 0
          for line in data:
              o_vec[r] = float(line[57])
              r += 1

          return o_vec

     # normalize input matrix
      def normalize_input_matrix(self, i_mat):
          return util.logarithmic_transformation(i_mat, True)

      # construct input matrix from data
      def construct_input_matrix(self, data):
          rows = len(data)
          cols = len(data[0])
          i_mat = np.zeros(shape=(rows, cols))

          for r in range(0, len(i_mat)):
              i_mat[r][0] = 1.0

          r = 0
          for line in data:
              c = 1
              for ele in line[:57]:
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
          r = 0
          for row in i_mat:
              p = np.dot(row, r_vec)
              ex = np.exp(p)
              prob = ex / (1 + ex)
              if (prob >= 0.5):
                 p_vec.append(1)
              else:
                 p_vec.append(0)
              r += 1

          print(p_vec)

          # return predicted output
          return o_vec, p_vec

      # ask logistic regression learner to learn from training data
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
