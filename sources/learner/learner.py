################################################################################
#                                                                              #
#                                  Learner:                                    #
#                                                                              #
################################################################################
#                                                                              #
# This module defines a class Learner which handles different learners         #
#                                                                              #
################################################################################



#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import os
import sys
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# import package modules here                                                  #
#------------------------------------------------------------------------------#
import sources.utility.util as util
import sources.learner.linear_regression as lin
import sources.learner.logistic_regression as log
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Learner Class                                                                #
#------------------------------------------------------------------------------#
class Learner:
      # data members
      training_data = None
      test_data = None
      kind = None

      # special init method
      def __init__(self, training_data, test_data, kind):
          self.training_data = training_data
          self.test_data = test_data
          self.kind = kind

      # logistic regression learner
      def __logistic_regression(self):
          lro = log.LogisticRegression()
          r_vec = lro.learn(self.training_data)
          o_vec, p_vec = lro.predict(self.test_data, r_vec)
          error, accuracy = util.compute_error_and_accuracy(o_vec, p_vec)
          util.print_logistic_regression_output(o_vec, p_vec, error, accuracy)

      # linear regression learner
      def __linear_regression(self):
          lro = lin.LinearRegression()
          r_vec = lro.learn(self.training_data)
          o_vec, p_vec = lro.predict(self.test_data, r_vec)
          mse = util.compute_mean_squared_error(o_vec, p_vec)
          util.print_linear_regression_output(o_vec, p_vec, mse)

      # public interface function of the class Learner
      def learn_and_predict(self):
          os.system("clear")
          print("\n\nBe patient... I am learning...")

          # call learner based on user choice
          if self.kind == 1:
             self.__linear_regression()
          elif self.kind == 2:
             self.__logistic_regression()
#------------------------------------------------------------------------------#
