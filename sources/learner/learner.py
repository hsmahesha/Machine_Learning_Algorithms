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

          print("\noutput:")
          print("actual class, predicted class, diff")
          for i in range(0, len(o_vec)):
              o = int(o_vec[i])
              p = int(p_vec[i])
              d = abs(o-p)
              print(o, p, d)

      # linear regression learner
      def __linear_regression(self):
          lro = lin.LinearRegression()
          r_vec = lro.learn(self.training_data)
          o_vec, p_vec = lro.predict(self.test_data, r_vec)

          print("\noutput:")
          print("actual year, predicted year, difference")
          for i in range(0, len(o_vec)):
              o = int(o_vec[i])
              p = int(p_vec[i])
              d = abs(o - p)
              print(o, p, d)

      # public interface function of the class Learner
      def learn_and_predict(self):
          # call learner based on user choice
          if self.kind == 1:
             self.__linear_regression()
          elif self.kind == 2:
             self.__logistic_regression()
#------------------------------------------------------------------------------#
