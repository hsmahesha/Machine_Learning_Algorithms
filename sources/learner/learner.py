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
import sources.learner.linear_regression as lr
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Learner Class                                                                #
#------------------------------------------------------------------------------#
class Learner:
      # data members
      training_data = None
      test_data = None

      # special init method
      def __init__(self, training_data, test_data):
          self.training_data = training_data
          self.test_data = test_data

      # linear regression learner
      def linear_regression(self):
          lro = lr.LinearRegression()
          r_vec = lro.learn(self.training_data)
          o_vec, p_vec = lro.predict(self.test_data, r_vec)

          print("\noutput:")
          print("actul year, predicted year, difference")
          for i in range(0, len(o_vec)):
              o = int(o_vec[i])
              p = int(p_vec[i])
              d = abs(o - p)
              print(o, p, d)

      # public interface function of the class Learner
      def learn_and_predict(self):
          # ask user which learner he wants to call
          l_kind = util.get_learner_kind()

          # call learner based on user choice
          if l_kind == util.LKind.linear_regression:
             self.linear_regression()
          elif l_kind == util.LKind.logistic_regression:
             print("Not yet implemented.")
             sys.exit()
#------------------------------------------------------------------------------#
