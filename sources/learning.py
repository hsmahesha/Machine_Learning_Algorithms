################################################################################
#                                                                              #
#                        Machine Learning System:                              #
#                                                                              #
################################################################################
#                                                                              #
# This project implements different machine larning techniques. Note that th-  #
# -is is an hobby project implemented in order to get practical hands on with  #
# different machine learning approaches. And it is the work in progress.       #
#                                                                              #
################################################################################


#------------------------------------------------------------------------------#
# import built-in system modules here                                          #
#------------------------------------------------------------------------------#
import sys
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# include main source directory path to system path                            #
#------------------------------------------------------------------------------#
sys.path.insert(1, '../sources')
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# import package modules here                                                  #
#------------------------------------------------------------------------------#
import sources.utility.util as util
import sources.file_handler.file_handler as fh
import sources.learner.learner as lr
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# main interface to machine learning system                                    #
#------------------------------------------------------------------------------#
def machine_learning_system(argv):
    # parse command line arguments and get the training data and test data file
    d_file, t_file = util.parse_command_line_arguments(argv)

    # construct file handler object
    fho = fh.FileHandler()

    # open training data and test data files
    fho.open_training_data_file(d_file)
    fho.open_test_data_file(t_file)

    # read training data and test data files
    train_data = fho.read_training_data_file()
    test_data = fho.read_test_data_file()

    # construct learner object
    learner = lr.Learner(train_data, test_data)

    # ask machine to learn from train data and make predictions for test data 
    learner.learn_and_predict()

    # close training data and test data file
    fho.close_training_data_file()
    fho.close_test_data_file()
#------------------------------------------------------------------------------#
