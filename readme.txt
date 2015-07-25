#------------------------------------------------------------------------------#
                         Machine Learning System:

This project implements different machine larning techniques. Note that this is
an hobby project implemented in order to get practical hands on with different
machine learning approaches. And it is a work in progress.
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
Description Of Files And Projects:
#------------------------------------------------------------------------------#
readme.txt: 
   this file.

rm_pycache.sh: 
   simple script to delete .pyc files and __pycache__ directories

main.py: 
   python file which calls the main interface function of our machine learning
   system

sources:
   directory which contains all the python source files which implements our
   machine learning system

data_set:
   directory which contains data sets for different learners in separate
   sub-directories
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
Usage:
#------------------------------------------------------------------------------#
command line input:
    python main.py Kind TrainingData.txt TestData.txt

    where

    Kind: Represents type of learner as follows
          Enter 1 for linear regression
          Enter 2 for logistic regression
          Enter 3 for k-mean clustering

    TrainingData.txt: Choose it based on 'Kind' from './data_set' directory

    TestData.txt: Choose it based on 'Kind' from './data_set' directory

Note: For unsupervised learners like k-mean clustering, the 'TestData.txt' 
      should be empty 'NA.txt' as test data is not applicable for these learners
#------------------------------------------------------------------------------#
