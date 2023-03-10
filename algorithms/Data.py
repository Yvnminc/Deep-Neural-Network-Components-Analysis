"""
File name: data.py
Authors: Yanming Guo
Description: Deal with the input data in npy format,
             also with some processing method.
"""

import numpy as np
import os

class Data:
  '''
  This class deal with the input data in npy format,
  also with some processing method.
  '''

  def __init__(self, path = "/Assignment1-Dataset"):
    os_path = os.getcwd()
    '''
    Read data from npy format.
    '''

    # Config the data path
    train_data_path = os_path + path + '/train_data.npy'
    train_label_path = os_path + path + '/train_label.npy'
    test_data_path = os_path + path + '/test_data.npy'
    test_label_path = os_path + path + '/test_label.npy'

    # Assignments
    self.train_data = np.load(train_data_path)
    self.train_label = np.load(train_label_path)
    self.test_data = np.load(test_data_path)
    self.test_label = np.load(test_label_path)
  
  def get_train_data(self):
    '''
    Get method.
    '''
    return self.train_data

  def print_shapes(self):
    '''
    Print shapes
    '''
    print(self.train_data.shape)
    print(self.train_label.shape)
    print(self.test_data.shape)
    print(self.test_label.shape)