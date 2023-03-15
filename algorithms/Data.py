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

  def __init__(self, path = "/Assignment1-Dataset", split_rate = 0.9):
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
    X = np.load(train_data_path)
    y = np.load(train_label_path).flatten()

    self.train_validation_split(X, y, split_rate)
    self.test_data = np.load(test_data_path)
    self.test_label = np.load(test_label_path).flatten()
    
    self.one_hot()

  def get_train_data(self):
    '''
    Get method.
    '''
    return self.train_data

  def one_hot(self):
    self.train_label = np.eye(np.max(self.train_label)+1)[self.train_label]
    self.validation_label = np.eye(np.max(self.validation_label)+1)[self.validation_label]
    self.test_label = np.eye(np.max(self.test_label)+1)[self.test_label]

  def print_shapes(self):
    '''
    Print shapes
    '''
    print(self.train_data.shape)
    print(self.train_label.shape)
    print(self.validation_data.shape)
    print(self.validation_label.shape)
    print(self.test_data.shape)
    print(self.test_label.shape)

  def train_validation_split(self, X, y, rate = 0.9):
      
      m = X.shape[0]

      '''
      # Random shuffle the index
      # Reference:
      https://stackoverflow.com/questions/43229034/
      randomly-shuffle-data-and-labels-from-different-files-in-the-same-order
      '''
      idx = np.random.permutation(m)
      n_train = round(m* rate)
      n_validation = m - n_train

      new_X, new_y = X[idx], y[idx]
      self.train_data = new_X[:n_train]
      self.train_label = new_y[:n_train]
      self.validation_data = new_X[-n_validation:]
      self.validation_label = new_y[-n_validation:]