"""
File name: Activations.py
Authors: Yanming Guo
Description: Defines different activation functions.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np

class Activation:
    # Tanh activation
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)   
        return 1.0 - a**2
    
    # Logistic activation
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x) 
        return  a * (1 - a )
    
    # Relu activation
    def __relu(self, x):
        mask = x > 0
        return x * mask

    def __relu_deriv(self, a):
        return np.int64(a > 0)
    
    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv

        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv