"""
File name: Loss.py
Authors: Yanming Guo
Description: Defines different loss functions.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""

from .Activation import *

class Loss:
    def __criterion_MSE(self,y,y_hat):
        activation_deriv = Activation(self.activation).f_deriv
        # MSE
        error = y-y_hat
        loss=error**2
        # calculate the MSE's delta of the output layer
        delta=-error*activation_deriv(y_hat)
        # return loss and delta
        return loss,delta

    def __init__(self, activation, loss = "MSE"):
        self.activation = activation
        self.cal_loss = self.__criterion_MSE