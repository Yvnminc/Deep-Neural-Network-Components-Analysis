# """
# File name: Loss.py
# Authors: Yanming Guo
# Description: Defines different loss functions.
# Reference: Week 2 tut sheet of COMP5329 Deep Learning,
#            University of Sydney
# """


# import numpy as np

# class Loss(object):

#     def dz(self, y, y_hat):
#         #we avoid using the derivative of softmax to save computing power
#         return y_hat - y

#     def __criterion_MSE(self,y,y_hat):
#         activation_deriv = Activation("").f_deriv
#         # MSE
#         error = y-y_hat
#         loss=error**2
#         # calculate the MSE's delta of the output layer
#         delta=-error*activation_deriv(y_hat)
#         # return loss and delta
#         return loss,delta
    
#     def loss(self, y, y_hat):
#         m = y.shape[1]

#         loss = (-1/m) * np.sum(np.sum((y * np.log(y_hat))))

#         #logs = -np.log(y_hat.T[range(len(y_hat.T)), np.argmax(y.T, axis=-1)])
#         #loss = np.mean(logs)
#         return loss
    
# from .Activation import *
# import numpy as np

# class Loss:
#     def __criterion_MSE(self,y,y_hat):
#         activation_deriv = Activation(self.activation).f_deriv
#         # MSE
#         error = y-y_hat
#         loss=error**2
#         # calculate the MSE's delta of the output layer
#         delta=-error*activation_deriv(y_hat)
#         # return loss and delta
#         return loss,delta
    
#     def __criterion_CE(self,y,y_hat):
#         """
#         y_hat: batch_size * n_class
#         y : batch_size * 1
#         y_actual_onehot: one hot encoding of y_hat, (batch_size * n_class)
#         """
#         activation_deriv = Activation(self.activation).f_deriv
#         batch_size = y.shape[0]
        
#         # cross entropy
#         y_actual_onehot = np.eye(self.n_out)[y].reshape(-1, self.n_out)
#         # avoid log() overflow problem
#         y_hat = np.clip(y_hat, 1e-12, 1-1e-12)
        
#         loss = - np.sum(np.multiply(y_actual_onehot, np.log(y_hat)))/batch_size
        
    
#         # derivative of cross entropy with softmax
#         # self.layers[-1] is the activation function
#         delta = activation_deriv(y_actual_onehot, y_hat)
#         # return loss and delta
#         return loss, delta

#     def __init__(self, activation, loss = "MSE"):
#         self.activation = activation
#         self.cal_loss = self.__criterion_MSE

#         if loss == "CE":
#             self.cal_loss = self.__criterion_CE