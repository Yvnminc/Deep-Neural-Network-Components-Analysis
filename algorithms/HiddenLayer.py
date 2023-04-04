"""
File name: Layer.py
Authors: Yanming Guo, Yongjiang Shi, Jiacheng Zhang
Description: Defines the layer class and its operation of the nn.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np
from .Dropout import Dropout
from .Activation import Activation
from .WeightDecay import *

class HiddenLayer:

    def __init__(self, n_in, n_out):
        '''
        Typical hidden layer of a MLP: units are fully-connected. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        '''
        # Randomly assign small values for the weights as the initiallization according to the Xavier Method
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        self.b = np.zeros(n_out,)
        

        # We set the size of weight gradients' matrix the same size as the weights'
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        self.n_in = n_in
        self.n_out = n_out
        self.activation=None
        self.optimizer = None
        self.m = None
        self.z = None
        self.z_norm = None
        self.a = None
        self.a_dropout = None
        
        # note here the self.drop is a dropout object, not a saclar probability
        self.drop = None
        self.input = None
    
        self.activation_deriv = None
        self.batchNormalizer = None
       

    def set_drop_out_layer(self, keep_prob):
        self.drop = Dropout(keep_prob)

    # note need to pass an optimizer object
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_activation(self, activation):
        self.activation = Activation(activation).f
           
    def set_activation_deriv(self, activation):
        self.activation_deriv = Activation(activation).f_deriv
    
    def set_batchNormalizer(self, norm):
        if norm != None:
            self.batchNormalizer = norm

    def forward(self, input, train_mode=True, regularizer=None):
        '''
        Input:
        input: a tensor containing the input/output from the previous layer
        train_mode: boolean value indicating if we are in the training stage
        '''
        # number of instance
        self.m = input.shape[0]
        self.input = input

        # store linear combination
        self.z = np.dot(input, self.W) + self.b

        # batch normalization
        self.z_norm = self.z

        if self.batchNormalizer is not None:
            self.z_norm = self.batchNormalizer.forward(self.z, train_mode)

            
        self.a = self.activation(self.z_norm)

        if train_mode:
            # this is model's regularizar that has been passed into layers for loss calculation
            if regularizer is not None:
                regularizer.forward(self.W)
            self.a_dropout = self.drop.forward(self.a)
        else:
            self.a_dropout = self.a
        
        return self.a_dropout
    
    def backward(self, delta, output_layer=False, regularizer = None):
        if output_layer == True:
            dz = delta
        else:

            # back propagate the dropout operation
            da = self.drop.backward(delta)

            # back propagate the activation function using its derivative
            if self.activation is not None:
                dz_norm = self.activation_deriv(self.a) * da
            else:
                dz_norm = da
            
            # back propagate the batch normalization operation
            if self.batchNormalizer is not None:
                dz = self.batchNormalizer.backward(dz_norm)
            else:
                dz = dz_norm

        

        # m should be the first dimension of the input
        m = self.input.shape[0]

        # calculate the gradient of weights
        self.grad_W = np.dot( self.input.T,dz)/m

        # back propagate the regularization operation
        if regularizer is not None:
            self.grad_W = regularizer.backward(self.grad_W, self.W, m)

        # calculate the gradient of bias term
        self.grad_b = np.sum(dz, axis = 0,keepdims = True)/m
       
        # calculate the gradient of activations
        dinput = np.dot(dz, self.W.T)
        return dinput

    def update(self, lr):
        
        self.W, self.b = self.optimizer.update(lr, self.W, self.b, self.grad_W, self.grad_b)
    
        # update normalizer as well
        if self.batchNormalizer is not None:
            self.batchNormalizer.update(lr)


