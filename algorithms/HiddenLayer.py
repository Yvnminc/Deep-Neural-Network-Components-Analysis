"""
File name: HiddenLayer.py
Authors: Yanming Guo, Yongjiang Shi
Description: Defines the layer operation of the nn.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np
from .Dropout import Dropout
from .Activation import Activation
from .WeightDecay import *

class HiddenLayer:

    def __init__(self,n_in, n_out,last = False):
        '''
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        '''
        # we randomly assign small values for the weights as the initiallization
        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        # if activation == 'logistic':
        #     self.W *= 4

        # we set the size of bias as the size of output dimension
        self.b = np.zeros(n_out,)
        
        # we set he size of weight gradation as the size of weight
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
        
        # dropout layer,not a prob
        self.drop = None
        self.input = None
        self.last = last

        # not sure if we need this
        # activation deriv of last layer
        # self.activation_deriv=None
        
        # if activation_last_layer:
        #    self.activation_deriv=Activation(activation_last_layer).f_deriv


           
    def set_drop_out_layer(self, keep_prob):
        self.drop = Dropout(keep_prob)

    # pass an optimizer object
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_activation(self,activation):
        self.activation = Activation(activation).f
           
           
    
    def forward(self, input, train_mode = True, regularizer = None):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        :mode: a string indicating if we are currently training, to indicate we are training , input "train"
        '''
        # number of instance
        self.m = input.shape[1]
        self.input = input
        

        
        self.z = np.dot(input,self.W) + self.b
      

        #batch normalize need to be done

        #not sure
        self.a  = self.activation(self.z)


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
            pass
        else:
            pass







    def update(self,lr):
                  
        self.W, self.b= self.optimizer.update(lr, self.W, self.b, self.grad_W, self.grad_b)
        

        #update normalizer as well
        if(self.BatchNormalizer is not None):
            self.BatchNormalizer.update(lr)
