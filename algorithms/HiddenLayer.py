"""
File name: Layer.py
Authors: Yanming Guo
Description: Defines the layer operation of the nn.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np
from .Dropout import Dropout
from .Activation import Activation

class HiddenLayer:

    def __init__(self,n_in, n_out,
                 activation_last_layer='tanh',activation='tanh',  W=None, b=None):
        '''
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        '''
        
        self.input=None
        self.activation=Activation(activation).f
        self.drop = None
        self.optimizer = None   
        
        # potentially redundent, we probably only need the activation type and check if == "softmax" then final layer, if not then hidden layer
        # take less argument for instantiation. or we could just add a method called add_last_layer to handle the last layer      Leo
        self.activation_type = activation
        self.last_layer_act_type = activation_last_layer
        
        # activation deriv of last layer
        self.activation_deriv=None
        
        

        
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

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
           
           
           
           
           
           
    def set_keep_prob(self, keep_prob):
        self.drop = Dropout(keep_prob)


    # pass an optimizer object
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
           
           
    
    def forward(self, input, mode):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        :mode: a string indicating if we are currently training, to indicate we are training , input "train"
        '''
        
        lin_output = np.dot(input, self.W) + self.b
        
        # if self.activation_type == self.last_layer_act_type:
        # check if this is the last layer by checking the activation type
        # could be written as 
        # if self.activation_type == "softmax"
        # do batch norm and dropout here
        
        

        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
           
        # dropout after activation, need to set a keep probability with set_keep_prob, for final layer, should be 0
        self.output = self.drop.forward(self.output, mode)
           
        self.input=input
        return self.output
    
    def backward(self, delta, output_layer=False):         
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
           
           
           
        # run drop out to backprob   
        delta = self.drop.backward(delta)   
           
           
           
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta






    def update(self,lr):
                  
        self.W, self.b= self.optimizer.update(lr, self.W, self.b, self.grad_W, self.grad_b)
        

        #update normalizer as well
        if(self.BatchNormalizer is not None):
            self.BatchNormalizer.update(lr)
