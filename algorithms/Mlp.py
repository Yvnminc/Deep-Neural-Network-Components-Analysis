"""
File name: MLP.py
Authors: Yanming Guo
Description: Defines the MLP model with full configs.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""

from .Activation import Activation
from .HiddenLayer import HiddenLayer
from .Dropout import Dropout
from .Optimizers import *
from .WeightDecay import *
import numpy as np


class Mlp:
    """
    """ 



    # for initiallization, the code will create all layers automatically based on the provided parameters.  
    # can we set all initilization parameter with a set_para method?   
    def __init__(self, learning_rate = 0.001,batch_size = 1, keep_prob = 1 , loss = "CE", epoch = 100):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """        
        ### initialize layers
        self.layers=[]
        #need to add training data size to index 0 in dims
        self.dims = []
        self.lr = learning_rate
        self.loss = loss
        self.optimizer = None
        self.regularizer = None
        self.epoch = epoch
        self.keep_prob = keep_prob
        self.norm = None
        self.n_out = 10

        # activation should be controled at layer level
        #self.activation = activation
        #last_act = self.activation[-1]
        #print(self.activation[-1])

        # subsituted with set loss
        # self.criterion_loss = Loss(last_act, self.loss).cal_loss
                       
                      
    def set_momentum(self, beta):
        self.optimizer = GD_with_Momentum(beta)
    
    def set_regularizer(self, lam, reg_type):
        if reg_type == "L2":
            self.regularizer = L2(lam)
        elif reg_type == "L1":
            self.regularizer = L1(lam)
        else:
            self.regularizer = None 
           
           
           
           
    # if it is last layer, activation should be set to softmax and keep_prob should be set to 1
    def add_layer(self, n_in,n_out, activation, keep_prob):

        
        layer = HiddenLayer(n_in, n_out)
        layer.set_activation(activation)
        


        if(self.norm is not None):
            layer.setBatchNormalizer(self.norm.clone())

        layer.set_drop_out_layer(keep_prob)

        if(self.optimizer != None):
            # calculations happen at each layer so every layer should have a optimizer object with the same beta
            layer.set_optimizer(self.optimizer.clone())
          

        self.dims.append(n_out)
        self.layers.append(layer)



    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self, input, train_mode = True):
       
        # reset regularizer for each epoch              
        if(self.regularizer is not None):
            self.regularizer.reset()   
           
        for layer in self.layers:
            output = layer.forward(input, regularizer = self.regularizer, mode = train_mode)
            input = output
        output = 0
        return output

    # define the objection/loss function, we use mean sqaure error (MSE) as the loss
    # you can try other loss, such as cross entropy.
    # when you try to change the loss, you should also consider the backward formula for the new loss as well!
    #def criterion_MSE(self,y,y_hat):
    #   activation_deriv=Activation(self.activation[-1]).f_deriv
    #    # MSE
    #    error = y-y_hat
    #   loss=error**2
    #    # calculate the MSE's delta of the output layer
    #    delta=-error*activation_deriv(y_hat)    
    #    # return loss and delta
    #   return loss,delta

    # backward progress  
    def backward(self,delta):
        delta = self.layers[-1].backward(delta,output_layer=True, regularizer = self.regularizer)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta, regularizer = self.regularizer)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!    
    def update(self):
        for layer in self.layers:
           layer.update(self.lr)


    # define the training function
    # it will return all losses within the whole training process.

    # to be change to allow mini batch training
    def fit(self,X,y,learning_rate=0.1, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """ 
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)
        
        for k in range(epochs):
            loss=np.zeros(X.shape[0])
            for it in range(X.shape[0]):
                i=np.random.randint(X.shape[0])
                
                # forward pass
                y_hat = self.forward(X[i])

                # backward pass
                # loss[it],delta=self.criterion_MSE(y[i],y_hat)

                loss[it],delta= self.criterion_cross_entropy(y[i],y_hat)

                self.backward(delta)
                
                # update
                self.update(learning_rate)
            to_return[k] = np.mean(loss)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        output = np.zeros(x.shape[0])
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i,:], train_mode=False)
        return output

    def criterion_cross_entropy(self,y,y_hat):
        """
        y_hat: batch_size * n_class
        y : batch_size * 1
        y_actual_onehot: one hot encoding of y_hat, (batch_size * n_class)
        """

        batch_size = y.shape[0]

        # avoid log() overflow problem
        y_hat = np.clip(y_hat, 1e-12, 1-1e-12)
        
        loss = - np.sum(np.multiply(y, np.log(y_hat)))/batch_size
        
    
        # derivative of cross entropy with softmax
        # self.layers[-1] is the activation function
        delta = self.layers[-1].f_deriv(y, y_hat)
        # return loss and delta
        return loss, delta