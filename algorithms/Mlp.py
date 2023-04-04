"""
File name: MLP.py
Authors: Yanming Guo, Yongjiang Shi, Jiacheng Zhang
Description: Defines the MLP model with full configs.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""

from .Activation import Activation
from .HiddenLayer import HiddenLayer
from .Dropout import Dropout
from .Optimizers import *
from .WeightDecay import *
from .BatchNormalization import *
from .MiniBatchTraining import *


import numpy as np
import time

class Mlp:
    """
    The Neural Network class that enables forward pass, back propagation, construction of a neural network and 
    method such as fit, predict and evaluate.
    It work similar to tensorflow to a extent.
    """

    def __init__(self, learning_rate=0.1, batch_size=1, keep_prob=1, norm=None):
        """
        For initilisation, we need some general information about the neural network
        such as learning rate, batch size , universal dropout rate (we use its complement)
        and whether we are using batch normalization
        """        

        # initialize layer container
        self.layers=[]

        # dimension container
        self.dims = []

        self.lr = learning_rate
        self.optimizer = None
        self.regularizer = None
        self.keep_prob = keep_prob
        self.norm = norm
        self.batch = None
        self.batch_size = batch_size
        self.n_out = 10
                
    def set_optimiser(self, opt_type, params):
        if opt_type == 'Momentum':
            self.optimizer = GD_with_Momentum(params[0])
        elif opt_type == 'Adam':
            self.optimizer = Adam(params[0],params[1])
        else:
            raise Exception("optimiser type not supported")
    
    def set_batchNormalizer(self, momentum = 0.9):
        self.norm = BatchNormalization(momentum=momentum)
        self.norm.set_optimizer(self.optimizer.clone())

    def set_regularizer(self, lam):
        self.regularizer = L2(lam)
       
    # if it is last layer, activation should be set to softmax and keep_prob should be set to 1
    def add_layer(self, n_in, n_out, activation, keep_prob):
        layer = HiddenLayer(n_in, n_out)
        layer.set_activation(activation)
        layer.set_activation_deriv(activation)

        if self.norm is not None:
            layer.set_batchNormalizer(self.norm.clone())

        layer.set_drop_out_layer(keep_prob)

        if self.optimizer is not None:
            # calculations happen at each layer so every layer should have a optimizer object with the same beta
            layer.set_optimizer(self.optimizer.clone())
        self.dims.append(n_out)
        self.layers.append(layer)

    # forward progress: pass the information through the layers and out the results of final output layer
    def forward(self, input, mode=True):
        output = 0
        # reset regularizer for each epoch              
        if self.regularizer is not None:
            self.regularizer.reset()   
           
        for layer in self.layers:
            output = layer.forward(input, regularizer=self.regularizer, train_mode=mode)
            input = output
        return output

    # backward progress  
    def backward(self, delta):
        # get the back propagation error term for the last layer
        delta = self.layers[-1].backward(delta, output_layer=True, regularizer=self.regularizer)

        # propagate backwards for other layers
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta, regularizer=self.regularizer)

    # update the network weights after back propagation
    # make sure you run the backward function before the update function!    
    def update(self):
        for layer in self.layers:
           layer.update(self.lr)


    # get the gradients of each layer's parameters
    def get_grads(self):
        """
        Return two lists that contain the gradients of W and gradient of b respectively
        """
        layer_grad_W = []
        layer_grad_b = []
        for i in range(len(self.layers)):
            layer_grad_W.append(self.layers[i].grad_W)
            layer_grad_b.append(self.layers[i].grad_b)
        return layer_grad_W, layer_grad_b

    # define the training function
    # it will return all losses during the whole training process.
    def fit(self, X, y, epochs):
        """
        mini-batch training process.
        Input:
        X: Features with shape (n_example, n_feature)
        Y: Labels with the shape (n_example, n_classes)
        epochs: number of iteration the model trains on the data
        """
        X = np.array(X)
        y = np.array(y)
        
        self.batch = MiniBatch(X,y)

        total_loss_train = []
        total_acc_train = []
        
        total_time_start = time.time()

        for k in range(epochs):
            time_start = time.time()
            if self.regularizer is not None:
                self.regularizer.reset()

            self.batch.fit(self, size = self.batch_size)

            # collect the loss and and other useful informations
            # you can collect other information (run time) you are interested in below and either return it as output or print it out
            mean_loss_train = np.mean(self.batch.getLoss())
            mean_acc_train = np.mean(self.batch.getAccuracy())
            total_loss_train.append(mean_loss_train)
            total_acc_train.append(mean_acc_train)

            # printing Training Loss every 5 epochs to observe convergence 
            if (k + 1) %5 == 0:
                running_time = time.time() - time_start
                print('Epoch:', k+1, ' Training Loss:', total_loss_train[k], ' Time (sec) per epoch:', running_time)

        total_time = time.time() - total_time_start
        return np.array(total_loss_train), total_time

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x, train_mode = False)
        return x
    
    # given test feature and y_true, return accuracy of the model
    def evaluate(self,X ,y):
        prediction = self.predict(X)
        prediction = np.argmax(prediction, 1)
        acc = sum(prediction == np.argmax(y, 1))/len(y)
        return acc

    def criterion_cross_entropy(self, y, y_hat):
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
        # self.layers[-1] is the last layer
        delta = self.layers[-1].activation_deriv(y, y_hat)
        
        # return loss and delta
        return loss, delta