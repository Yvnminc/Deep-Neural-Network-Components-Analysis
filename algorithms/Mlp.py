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
    """
    # for initiallization, the code will create all layers automatically based on the provided parameters.  
    # can we set all initilization parameter with a set_para method?   
    def __init__(self, learning_rate=0.1, batch_size=1, keep_prob=1, loss="CE", norm=None):
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
        self.keep_prob = keep_prob
        self.norm = norm
        self.batch = MiniBatch()
        self.batch_size = batch_size
        self.n_out = 10

        # activation should be controled at layer level
        #self.activation = activation
        #last_act = self.activation[-1]
        #print(self.activation[-1])

        # subsituted with set loss
        # self.criterion_loss = Loss(last_act, self.loss).cal_loss
        if self.optimizer is not None and self.norm is not None:
            self.norm.set_optimizer(self.optimizer.clone())
                      
    def set_momentum(self, beta):
        self.optimizer = GD_with_Momentum(beta)
    
    def set_batchNormalizer(self):
        self.norm = BatchNormalization()

    def set_regularizer(self, lam, reg_type):
        if reg_type == "L2":
            self.regularizer = L2(lam)
        elif reg_type == "L1":
            self.regularizer = L1(lam)
        else:
            self.regularizer = None
           
    # if it is last layer, activation should be set to softmax and keep_prob should be set to 1
    def add_layer(self, n_in, n_out, activation, keep_prob):
        layer = HiddenLayer(n_in, n_out)
        layer.set_activation(activation)
        layer.set_activation_deriv(activation)

        if self.norm is not None:
            layer.set_batchNormalizer(self.norm.clone())

        layer.set_drop_out_layer(keep_prob)

        if self.optimizer != None:
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
            #print(input.shape)
            output = layer.forward(input, regularizer=self.regularizer, train_mode=mode)
            input = output
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
    def backward(self, delta):
        delta = self.layers[-1].backward(delta, output_layer=True, regularizer=self.regularizer)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta, regularizer=self.regularizer)

    # update the network weights after backward.
    # make sure you run the backward function before the update function!    
    def update(self):
        for layer in self.layers:
           layer.update(self.lr)

    # update a batch of parameters
    # def batch_update(self, dW, db):
    #     return

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
    # it will return all losses within the whole training process.
    def fit(self, X, y, epochs):
        """
        mini-batch trainnig process.
        :param X: Input data or features, assume with shape (n_features, n_examples)
        :param Y: Input targets, assume with the shape (n_classes, n_example)
        :param lr: a hyperparameter that defines the speed of learning (i.e., learning rate)
        :param epochs: a hyperparameter that defines number of times the dataset is presented to the network for learning
        :param batch_size: a hyperparameter that defines the size of a batch
        :param momentum: a hyperparameter that controls the momentum in gradient descent based optimization
        """
        X = np.array(X)
        y = np.array(y)
        to_return = np.zeros(epochs)

        for k in range(epochs):
            time_start = time.time()
            batches = self.batch.get_batch(X, y, self.batch_size)
            loss = np.zeros(len(batches))
            index = 0
            for batch in batches:
                X_b = np.array(batch[0])
                Y_b = np.array(batch[1])
                y_hat = self.forward(X_b)
                batch_loss, delta = self.criterion_cross_entropy(Y_b, y_hat)
                self.backward(delta)
                self.update()
                loss[index] = np.mean(batch_loss)
                index += 1
            to_return[k] = np.mean(loss)
            print('Epoch:', k+1, ' Training Loss:', to_return[k], ' Time (sec):', time.time() - time_start)
        return to_return

    # define the prediction function
    # we can use predict function to predict the results of new data, by using the well-trained network.
    # def predict(self, x):
    #     x = np.array(x)
    #     output = np.zeros(x.shape[0])
    #     for i in np.arange(x.shape[0]):
    #         output[i] = self.forward(x[i,:], mode=False)
    #     return output
    def predict(self, x):
        x = np.array(x)
        for layer in self.layers:
            x = layer.forward(x, train_mode = False)#regularizer collect W during forward
        return x

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
        # self.layers[-1] is the activation function
        delta = self.layers[-1].activation_deriv(y, y_hat)
        # return loss and delta
        return loss, delta