"""
File name: BatchNomalization.py
Authors: Jiacheng Zhang
Description: Defines the process of batch normalization.
Reference: https://arxiv.org/pdf/1502.03167.pdf
"""
import numpy as np

class BatchNormalization:
    def __init__(self, momentum=0.9, optimizer=None):
        """
        initialization of batch normalization
        :param momentum: a hyperparameter that controls the momentum in gradient descent based optimization
        :param optimizer: a specific optimization algorithm, e.g., SGD
        m: the size of a mini-batch
        gamma: a parameter that scales the normalized value
        beta: a parameter that shifts the normalized value
        dgamma: the derivative of gamma
        dbeta: the derivative of beta
        X: a batch of input values, X = {x1, ... ,xm}
        X_hat: the normalized X
        mean: mini-batch mean
        var: mini-batch variance
        std: mini-batch standard deviation
        avg_mean: weighted average mean by considering momentum
        avg_var: weighted average variance by considering momentum
        """
        self.momentum = momentum
        self.optimizer = optimizer
        self.m = None
        self.gamma = None
        self.beta = None
        self.dgamma = 0
        self.dbeta = 0
        self.x = None
        self.x_hat = None
        self.mean = None
        self.var = None
        self.std = None
        self.avg_mean = 0
        self.avg_var = 0

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def clone(self):
        opt = None
        if self.optimizer is not None:
            opt = self.optimizer.clone()
        return BatchNormalization(self.momentum, opt)

    def __init_param(self, d):
        self.gamma = np.ones((d, 1))
        self.beta = np.zeros((d, 1))
        self.avg_mean = np.zeros((d, 1))
        self.avg_var = np.zeros((d, 1))

    def forward(self, x, training, epsilon=1e-8):
        '''
        Batch Nomalization Transform, applied to activation x over a mini-batch
        '''
        if self.gamma is None and self.beta is None:
            self.__init_param(x.shape[0])

        if not training:
            # print(x.shape)
            # print(self.avg_mean.shape)
            x_hat = (x - self.avg_mean)/np.sqrt(self.avg_var + epsilon)
            return (self.gamma * x_hat) + self.beta

        self.x = x

        # Step1: Calculate the mini-batch mean
        self.mean = np.mean(x, axis=1, keepdims=True)

        # Step2: calculate the mini-batch variance
        self.var = np.var(x, axis=1, keepdims=True)

        # Step3: Calculate the standard deviation
        self.std = np.sqrt(self.var + epsilon)

        # Step4: Normalize
        self.x_hat = (self.x - self.mean)/self.std
        self.avg_mean = self.momentum * self.avg_mean + (1 - self.momentum) * self.mean
        self.avg_var = self.momentum * self.avg_var + (1 - self.momentum) * self.var

        # Step5: Scale and Shift
        return (self.gamma * self.x_hat) + self.beta

    def backward(self, dx_norm):
        '''
        During training we need to backpropagate the gradient of loss through batch normalization transformation,
        as well as compute the gradients with respect to the parameters of the BN transform.
        '''

        # x_norm = gamma * X_hat + beta
        # Therefore, dx_hat = dx_norm * gamma
        dx_hat = dx_norm * self.gamma

        m = self.x.shape[1]
        # dgamma = dx_norm * input_hat
        # dx_norm is (n_feature, m), X_hat is (n_feature, m)
        # where gamma is (n_feature, 1), sum the product along axis 1
        self.dgamma = np.sum(dx_norm * self.x_hat, axis=1, keepdims=True)/m
        # the same applied to dbeta
        self.dbeta = np.sum(dx_norm, axis=1, keepdims=True)/m

        # The following derivations can be found in paper https://arxiv.org/pdf/1502.03167.pdf
        dvar = np.sum(dx_hat * (self.x - self.mean), axis=1, keepdims=True) * ((self.std**-3)/-2)

        dmean = np.sum(dx_hat * (-1/self.std), axis=1, keepdims=True) + dvar * np.sum(-2*(self.x - self.mean),
                                                                                      axis=1, keepdims=True)/m

        dx = dx_hat/self.std + (dvar * (2*(self.x-self.mean)/m)) + dmean/m
        return dx

    def update(self, lr):
        if self.optimizer is None:
            self.gamma = self.gamma - lr * self.dgamma
            self.beta = self.beta - lr * self.dbeta
        else:
            self.gamma, self.beta = self.optimizer.update(lr, self.gamma, self.beta, self.dgamma, self.dbeta)