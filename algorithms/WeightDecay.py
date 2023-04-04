"""
File name: WeightDecay.py
Authors: Yongjiang Shi
Description: Defines the regularizer (L2) that could been used for the model as weight decay term. 
"""

import numpy as np

class L2:
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss= 0

    def reset(self):
        self.loss = 0

    def forward(self, W):
        # accumalate the square of weights during the forward pass 
        self.loss += np.sum(np.square(W))      
    
    def get_loss(self, m):
        return self.loss * (self.lamda/(2*m))

    def backward(self, grad_W, W, m):
        return grad_W + self.lamda * W / m 

