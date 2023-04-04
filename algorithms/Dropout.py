"""
File name: Dropout.py
Authors: Yongjiang Shi
Description: The Dropout class allows performing regularization on the activation result through droping 
some nodes in a layer during training. In back propagation, we adjust the weight for nodes in the same layer that 
was not been masked during the forward phase.

"""

import numpy as np

class Dropout:
    """
    Description: The Dropout class allows performing regularization on the activation result through droping 
    some nodes in a layer during training.
    
    Attribute:
    - keep_prob: the probability of a node not being dropped
    
    Method:
    - forward(A, mode): takes a activation result for one layer and apply a dropout mask on it so some of the activation becomes 0, 
    this only happens when mode = "train"
    - backward(dA): apply back prop on the layer that has went through the dropout process
    
    """
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None



    def forward(self, A):
        # Mask has the same shape as A
        self.mask = np.random.rand(A.shape[0], A.shape[1])
        # Turn the random number in mask into 0 and 1 
        self.mask = (self.mask < self.keep_prob)
        # Perfrom the element-wise multiplication and drop out some nodes in A
        A = A * self.mask
        # scale the remaing node up by the keep probability
        A = A / self.keep_prob
        return  A
          

    def backward(self, dA):
        # multiply by the same drop out binary matrix in the forward pass
        dA = dA * self.mask 
        # scale by the keep probability
        dA = dA / self.keep_prob
        return dA
