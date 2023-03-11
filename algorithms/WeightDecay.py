import numpy as np

class L2:
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss= 0

    def reset(self):
        self.loss = 0

    def forward(self, W):
        self.loss += np.sum(np.square(W))
        
    # not sure if we need this
    #def get_loss(self, m):
    #   return self.loss * (self.lamda/(2*m))

    def backward(self, grad_W, W, m):
        return grad_W + self.lamda * W / m 

class L1:
    def __init__(self, lamda = 1):
        self.lamda = lamda
        self.loss = 0

    def reset(self):
        self.loss = 0

    def forward(self, W):
        self.loss += np.sum(np.abs(W))
    
    #def get_loss(self, m):
    #   return self.loss * self.lamda/m

    def backward(self, grad_W, W, m):
        return grad_W + self.lamda * np.sign(W) / m
