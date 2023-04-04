"""
File name: Optimizers.py
Authors: Yongjiang Shi
Description: Defines the gradient descent optimizer with momentum setting allowed, 
if beta is set to 0 then it is standard gradient descent, otherwise it will be gradient 
descent with momentum.

"""
import numpy as np

class GD_with_Momentum:
    def __init__(self, beta = 0):
        # with beta = 0, it will be gradient descent without momentum
        # if want to do GD with momentum, the default value is usually 0.9

        self.beta = beta
        self.v_W = 0
        self.v_b = 0
        
    # for copying the optimizer at model level and set optimizer for each layer 
    def clone(self):
        return GD_with_Momentum(self.beta)

    # update rule for gradient descent with momentum
    def update(self, lr, W, b, grad_W, grad_b):
        self.v_W = self.beta * self.v_W +  (1 - self.beta) * grad_W
        self.v_b = self.beta * self.v_b +  (1 - self.beta) * grad_b
        return W - lr * self.v_W, b - lr * self.v_b



class Adam:
    def __init__(self, beta1 = 0.9, beta2 = 0.99 ):
        self.epsilon = 1e-12

        self.beta1 = beta1
        self.beta2 = beta2

        # first and second moments for weight and bias
        self.vW = 0
        self.mW = 0
        self.vb = 0
        self.mb = 0

        # timestamp
        self.t = 0
    
    # for copying the optimizer at model level and set optimizer for each layer 
    def clone(self):
        return Adam(beta1 = self.beta1, beta2 = self.beta2)


    # adam optimizer update rule with bias correction
    def update(self, lr, W, b, grad_W, grad_b):

        self.t+=1

        self.mW = (self.beta1 * self.mW + (1 - self.beta1) * grad_W)
        self.vW = (self.beta2 * self.vW + (1 - self.beta2) * (grad_W ** 2))
        self.mb = (self.beta1 * self.mb + (1 - self.beta1) * grad_b)
        self.vb = (self.beta2 * self.vb + (1 - self.beta2) * (grad_b) ** 2)

        
        # perform bias correction
        mW_hat = self.mW/(1 - (self.beta1 ** self.t))
        vW_hat = self.vW/(1 - (self.beta2 ** self.t))

        mb_hat = self.mb/(1 - (self.beta1 ** self.t))
        vb_hat = self.vb/(1 - (self.beta2 ** self.t))


        return W - lr * (mW_hat/np.sqrt(vW_hat + self.epsilon)), b - lr * (mb_hat/np.sqrt(vb_hat + self.epsilon))



        
        
   

            

