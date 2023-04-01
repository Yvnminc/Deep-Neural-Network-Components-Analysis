"""
File name: Activations.py
Authors: Yanming Guo
Description: Defines different activation functions.
Reference: Week 2 tut sheet of COMP5329 Deep Learning, University of Sydney
"""
import numpy as np

class Activation:
    # Tanh activation
    def __tanh(self, x):
        '''
        Derivative of tanh function.
        Accroding to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        '''
        return np.tanh(x)
    
    # Derivative of tanh function
    def __tanh_deriv(self, a):
        '''
        Derivative of tanh function.
        Accroding to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f'(x) = 1 - tanh^2(x)
        '''
        return 1 - np.square(a)
    
    # Logistic activation
    def __logistic(self, x):
        '''
        Logistic function.
        According to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f(x) = 1 / (1 + exp(-x))
        '''
        return 1.0 / (1.0 + np.exp(-x))

    # Derivative of logistic function as known as sigmoid function
    def __logistic_deriv(self, a):
        '''
        Derivative of logistic function.
        According to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f'(x) = f(x) * (1 - f(x))
        '''
        return a * (1.0 - a)
    
    # Relu activation
    def __relu(self, x):
        '''
        Relu function.
        According to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f(x) = max(0, x)
        '''
        return np.maximum(0, x)

    # Derivative of relu function
    def __relu_deriv(self, a):
        '''
        Derivative of ReLU function.
        According to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f'(x) = 1 if x > 0 else 0
        '''
        return (a > 0).astype(int)
    
    # Leaky Relu activation
    def __leaky_relu(self, x, alpha=0.01):
        '''
        Leaky Relu function.
        According to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f(x) = x if x > 0 else alpha * x
        '''
        return np.maximum(alpha * x, x)
    
    # Derivative of leaky relu function
    def __leaky_relu_deriv(self, a, alpha=0.01):
        '''
        Derivative of Leaky ReLU function.
        According to Review and Comparison of Activation Functions in Deep Neural Networks
        https://arxiv.org/pdf/2010.09458.pdf

        f'(x) = 1 if x > 0 else alpha
        '''
        return (a > 0).astype(int) + alpha * (a <= 0).astype(int)
    
    def __elu(self, x, alpha=0.01):
        '''
        ELU function.
        According to Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        https://arxiv.org/pdf/1511.07289.pdf

        f(x) = x if x > 0 else alpha * (exp(x) - 1)
        '''

        x = np.clip(x, -10, 10)
        value = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        # print(value)
        return value
    
    # Derivative of elu function
    def __elu_deriv(self, a, alpha=0.01):
        '''
        Derivative of ELU function.
        According to Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        https://arxiv.org/pdf/1511.07289.pdf

        f'(x) = 1 if x > 0 else f(x) + alpha
        '''
        return (a > 0).astype(int) + (a <= 0).astype(int) * (a + alpha)
    
    # GELU activation
    def __gelu(self, x, simple = True):
        '''
        GELU function.
        Accroding to https://arxiv.org/pdf/1606.08415.pdf

        f(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

        or for efficiency, the formula could be simplified as:
        f(x) = x * sigmoid(1.702 * x)
        '''
        if simple:
            return x * self.__logistic(1.702 * x)
        else:
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def __gelu_deriv(self, a, simple = True):
        '''
        Derivative of GELU function.
        Accroding to https://arxiv.org/pdf/1606.08415.pdf

        f'(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))) + (x * pi^(-0.5) * exp(-x^2 / 2) * (x^2 + 1.26551223 * x + 1.00002368)) / (sqrt(2) * sqrt(pi))

        or for efficiency, the formula could be simplified as:
        f'(x) = sigmoid(1.702 * x) + x * sigmoid(1.702 * x) * (1 - sigmoid(1.702 * x))
        '''

        if simple:
            return self.__logistic(1.702 * a) + a * self.__logistic(1.702 * a) * (1 - self.__logistic(1.702 * a))
        else:
            return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a**3)) + (a * np.pi**(-0.5) * np.exp(-a**2 / 2) * (a**2 + 1.26551223 * a + 1.00002368)) / (np.sqrt(2) * np.sqrt(np.pi)))
    
    def __swish(self, x, beta=1):
        '''
        Swish function.
        Accroding to https://arxiv.org/pdf/1710.05941.pdf

        f(x) = x * sigmoid(beta * x)
        '''
        return x * self.__logistic(beta * x)
    
    def __swish_deriv(self, a, beta=1):
        '''
        Derivative of Swish function.
        Accroding to https://arxiv.org/pdf/1710.05941.pdf

        f'(x) = beta * swish(x) + sigmoid(beta * x) * (1 - swish(x))
        '''
        return beta * a + 1  - beta * a
    
    # Softmax activation
    def __softmax(self, x):
        """
        Softmax function
        Accoording to https://en.wikipedia.org/wiki/Softmax_function

        f(x) = exp(x) / sum(exp(x)
        """
        return np.exp(x - np.max(x, axis = 1, keepdims = True)) / np.sum(np.exp(x - np.max(x, axis = 1, keepdims = True)), axis = 1, keepdims = True)
       
    def __softmax_deriv(self, y, y_pred):
        '''
        Derivative of softmax function, conbined with cross entropy loss function.
        According to https://deepnotes.io/softmax-crossentropy

        f'(x) = y_pred - y
        '''
        return y_pred - y
    
    def __init__(self,activation='tanh'):
        '''
        Init function for the activation class.
        '''
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv

        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv

        elif activation == 'leaky_relu':
            self.f = self.__leaky_relu
            self.f_deriv = self.__leaky_relu_deriv

        elif activation == 'elu':
            self.f = self.__elu
            self.f_deriv = self.__elu_deriv

        elif activation == 'gelu':
            self.f = self.__gelu
            self.f_deriv = self.__gelu_deriv

        elif activation == 'swish':
            self.f = self.__swish
            self.f_deriv = self.__swish_deriv