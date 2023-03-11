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

    def update(self, lr, W, b, grad_W, grad_b):
        self.v_W = self.beta * self.v_W +  (1 - self.beta) * grad_W
        self.v_b = self.beta * self.v_b +  (1 - self.beta) * grad_b
        return W - lr * self.v_W, b - lr * self.v_b

   


  
