"""
File name: MiniBatchTraining.py
Authors: Jiacheng Zhang
Description: Defines the process of mini-batch training.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np

class MiniBatchV2:
    def __init__(self, X, Y):
        #assume X is of the shape (n_features, n_example)
        #Y is of the shape (n_classes, n_example)
        self.x_features = X.shape[1]
        self.y_classes = Y.shape[1]
        self.m = X.shape[0]
        self.map = np.concatenate([X, Y], axis=1)
        self.loss = []
        self.accuracy = []

    def shuffle(self, boo = True):

        #shuffle only shuffles the array along the first axis
        #must reshape map first
       
        np.random.shuffle(self.map)
        



    def getX(self):
        return self.map[:, :self.x_features]

    def getY(self):
        return self.map[:, self.x_features:]

    def getLoss(self):
        return self.loss

    def getAccuracy(self):
        return self.accuracy



    def reset(self):
        self.loss = []
        self.accuracy = []



    def fit(self, model, size = None):
        #reset loss and accuracy
        self.reset()

        #if batch size is not given then use one batch per epoch
        if(size == None):
            size = self.m

        #shuffle
        self.shuffle()

        #get the number of batch, X  and Y
        batch_num = self.m//size
        shuff_X = self.getX()
        shuff_Y = self.getY()

        for i in range(batch_num):


            start = i * size
            end = start + size
            mini_X = shuff_X[start:end,:]
            mini_Y = shuff_Y[start:end,:]


            mini_Y_hat = model.forward(mini_X, mode = True)
            if model.regularizer is not None:
                self.loss.append( model.criterion_cross_entropy(mini_Y, mini_Y_hat)[0] + model.regularizer.get_reg_loss(self.m))
            else:
                self.loss.append( model.criterion_cross_entropy(mini_Y, mini_Y_hat)[0])
            
            #print(self.loss)

            

            #compute accuracy for this mini batch
            self.accuracy.append(np.mean( np.equal(np.argmax(mini_Y, 1), np.argmax(mini_Y_hat, 1))))

            mini_dz = mini_Y_hat - mini_Y
            model.backward(mini_dz)
            model.update()