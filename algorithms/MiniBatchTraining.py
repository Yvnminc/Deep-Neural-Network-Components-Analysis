"""
File name: MiniBatchTraining.py
Authors: Jiacheng Zhang
Description: Defines the process of mini-batch training.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np

class MiniBatch:
    """
    Allowing training with batches and perform forward pass and back propagation on those batches.
    """

    def __init__(self, X, Y):
        self.x_features = X.shape[1]
        self.y_classes = Y.shape[1]
        self.m = X.shape[0]
        self.map = np.concatenate([X, Y], axis=1)
        self.loss = []
        self.accuracy = []

    def shuffle(self):
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
        self.reset()

        # if there is no batch size provided then use all data as a batch
        if(size == None):
            size = self.m

        self.shuffle()

        batch_num = self.m//size
        X_shuffled = self.getX()
        Y_shuffled = self.getY()

        for i in range(batch_num):
            # work out the starting indice and the ending indice
            start = i * size
            end = start + size
            mini_X = X_shuffled[start:end,:]
            mini_Y = Y_shuffled[start:end,:]

            mini_Y_hat = model.forward(mini_X, mode = True)

            # if there is a regularizer, add the regularizer loss to the loss
            if model.regularizer is not None:
                self.loss.append( model.criterion_cross_entropy(mini_Y, mini_Y_hat)[0] + model.regularizer.get_loss(self.m))
            else:
                self.loss.append( model.criterion_cross_entropy(mini_Y, mini_Y_hat)[0])

            self.accuracy.append(np.mean(np.equal(np.argmax(mini_Y, 1), np.argmax(mini_Y_hat, 1))))

            mini_dz = mini_Y_hat - mini_Y
            model.backward(mini_dz)
            model.update()