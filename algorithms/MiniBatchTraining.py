"""
File name: MiniBatchTraining.py
Authors: Jiacheng Zhang
Description: Defines the process of mini-batch training.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np

class MiniBatch:

    def get_batch(self, X, Y, batch_size):
        """
        Divide data into batches.
        :param X: Input data or features, assume with shape (n_features, n_examples)
        :param Y: Input targets, assume with the shape (n_classes, n_example)
        :param batch_size: a hyperparameter that defines the size of a batch
        """
        mini_batches = []
        data = np.stack((X, Y), axis=1)
        np.random.shuffle(data)
        num_batches = X.shape[0] // batch_size
        for i in range(num_batches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size]
            mini_batches.append((mini_batch[:, 0], mini_batch[:, 1]))
        if X.shape[0] % batch_size != 0:
            mini_batch = data[(i + 1) * batch_size:]
            mini_batches.append((mini_batch[:, 0], mini_batch[:, 1]))
        return mini_batches

    # Add this function to optimizer.py
    # To Be Done
    def batch_update(self, dW, db):
        return

    # Add this function to Mlp.py
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

    # Add this function to Mlp.py
    def fit(self, X, y, lr, epochs, batch_size):
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
            batches = self.get_batch(X, y, batch_size)
            loss = np.zeros(len(batches))
            index = 0

            for batch in batches:
                X_b = np.array(batch[0])
                Y_b = np.array(batch[1])
                dW = []
                db = []
                batch_loss = np.zeros(X_b.shape[0])
                for j in range(X_b.shape[0]):
                    # forward pass
                    y_hat = self.forward(X_b[j])
                    # backward pass
                    batch_loss[j], delta = self.criterion_loss(Y_b[j], y_hat)
                    self.backward(delta)
                    layer_grad_W, layer_grad_b = self.get_grads()
                    dW.append(layer_grad_W)
                    db.append(layer_grad_b)
                loss[index] = np.mean(batch_loss)
                index += 1
                gradients_W = {}
                gradients_b = {}
                for i in range(len(self.layers)):
                    gradients_W[i] = np.array([j[i] for j in dW]).mean(axis=0)
                    gradients_b[i] = np.array([j[i] for j in db]).mean(axis=0)
                DW = [i for j, i in gradients_W.items()]
                Db = [i for j, i in gradients_b.items()]
                # update weights with batch gradient
                self.batch_update(DW, Db)
            to_return[k] = np.mean(loss)
        return to_return


