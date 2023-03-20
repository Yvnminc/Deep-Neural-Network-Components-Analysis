"""
File name: MiniBatchTraining.py
Authors: Jiacheng Zhang
Description: Defines the process of mini-batch training.
Reference: Week 2 tut sheet of COMP5329 Deep Learning,
           University of Sydney
"""
import numpy as np

class MiniBatch:

    def get_batch(self, X, y, batch_size):
        """
        Divide data into batches.
        :param X: Input data or features, assume with shape (n_examples, n_features)
        :param Y: Input targets, assume with the shape (n_example, n_classes)
        :param batch_size: a hyperparameter that defines the size of a batch
        """
        mini_batches = []
        n_features = X.shape[1]
        data = np.concatenate((X, y), axis=1)
        np.random.shuffle(data)

        num_batches = X.shape[0] // batch_size
        for i in range(num_batches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size]
            mini_batches.append((mini_batch[:, :n_features], mini_batch[:, n_features:]))
        # if X.shape[0] % batch_size != 0:
        #     mini_batch = data[(i + 1) * batch_size:]
        #     mini_batches.append((mini_batch[:, :n_features], mini_batch[:, n_features:]))
        return mini_batches
