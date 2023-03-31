"""
File name: Evals.py
Authors: Yanming Guo
Description: Set up the experiment and run the experiments with sklearn metrics.
             The output of the experiment is a dataframe.
"""
from .Data import Data
from .MlpV2 import MlpV2
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the batch size experiment
def set_batch(lrs, batchs):
    nn_list = []
    hyperparams = []

    for batch in batchs:
        for lr in lrs:
            nn = set_nn(batch=batch, lr = lr)
            nn_list.append(nn)
            hyperparams.append([lr, batch])

    return nn_list, hyperparams

# Define the activation experiment
def set_exp(exps, exp_name = "activation"):
    nn_list = []
    hyperparams = []

    for exp in exps:
        if exp_name == "activation":
            nn = set_nn(act=exp)
        elif exp_name == "optimiser":
            nn = set_nn(opt=exp)
        elif exp_name == "structure":
            nn = set_nn(structure=exp)
        elif exp_name == "keep_prob":
            nn = set_nn(keeprob=exp)

        nn_list.append(nn)
        hyperparams.append(exp)
    
    return nn_list, hyperparams

# Set the default values for the hyperparameters of the neural network
def set_nn(lr = 0.01, batch = 128, act = "relu", opt = ["Momentum", [0.9]],
           bn = True, structure = [512, 256, 128, 64, 10], keeprob = 1):
    
    # Set the optimiser
    opt_type = opt[0]
    params = opt[1]

    # Set the structure
    first_layer = structure[0]
    last_second_layer = structure[-2]
    last_layer = structure[-1]

    # Set the neural network
    nn = MlpV2(learning_rate = lr, batch_size=batch)

    # Set the optimiser
    nn.set_optimiser(opt_type= opt_type, params = params)

    # Set batch normalizer
    if bn == True:
        nn.set_batchNormalizer()

    # Add first layers
    nn.add_layer(128,first_layer,act,keeprob)

    # Add hidden layers
    for i in range(len(structure)-2):
        nn.add_layer(structure[i],structure[i+1],act,keeprob)

    # Add last layer
    nn.add_layer(last_second_layer,last_layer,"softmax",keeprob)
    return nn

# Run the experiment
def run_exp(data = Data(), epochs = 5, hyperparams = None, nns = None):
    # Metrics
    loss = []
    train_acc = []
    valid_acc = []
    train_precision = []
    valid_precision = []
    train_recall = []
    valid_recall = []
    train_f1 = []
    valid_f1 = []
    times = []

    # Training and validation
    X_train = data.train_data
    y_train = data.train_label
    X_valid = data.validation_data
    y_valid = data.validation_label

    # Testing
    X_test = data.test_data
    y_test = data.test_label

    for nn in nns:
        train_loss,time = nn.fit(X_train, y_train, epochs= epochs)
        loss.append(train_loss)

        acc = nn.evaluate(X_train, y_train)
        train_acc.append(acc)

        v_acc = nn.evaluate(X_valid, y_valid)
        valid_acc.append(v_acc)
        
        y_pred_train = np.argmax(nn.predict(X_train), axis=1)
        y_train_transformed = np.argmax(y_train, axis=1)
        y_pred_valid = np.argmax(nn.predict(X_valid), axis=1)
        y_valid_transformed = np.argmax(y_valid, axis=1)
        
        precision = precision_score(y_train_transformed, y_pred_train, average='macro')
        recall = recall_score(y_train_transformed, y_pred_train, average='macro')
        f1 = f1_score(y_train_transformed, y_pred_train, average='macro')
        
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)
        
        v_precision = precision_score(y_valid_transformed, y_pred_valid, average='macro')
        v_recall = recall_score(y_valid_transformed, y_pred_valid, average='macro')
        v_f1 = f1_score(y_valid_transformed, y_pred_valid, average='macro')
        
        valid_precision.append(v_precision)
        valid_recall.append(v_recall)
        valid_f1.append(v_f1)
        times.append(time)
    
    eval_dict = {"loss": loss, "train_acc": train_acc, "valid_acc": valid_acc, "train_precision": train_precision, "valid_precision": valid_precision, "train_recall": train_recall, "valid_recall": valid_recall, "train_f1": train_f1, "valid_f1": valid_f1, "times": times}

    eval_df = pd.DataFrame(eval_dict)
    eval_df.insert(0, "Hyperparameters", hyperparams, True)

    return eval_df

# Draw the training and validation accuracy for different hyperparameters
def plot_train_valid_acc_bar(eval_df):
    '''
    The training accuracy is represented by the blue bar
    The validation accuracy is represented by the orange bar
    '''
    eval_df.plot(x="Hyperparameters", y=["train_acc", "valid_acc"], kind="bar")
    plt.show()

# Draw the training and validation accuracy for different hyperparameters
def plot_train_valid_precision_bar(eval_df):
    '''
    The training precision is represented by the blue bar
    The validation precision is represented by the orange bar
    '''
    eval_df.plot(x="Hyperparameters", y=["train_acc", "valid_acc"], kind="bar")
    plt.show()