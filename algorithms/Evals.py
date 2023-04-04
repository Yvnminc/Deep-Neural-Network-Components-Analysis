"""
File name: Evals.py
Authors: Yanming Guo
Description: Set up the experiment and run the experiments with sklearn metrics.
             The output of the experiment is a dataframe and visualisations.
"""
from .Data import Data
from .Mlp import Mlp
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
def set_exp(exps, exp_name = "activation", deep = False, bn = True):
    nn_list = []
    hyperparams = []
    exp_dict = {"exp_name": exp_name, "nn_list": nn_list, "hyperparams": hyperparams}

    opt_param_dict = {"SGD":("Momentum",[0,0]),"Momentum":("Momentum",[0.9,0.9]),"Adam":("Adam",[0.9,0.99])}
    if exp_name == "batch":
        for batch in exps[0]:
            for lr in exps[1]:
                nn = set_nn(batch=batch, lr = lr, bn = bn)
                nn_list.append(nn)
                hyperparams.append([lr, batch])
        return exp_dict
    
    for exp in exps:
        if exp_name == "activation":
            nn = set_nn(act=exp)
            if deep == True:
                nn = set_nn(act=exp, structure = [256, 128, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10])

        elif exp_name == "optimiser":
            exp_param = opt_param_dict[exp]
            nn = set_nn(opt = exp_param, act = "leaky_relu")
        elif exp_name == "structure":
            nn = set_nn(structure=exp)
        elif exp_name == "keep_prob":
            nn = set_nn(keeprob=exp, opt = ["Adam", [0.9, 0.99]], act = "leaky_relu")
        elif exp_name == "batch_normalizer":
            nn = set_nn(bn=exp)
        elif exp_name == "weight_decay":
            nn = set_nn(weight_decay=exp, opt = ["Adam", [0.9, 0.99]], act = "leaky_relu", keeprob=0.8)

        nn_list.append(nn)
        hyperparams.append(exp)
    
    return exp_dict

# Set the default values for the hyperparameters of the neural network
def set_nn(lr = 0.01, batch = 128, act = "relu", opt = ["Momentum", [0.9, 0]],
           bn = True, structure = [512, 256, 128, 64, 10], keeprob = 1, weight_decay = 0):
    '''
    Set the neural network with the given hyperparameters.
    '''
    # Set the optimiser
    opt_type = opt[0]
    params = opt[1]

    # Set the structure
    first_layer = structure[0]
    last_second_layer = structure[-2]
    last_layer = structure[-1]

    # Set the neural network
    nn = Mlp(learning_rate = lr, batch_size=batch)

    # set the regularizer
    if weight_decay > 0:
        nn.set_regularizer(weight_decay)

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
    nn.add_layer(last_second_layer,last_layer,"softmax",1)
    return nn

# Run the experiment
def run_exp(data = Data(), epochs = 5, exp_dict = None,
            plot = True, metric = "all"):
    '''
    Run the experiment with the given hyperparameters and neural networks.
    '''
    nns = exp_dict["nn_list"]
    hyperparams = exp_dict["hyperparams"]
    exp_name = exp_dict["exp_name"]

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

    for nn, hyperparam in zip(nns, hyperparams):
        print(f"-----------------Running: {hyperparam}----------------------")
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

    if plot == True:
        plot_train_valid_bar(eval_df, metric = metric, exp_name = exp_name)
        plot_time_bar(eval_df, exp_name = exp_name)
        plot_loss_line(eval_df, exp_name = exp_name)
    return eval_df
    
# Draw the training and validation metrics for different hyperparameters
def plot_train_valid_bar(eval_df, metric = "all", exp_name = "batch"):
    '''
    The training metric is represented by the blue bar
    The validation metric is represented by the orange bar
    '''
    if metric == "accuracy":
        eval_df.plot(x="Hyperparameters", y=["train_acc", "valid_acc"], kind="bar")

        # Assign value to the bar
        for i in range(len(eval_df)):
            plt.text(x = i-0.1 , y = eval_df["train_acc"][i] + 0.01, s = round(eval_df["train_acc"][i], 3), size = 10)
            plt.text(x = i+0.1 , y = eval_df["valid_acc"][i] + 0.01, s = round(eval_df["valid_acc"][i], 3), size = 10)
        plt.ylabel("Accuracy")

        # Move the legend to dwon right
        plt.legend(loc='lower right')

    if metric == "precision":
        eval_df.plot(x="Hyperparameters", y=["train_precision", "valid_precision"], kind="bar")

        # Assign value to the bar
        for i in range(len(eval_df)):
            plt.text(x = i-0.1 , y = eval_df["train_precision"][i] + 0.01, s = round(eval_df["train_precision"][i], 3), size = 10)
            plt.text(x = i+0.1 , y = eval_df["valid_precision"][i] + 0.01, s = round(eval_df["valid_precision"][i], 3), size = 10)

        plt.ylabel("Precision")

        # Move the legend to dwon right
        plt.legend(loc='lower right')

    if metric == "recall":
        eval_df.plot(x="Hyperparameters", y=["train_recall", "valid_recall"], kind="bar")

        # Assign value to the bar
        for i in range(len(eval_df)):
            plt.text(x = i-0.1 , y = eval_df["train_recall"][i] + 0.01, s = round(eval_df["train_recall"][i], 3), size = 10)
            plt.text(x = i+0.1 , y = eval_df["valid_recall"][i] + 0.01, s = round(eval_df["valid_recall"][i], 3), size = 10)

        plt.ylabel("Recall")

        # Move the legend to dwon right
        plt.legend(loc='lower right')

    if metric == "f1":
        eval_df.plot(x="Hyperparameters", y=["train_f1", "valid_f1"], kind="bar")

        # Assign value to the bar
        for i in range(len(eval_df)):
            plt.text(x = i-0.2 , y = eval_df["train_f1"][i] + 0.01, s = round(eval_df["train_f1"][i], 3), size = 10)
            plt.text(x = i+0.2 , y = eval_df["valid_f1"][i] + 0.01, s = round(eval_df["valid_f1"][i], 3), size = 10)

        plt.ylabel("F1")

        # Move the legend to dwon right
        plt.legend(loc='lower right')

    if metric == "all":
        # Plot the figure in (2 x 2) grid
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Metrics by different hyperparameters')

        # Plot accuracy
        eval_df.plot(y=["train_acc", "valid_acc"], kind="bar", ax=axs[0, 0], x="Hyperparameters")

        # Assign value to the bar
        for i in range(len(eval_df)):
            axs[0, 0].text(x = i - 0.25, y = eval_df["train_acc"][i], s = round(eval_df["train_acc"][i], 2), size = 10)
            axs[0, 0].text(x = i + 0.05, y = eval_df["valid_acc"][i], s = round(eval_df["valid_acc"][i], 2), size = 10)

        axs[0, 0].set_title("Accuracy")
        axs[0, 0].set_ylabel("Accuracy")

        # Remove the x label
        axs[0, 0].set_xlabel("")

        # Move the legend to dwon right
        axs[0, 0].legend(loc='lower right')
        
        # Plot precision
        eval_df.plot(y=["train_precision", "valid_precision"], kind="bar", ax=axs[0, 1], x="Hyperparameters")

        # Assign value to the bar
        for i in range(len(eval_df)):
            axs[0, 1].text(x = i - 0.25, y = eval_df["train_precision"][i], s = round(eval_df["train_precision"][i], 2), size = 10)
            axs[0, 1].text(x = i + 0.05, y = eval_df["valid_precision"][i], s = round(eval_df["valid_precision"][i], 2), size = 10)

        axs[0, 1].set_title("Precision")
        axs[0, 1].set_ylabel("Precision")

        # Remove the x label
        axs[0, 1].set_xlabel("")

        # Move the legend to dwon right
        axs[0, 1].legend(loc='lower right')

        # Plot recall
        eval_df.plot(y=["train_recall", "valid_recall"], kind="bar", ax=axs[1, 0], x = "Hyperparameters")

        # Assign value to the bar
        for i in range(len(eval_df)):
            axs[1, 0].text(x = i -0.25, y = eval_df["train_recall"][i], s = round(eval_df["train_recall"][i], 2), size = 10)
            axs[1, 0].text(x = i + 0.05, y = eval_df["valid_recall"][i], s = round(eval_df["valid_recall"][i], 2), size = 10)

        axs[1, 0].set_title("Recall")
        axs[1, 0].set_ylabel("Recall")

        # Remove the x label
        axs[1, 0].set_xlabel("")

        # Move the legend to dwon right
        axs[1, 0].legend(loc='lower right')

        # Plot F1 score
        eval_df.plot(y=["train_f1", "valid_f1"], kind="bar", ax=axs[1, 1], x = "Hyperparameters")

        # Assign value to the bar with 1 decimal places
        for i in range(len(eval_df)):
            axs[1, 1].text(x = i-0.25 , y = eval_df["train_f1"][i], s = round(eval_df["train_f1"][i], 2), size = 10)
            axs[1, 1].text(x = i+0.05 , y = eval_df["valid_f1"][i], s = round(eval_df["valid_f1"][i], 2), size = 10)

        axs[1, 1].set_title("F1 Score")
        axs[1, 1].set_ylabel("F1 Score")
        
        # Remove the x label
        axs[1, 1].set_xlabel("")

        # Move the legend to dwon right
        axs[1, 1].legend(loc='lower right')

        # Adjust layout to avoid overlapping the texts
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)


    plt.savefig(f'../visual_outputs/{exp_name}_{metric}.png')
    plt.show()
    

# Draw time by different hyperparameters
def plot_time_bar(eval_df, exp_name = "batch"):
    '''
    Plot the time by different hyperparameters in bar chart
    '''
    eval_df.plot(x="Hyperparameters", y=["times"], kind="bar")

    # Assign value to the bar
    for i in range(len(eval_df)):
        plt.text(x = i - 0.2, y = eval_df["times"][i] + 0.05, s = round(eval_df["times"][i], 3), size = 10)

    plt.ylabel('Time (s)')
    plt.savefig(f'../visual_outputs/{exp_name}_times.png')

    # Move the legend to dwon right
    plt.legend(loc='lower right')
    plt.show()

# Draw the training loss by epochs
def plot_loss_line(eval_df, exp_name = "batch"):
    '''
    Plot the training loss by epochs with different hyperparameters in line chart
    '''
    loss = eval_df["loss"]
    hyperparams = eval_df["Hyperparameters"]

    for hyperparam in range(len(hyperparams)):
        plt.plot(loss[hyperparam],label=f'{hyperparams[hyperparam]}')

    plt.legend()
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig(f'../visual_outputs/{exp_name}_loss.png')
    plt.show()

def experiment(exp_para, exp_name, epochs = 10):
    exp_dict = set_exp(exp_para, exp_name)
    print(exp_dict)
    exp_df = run_exp(exp_dict, epochs = epochs)
    return exp_df