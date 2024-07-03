import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from adaboost import AdaBoost
from decision_stump import DecisionStump
from tqdm import tqdm


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    Q1(train_X, train_y, test_X, test_y, n_learners)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # raise NotImplementedError()

    # # Question 2: Plotting decision surfaces
    # T = [5, 50, 100, 250]
    # lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # raise NotImplementedError()

    # # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()

    # # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()
def plot_train_validation_error(train_error, validation_error,num_of_wls,file_name):
    plt.figure()
    plt.plot(range(1, num_of_wls+1), train_error, label="Train Error")
    plt.plot(range(1, num_of_wls+1), validation_error, label="Validation Error")
    plt.xlabel("Weak learners")
    plt.ylabel("Error")
    plt.title("Train and Validation Error as a Function of Number of Weak Learners")
    plt.legend()
    plt.show()
    plt.savefig(file_name)

def Q1(train_X, train_y, val_X, val_y, n_learners):
    # generate 5000 train samples and 500 test samples with noise, train ensemble adaboost of 250
    # weak learners via decision stump, plot the train and test error as func of number of weak learners
    model = AdaBoost(DecisionStump,n_learners).fit(train_X,train_y)
    train_error = [model.partial_loss(train_X,train_y,i) for i in range(1,n_learners+1)]
    val_error =[model.partial_loss(val_X,val_y,i) for i in range(1,n_learners+1)]
    plot_train_validation_error(train_error,val_error,n_learners,"Q1_no_noise.png")
if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
