import numpy as np


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples o sr not

    Returns
    -------
    Misclassification of given predictions 
    """
    try:
        if normalize:
            return np.sum(y_true != y_pred)/len(y_true)
        else:
            return np.sum(y_true != y_pred)
    except:
        raise ValueError("Input arrays must be of same shape")

