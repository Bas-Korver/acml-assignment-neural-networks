import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> float:
    """
    Calculates the mean squared error (MSE) between the true and predicted values.

    :param y_true: The true values.
    :param y_pred: The predicted values.
    :param derivative: Whether to calculate the derivative or not.
    :return: The mean squared error.
    """

    if derivative:
        return np.sum(y_pred - y_true) / len(y_true)

    return np.sum(np.square(y_true - y_pred)) / len(y_true) * 2


def squared_error(y_true: np.ndarray, y_pred: np.ndarray, derivative: bool = False) -> np.ndarray:
    """
    Calculates the squared error (SE) between the true and predicted values.

    :param y_true: The true values.
    :param y_pred: The predicted values.
    :param derivative: Whether to calculate the derivative or not.
    :return: The squared error.
    """

    if derivative:
        return y_pred - y_true

    return np.square(y_true - y_pred) / 2
