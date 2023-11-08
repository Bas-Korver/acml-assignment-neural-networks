import numpy as np


def sigmoid(x, derivative: bool = False):
    """
    Calculates the sigmoid of x.

    :param x: The input.
    :param derivative: Whether to calculate the derivative or not.

    :return: The sigmoid of x.
    """

    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))
