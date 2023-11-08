import numpy as np


def zeros(output_shape: int, input_shape: int, **kwargs):
    """
    Generates a matrix of zeros.

    :param output_shape: Amount of output Neurons.
    :param input_shape: Amount of input Neurons.

    :return: Matrix of zeros.
    """

    return np.zeros((output_shape, input_shape))


def normal(output_shape: int, input_shape: int, **kwargs):
    """
    Generates a matrix of random values from a normal distribution.

    :param output_shape: Amount of output Neurons.
    :param input_shape: Amount of input Neurons.

    :return: Matrix of random values from a normal distribution.
    """

    return np.random.normal(kwargs["mu"], kwargs["sigma"], size=(output_shape, input_shape))