import numpy as np


def zeros(output_shape, input_shape, **kwargs):
    return np.zeros((output_shape, input_shape))


def normal(output_shape, input_shape, **kwargs):
    print(kwargs)
    return np.random.normal(kwargs["mu"], kwargs["sigma"], size=(output_shape, input_shape))