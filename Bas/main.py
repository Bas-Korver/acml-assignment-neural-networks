import numpy as np

import ann
from activation_functions import sigmoid
from cost_functions import squared_error


def generate_training_data(examples: np.ndarray, n: int, shuffel: bool = True) -> np.ndarray:
    """
    Generates n copies of the training examples.

    :param examples: The training examples.
    :param n: The number of copies.
    :param shuffel: Whether to shuffel the data or not.
    :return: The training examples.
    """

    data = np.tile(examples, (n, 1))

    if shuffel:
        return np.random.permutation(data)
    else:
        return data


if __name__ == "__main__":
    training_examples = np.identity(8)
    training_data = generate_training_data(training_examples, 1, shuffel=False)

    ann = ann.ANN(8, [3], 8, sigmoid, squared_error, bias=True)

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])


    ann.feed_forward(a)
    ann.backpropagate(a, a)

    # ann.train(training_data, training_data, 100, 0.1, 32)

