import numpy as np

import ann
from activation_functions import sigmoid
from cost_functions import squared_error, mean_squared_error
from number_generation import zeros, normal


if __name__ == "__main__":
    training_examples = np.identity(8)

    number_generation_kwargs = {'mu': 0, 'sigma': 0.1}
    ann = ann.ANN(8, [3], 8, sigmoid, squared_error, normal, zeros, add_bias=False, **number_generation_kwargs)
    ann.train(training_examples, training_examples, 10000, 0.1)

    predictions = ann.feed_forward(training_examples)

    for i, prediction in enumerate(predictions):
        print(f"input: {training_examples[i]}, output: {np.around(prediction, 2)}")





