import numpy as np

import ann
from activation_functions import sigmoid
from cost_functions import squared_error, mean_squared_error


if __name__ == "__main__":
    training_examples = np.identity(8)

    ann = ann.ANN(8, [8], 8, sigmoid, squared_error, add_bias=True)
    ann.train(training_examples, training_examples, 10000, 0.1)

    predictions = ann.feed_forward(training_examples)

    for i, prediction in enumerate(predictions):
        print(f"input: {training_examples[i]}, output: {np.around(prediction, 2)}")



