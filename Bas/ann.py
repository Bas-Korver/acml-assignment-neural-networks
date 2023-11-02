from typing import Callable

import numpy as np


class ANN:
    def __init__(self, input_num: int, hidden_layers: list[int], output_num: int,
                 activation_function: list[Callable] | Callable, bias: bool = True, mu: float = 0, sigma: float = 0.1):
        self.input_num = input_num
        self.hidden_layers = np.array(hidden_layers)
        self.output_num = output_num
        self.layers = np.concatenate((np.array([self.input_num]), self.hidden_layers, np.array([self.output_num])))
        self.bias = bias
        self.mu = mu
        self.sigma = sigma
        self.activation_function = activation_function

        if type(activation_function) is not list:
            self.activation_function = [activation_function for _ in range(len(self.layers) - 1)]
        elif len(activation_function) != len(self.layers) - 1:
            raise ValueError("The number of activation functions must be equal to the number of layers - 1")

        if bias:
            self.network = [np.random.normal(self.mu, self.sigma, size=(self.layers[i + 1], self.layers[i] + 1)) for i
                            in range(len(self.layers) - 1)]
        else:
            self.network = [np.random.normal(self.mu, self.sigma, size=(self.layers[i + 1], self.layers[i])) for i in
                            range(len(self.layers) - 1)]

    def feed_forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Feeds the input data through the network

        :param input_data: An array of input data
        :return: The output of the network
        """

        output = input_data

        for i in range(len(self.network)):
            if self.bias:
                output = np.concatenate((output, np.ones((output.shape[0], 1))), axis=1)

            output = self.activation_function[i](np.dot(output, self.network[i].T))

        return output
