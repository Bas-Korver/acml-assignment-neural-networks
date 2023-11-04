from typing import Callable

import numpy as np


class Layer:
    def __init__(self, input_num: int, output_shape: int, activation_function: Callable, bias: bool, mu: float,
                 sigma: float):
        """
        Initializes a layer in the neural network.

        :param input_num: Amount of input Neurons (amount of neurons in the previous layer).
        :param output_shape: Amount of output Neurons (amount of neurons in the current layer).
        :param activation_function: Activation function to use.
        :param bias: Whether to use bias.
        :param mu: Mean of the normal distribution used to initialize weights.
        :param sigma: Standard deviation of the normal distribution used to initialize weights.
        """

        self.activation_function = activation_function
        self.z = 0  # Sum of weighted activations.
        self.a = 0  # Activation function applied to z.

        if bias:
            self.weights = np.random.normal(mu, sigma, size=(output_shape, input_num + 1))
        else:
            self.weights = np.random.normal(mu, sigma, size=(output_shape, input_num))

    def step(self, inputs: np.ndarray) -> np.ndarray:
        self.z = np.dot(inputs, self.weights.T)
        self.a = self.activation_function(self.z)
        return self.a


class ANN:
    def __init__(self, input_num: int, hidden_layers: list[int], output_num: int,
                 activation_functions: list[Callable] | Callable, bias: bool = True, mu: float = 0, sigma: float = 0.1):
        """
        Initializes the network.

        :param input_num: Amount of input Neurons.
        :param hidden_layers: Amount of hidden layers and Neurons per layer.
        :param output_num: Amount of output Neurons.
        :param activation_functions: Activation function(s) to use.
        :param bias: Whether to use bias.
        :param mu: Mean of the normal distribution used to initialize weights.
        :param sigma: Standard deviation of the normal distribution used to initialize weights.
        """

        self.shape = [input_num] + hidden_layers + [output_num]
        self.bias = bias

        if type(activation_functions) is not list:
            activation_functions = [activation_functions for _ in range(len(self.shape) - 1)]
        elif len(activation_functions) != len(self.shape) - 1:
            raise ValueError("The number of activation functions must be equal to the number of layers - 1")

        self.network = [
            Layer(self.shape[i], self.shape[i + 1], activation_functions[i], self.bias, mu, sigma) for
            i in range(len(self.shape) - 1)]

    def feed_forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Feeds the input data through the network

        :param input_data: An array of input data
        :return: The output of the network
        """

        activation = input_data

        for layer in self.network:
            if self.bias:
                activation = np.concatenate((np.ones((activation.shape[0], 1)), activation), axis=1)
            activation = layer.step(activation)

        return activation
