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
        self.z = np.ndarray  # Sum of weighted activations.
        self.a = np.ndarray  # Activation function applied to z.

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
                 activation_functions: list[Callable] | Callable, cost_function: Callable, bias: bool = True,
                 mu: float = 0, sigma: float = 0.1):
        """
        Initializes the network.

        :param input_num: Amount of input Neurons.
        :param hidden_layers: Amount of hidden layers and Neurons per layer.
        :param output_num: Amount of output Neurons.
        :param activation_functions: Activation function(s) to use.
        :param cost_function: Cost function to use.
        :param bias: Whether to use bias.
        :param mu: Mean of the normal distribution used to initialize weights.
        :param sigma: Standard deviation of the normal distribution used to initialize weights.
        """

        self.shape = [input_num] + hidden_layers + [output_num]
        self.bias = bias
        self.cost_function = cost_function

        if type(activation_functions) is not list:
            activation_functions = [activation_functions for _ in range(len(self.shape) - 1)]
        elif len(activation_functions) != len(self.shape) - 1:
            raise ValueError("The number of activation functions must be equal to the number of layers - 1")

        self.network = [Layer(self.shape[i], self.shape[i + 1], activation_functions[i], self.bias, mu, sigma) for i in
                        range(len(self.shape) - 1)]

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

    def backpropagate(self, x: np.ndarray, y_true: np.ndarray):
        delta = self.cost_function(self.network[-1].a, y_true, derivative=True)
        delta = delta * self.network[-1].activation_function(self.network[-1].z, derivative=True)
        gradients_list = [0 for _ in range(len(self.network))]

        gradients_list[-1] = np.dot(delta.T, self.network[-2].a)

        for i in range(len(self.network[:-2]), -1, -1):
            delta = np.dot(delta, self.network[i + 1].weights) * self.network[i].activation_function(self.network[i].z,
                                                                                                     derivative=True)

            if i == 0:
                gradient = np.dot(delta.T, x)
            else:
                gradient = np.dot(delta.T, self.network[i - 1].a)

            gradients_list[i] = gradient

        return gradients_list

    def train(self, x, y, epochs, learning_rate, batch_size):
        """
        Train the neural network using backpropagation with mini-batch training.

        :param x: A list of (input, target) pairs for training
        :param epochs: The number of training epochs
        :param learning_rate: The learning rate for weight updates
        :param batch_size: The size of each mini-batch
        """

        for epoch in range(epochs):
            np.random.shuffle(x)  # Shuffle the training data for each epoch

            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                self.feed_forward(batch_x)

                gradients_list = self.backpropagate(batch_x, batch_y)

                # Update weights with accumulated gradients
                for j in range(len(self.network)):
                    self.network[j].weights -= learning_rate * gradients_list[j]
