from typing import Callable

import numpy as np


class Layer:
    def __init__(self, input_shape: int, output_shape: int, activation_function: Callable, weight_generation: Callable,
                 bias_generation: Callable, add_bias: bool, **kwargs):
        """
        Initializes a layer in the neural network.

        :param input_shape: Amount of input Neurons (amount of neurons in the previous layer).
        :param output_shape: Amount of output Neurons (amount of neurons in the current layer).
        :param activation_function: Activation function to use.
        :param add_bias: Whether to use bias.
        :param mu: Mean of the normal distribution used to initialize weights.
        :param sigma: Standard deviation of the normal distribution used to initialize weights.
        """

        self.activation_function = activation_function
        self.add_bias = add_bias

        self.z = np.ndarray  # Sum of weighted activations.
        self.a = np.ndarray  # Activation function applied to z.
        self.weights = weight_generation(output_shape, input_shape, **kwargs)
        # self.weights = np.random.normal(mu, sigma, size=(output_shape, input_shape))
        # self.weights = np.zeros((output_shape, input_shape))
        self.bias = 0
        if self.add_bias:
            self.bias = bias_generation(output_shape, input_shape, **kwargs)
            self.bias = np.zeros((output_shape, 1))
            # self.bias = np.random.normal(mu, sigma, size=(output_shape, 1))

    def step(self, inputs: np.ndarray) -> np.ndarray:
        self.z = np.dot(inputs, self.weights.T)
        if self.add_bias:
            self.z += self.bias.T

        self.a = self.activation_function(self.z)
        return self.a


class ANN:
    def __init__(self, input_num: int, hidden_layers: list[int], output_num: int,
                 activation_functions: list[Callable] | Callable, cost_function: Callable, weight_generation: Callable,
                 bias_generation: Callable, add_bias: bool = True, **kwargs):
        """
        Initializes the network.

        :param input_num: Amount of input Neurons.
        :param hidden_layers: Amount of hidden layers and Neurons per layer.
        :param output_num: Amount of output Neurons.
        :param activation_functions: Activation function(s) to use.
        :param cost_function: Cost function to use.
        :param weight_generation: Function to generate weights.
        :param bias_generation: Function to generate biases.
        :param add_bias: Whether to use bias.
        :param mu: Mean of the normal distribution used to initialize weights.
        :param sigma: Standard deviation of the normal distribution used to initialize weights.
        """

        self.shape = [input_num] + hidden_layers + [output_num]
        self.add_bias = add_bias
        self.cost_function = cost_function
        self.network = []

        if type(activation_functions) is not list:
            activation_functions = [activation_functions for _ in range(len(self.shape) - 1)]
        elif len(activation_functions) != len(self.shape) - 1:
            raise ValueError("The number of activation functions must be equal to the number of layers - 1")

        # self.network.append(Layer(self.shape[0], self.shape[1], activation_functions[0], False, mu, sigma))
        self.network.append(Layer(self.shape[0], self.shape[1], activation_functions[0], weight_generation, bias_generation, False, **kwargs))
        self.network.extend(
            [Layer(self.shape[i + 1], self.shape[i + 2], activation_functions[i], weight_generation, bias_generation, self.add_bias, **kwargs) for i in
             range(len(self.shape[1:]) - 1)])

    def feed_forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Feeds the input data through the network

        :param input_data: An array of input data
        :return: The output of the network
        """

        activation = input_data

        for layer in self.network:
            activation = layer.step(activation)

        return activation

    def backpropagate(self, x: np.ndarray, y_true: np.ndarray):
        deltas = [0 for _ in range(len(self.network))]
        weight_gradients = [0 for _ in range(len(self.network))]
        bias_gradients = [0 for _ in range(len(self.network) - 1)]

        deltas[-1] = np.multiply(self.cost_function(self.network[-1].a, y_true, derivative=True),
            self.network[-1].activation_function(self.network[-1].z, derivative=True))

        for i in reversed(range(len(self.network[:-1]))):
            deltas[i] = np.multiply(np.dot(deltas[i + 1], self.network[i + 1].weights),
                self.network[i].activation_function(self.network[i].z, derivative=True))

        for i in range(len(deltas)):
            if i == 0:
                weight_gradients[i] = np.dot(x.T, deltas[i])
            else:
                weight_gradients[i] = np.dot(self.network[i - 1].a.T, deltas[i])

        if self.add_bias:
            for i in range(len(deltas[1:])):
                bias_gradients[i] = deltas[i + 1]

        return weight_gradients, bias_gradients

    def train(self, x, y, epochs, learning_rate):
        """
        Train the neural network using backpropagation with mini-batch training.

        :param x: A list of (input, target) pairs for training.
        :param epochs: The number of training epochs.
        :param batch_size: The size of each mini-batch.
        """

        for epoch in range(epochs):
            randomize = np.arange(len(x))
            np.random.shuffle(randomize)  # Shuffle the training data for each epoch.
            x = x[randomize]
            y = y[randomize]

            for i in range(0, len(x)):
                batch_x = x[i:i + 1]
                batch_y = y[i:i + 1]

                self.feed_forward(batch_x)

                weight_gradients, bias_gradients = self.backpropagate(batch_x, batch_y)

                for j in range(len(self.network)):
                    self.network[j].weights -= learning_rate * weight_gradients[j].T

                if self.add_bias:
                    for j in range(len(self.network[1:])):
                        self.network[j + 1].bias -= learning_rate * bias_gradients[j].T
