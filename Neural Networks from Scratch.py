# https://youtu.be/Wo5dMEP_BbI?si=0q4UUdrrfm7YoZ2o
# Neural Networks from Scratch

import sys
import numpy as np
import matplotlib

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -.8]]

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)

# weights = [[0.2, 0.8, -.5, 1],[0.5, -.91, .26, -.5],[-.26, -.27, .17, .87]]
# biases = [2, 3, .5]
#
# weights2 = [[0.1, -.14, .5],[-.5, .12, -.33],[-.44, .73, -.13]]
# biases2 = [-1, 2, -0.5]
#
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
#
# layer_ouputs = []
# # for neuron_weights, neuron_bias in zip(weights, biases):
# #     neuron_output = 0
# #     for n_input, weight in zip(inputs, neuron_weights):
#
# print(layer1_outputs)
# print(layer2_outputs)