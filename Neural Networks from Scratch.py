# https://youtu.be/Wo5dMEP_BbI?si=0q4UUdrrfm7YoZ2o
# Neural Networks from Scratch

import sys

import nnfs
import numpy as np
from nnfs.datasets import spiral_data
import matplotlib

nnfs.init()

# 퍼셉트론 연산을 수행하는 부분 input*weight + bias
class Layer_Dense:

    # 입력값, 뉴런 수를 순서대로 입력받고
    # 그에 대해 무작위 가중치 벡터 생성 + scaling(0.1)
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # 위에서 생성한 weight와 bias를 기반으로 퍼셉트론을 수행해 ouput 추출
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activation Function
class Activation_ReLU:
    # 0과 비교해서 0보다 큰 값만 그대로 출력, 0보다 작으면 0
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

# input, hidden layer
# input, weight, bias를 정한 후 activation 함수 실행
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# hidden, output layer
# 실행한 activation 함수를 정규화
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# layer1.forward(X)
# print(layer1.output)

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