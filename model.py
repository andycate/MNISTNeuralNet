import numpy as np
from enum import Enum
import math

class InitType(Enum):
    NORMAL = 1
    ZERO = 2
    RANDOM = 3
    EPSILON = 4

class Model:
    def __init__(self, learning_rate, weight_decay_rate, shape, init_type=InitType.RANDOM, epsilon=0.01):
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        if init_type == InitType.NORMAL:
            self.weights = ()
            self.biases = ()
            for i in range(len(shape) - 1):
                self.weights += (np.random.normal(0, math.pow(epsilon, 2), (shape[i], shape[i + 1])),)
                self.biases += (np.random.normal(0, math.pow(epsilon, 2), (shape[i + 1])),)
        elif init_type == InitType.ZERO:
            for i in range(len(shape) - 1):
                self.weights += (np.zeros((shape[i], shape[i + 1]), dtype=np.float32),)
                self.biases += (np.zeros((shape[i + 1]), dtype=np.float32),)
        elif init_type == InitType.RANDOM:
            for i in range(len(shape) - 1):
                self.weights += (np.random.rand(shape[i], shape[i + 1]) * epsilon,)
                self.biases += (np.random.rand(shape[i + 1]) * epsilon,)
        elif init_type == InitType.EPSILON:
            for i in range(len(shape) - 1):
                self.weights += (np.ones((shape[i], shape[i + 1]), dtype=np.float32) * epsilon,)
                self.biases += (np.ones((shape[i + 1]), dtype=np.float32) * epsilon,)

    def __sigmoid(self, out): # good
        return 1 / (1 + np.exp(-out))

    def __softmax(self, out): # good
        raw = np.exp(out)
        return raw / np.repeat(np.sum(raw, axis=0).reshape(1, -1), raw.shape[0], axis=0)

    def __loss(self, predictions, lbls_logits): # good
        return -np.sum(np.log(predictions) * lbls_logits)

    def __ave_loss(self, predictions, lbls_logits):
        return self.__loss(predictions, lbls_logits) / predictions.shape[1]

    def predict(self, imgs):
        z_sum = ()
        activations = (imgs,)
        for i in range(len(self.weights) - 1):
            z_sum += ((activations[-1].transpose().dot(self.weights[i]) + np.repeat(self.biases[i].reshape(1, -1), activations[-1].shape[1], axis=0)).transpose(),)
            activations += (self.__sigmoid(z_sum[-1]),)
        z_sum += ((activations[-1].transpose().dot(self.weights[-1]) + np.repeat(self.biases[-1].reshape(1, -1), activations[-1].shape[1], axis=0)).transpose(),)
        activations += (self.__softmax(z_sum[-1]),)
        return z_sum, activations

    def minimize(self, imgs, lbls_logits):
        # run a forward network pass
        sums, activations = self.predict(imgs) # works

        cross_entropy = (activations[-1] - lbls_logits,)
        for i in range(2, len(self.weights) + 1):
            cross_entropy += (self.weights[-i + 1].dot(cross_entropy[-1]) * activations[-i] * (1 - activations[-i]),)

        weight_gradients = ()
        bias_gradients = ()
        for i in range(1, len(self.weights) + 1):
            weight_gradients += ((activations[-i - 1].dot(cross_entropy[i - 1].transpose()) / imgs.shape[1]) + (self.weight_decay_rate * self.weights[-i]),)
            bias_gradients += (np.average(cross_entropy[i - 1], axis=1),)

        self.weights = list(self.weights)
        self.biases = list(self.biases)
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * weight_gradients[-i - 1])
            self.biases[i] -= (self.learning_rate * bias_gradients[-i - 1])
        self.weights = tuple(self.weights)
        self.biases = tuple(self.biases)