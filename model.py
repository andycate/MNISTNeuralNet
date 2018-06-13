import numpy as np
from enum import Enum
import math

class InitType(Enum):
    NORMAL = 1
    ZERO = 2
    RANDOM = 3
    EPSILON = 4

class Model:
    def __init__(self, learning_rate, weight_decay_rate, init_type=InitType.RANDOM, epsilon=0.0001):
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        if init_type == InitType.NORMAL:
            self.theta_0 = np.random.normal(0, math.pow(epsilon, 2), (784, 256))
            self.theta_1 = np.random.normal(0, math.pow(epsilon, 2), (256, 10))
            self.bias_0 = np.random.normal(0, math.pow(epsilon, 2), (256))
            self.bias_1 = np.random.normal(0, math.pow(epsilon, 2), (10))
        elif init_type == InitType.ZERO:
            self.theta_0 = np.zeros((784, 256), dtype=np.float32)
            self.theta_1 = np.zeros((256, 10), dtype=np.float32)
            self.bias_0 = np.zeros((256), dtype=np.float32)
            self.bias_1 = np.zeros((10), dtype=np.float32)
        elif init_type == InitType.RANDOM:
            self.theta_0 = np.random.rand(784, 256) * epsilon
            self.theta_1 = np.random.rand(256, 10) * epsilon
            self.bias_0 = np.random.rand(256) * epsilon
            self.bias_1 = np.random.rand(10) * epsilon
        elif init_type == InitType.EPSILON:
            self.theta_0 = np.ones((784, 256), dtype=np.float32) * epsilon
            self.theta_1 = np.ones((256, 10), dtype=np.float32) * epsilon
            self.bias_0 = np.ones((256), dtype=np.float32) * epsilon
            self.bias_1 = np.ones((10), dtype=np.float32) * epsilon

    def __sigmoid(self, out):
        return 1 / (1 + np.exp(-out))

    def __softmax(self, out):
        raw = np.exp(out)
        return raw / np.repeat(np.sum(raw, axis=0).reshape(1, -1), 10, axis=0)

    def __loss(self, predictions, lbls_logits):
        return -np.sum(np.log(predictions) * lbls_logits)

    def minimize(self, imgs, lbls_logits):
        # run a forward network pass
        sum_0 = (imgs.transpose().dot(self.theta_0) + np.repeat(self.bias_0.reshape(1, -1), imgs.shape[1], axis=0)).transpose()
        activation_0 = self.__sigmoid(sum_0) # output from hidden layer, for each image (256, <# images>)
        sum_1 = (activation_0.transpose().dot(self.theta_1) + np.repeat(self.bias_1.reshape(1, -1), activation_0.shape[1], axis=0)).transpose()
        activation_1 = self.__softmax(sum_1) # predictions (10, <# images>)

        cross_entropy_1 = -np.sum(lbls_logits - activation_1, axis=1).reshape(-1, 1) # total cross entropy across all images
        cross_entropy_0 = self.theta_1.dot(cross_entropy_1) * np.sum(activation_0 * (1 - activation_0), axis=1).reshape(-1, 1)

        theta_gradient_1 = np.repeat(cross_entropy_1.transpose(), self.theta_1.shape[0], axis=0) * np.repeat(np.sum(activation_0, axis=1).reshape(-1, 1), self.theta_1.shape[1], axis=1)
        bias_gradient_1 = cross_entropy_1.reshape(-1)
        theta_gradient_0 = np.repeat(cross_entropy_0.transpose(), self.theta_0.shape[0], axis=0) * np.repeat(np.sum(imgs, axis=1).reshape(-1, 1), self.theta_0.shape[1], axis=1)
        bias_gradient_0 = cross_entropy_0.reshape(-1)

        self.theta_1 -= theta_gradient_1
        self.bias_1 -= bias_gradient_1
        self.theta_0 -= theta_gradient_0
        self.bias_0 -= bias_gradient_0

    def predict(self, imgs):
        sum_0 = (imgs.transpose().dot(self.theta_0) + np.repeat(self.bias_0.reshape(1, -1), imgs.shape[1], axis=0)).transpose()
        activation_0 = self.__sigmoid(sum_0)
        sum_1 = (activation_0.transpose().dot(self.theta_1) + np.repeat(self.bias_1.reshape(1, -1), activation_0.shape[1], axis=0)).transpose()
        activation_1 = self.__softmax(sum_1)
        return activation_1