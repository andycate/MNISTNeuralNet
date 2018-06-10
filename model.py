import numpy as np
from enum import Enum

class InitType(Enum):
    NORMAL = 1
    ZERO = 2
    RANDOM = 3
    EPSILON = 4

class Model:
    def __init__(self, learning_rate, init_type=InitType.RANDOM, epsilon=0.01):
        self.learning_rate = learning_rate
        if init_type == InitType.NORMAL:
            self.theta_0 = np.random.normal(0, epsilon, (256, 784))
            self.theta_1 = np.random.normal(0, epsilon, (10, 784))
        elif init_type == InitType.ZERO:
            self.theta_0 = np.zeros((10, 785), dtype=np.float32)
            self.theta_1 = np.zeros((10, 785), dtype=np.float32)
        elif init_type == InitType.RANDOM:
            self.theta_0 = np.random.rand(10, 785) * epsilon
            self.theta_1 = np.random.rand(10, 785) * epsilon
        elif init_type == InitType.EPSILON:
            self.theta_0 = np.ones((10, 785), dtype=np.float32) * epsilon
            self.theta_1 = np.ones((10, 785), dtype=np.float32) * epsilon
