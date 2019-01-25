# reinforcement learning neural network
# Network consists of only numeric activation states.
# Training is in a reinforcement style, just like
# the hormone regulation in brain. Actions are led
# by such kind of global adjustment which is so
# much different from BackProp theory.

import numpy as np


ACTIVATE_LEVEL = 1


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def activate(x):
    print(sigmoid(x - ACTIVATE_LEVEL))
    return np.uint32(np.random.rand(len(x)) < sigmoid(x - ACTIVATE_LEVEL))


# The layer activation states are of binary type.
# The connection states are of integer type(+/-127).
class RLNN(object):
    def __init__(self, dims):
        self.layers_ = []
        self.params_ = []
        self.dims_ = dims
        for i in range(len(dims)):
            self.layers_.append(np.zeros([dims[i]], dtype=np.uint8))
            if i > 0:
                self.params_.append(np.zeros([dims[i], dims[i-1]], dtype=np.int8))
        print('Building model done.')

    def forward(self):
        for i in range(len(self.dims_)-1):
            self.layers_[i+1] = activate(np.dot(self.params_[i], self.layers_[i]))


if __name__ == '__main__':
    nn = RLNN([16, 8, 4, 2])
    nn.layers_[0] = np.random.randint(0, 2, [nn.dims_[0]], dtype=np.uint8)
    nn.forward()
    print(nn.layers_[0])
    print(nn.layers_[1])
    print(nn.layers_[2])
    print(nn.layers_[3])
