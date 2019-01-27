# reinforcement learning neural network
# Network consists of only numeric activation states.
# Training is in a reinforcement style, just like
# the hormone regulation in brain. Actions are led
# by such kind of global adjustment which is so
# much different from BackProp theory.

# TO DO LIST:
# This training is bad, since reward or punishment is
# given when there exists no activated route!
# To add this constraint in training.


import numpy as np


ACTIVATE_LEVEL = 8
TOP_LEVEL = 120
BOTTOM_LEVEL = -120

global g_reward_score
global g_punish_score
g_punish_score = 0.2
g_reward_score = g_punish_score
boost_ratio = 0.2


# In my vision, reward score and punish score will change with
# learning progress.
def update_globals(scores):
    global g_reward_score
    global g_punish_score
    mean_score_ = np.mean(scores)
    g_reward_score = mean_score_ + boost_ratio * (1 - mean_score_)
    print('REWARD_SCORE= %.5f' % g_reward_score)
    g_punish_score = mean_score_


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x/30.0))


def activate(x):
    # print(sigmoid(x - ACTIVATE_LEVEL))
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

    def forward(self, x):
        self.layers_[0] = x
        for i in range(len(self.dims_)-1):
            self.layers_[i+1] = activate(np.dot(self.params_[i], self.layers_[i]))

    def train(self, iters, samples_x, samples_y):
        assert len(samples_x) == len(samples_y)
        global g_reward_score
        global g_punish_score
        scores_ = np.zeros([len(samples_x)], dtype=np.float32)
        for _ in range(iters):
            ids_ = np.random.permutation(len(samples_x))
            for i in ids_:
                x_, y_ = samples_x[i], samples_y[i]
                self.forward(x_)
                scores_[i] = np.mean(self.layers_[-1] == y_)
                reward_ = np.int8(scores_[i] >= g_reward_score) - np.int8(scores_[i] < g_punish_score)
                for k in range(len(self.params_)):
                    self.params_[k] += reward_ * np.outer(self.layers_[k+1], self.layers_[k])
                    self.params_[k] = np.minimum(np.maximum(self.params_[k], BOTTOM_LEVEL), TOP_LEVEL)
            update_globals(scores_)
            print('mean score: %.6f' % np.mean(scores_))


def int2bins(x):
    x = np.uint8(x)
    op = 0b10000000
    bins = np.array([0.0] * 8)
    for i in range(8):
        if op & x == op:
            bins[i]=1
        else:
            bins[i]=0
        op = op >> 1
    return bins


def concat(bins_1, bins_2):
    return np.concatenate((bins_1, bins_2), axis=0)


def observe(size):
    x = np.random.randint(0, 256,[size, 2])
    _x = np.zeros([size, 16], dtype=np.uint8)
    _y = np.zeros([size, 2], dtype=np.uint8)
    for i in range(size):
        _x[i] = concat(int2bins(x[i, 0]), int2bins(x[i, 1]))
        if x[i,0] > x[i, 1]:
            _y[i, 0] = 0
            _y[i, 1] = 1
        elif x[i, 0] <= x[i, 1]:
            _y[i, 0] = 1
            _y[i, 1] = 0
        else:
            _y[i, 0] = 1
            _y[i, 1] = 1
    return _x, _y


if __name__ == '__main__':
    nn = RLNN([16, 8, 4, 2])
    # nn.forward(np.random.randint(0, 2, [nn.dims_[0]], dtype=np.uint8))
    samples_x, samples_y = observe(10)
    nn.train(10000, samples_x, samples_y)
    print(nn.params_)
