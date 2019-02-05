# build_concept.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Ideas are :
# observations => encoder => self organinization + decoder

bias = 0.5


def f(x):
    return np.minimum(np.maximum(x, 0), 1)
    # return np.maximum(x, 0)


def rand_trans_matrix(n):
    conns_ = 2*(np.random.rand(n, n) - 0.5)
    diag_ = 1 - np.diag([1]*n)
    return conns_ * diag_


def rand_init_states(n):
    return np.random.rand(n)


def trans(x, x0, w):
    return (x0 - bias) + w.dot(f(x))/len(x)


def sparsity(x):
    assert len(x.shape)==1
    return np.sum(x == 0)/x.shape[0]


if __name__ == '__main__':
    N = 32
    repeats = 1000
    steps = 5
    errs_ = np.zeros([repeats, steps])
    spa_inc_ = np.zeros([repeats])
    for k in range(repeats):
        conns = rand_trans_matrix(N)
        inits = rand_init_states(N)
        state = inits - bias
        last_state = np.copy(state)
        for i in range(steps):
            last_state[:] = state[:]
            state = trans(last_state, inits, conns)
            errs_[k, i] = np.sum(np.abs(state - last_state))/N
        plt.plot(errs_[k, :])
        spa_inc_[k] = sparsity(f(state)) - sparsity(f(inits-bias))
    fig_spa = plt.figure()
    plt.plot(spa_inc_)
    plt.show()
    print(spa_inc_.mean())
