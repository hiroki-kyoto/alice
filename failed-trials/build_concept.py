# build_concept.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from struct import unpack
from PIL import Image
import gzip

# Ideas are :
# observations => encoder => self organization + decoder

bias = 0.1


def f(x):
    # return np.minimum(np.maximum(x, 0), 1)
    return np.maximum(x, 0)


def tf_act(t_x):
    #return tf.minimum(tf.maximum(t_x, 0), 1)
    return tf.maximum(t_x, 0)


def rand_trans_matrix(n):
    conns_ = 2*(np.random.rand(n, n) - 0.5)
    diag_ = 1 - np.diag([1]*n)
    return conns_ * diag_


def tf_init_trans_weights(n):
    return tf.constant(rand_trans_matrix(n), dtype=tf.float32)


def rand_init_states(n):
    return np.random.rand(n)


def trans(x, x0, w):
    return (x0 - bias) + w.dot(f(x))/len(x)


def tf_trans(t_x, t_x0, t_w):
    assert len(t_x.shape) == 2
    # return (t_x0 - bias) + tf.matmul(t_w, tf_act(t_x))/t_x.shape.as_list()[0]
    return (t_x0 - bias) + tf.matmul(t_w, tf_act(t_x))


def tf_stablize(t_x, t_w, depth):
    t_x0 = t_x
    for _ in range(depth):
        t_x = tf_trans(t_x, t_x0, t_w)
    return tf_act(t_x)


def sparsity(x):
    assert len(x.shape)==1
    return np.sum(x==0)/x.shape[0]


def test_steady_field():
    N = 16
    repeats = 1000
    steps = 4
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


def read_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        assert rows == 28
        assert cols == 28
        return np.float32(np.fromfile(f, dtype=np.uint8)).reshape(num, rows, cols)/255.0


def read_labels(path):
    with open(path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.zeros([num, 10])
        ids = np.fromfile(f, dtype=np.uint8)
        for i in range(num):
            labels[i, ids[i]] = 1.0
        return labels


if __name__ == '__main__':
    # load the dataset
    data_dir = 'E:/CodeHub/Datasets/MNIST'
    train_image_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    ims = read_images(train_image_path)
    lbs = read_labels(train_label_path)
    # build the network
    dims = [ims.shape[1]*ims.shape[2], 16, lbs.shape[1]]
    steps = 4
    layers = []

    t_input = tf.placeholder(dtype=tf.float32, shape=[dims[0], 1])
    layers.append(t_input)
    t_w1 = tf.get_variable(
        name='w1',
        shape=[dims[1], dims[0]],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.1))
    layers.append(tf.matmul(t_w1, layers[-1]))
    t_w2 = tf.get_variable(
        name='w2',
        dtype=tf.float32,
        initializer=tf_init_trans_weights(dims[1]))
    layers.append(tf_stablize(layers[-1], t_w2, steps))
    t_w3 = tf.get_variable(
        name='w3',
        shape=[dims[2], dims[1]],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.1))
    layers.append(tf.matmul(t_w3, layers[-1]))
    t_w4 = tf.get_variable(
        name='w4',
        dtype=tf.float32,
        initializer=tf_init_trans_weights(dims[2]))
    layers.append(tf_stablize(layers[-1], t_w4, steps))
    t_feedback = tf.placeholder(dtype=tf.float32, shape=[dims[2]])
    t_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=t_feedback,
        logits=tf.reshape(layers[-1], [dims[-1]]))
    t_w_reg = tf.reduce_mean(tf.maximum(tf.abs(t_w1) - 1.0, 0)) + \
              tf.reduce_mean(tf.maximum(tf.abs(t_w2) - 1.0, 0)) + \
              tf.reduce_mean(tf.maximum(tf.abs(t_w3) - 1.0, 0)) + \
              tf.reduce_mean(tf.maximum(tf.abs(t_w4) - 1.0, 0))
    t_out_reg = tf.reduce_mean(tf.maximum(tf.abs(layers[1]) - 1.0, 0)) + \
                tf.reduce_mean(tf.maximum(tf.abs(layers[2]) - 1.0, 0)) + \
                tf.reduce_mean(tf.maximum(tf.abs(layers[3]) - 1.0, 0)) + \
                tf.reduce_mean(tf.maximum(tf.abs(layers[4]) - 1.0, 0))

    t_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(t_loss + t_w_reg + t_out_reg)

    # train the model with data
    repeats = 600000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    loss_av_ = np.zeros([500])
    for i in range(repeats):
        id_ = np.random.randint(len(ims))
        loss_, w_reg_, out_reg_, _ = sess.run([t_loss, t_w_reg, t_out_reg, t_opt], feed_dict={
            t_input: np.reshape(ims[id_], [dims[0], 1]),
            t_feedback: lbs[id_]
        })
        loss_av_[i%len(loss_av_)] = loss_
        print('#%d\t loss: %f\t wreg: %f\t oreg: %f' %
              (i, loss_av_[np.where(loss_av_)].mean(), w_reg_, out_reg_))
