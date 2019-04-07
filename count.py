# filename: macro2micro_image2class.py

import numpy as np
import tensorflow as tf
import struct
import os
from PIL import Image


def conv2d(input_, ksize, stride, out_channels, scope):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        w_ = tf.get_variable(
            name='w',
            shape=[ksize, ksize, input_.shape.as_list()[-1], out_channels],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(0.02))
        b_ = tf.get_variable(
            name='b',
            shape=[out_channels],
            dtype=tf.float32,
            initializer=tf.initializers.constant(0.0))
        return tf.nn.leaky_relu(
            tf.nn.conv2d(
                input_,
                w_,
                (1, stride, stride, 1),
                padding='SAME')
            + b_)


def deconv2d(input_, ksize, stride, out_channels, scope):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        n = input_.shape.as_list()[0]
        h = input_.shape.as_list()[1]
        w = input_.shape.as_list()[2]
        c = input_.shape.as_list()[3]
        filters = tf.get_variable(
            name='w',
            shape=[ksize, ksize, out_channels, c],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(0.02))
        biases = tf.get_variable(
            name='b',
            shape=[out_channels],
            dtype=tf.float32,
            initializer=tf.initializers.constant(0.0))
        h = int(h * stride)
        w = int(w * stride)
        return tf.nn.leaky_relu(
            tf.nn.conv2d_transpose(
                input_,
                filters,
                [n, h, w, out_channels],
                [1, stride, stride, 1],
                'SAME') + biases)


def fully_connect(input_, units, scope):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        w_ = tf.get_variable(
            name='w',
            shape=[input_.shape.as_list()[-1], units],
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(0.02))
        b_ = tf.get_variable(
            name='b',
            shape=[units],
            dtype=tf.float32,
            initializer=tf.initializers.constant(0.0))
        return tf.nn.leaky_relu(tf.matmul(input_, w_) + b_)


def random_down_sample(x, rate):
    assert len(x.shape)==4
    # G
    pass


def random_up_sample(x, rate):
    # Gaussian distribution sampling
    pass


def encoder(input_, scope_):
    out_ = conv2d(input_, 3, 2, 8, scope_ + '/encoder/conv1')
    out_ = conv2d(out_, 3, 1, 4, scope_ + '/encoder/conv2')
    out_ = conv2d(out_, 3, 2, 8, scope_ + '/encoder/conv3')
    out_ = conv2d(out_, 3, 1, 4, scope_ + '/encoder/conv4')
    channels = out_.shape[1] * out_.shape[2] * out_.shape[3]
    out_ = tf.reshape(out_, [out_.shape[0], channels])
    out_ = fully_connect(out_, 4, scope_ + '/encoder/fc1')
    out_ = fully_connect(out_, 4, scope_ + '/encoder/fc2')
    return out_


def decoder(input_, scope_):
    out_ = fully_connect(input_, 8, scope_ + '/decoder/fc1')
    batch_ = out_.shape.as_list()[0]
    h_ = 7
    w_ = 7
    c_ = 4
    channels = int(h_ * w_ * c_)
    out_ = fully_connect(out_, channels, scope_ + '/decoder/fc2')
    out_ = tf.reshape(out_, [batch_, h_, w_, c_])
    out_ = deconv2d(out_, 3, 2, 4, scope_ + '/decoder/deconv1')
    out_ = deconv2d(out_, 3, 1, 4, scope_ + '/decoder/deconv2')
    out_ = deconv2d(out_, 3, 2, 4, scope_ + '/decoder/deconv3')
    out_ = deconv2d(out_, 3, 1, 1, scope_ + '/decoder/deconv4')
    return out_


def read_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>4I', f.read(16))
        assert rows == 28
        assert cols == 28
        return np.float32(np.fromfile(f, dtype=np.uint8)).reshape(num, rows, cols)/255.0


def read_labels(path):
    with open(path, 'rb') as f:
        _, num = struct.unpack('>2I', f.read(8))
        labels = np.zeros([num, 10])
        ids = np.fromfile(f, dtype=np.uint8)
        for i in range(num):
            labels[i, ids[i]] = 1.0
        return labels


def gray2rgb(gray):
    return np.stack([gray, gray, gray], axis=-1)


if __name__ == '__main__':
    # load the dataset
    data_dir = 'E:/CodeHub/Datasets/MNIST'
    train_image_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    ims = read_images(train_image_path)
    lbs = read_labels(train_label_path)

    # build the network
    t_in_ = tf.placeholder(dtype=tf.float32, shape=[1, ims.shape[1], ims.shape[2], 1])
    # build encoder
    t_features = encoder(t_in_, scope_='mnist')
    print(t_features.shape.as_list())
    t_in_rec = decoder(t_features, scope_='mnist')
    t_loss = tf.reduce_mean(tf.abs(t_in_ - t_in_rec))
    t_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(t_loss)

    # train the model with data
    repeats = 5000000
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    loss_av_ = np.zeros([500], dtype=np.float32)

    for i in range(repeats):
        id_ = np.random.randint(len(ims))
        loss_, _, out_ = sess.run([t_loss, t_opt, t_in_rec], feed_dict={
            t_in_: np.expand_dims(np.expand_dims(ims[id_], axis=0), axis=-1)
        })
        loss_av_[i % len(loss_av_)] = loss_

        if i % 1000 == 0:
            print('#%d\t loss: %f' % (i, loss_av_[np.where(loss_av_)].mean()))
            # save the reconstructed image
            out_ = np.maximum(np.minimum(out_, 1.0), 0.0)
            Image.fromarray(gray2rgb(np.uint8(out_[0, :, :, 0] * 255))).save('rec_%d.jpg' % i)

    # save the model
    saver.save(sess, './model_mnist.ckpt')

    # stage II: enhance autoencoder with self produced input
    # stage III: recursively repeat such training until AE converged.
