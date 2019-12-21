# filename: macro2micro_image2class.py

import numpy as np
import tensorflow as tf
import struct
import os
from PIL import Image
import glob


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
    out_ = fully_connect(out_, 2, scope_ + '/encoder/fc2')
    return out_


def decoder(input_, scope_, h, w):
    out_ = fully_connect(input_, 8, scope_ + '/decoder/fc1')
    batch_ = out_.shape.as_list()[0]
    h_ = h // 4
    w_ = w // 4
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


def train_mnist():
    # load the dataset
    data_dir = 'E:/CodeHub/Datasets/MNIST'
    train_image_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    ims = read_images(train_image_path)
    lbs = read_labels(train_label_path)

    ids = [-1] * 10
    i = 0
    while np.min(ids) < 0:
        id_ = np.argmax(lbs[i])
        ids[id_] = i
        i += 1
    print(ids)

    # build the network
    t_in_ = tf.placeholder(dtype=tf.float32, shape=[1, ims.shape[1], ims.shape[2], 1])
    # build encoder
    t_features = encoder(t_in_, 'mnist')
    print(t_features.shape.as_list())
    t_in_rec = decoder(t_features, 'mnist', ims.shape[1], ims.shape[2])
    t_loss = tf.reduce_mean(tf.abs(t_in_ - t_in_rec))
    t_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(t_loss)

    # train the model with data
    repeats = 1000000
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt_prefix = 'models/mnist.ckpt'
    if tf.train.checkpoint_exists(ckpt_prefix):
        saver.restore(sess, ckpt_prefix)
    else:
        sess.run(tf.global_variables_initializer())

    # test the result with latent interpolation
    # fea_0 = sess.run(t_features, feed_dict={
    #     t_in_: np.expand_dims(np.expand_dims(ims[ids[7]], axis=0), axis=-1)})
    # fea_1 = sess.run(t_features, feed_dict={
    #     t_in_: np.expand_dims(np.expand_dims(ims[ids[8]], axis=0), axis=-1)})
    # out_ = sess.run(t_in_rec, feed_dict={
    #     t_features: 0.5 * fea_0 + 0.5 * fea_1
    # })
    # out_ = np.maximum(np.minimum(out_, 1.0), 0.0)
    # out_ = out_[0, :, :, 0]
    # Image.fromarray(gray2rgb(np.uint8(out_ * 255))).show()
    # exit(0)

    loss_av_ = np.zeros([500], dtype=np.float32)
    for i in range(repeats):
        id_ = np.random.randint(len(ids))
        loss_, _, out_ = sess.run([t_loss, t_opt, t_in_rec], feed_dict={
            t_in_: np.expand_dims(np.expand_dims(ims[ids[id_]], axis=0), axis=-1)
        })
        loss_av_[i % len(loss_av_)] = loss_

        if i % 10000 == 0:
            print('#%d\t loss: %f' % (i, loss_av_[np.where(loss_av_)].mean()))
            # save the reconstructed image
            out_ = np.maximum(np.minimum(out_, 1.0), 0.0)
            Image.fromarray(gray2rgb(np.uint8(out_[0, :, :, 0] * 255))).save('shots/rec_%d.jpg' % i)

    # save the model
    saver.save(sess, ckpt_prefix)

    # stage II: enhance autoencoder with self produced input
    # stage III: recursively repeat such training until AE converged.


def train_drawing():
    # load the dataset
    data_dir = 'E:/CodeHub/Datasets/Drawing'
    train_image_path = glob.glob(data_dir + '/*.jpg')
    ims = []
    for path_ in train_image_path:
        ims.append(np.array(Image.open(path_), dtype=np.float32)[:, :, 0] / 255.0)
    ims = np.stack(ims, axis=0)

    # build the network
    t_in_ = tf.placeholder(dtype=tf.float32, shape=[1, ims.shape[1], ims.shape[2], 1])
    # build encoder
    t_features = encoder(t_in_, 'drawing')
    print(t_features.shape.as_list())
    t_in_rec = decoder(t_features, 'drawing', ims.shape[1], ims.shape[2])
    t_loss = tf.reduce_mean(tf.abs(t_in_ - t_in_rec))
    t_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(t_loss)

    # train the model with data
    repeats = 500000
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt_prefix = 'models/drawing.ckpt'
    if tf.train.checkpoint_exists(ckpt_prefix):
        saver.restore(sess, ckpt_prefix)
    else:
        sess.run(tf.global_variables_initializer())

    # test the result with latent interpolation
    # out_ = sess.run(t_in_rec, feed_dict={
    #     t_in_: np.expand_dims(np.expand_dims(ims[ids[7]], axis=0), axis=-1)
    # })
    # out_ = np.maximum(np.minimum(out_, 1.0), 0.0)
    # out_ = out_[0, :, :, 0]
    # Image.fromarray(gray2rgb(np.uint8(out_ * 255))).show()
    # exit(0)

    loss_av_ = np.zeros([500], dtype=np.float32)
    for i in range(repeats):
        id_ = np.random.randint(len(ims))
        loss_, _, out_ = sess.run([t_loss, t_opt, t_in_rec], feed_dict={
            t_in_: np.expand_dims(np.expand_dims(ims[id_], axis=0), axis=-1)
        })
        loss_av_[i % len(loss_av_)] = loss_

        if i % 10000 == 0:
            print('#%d\t loss: %f' % (i, loss_av_[np.where(loss_av_)].mean()))
            # save the reconstructed image
            out_ = np.maximum(np.minimum(out_, 1.0), 0.0)
            Image.fromarray(
                gray2rgb(
                    np.uint8(out_[0, :, :, 0] * 255)
                )
            ).save('shots/rec_%d.jpg' % i)

    # save the model
    saver.save(sess, ckpt_prefix)


def connect_encoders(fea_a, fea_b):
    out_a2b = fully_connect(fea_a, fea_a.shape[1] * 2, 'connector/a2b/fc1')
    out_a2b = fully_connect(out_a2b, fea_a.shape[1] * 4, 'connector/a2b/fc2')
    out_a2b = fully_connect(out_a2b, fea_a.shape[1] * 2, 'connector/a2b/fc3')
    out_a2b = fully_connect(out_a2b, fea_b.shape[1], 'connector/a2b/fc4')

    out_b2a = fully_connect(fea_b, fea_b.shape[1] * 2, 'connector/b2a/fc1')
    out_b2a = fully_connect(out_b2a, fea_b.shape[1] * 4, 'connector/b2a/fc2')
    out_b2a = fully_connect(out_b2a, fea_b.shape[1] * 2, 'connector/b2a/fc3')
    out_b2a = fully_connect(out_b2a, fea_a.shape[1], 'connector/b2a/fc4')
    return out_a2b, out_b2a


def initialize_unset_vars(sess):
    global_vars = tf.global_variables()
    bool_inits = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    uninit_vars = [v for (v, b) in zip(global_vars, bool_inits) if not b]
    for v in uninit_vars:
        print(str(v.name))
    if len(uninit_vars):
        sess.run(tf.variables_initializer(uninit_vars))


def train_connector():
    # mnist dataset
    data_dir = 'E:/CodeHub/Datasets/MNIST'
    train_image_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    ims_mnist = read_images(train_image_path)
    lbs_mnist = read_labels(train_label_path)
    ids_mnist = [-1] * 10
    i = 0
    while np.min(ids_mnist) < 0:
        id_ = np.argmax(lbs_mnist[i])
        ids_mnist[id_] = i
        i += 1

    # drawing dataset
    data_dir = 'E:/CodeHub/Datasets/Drawing'
    train_image_path = glob.glob(data_dir + '/*.jpg')
    ims_drawing = []
    for path_ in train_image_path:
        ims_drawing.append(np.array(Image.open(path_), dtype=np.float32)[:, :, 0] / 255.0)
    ims_drawing = np.stack(ims_drawing, axis=0)

    t_in_mnist = tf.placeholder(
        dtype=tf.float32,
        shape=[1, ims_mnist.shape[1], ims_mnist.shape[2], 1])
    t_fea_mnist = encoder(t_in_mnist, 'mnist')

    t_in_drawing = tf.placeholder(
        dtype=tf.float32,
        shape=[1, ims_drawing.shape[1], ims_drawing.shape[2], 1])
    t_fea_drawing = encoder(t_in_drawing, 'drawing')

    out_a2b, out_b2a = connect_encoders(t_fea_mnist, t_fea_drawing)

    t_out_mnist = decoder(
        out_b2a,
        'mnist',
        ims_mnist.shape[1],
        ims_mnist.shape[2])

    t_out_drawing = decoder(
        out_a2b,
        'drawing',
        ims_drawing.shape[1],
        ims_drawing.shape[2])

    loss_a2b = tf.reduce_mean(tf.abs(out_a2b - t_fea_drawing))
    loss_b2a = tf.reduce_mean(tf.abs(out_b2a - t_fea_mnist))
    loss_overall = loss_a2b + loss_b2a

    all_vars = tf.trainable_variables()
    train_vars = []
    for v in all_vars:
        if v.name.startswith('connector'):
            train_vars.append(v)
            print(v.name)

    minimizer_ = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        loss_overall,
        var_list=train_vars
    )

    repeats = 500000
    sess = tf.Session()
    ckpt_prefix = 'models/connected.ckpt'
    if tf.train.checkpoint_exists(ckpt_prefix):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, ckpt_prefix)
    else:
        # initialize the mnist model
        vars_mnist = dict()
        for v in all_vars:
            if v.name.startswith('mnist'):
                vars_mnist[v.name[:-2]] = v
        saver = tf.train.Saver(var_list=vars_mnist)
        saver.restore(sess, 'models/mnist.ckpt')
        # initialize the drawing model
        vars_drawing = dict()
        for v in all_vars:
            if v.name.startswith('drawing'):
                vars_drawing[v.name[:-2]] = v
        saver = tf.train.Saver(var_list=vars_drawing)
        saver.restore(sess, 'models/drawing.ckpt')
        initialize_unset_vars(sess)

    loss_av_ = np.zeros([500], dtype=np.float32)
    for i in range(repeats):
        id_ = np.random.randint(len(ids_mnist))
        loss_, _, out_mnist, out_drawing = sess.run([
            loss_overall,
            minimizer_,
            t_out_mnist,
            t_out_drawing
        ], feed_dict={
            t_in_mnist: np.expand_dims(
                np.expand_dims(
                    ims_mnist[ids_mnist[id_]],
                    axis=0),
                axis=-1),
            t_in_drawing: np.expand_dims(
                np.expand_dims(
                    ims_drawing[id_],
                    axis=0),
                axis=-1)
        })
        loss_av_[i % len(loss_av_)] = loss_

        if i % 10000 == 0:
            print('#%d\t loss: %f' % (i, loss_av_[np.where(loss_av_)].mean()))
            # save the reconstructed image
            out_mnist_sup = np.zeros([out_drawing.shape[1], out_drawing.shape[2]])
            out_mnist_sup[:out_mnist.shape[1], :out_mnist.shape[2]] = out_mnist[0, :, :, 0]
            out_drawing = out_drawing[0, :, :, 0]
            out_ = np.concatenate((out_mnist_sup, out_drawing), axis=1)
            out_ = np.maximum(np.minimum(out_, 1.0), 0.0)
            Image.fromarray(
                gray2rgb(
                    np.uint8(out_ * 255)
                )
            ).save('shots/rec_%d.jpg' % i)

    # save the model
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, ckpt_prefix)


if __name__ == '__main__':
    train_mnist()
    # train_drawing()
    # train_connector()