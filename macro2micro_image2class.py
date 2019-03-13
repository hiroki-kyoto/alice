# filename: macro2micro_image2class.py

import numpy as np
import tensorflow as tf
import struct
import os


def recursive_split(input_, depth_):
    if len(input_.shape.as_list()) != 4:
        print('shape of input must be in 4 dimension!')
        assert False
    shape_of_input = input_.shape.as_list()
    h_, w_ = shape_of_input[1], shape_of_input[2]
    base_h = h_ >> depth_
    base_w = w_ >> depth_
    all_splits = []
    for i in range(depth_):
        thumbnail = tf.image.resize_bilinear(input_, size=(base_h, base_w))
        all_splits.append(thumbnail)
        shape_of_input = input_.shape.as_list()
        h_, w_ = shape_of_input[1], shape_of_input[2]
        unit_h, unit_w = h_ // 4, w_ // 4
        splits = []
        splits.append(input_[:, 0:2 * unit_h, 0:2 * unit_w, :])
        splits.append(input_[:, 0:2 * unit_h, unit_w:3 * unit_w, :])
        splits.append(input_[:, 0:2 * unit_h, 2 * unit_w:4 * unit_w, :])
        splits.append(input_[:, unit_h:3 * unit_h, 0:2 * unit_w, :])
        splits.append(input_[:, unit_h:3 * unit_h, unit_w:3 * unit_w, :])
        splits.append(input_[:, unit_h:3 * unit_h, 2 * unit_w:4 * unit_w, :])
        splits.append(input_[:, 2 * unit_h:4 * unit_h, 0:2 * unit_w, :])
        splits.append(input_[:, 2 * unit_h:4 * unit_h, unit_w:3 * unit_w, :])
        splits.append(input_[:, 2 * unit_h:4 * unit_h, 2 * unit_w:4 * unit_w, :])
        input_ = tf.concat(splits, axis=0)
    shape_of_input = input_.shape.as_list()
    h_, w_ = shape_of_input[1], shape_of_input[2]
    assert h_ == base_h
    assert w_ == base_w
    all_splits.append(input_)
    return tf.concat(all_splits, axis=0)


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
        return tf.nn.relu(tf.nn.conv2d(input_, w_, (1, stride, stride, 1), padding='SAME') + b_)


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
        return tf.nn.relu(tf.matmul(input_, w_) + b_)


def simple_cnn(input_):
    return conv2d(conv2d(conv2d(input_, 2, 1, 16, 'conv1'), 2, 1, 4, 'conv2'), 2, 1, 1, 'conv3')


def simple_classfier(input_, classes):
    return fully_connect(fully_connect(input_, 2, 'fc1'), classes, 'fc2')


def macro2micro_image2class(input_, depth_, classes):
    shape_of_input = input_.shape.as_list()
    if shape_of_input[0] != 1:
        print('Such network only support single batch!!!')
        assert False
    if len(shape_of_input) != 4:
        print('shape of input must be in 4 dimension!')
        assert False
    h_, w_ = shape_of_input[1], shape_of_input[2]
    shrinkage = 2 << depth_
    if h_ % shrinkage or w_ % shrinkage:
        print('network depth setting warning: input shape does not perfectly fit into this depth!')
        h_ = int(np.ceil(h_ / shrinkage) * shrinkage)
        w_ = int(np.ceil(w_ / shrinkage) * shrinkage)
        print('image resized into :[%d x %d]!' % (h_, w_))
        input_ = tf.image.resize_bilinear(input_, (h_, w_))
    batches = recursive_split(input_, depth_)
    batches = simple_cnn(batches)
    print(batches.shape.as_list())
    output_ = tf.reduce_mean(batches, axis=[1, 2])
    print(output_.shape.as_list())
    output_ = tf.transpose(output_, perm=[1, 0])
    output_ = simple_classfier(output_, classes)
    print(output_.shape.as_list())
    return output_


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


if __name__ == '__main__':
    # load the dataset
    data_dir = 'E:/CodeHub/Datasets/MNIST'
    train_image_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    ims = read_images(train_image_path)
    lbs = read_labels(train_label_path)
    # build the network
    t_in_ = tf.placeholder(dtype=tf.float32, shape=[1, ims.shape[1], ims.shape[2], 1])
    t_out_ = macro2micro_image2class(t_in_, 3, lbs.shape[1])
    t_feedback = tf.placeholder(dtype=tf.float32, shape=[1, lbs.shape[1]])
    #t_loss = tf.nn.softmax_cross_entropy_with_logits_v2(onehot_labels=t_feedback, logits=t_out_)
    t_loss = tf.losses.softmax_cross_entropy(t_feedback, t_out_)
    t_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(t_loss)

    # train the model with data
    repeats = 600000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    loss_av_ = np.zeros([500], dtype=np.float32)
    corr_ = np.zeros([500], dtype=np.float32)
    for i in range(repeats):
        id_ = np.random.randint(len(ims))
        loss_, _, out_ = sess.run([t_loss, t_opt, t_out_], feed_dict={
            t_in_: np.expand_dims(np.expand_dims(ims[id_], axis=0), axis=-1),
            t_feedback: np.expand_dims(lbs[id_], axis=0)
        })
        loss_av_[i%len(loss_av_)] = loss_
        if np.argmax(out_) == np.argmax(lbs[id_]):
            corr_[i%len(corr_)] = 1
        else:
            corr_[i % len(corr_)] = 0
        print('#%d\t loss: %f\t acc= %f' % (i, loss_av_[np.where(loss_av_)].mean(), np.sum(corr_)/len(corr_)))

