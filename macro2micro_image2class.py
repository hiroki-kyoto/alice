# filename: macro2micro_image2class.py

import numpy as np
import tensorflow as tf


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
            initializer=tf.initializers.random_normal(0.2))
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
            initializer=tf.initializers.random_normal(0.2))
        b_ = tf.get_variable(
            name='b',
            shape=[units],
            dtype=tf.float32,
            initializer=tf.initializers.constant(0.0))
        return tf.nn.relu(tf.matmul(input_, w_) + b_)


def simple_cnn(input_):
    return conv2d(conv2d(conv2d(input_, 2, 1, 16, 'conv1'), 2, 1, 4, 'conv2'), 2, 1, 1, 'conv3')


def simple_classfier(input_):
    return fully_connect(fully_connect(input_, 2, 'fc1'), 10, 'fc2')


def macro2micro_image2class(input_, depth_):
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
    output_ = simple_classfier(output_)
    print(output_.shape.as_list())
    return output_


def main():
    in_ = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 1])
    out_ = macro2micro_image2class(in_, 3)
    sess = tf.Session()
    saver = tf.train.Saver()

    # load the dataset


main()
