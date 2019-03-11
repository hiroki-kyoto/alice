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
    # batches = simple_cnn(batches)
    output_ = None
    return output_


def main():
    in_ = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 1])
    out_ = model_macro2micro_image2class(in_, 3)
