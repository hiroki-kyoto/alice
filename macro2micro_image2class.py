# filename: macro2micro_image2class.py

import numpy as np
import tensorflow as tf


def resize_to_fit_depth(input_, depth):
    shape_of_input = input_.shape.as_list()
    h, w = shape_of_input[1], shape_of_input[2]
    # resize if needed
    shrinkage = 2 << depth
    if h % shrinkage or w % shrinkage:
        h = int(np.ceil(h / shrinkage) * shrinkage)
        w = int(np.ceil(w / shrinkage) * shrinkage)
        input_ = tf.image.resize_bilinear(input_, (h, w))
    return input_


def model_macro2micro_image2class(input_, depth):
    shape_of_input = input_.shape.as_list()
    assert len(shape_of_input) == 4
    h, w = shape_of_input[1], shape_of_input[2]
    base_h = h / (2 << depth)
    base_w = w / (2 << depth)
    input_ = resize_to_fit_depth(input_)
    # decompose the input
    thumbnail = tf.image.resize_bilinear(input_, (base_h, base_w))
    spilts = tf.strided_slice() ##### how to fix this by not using slicing...

    output_ = None
    return output_


def main():
    in_ = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 1])
    out_ = model_macro2micro_image2class(in_, 3)
