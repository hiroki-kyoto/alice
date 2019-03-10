# filename: macro2micro_image2class.py

import numpy as np
import tensorflow as tf


def model_macro2micro_image2class(input_, depth):
    shape_of_input = input_.shape.as_list()
    assert len(shape_of_input) == 4
    h, w = shape_of_input[1], shape_of_input[2]
    # resize if needed
    output_ = None
    return output_


def main():
    in_ = tf.placeholder(dtype=tf.float32, shape=[1, 32, 32, 1])
    out_ = model_macro2micro_image2class(in_, 3)
