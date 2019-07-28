import numpy as np
import tensorflow as tf


def create_variable(name, shape, trainable):
    # check if the variable do exist or not
    vars = tf.global_variables()
    if len(tf.get_variable_scope().name) > 0:
        var_name = tf.get_variable_scope().name + '/' + name
    else:
        var_name = name
    for v in vars:
        if v.name == var_name:
            raise NameError("create_variable: failed, since specified name is already in use.")
    return tf.get_variable(var_name, shape=shape, dtype=tf.float32, trainable=trainable)


def down_sample(x, down_scale):
    ksize = 3*down_scale[0]//2, 3*down_scale[1]//2
    return tf.layers.average_pooling2d(x, ksize, down_scale, 'same')


