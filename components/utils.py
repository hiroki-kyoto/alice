import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


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
    ksize = int(3*down_scale[0]//2), int(3*down_scale[1]//2)
    return tf.layers.average_pooling2d(x, ksize, down_scale, 'same')


def up_sample(x, up_scale):
    _shape = x.shape.as_list()
    assert len(_shape) == 4
    size_new = int(up_scale[0] * _shape[1]), int(up_scale[1] * _shape[2])
    return tf.image.resize_bilinear(x, size_new)


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def show_rgb(rgb: np.ndarray) -> None:
    if np.max(rgb) <= 1.0:
        Image.fromarray(np.uint8(rgb * 255)).show()
    else:
        Image.fromarray(np.uint8(rgb)).show()


def gray2rgb(gray):
    return np.stack([gray, gray, gray], axis=-1)


def show_gray(gray, min=0, max=255):
    assert max - min > 0
    gray_std = np.minimum(np.maximum(gray, min), max) / (max - min)
    return show_rgb(gray2rgb(gray_std))


def normalize(x):
    _min = np.min(x)
    _max = np.max(x)
    if (_max > _min):
        return (x - _min) / (_max - _min)
    else:
        return x


# [i] hsv : hsv image of float data type in [0, 1]
# [i] mode: full(0) or half(1) for channel HUE
# [o] rgb : rgb image of float in [0, 1]
def hsv2rgb(hsv, mode=0):
    hsv_u8 = np.uint8(hsv * 255)
    if mode == 1 or mode == 'full' or mode == 'FULL':
        rgb = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB_FULL)
    else:
        rgb = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)
    return np.float32(rgb) / 255.0


# convert any nested list(or tuple) into a flat list
def flatten(x: list):
    y = x.copy()
    item_list_found = True
    while item_list_found:
        item_list_found = False
        z = []
        for i in y:
            if type(i) is list or type(i) is tuple:
                item_list_found = True
                for ii in i:
                    z.append(ii)
            else:
                z.append(i)
        y = z.copy()
    return y
