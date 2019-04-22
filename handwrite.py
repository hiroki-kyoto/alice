# handwrite.py
import numpy as np
import tensorflow as tf
from PIL import Image

# canvas setting: canvas height and width, and pen radius
h, w = 64, 64
r = w // 16
color_bound = 0.5
sim_c = 0.5 # the speed of light in simulation: the maximum of speed enabled
sim_d = 1.0/w # the minimum of simulation in space
sim_t = sim_d / sim_c
num_moves = 128


def depth2color(depth):
    if depth < color_bound:
        return depth / color_bound
    else:
        return 1.0


def dot(bmp, x, y, p):
    x_int = int(x * w)
    y_int = int(y * h)
    p_int = int(p * r)
    if p > 0:
        for i in range(y_int - p_int, y_int + p_int + 1):
            for j in range(x_int - p_int, x_int + p_int + 1):
                if 0 <= i < h and 0 <= j < w:
                    if (i - y_int) * (i - y_int) + (j - x_int) * (j - x_int) <= p_int * p_int:
                        bmp[i, j] = np.minimum(1.0, bmp[i, j] + depth2color(p))


# draw black lines on white sheet
def draw_on_sheet(start_, moves):
    # start_ includes initial position, pressure.
    # moves includes: acceleration over position and pressure.
    # the velocity and position over sheet surface and along the direction
    # erected to the sheet.
    x, y, p = start_[0], start_[1], start_[2]
    bmp = np.zeros([h, w], dtype=np.float32)
    v_x, v_y, v_p = 0, 0, 0
    dot(bmp, x, y, p)
    last_x = x
    last_y = y
    for m_ in moves:
        a_x, a_y, a_p = m_[0], m_[1], m_[2]
        for t in np.arange(0, 1, sim_t):
            x_t = x + v_x * t + 0.5 * a_x * t * t
            y_t = y + v_y * t + 0.5 * a_y * t * t
            p_t = p + v_p * t + 0.5 * a_p * t * t
            if x_t != last_x or y_t == last_y:
                dot(bmp, x_t, y_t, p_t)
                last_x = x_t
                last_y = y_t
        x = x + v_x + 0.5 * a_x
        y = y + v_y + 0.5 * a_y
        p = p + v_p + 0.5 * a_p
        v_x = v_x + a_x
        v_y = v_y + a_y
        v_p = v_p + a_p
        # assert np.abs(v_x) < sim_c
        # assert np.abs(v_y) < sim_c
        if p >= 1 or p <= -1:
            v_p = 0
            p = p / np.abs(p)
    return bmp


def simulator():
    t_move = tf.placeholder(dtype=tf.float32, shape=[1, 1, 1, 3])
    t_state = tf.placeholder(dtype=tf.float32, shape=[1, 1, 2, 3])
    t_obsv = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 1])
    t_pred = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 1])
    t_next_state = tf.placeholder(dtype=tf.float32, shape=[1, 1, 2, 3])



if __name__ == '__main__':
    bmp = draw_on_sheet(
        [0.3, 0.3, 0.0],
        [
            [0.01, 0.02, 0.2],
            [0.0, -0.02, 0.0],
            [0.3, -0.02, -0.07],
            [-0.3, 0.04, 0.03],
            [-0.1, 0.25, -0.5],
            [-0.05, 0.0, 0.5],
            [0.1, -0.7, -1.0],
            [0.1, 0.0, 1.0],
            [0.05, 0.6, 0.8],
            [-0.35, 0.3, -1.7],
            [0.0, -0.3, 1.0]
        ])
    print(bmp.shape)
    Image.fromarray(np.uint8(bmp*255)).show()