# handwrite.py
import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib.pyplot as plt
import time
import math


# canvas setting: canvas height and width, and pen radius
h, w = 256, 256
r = w // 32
color_bound = 0.5
sim_c = 0.5 # the speed of light in simulation: the maximum of speed enabled
sim_d = 1.0 / w # the minimum of simulation in space
sim_t = sim_d / sim_c
num_moves = 128


# define all the possible moves for robot
_M_X = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) / w
_M_Y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) / h
_M_Z = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


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
def update_sheet(bmp_, pos_, vel_, mov_):
    # start_ includes initial position, pressure.
    # moves includes: acceleration over position and pressure.
    # the velocity and position over sheet surface and along the direction
    # erected to the sheet.
    x, y, p = pos_[0], pos_[1], pos_[2]
    v_x, v_y, v_p = vel_[0], vel_[1], vel_[2]
    a_x, a_y, a_p = mov_[0], mov_[1], mov_[2]
    last_x = x
    last_y = y
    for t in np.arange(0, 1, sim_t):
        x_t = x + v_x * t + 0.5 * a_x * t * t
        y_t = y + v_y * t + 0.5 * a_y * t * t
        p_t = p + v_p * t + 0.5 * a_p * t * t
        if x_t != last_x or y_t == last_y:
            dot(bmp_, x_t, y_t, p_t)
            last_x = x_t
            last_y = y_t
    x = x + v_x + 0.5 * a_x
    y = y + v_y + 0.5 * a_y
    p = p + v_p + 0.5 * a_p
    v_x = v_x + a_x
    v_y = v_y + a_y
    v_p = v_p + a_p
    if p > 1 or p < 0:
        v_p = 0
        p = np.minimum(np.maximum(p, 0), 1)
    if x > 1 or x < 0:
        v_x = 0
        x = np.minimum(np.maximum(x, 0), 1)
    if y > 1 or y < 0:
        v_y = 0
        y = np.minimum(np.maximum(y, 0), 1)
    pos_[0] = x
    pos_[1] = y
    pos_[2] = p
    vel_[0] = v_x
    vel_[1] = v_y
    vel_[2] = v_p


def act_fn():
    return tf.nn.leaky_relu


def ini_fn():
    return tf.initializers.truncated_normal(0.0, 0.1)


def dense_block(input_, dims, norm):
    if norm:
        out_ = tf.layers.batch_normalization(input_)
    else:
        out_ = input_
    for i in range(len(dims)-1):
        out_ = tf.layers.dense(
            out_,
            dims[i],
            act_fn(),
            True,
            kernel_initializer=ini_fn())
    out_ = tf.layers.dense(
        out_,
        dims[-1],
        None,
        True,
        kernel_initializer=ini_fn())
    return out_


def conv_block(input_, filters, strides, norm):
    if norm:
        out_ = tf.layers.batch_normalization(input_)
    else:
        out_ = input_
    for i in range(len(filters)-1):
        out_ = tf.layers.conv2d(
            out_,
            filters[i],
            3,
            strides[i],
            'same',
            activation=act_fn(),
            kernel_initializer=ini_fn())
    out_ = tf.layers.conv2d(
        out_,
        filters[-1],
        3,
        strides[-1],
        'same',
        kernel_initializer=ini_fn())
    return out_


def deconv_block(input_, filters, strides, norm):
    if norm:
        out_ = tf.layers.batch_normalization(input_)
    else:
        out_ = input_
    for i in range(len(filters)-1):
        out_ = tf.layers.conv2d(
            out_,
            filters[i],
            3,
            strides[i],
            'same',
            activation=act_fn(),
            kernel_initializer=ini_fn())
    out_ = tf.layers.conv2d_transpose(
        out_,
        filters[-1],
        3,
        strides[-1],
        'same',
        kernel_initializer=ini_fn())
    return out_


def visualize_bmp(bmp):
    Image.fromarray(np.uint8((1 - bmp) * 255)).show()


def save_bmp(bmp, itr, dir_):
    Image.fromarray(np.uint8((1 - bmp) * 255)).save('%s/%d.jpg' % (dir_, itr))


def merge_bmp(bmp_ori, bmp_left, bmp_right):
    seg_width = 3
    seg_band = np.zeros([bmp_ori.shape[0], seg_width, 3])
    seg_band[:, :, 0] = 0.0
    seg_band[:, :, 1] = 1.0
    seg_band[:, :, 2] = 1.0
    bmp_ori = np.stack([bmp_ori, bmp_ori, bmp_ori], axis=-1)
    bmp_left = np.stack([bmp_left, bmp_left, bmp_left], axis=-1)
    bmp_right = np.stack([bmp_right, bmp_right, bmp_right], axis=-1)
    return np.concatenate((bmp_ori, seg_band, bmp_left, seg_band, bmp_right), axis=1)


def expand_dims(tensor, axises):
    for i in range(len(axises)):
        tensor = np.expand_dims(tensor, axis=axises[i])
    return tensor


def cut(bmp):
    return np.maximum(np.minimum(bmp, 1), 0)


def action_generator(n):
    # generate action over x axis
    x_ = np.argmax(np.random.uniform(0.0, 1.0, [n, len(_M_X)]), axis=-1)
    y_ = np.argmax(np.random.uniform(0.0, 1.0, [n, len(_M_Y)]), axis=-1)
    z_ = np.argmax(np.random.uniform(0.0, 1.0, [n, len(_M_Z)]), axis=-1)
    return np.stack([x_, y_, z_], axis=-1)


def render_step(im_, pos_, move_):
    delta_x = _M_X[move_[0]]
    delta_y = _M_Y[move_[1]]
    z = _M_Z[move_[2]]
    pos_[0] = pos_[0] + delta_x
    pos_[1] = pos_[1] + delta_y
    pos_[2] = z
    pos_ = np.maximum(np.minimum(pos_, 1.0), 0.0)
    dot(im_, pos_[0], pos_[1], pos_[2])
    return im_, pos_


def index_to_onehot(inds, dims):
    onehot = np.zeros([np.sum(dims, keepdims=True)[0]], dtype=np.float32)
    offset = 0
    for i in range(len(inds)):
        onehot[inds[i] + offset] = 1
        offset += dims[i]
    return onehot


class Render:
    def __init__(self):
        self.t_action = tf.placeholder(
            dtype=tf.float32,
            shape=[1, len(_M_X) + len(_M_Y) + len(_M_Z)])
        self.patch_r = r + 1
        self.patch_h = 2 * self.patch_r + 1
        self.patch_w = 2 * self.patch_r + 1
        self.t_observ = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.patch_h * self.patch_w])
        self.t_next_observ = tf.placeholder(
            dtype=tf.float32,
            shape=[1, self.patch_h * self.patch_w])

        t_feat = tf.layers.dense(
            self.t_action,
            16,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        t_feat = tf.layers.dense(
            t_feat,
            self.patch_h * self.patch_w,
            act_fn(),
            True,
            kernel_initializer=ini_fn())

        self.t_pred_observ = tf.maximum(
            tf.minimum(t_feat + self.t_observ, 1.0), 0)

        self.t_loss = tf.reduce_mean(
            tf.abs(self.t_pred_observ - self.t_next_observ))
        self.t_opt = tf.train.AdamOptimizer(
            learning_rate=1e-4).minimize(self.t_loss)
        self.sess = tf.Session()

    def train(self, model_path, dump_path):
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(model_path):
            saver.restore(self.sess, model_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_sessions = 1000
        train_episodes = 100
        train_steps = 30
        loss_cache = np.zeros([100])
        loss_means = []
        loss_varis = []
        counter = 0

        plt.ion()
        plt.figure(1)

        for _ in range(train_sessions):
            hit_wall = False
            im = np.zeros([h, w], dtype=np.float32)
            pos = np.array([0.5, 0.5, 0.0])
            moves = action_generator(train_episodes)
            for i in range(train_episodes):
                if hit_wall:
                    break
                steps_ = int(train_steps * np.random.rand())
                for _ in range(steps_):
                    ppos = [int(pos[0]*w), int(pos[1]*h)]
                    pbeg = [ppos[0] - self.patch_r,
                            ppos[1] - self.patch_r]
                    pend = [ppos[0] + self.patch_r + 1,
                            ppos[1] + self.patch_r + 1]
                    if pbeg[0] < 0 or pbeg[1] < 0\
                        or pend[0] > w or pend[1] > h:
                        hit_wall = True
                        print("===== hit the wall, start new session =====")
                        break
                    curr = np.copy(im[pbeg[1]:pend[1], pbeg[0]:pend[0]])
                    im, pos = render_step(im, pos, moves[i])
                    next = np.copy(im[pbeg[1]:pend[1], pbeg[0]:pend[0]])

                    _, loss = self.sess.run([
                        self.t_opt,
                        self.t_loss],
                        feed_dict={
                            self.t_action: expand_dims(
                                index_to_onehot(
                                    moves[i],
                                    [len(_M_X), len(_M_Y), len(_M_Z)]),
                                axises=[0]),
                            self.t_observ: np.reshape(
                                curr,
                                [1, self.patch_h * self.patch_w]),
                            self.t_next_observ: np.reshape(
                                next,
                                [1, self.patch_h * self.patch_w])
                        })

                    loss_cache[counter % len(loss_cache)] = loss

                    if (counter + 1) % 100 == 0:
                        loss_means.append(np.mean(loss_cache))
                        loss_varis.append(np.sqrt(np.sum(
                            np.square(loss_cache - loss_means[-1])) /
                                                  (len(loss_cache) - 1)))
                        plt.clf()
                        plt.plot(
                            range(len(loss_means)),
                            loss_means,
                            '-*',
                            label='mean')
                        plt.plot(
                            range(len(loss_varis)),
                            loss_varis,
                            '+-',
                            label='vari')
                        plt.legend()
                        plt.pause(0.01)
                    if (counter + 1) % 1000 == 0:
                        saver.save(self.sess, model_path)
                    counter += 1

    def test(self, model_path, dump_path):
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(model_path):
            saver.restore(self.sess, model_path)
        else:
            assert False

        train_sessions = 1000
        train_episodes = 100
        train_steps = 50

        plt.ion()
        plt.figure(1)

        for _ in range(train_sessions):
            hit_wall = False
            im = np.zeros([h, w], dtype=np.float32)
            im_o = np.copy(im)
            pos = np.array([0.5, 0.5, 0.0])
            moves = action_generator(train_episodes)
            for i in range(train_episodes):
                if hit_wall:
                    break
                steps_ = int(train_steps * np.random.rand())
                for _ in range(steps_):
                    ppos = [int(pos[0] * w), int(pos[1] * h)]
                    pbeg = [ppos[0] - self.patch_r,
                            ppos[1] - self.patch_r]
                    pend = [ppos[0] + self.patch_r + 1,
                            ppos[1] + self.patch_r + 1]
                    if pbeg[0] < 0 or pbeg[1] < 0 \
                            or pend[0] > w or pend[1] > h:
                        hit_wall = True
                        print("===== hit the wall, start new session =====")
                        break
                    curr = np.copy(im_o[pbeg[1]:pend[1], pbeg[0]:pend[0]])
                    im, pos = render_step(im, pos, moves[i])
                    pred = self.sess.run(
                        [self.t_pred_observ],
                        feed_dict={
                            self.t_action: expand_dims(
                                index_to_onehot(
                                    moves[i],
                                    [len(_M_X), len(_M_Y), len(_M_Z)]),
                                axises=[0]),
                            self.t_observ: np.reshape(
                                curr,
                                [1, self.patch_h * self.patch_w])
                        })
                    pred = np.reshape(pred, [self.patch_h, self.patch_w])
                    im_o[pbeg[1]:pend[1], pbeg[0]:pend[0]] = pred
                    out = np.concatenate((im, im_o), axis=1)
                    plt.clf()
                    plt.imshow(1 - out, cmap="gray", vmin=0.0, vmax=1.0)
                    plt.pause(0.01)


def test_dynamic_disp():
    plt.ion()
    plt.figure(1)
    im = np.zeros([h, w], dtype=np.float32)
    pos = np.array([0.5, 0.5, 0.0])
    moves = action_generator(100)
    episode = 50

    for i in range(len(moves)):
        print(moves[i])
        for j in range(int(episode * np.random.rand())):
            im, pos = render_step(im, pos, moves[i])
            plt.clf()
            plt.imshow(1 - im, cmap="gray", vmin=0.0, vmax=1.0)
            plt.pause(0.01)


if __name__ == '__main__':
    render = Render()
    # render.train('models/render.ckpt', 'shots')
    render.test('models/render.ckpt', 'shots')
