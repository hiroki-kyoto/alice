# yinyang.py
# Yin and Yang are separated and also connected
# Yin and Yang are contradicted and yet combined

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from happy_loop import Model, act_fn, ini_fn, initialize_uninitialized


class Generator(Model):
    # graph: On which graph should this model be built in.
    @staticmethod
    def build(inputs):
        t_action, t_observ = inputs
        # MODELS
        t_feat = tf.layers.dense(
            t_action,
            16,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        t_feat = tf.layers.dense(
            t_feat,
            16,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        t_feat = tf.layers.dense(
            t_feat,
            t_observ.shape.as_list()[1],
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        # addition required to be non-negative
        t_feat = tf.maximum(t_feat, 0)
        # OUTPUTS
        t_pred_observ = tf.minimum(t_feat + t_observ, 1.0)
        return t_pred_observ

    def __init__(self):
        super().__init__()
        self.patch_h = 2 * (STEP_SIZE + r) + 1
        self.patch_w = 2 * (STEP_SIZE + r) + 1
        # INPUTS
        with self.graph.as_default():
            self.t_action = tf.placeholder(
                dtype=tf.float32,
                shape=[1, len(_M_X) + len(_M_Y) + len(_M_Z)])
            self.t_observ = tf.placeholder(
                dtype=tf.float32,
                shape=[1, self.patch_h * self.patch_w])
            self.t_next_observ = tf.placeholder(
                dtype=tf.float32,
                shape=[1, self.patch_h * self.patch_w])
            self.t_pred_observ = self.build([self.t_action,
                                             self.t_observ])

    def train(self, ckpt_paths, dump_path):
        with self.graph.as_default():
            self.t_loss = tf.reduce_mean(
                tf.abs(self.t_pred_observ - self.t_next_observ))
            self.t_opt = tf.train.AdamOptimizer(
                learning_rate=1e-3).minimize(self.t_loss)
            saver = tf.train.Saver()
            if tf.train.checkpoint_exists(ckpt_paths[0]):
                saver.restore(self.sess, ckpt_paths[0])
            else:
                self.sess.run(tf.global_variables_initializer())

        train_sessions = 1000
        train_episodes = 100
        train_steps = 8
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
                    ppos = [int(pos[0] * w), int(pos[1] * h)]
                    pbeg = [ppos[0] - STEP_SIZE - r,
                            ppos[1] - STEP_SIZE - r]
                    pend = [ppos[0] + STEP_SIZE + r + 1,
                            ppos[1] + STEP_SIZE + r + 1]
                    if pbeg[0] < 0 or pbeg[1] < 0 \
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
                            '--',
                            label='mean')
                        plt.plot(
                            range(len(loss_varis)),
                            loss_varis,
                            '-*',
                            label='vari')
                        plt.legend()
                        plt.pause(0.01)
                    if (counter + 1) % 1000 == 0:
                        saver.save(self.sess, ckpt_paths[0])
                    counter += 1

    def test(self, ckpt_paths, dump_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            if tf.train.checkpoint_exists(ckpt_paths[0]):
                saver.restore(self.sess, ckpt_paths[0])
            else:
                print("========= NO VALID CHECK POINT FOUND !!! ========")
                print("========= RUNNING TEST IN RANDOM INITIAL ========")
                self.sess.run(tf.global_variables_initializer())

        train_sessions = 1000
        train_episodes = 100
        train_steps = 8

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
                    pbeg = [ppos[0] - STEP_SIZE - r,
                            ppos[1] - STEP_SIZE - r]
                    pend = [ppos[0] + STEP_SIZE + r + 1,
                            ppos[1] + STEP_SIZE + r + 1]
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
