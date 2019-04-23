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
        if p > 1 or p < 0:
            v_p = 0
            p = np.minimum(np.maximum(p, 0), 1)
        if x > 1 or x < 0:
            v_x = 0
            x = np.minimum(np.maximum(x, 0), 1)
        if y > 1 or y < 0:
            v_y = 0
            y = np.minimum(np.maximum(y, 0), 1)

    return bmp


def act_fn():
    return tf.nn.tanh


def action_encoder(t_action):
    with tf.variable_scope(name_or_scope='action/encoder', reuse=tf.AUTO_REUSE):
        t_out = tf.layers.dense(t_action, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        t_out = tf.layers.dense(t_out, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        return t_out


def states_encoder(t_states):
    with tf.variable_scope(name_or_scope='states/encoder', reuse=tf.AUTO_REUSE):
        # reshape into one-dimension vector in such form:
        # (v_x, v_y, v_p, x, y, p), a 6-item group.
        t_out = tf.reshape(t_states, shape=[1, 1, 1, 6])
        t_out = tf.layers.dense(t_out, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        t_out = tf.layers.dense(t_out, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        return t_out


def observ_encoder(t_observ):
    with tf.variable_scope(name_or_scope='observ/encoder', reuse=tf.AUTO_REUSE):
        t_out = tf.layers.conv2d(t_observ, 8, 3, 2, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d(t_out, 16, 3, 2, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d(t_out, 8, 3, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d(t_out, 16, 3, 2, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d(t_out, 8, 1, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d(t_out, 4, 1, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d(t_out, 1, 1, 1, 'same', activation=tf.nn.relu)
        shape_ = t_out.shape.as_list()
        t_out = tf.reshape(t_out, shape=[1, 1, 1, shape_[1] * shape_[2]])
        return t_out


def merge_features(t_feat_action, t_feat_states, t_feat_observ):
    with tf.variable_scope(name_or_scope='merge', reuse=tf.AUTO_REUSE):
        t_out = tf.concat([t_feat_action, t_feat_states, t_feat_observ], axis=-1)
        t_out = tf.layers.dense(t_out, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        t_out = tf.layers.dense(t_out, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        return t_out


def states_decoder(t_feat_merged):
    with tf.variable_scope(name_or_scope='states/decoder', reuse=tf.AUTO_REUSE):
        t_out = tf.layers.dense(t_feat_merged, 8, None, True)
        t_out = tf.layers.dense(t_out, 16, act_fn(), True)
        t_out = tf.layers.dense(t_out, 8, None, True)
        t_out = tf.layers.dense(t_out, 6, act_fn(), True)
        # reshape into 2-dimension array in such form:
        # [[v_x, v_y, v_p], [x, y, p]], a 2x3 array.
        t_out = tf.reshape(t_out, shape=[1, 1, 2, 3])
        return t_out


def observ_decoder(t_feat_merged):
    with tf.variable_scope(name_or_scope='observ/decoder', reuse=tf.AUTO_REUSE):
        t_out = tf.reshape(t_feat_merged, shape=[1, 4, 4, 1])
        t_out = tf.layers.conv2d(t_out, 4, 3, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d_transpose(t_out, 1, 3, 2, 'same')
        t_out = tf.layers.conv2d(t_out, 4, 3, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d_transpose(t_out, 1, 3, 2, 'same')
        t_out = tf.layers.conv2d(t_out, 4, 3, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d_transpose(t_out, 1, 3, 2, 'same')
        t_out = tf.layers.conv2d(t_out, 4, 3, 1, 'same', activation=tf.nn.relu)
        t_out = tf.layers.conv2d_transpose(t_out, 1, 3, 2, 'same')
        return t_out


class simulator:
    def __init__(self):
        self.t_action = tf.placeholder(dtype=tf.float32, shape=[1, 1, 1, 3])
        self.t_states = tf.placeholder(dtype=tf.float32, shape=[1, 1, 2, 3])
        self.t_observ = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 1])
        self.t_next_states = tf.placeholder(dtype=tf.float32, shape=[1, 1, 2, 3])
        self.t_next_observ = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 1])

        # build the encoder model
        t_feat_action = action_encoder(self.t_action)
        t_feat_states = states_encoder(self.t_states)
        t_feat_observ = observ_encoder(self.t_observ)
        print(t_feat_action.shape)
        print(t_feat_states.shape)
        print(t_feat_observ.shape)

        t_feat_merged = merge_features(t_feat_action, t_feat_states, t_feat_observ)
        print(t_feat_merged.shape)

        # build the decoder model
        self.t_pred_states = states_decoder(t_feat_merged)
        self.t_pred_observ = observ_decoder(t_feat_merged)
        print(self.t_pred_states.shape)
        print(self.t_pred_observ.shape)

        self.t_loss_states = tf.reduce_mean(tf.abs(self.t_pred_states - self.t_next_states))
        self.t_loss_observ = tf.reduce_mean(tf.abs(self.t_pred_observ - self.t_next_observ))
        alpha = 0.5
        self.t_loss_global = self.t_loss_states * alpha + self.t_loss_observ * (1 - alpha)

        self.t_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.t_loss_global)

    def train(self, samples):
        pass

    def load(self, model_path):
        pass

    def test(self, samples):
        pass


if __name__ == '__main__':
    # bmp = draw_on_sheet(
    #     [0.3, 0.3, 0.0],
    #     [
    #         [0.01, 0.02, 0.2],
    #         [0.0, -0.02, 0.0],
    #         [0.3, -0.02, -0.07],
    #         [-0.3, 0.04, 0.03],
    #         [-0.1, 0.25, -0.5],
    #         [-0.05, 0.0, 0.5],
    #         [0.1, -0.7, -1.0],
    #         [0.1, 0.0, 1.0],
    #         [0.05, 0.6, 0.8],
    #         [-0.35, 0.3, -1.5],
    #         [0.0, -0.3, 1.0]
    #     ])
    # pos = np.random.rand(3)
    # moves = 0.1 * (np.random.rand(100, 3) - 0.3)
    # bmp = draw_on_sheet(pos, moves)
    # Image.fromarray(np.uint8(bmp*255)).show()
    sim = simulator()
    samples = []
    # cache samples in separate threads
    sim.train(samples)