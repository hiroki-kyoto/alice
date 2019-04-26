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


def action_encoder(t_action):
    with tf.variable_scope(
            name_or_scope='action/encoder',
            reuse=tf.AUTO_REUSE):
        t_out = dense_block(t_action, [8, 16], False)
        t_out = dense_block(t_out, [8, 16], True)
        return t_out


def states_encoder(t_states):
    with tf.variable_scope(
            name_or_scope='states/encoder',
            reuse=tf.AUTO_REUSE):
        # reshape into one-dimension vector in such form:
        # (v_x, v_y, v_p, x, y, p), a 6-item group.
        t_out = tf.reshape(t_states, shape=[1, 1, 1, 6])
        t_out = dense_block(t_out, [8, 16], False)
        t_out = dense_block(t_out, [8, 16], True)
        return t_out


def observ_encoder(t_observ):
    with tf.variable_scope(
            name_or_scope='observ/encoder',
            reuse=tf.AUTO_REUSE):
        t_out = conv_block(t_observ, [8, 16], [2, 2], False)
        t_out = conv_block(t_out, [8, 16], [1, 2], True)
        t_out = conv_block(t_out, [8, 4, 1], [1, 1, 1], True)
        shape_ = t_out.shape.as_list()
        t_out = tf.reshape(t_out, shape=[1, 1, 1, shape_[1] * shape_[2]])
        return t_out


def merge_features(t_feat_action, t_feat_states, t_feat_observ):
    with tf.variable_scope(
            name_or_scope='merge',
            reuse=tf.AUTO_REUSE):
        t_out = tf.concat(
            [t_feat_action, t_feat_states, t_feat_observ],
            axis=-1)
        t_out = dense_block(t_out, [8, 16], True)
        t_out = dense_block(t_out, [8, 16], True)
        return t_out


def states_decoder(t_feat_merged):
    with tf.variable_scope(
            name_or_scope='states/decoder',
            reuse=tf.AUTO_REUSE):
        t_out = dense_block(t_feat_merged, [8, 16], True)
        t_out = dense_block(t_out, [8, 6], False)
        # reshape into 2-dimension array in such form:
        # [[v_x, v_y, v_p], [x, y, p]], a 2x3 array.
        t_out = tf.reshape(t_out, shape=[1, 1, 2, 3])
        return t_out


def observ_decoder(t_feat_merged):
    with tf.variable_scope(
            name_or_scope='observ/decoder',
            reuse=tf.AUTO_REUSE):
        t_out = tf.reshape(t_feat_merged, shape=[1, 4, 4, 1])
        t_out = deconv_block(t_out, [4, 1], [1, 2], True)
        t_out = deconv_block(t_out, [4, 1], [1, 2], True)
        t_out = deconv_block(t_out, [4, 1], [1, 2], True)
        t_out = deconv_block(t_out, [4, 1], [1, 2], True)
        return t_out


def visualize_bmp(bmp):
    Image.fromarray(np.uint8(bmp * 255)).show()


def save_bmp(bmp, itr, dir_):
    Image.fromarray(np.uint8(bmp * 255)).save('%s/%d.jpg' % (dir_, itr))


def merge_bmp(bmp_left, bmp_right):
    return np.concatenate((bmp_left, bmp_right), axis=1)


def expand_dims(tensor, axises):
    for i in range(len(axises)):
        tensor = np.expand_dims(tensor, axis=axises[i])
    return tensor


def cut(bmp):
    return np.maximum(np.minimum(bmp, 1), 0)


class Simulator:
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

        self.t_loss_states = tf.reduce_mean(
            tf.abs(self.t_pred_states - self.t_next_states))
        self.t_loss_observ = tf.reduce_mean(
            tf.abs(self.t_pred_observ - self.t_next_observ))
        alpha = 1.0
        self.t_loss_global = self.t_loss_states * alpha + self.t_loss_observ * (1 - alpha)

        self.t_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.t_loss_global)
        self.sess = tf.Session()

    def train(self, model_path, dump_path):
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(model_path):
            saver.restore(self.sess, model_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_step = 100000
        reset_prob = 0.01

        bmp = np.zeros([h, w], dtype=np.float32)
        bmp_last = np.zeros([h, w], dtype=np.float32)

        pos = np.random.rand(3)
        vel = np.random.rand(3)

        states = np.stack([vel, pos], axis=0)
        states_last = np.copy(states)

        loss_s_av = 0
        loss_o_av = 0

        for i in range(train_step):
            if np.random.rand() < reset_prob:
                bmp[:, :] = 0
                pos = np.random.rand(3)
                vel = np.random.rand(3)
                states[0, :] = vel
                states[1, :] = pos

            bmp_last[:, :] = bmp[:, :]
            states_last[:, :] = states[:, :]

            action_ = np.random.rand(3) - 0.5
            action_[:2] = 0.05 * action_[:2]
            action_[2] = 0.5 * action_[2]
            update_sheet(bmp, pos, vel, action_)
            states[0, :] = vel
            states[1, :] = pos

            pred, _, loss_s, loss_o = self.sess.run(
                [self.t_pred_observ,
                 self.t_opt,
                 self.t_loss_states,
                 self.t_loss_observ],
                feed_dict={
                    self.t_action: expand_dims(action_, axises=[0, 0, 0]),
                    self.t_states: expand_dims(states_last, axises=[0, 0]),
                    self.t_next_states: expand_dims(states, axises=[0, 0]),
                    self.t_observ: expand_dims(bmp_last, axises=[0, -1]),
                    self.t_next_observ: expand_dims(bmp, axises=[0, -1]),
                }
            )

            m = 100.0
            if i < m:
                loss_s_av = loss_s_av * (i / m) + loss_s * (1 - i / m)
                loss_o_av = loss_o_av * (i / m) + loss_o * (1 - i / m)
            else:
                loss_s_av = loss_s_av * ((m - 1) / m) + loss_s * (1 / m)
                loss_o_av = loss_o_av * ((m - 1) / m) + loss_o * (1 / m)

            if i % 1000 == 0:
                print("Itr=%d States=%.5f Observ=%.5f" % (i, loss_s_av, loss_o_av))
                bmp_merged = merge_bmp(bmp, cut(pred[0, :, :, 0]))
                save_bmp(bmp_merged, i, dump_path)
                # print('acceleration=%s' % str(action_))
                # print('previous velocity=%s' % str(states_last[0, :]))
                # print('previous position=%s' % str(states_last[1, :]))
                # print('velocity=%s' % str(states[0, :]))
                # print('position=%s' % str(states[1, :]))

        saver.save(self.sess, model_path)

    def load(self, model_path):
        pass

    def test(self, samples):
        pass


class StatePredictor:
    def __init__(self):
        self.t_action = tf.placeholder(dtype=tf.float32, shape=[1, 3])
        self.t_states = tf.placeholder(dtype=tf.float32, shape=[2, 3])
        self.t_next_states = tf.placeholder(dtype=tf.float32, shape=[2, 3])

        t_feat = tf.concat((self.t_action, self.t_states), axis=0)
        t_feat = tf.reshape(t_feat, shape=[1, 9])
        print(t_feat.shape.as_list())

        t_feat = tf.layers.dense(
            t_feat,
            8,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        t_feat = tf.layers.dense(
            t_feat,
            8,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        t_feat = tf.layers.dense(
            t_feat,
            8,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        t_feat = tf.layers.dense(
            t_feat,
            6,
            act_fn(),
            True,
            kernel_initializer=ini_fn())

        self.t_pred_states = tf.reshape(t_feat, shape=[2, 3])

        self.t_loss = tf.reduce_mean(tf.abs(self.t_pred_states - self.t_next_states))

        self.t_opt = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.t_loss)
        self.sess = tf.Session()

    def train(self, model_path, dump_path):
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(model_path):
            saver.restore(self.sess, model_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_step = 1000000
        reset_prob = 0.01

        pos = np.random.rand(3)
        vel = np.random.rand(3)

        states = np.stack([vel, pos], axis=0)
        states_last = np.copy(states)

        loss_av = 0

        for i in range(train_step):
            if np.random.rand() < reset_prob:
                pos = np.random.rand(3)
                vel = np.random.rand(3)
                states[0, :] = vel[:]
                states[1, :] = pos[:]

            states_last[:, :] = states[:, :]

            action_ = np.random.rand(3) - 0.5
            action_[:2] = 0.5 * action_[:2]
            action_[2] = 0.5 * action_[2]

            # update the states with physical rules
            pos = pos + vel + 0.5 * action_
            vel = vel + action_
            valid_mask = np.float32(pos >= 0)
            valid_mask = valid_mask * np.float32(pos <= 1)
            vel = vel * valid_mask
            pos = np.maximum(np.minimum(pos, 1), 0)

            states[0, :] = vel[:]
            states[1, :] = pos[:]

            pred, _, loss = self.sess.run(
                [
                    self.t_pred_states,
                    self.t_opt,
                    self.t_loss
                 ],
                feed_dict={
                    self.t_action: expand_dims(action_, axises=[0]),
                    self.t_states: states_last,
                    self.t_next_states: states
                }
            )

            m = 100.0
            if i < m:
                loss_av = loss_av * (i / m) + loss * (1 - i / m)
            else:
                loss_av = loss_av * ((m - 1) / m) + loss * (1 / m)

            if i % 1000 == 0:
                print("Itr=%d Loss=%.5f" % (i, loss_av))
                print('velocity: %s - %s' % (str(states[0, :]), str(pred[0, :])))
                print('position: %s - %s' % (str(states[1, :]), str(pred[1, :])))

        saver.save(self.sess, model_path)


if __name__ == '__main__':
    # bmp = np.zeros([h, w], dtype=np.float32)
    # pos = np.array([0.3, 0.3, 0.0])
    # moves = np.array([
    #     [0.01, 0.02, 0.2],
    #     [0.0, -0.02, 0.0],
    #     [0.3, -0.02, -0.07],
    #     [-0.3, 0.04, 0.03],
    #     [-0.1, 0.25, -0.5],
    #     [-0.05, 0.0, 0.5],
    #     [0.1, -0.7, -1.0],
    #     [0.1, 0.0, 1.0],
    #     [0.05, 0.6, 0.8],
    #     [-0.35, 0.3, -1.5],
    #     [0.0, -0.3, 1.0]
    # ])
    # vel = np.zeros([3])
    # for mov_ in moves:
    #     update_sheet(bmp, pos, vel, mov_)
    # visualize_bmp(bmp)

    # sim = Simulator()
    # sim.train('models/simulator.ckpt', 'shots')

    state_predictor = StatePredictor()
    state_predictor.train('models/state_predictor.ckpt',  'shots')

