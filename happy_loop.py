# handwrite.py
import numpy as np
import tensorflow as tf
from PIL import Image

# canvas setting: canvas height and width, and pen radius
h, w = 256, 256
r = w // 16
color_bound = 0.5
sim_c = 0.5 # the speed of light in simulation: the maximum of speed enabled
sim_d = 1.0/w # the minimum of simulation in space
sim_t = sim_d / sim_c
num_moves = 128


# define all the possible moves for robot
_ACT_ = [-1, 0, 1]


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


class ObservationPredictor:
    def __init__(self):
        self.t_action = tf.placeholder(dtype=tf.float32, shape=[1, 3])
        self.t_states = tf.placeholder(dtype=tf.float32, shape=[2, 3])
        self.t_observ = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 1])
        self.t_next_observ = tf.placeholder(dtype=tf.float32, shape=[1, h, w, 1])

        t_feat = tf.concat((self.t_action, self.t_states), axis=0)
        t_feat = tf.reshape(t_feat, shape=[1, 9])

        t_feat = tf.layers.dense(
            t_feat,
            8,
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
            64,
            act_fn(),
            True,
            kernel_initializer=ini_fn())
        # convert into a image
        t_feat = tf.reshape(t_feat, [1, 8, 8, 1])
        t_feat = tf.image.resize_bilinear(t_feat, [h, w])
        # t_feat = tf.layers.conv2d_transpose(
        #     inputs=t_feat,
        #     filters=4,
        #     kernel_size=3,
        #     strides=2,
        #     padding='same',
        #     activation=act_fn(),
        #     kernel_initializer=ini_fn())
        # t_feat = tf.layers.conv2d_transpose(
        #     inputs=t_feat,
        #     filters=4,
        #     kernel_size=3,
        #     strides=2,
        #     padding='same',
        #     activation=act_fn(),
        #     kernel_initializer=ini_fn())
        # t_feat = tf.layers.conv2d_transpose(
        #     inputs=t_feat,
        #     filters=4,
        #     kernel_size=3,
        #     strides=2,
        #     padding='same',
        #     activation=act_fn(),
        #     kernel_initializer=ini_fn())
        # t_feat = tf.layers.conv2d_transpose(
        #     inputs=t_feat,
        #     filters=1,
        #     kernel_size=3,
        #     strides=2,
        #     padding='same',
        #     activation=act_fn(),
        #     kernel_initializer=ini_fn())

        self.t_pred_observ = tf.minimum(t_feat + self.t_observ, 1.0)

        self.t_loss = tf.reduce_sum(tf.abs(self.t_pred_observ - self.t_next_observ))
        self.t_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.t_loss)
        self.sess = tf.Session()

    def train(self, model_path, dump_path):
        saver = tf.train.Saver()
        if tf.train.checkpoint_exists(model_path):
            saver.restore(self.sess, model_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        train_step = 100000
        reset_prob = 1.0

        bmp = np.zeros([h, w], dtype=np.float32)
        bmp_last = np.zeros([h, w], dtype=np.float32)

        pos = np.random.rand(3)
        vel = np.random.rand(3)

        states = np.stack([vel, pos], axis=0)
        states_last = np.copy(states)

        loss_cache = np.zeros([1000])

        for i in range(train_step):
            if np.random.rand() < reset_prob:
                bmp[:, :] = 0
                pos = np.random.rand(3)
                vel = np.random.rand(3) - 0.5
                states[0, :] = vel[:]
                states[1, :] = pos[:]

            bmp_last[:, :] = bmp[:, :]
            states_last[:, :] = states[:, :]

            action_ = np.random.rand(3) - 0.5
            action_[:2] = 0.1 * action_[:2]
            action_[2] = 0.5 * action_[2]

            # update the states with physical rules
            update_sheet(bmp, pos, vel, action_)
            states[0, :] = vel
            states[1, :] = pos

            pred, _, loss = self.sess.run(
                [
                    self.t_pred_observ,
                    self.t_opt,
                    self.t_loss
                 ],
                feed_dict={
                    self.t_action: expand_dims(action_, axises=[0]),
                    self.t_states: states_last,
                    self.t_observ: expand_dims(bmp_last, axises=[-1, 0]),
                    self.t_next_observ: expand_dims(bmp, axises=[-1, 0])
                }
            )

            loss_cache[i%len(loss_cache)] = loss

            if (i + 1) % 1000 == 0:
                loss_mean = np.mean(loss_cache)
                loss_vari = np.sqrt(np.sum(np.square(loss_cache - loss_mean)) / (len(loss_cache) - 1))
                print("Itr=%d Loss=%.5f(+/-%.5f)" % (i, loss_mean, loss_vari))
                bmp_merged = merge_bmp(bmp, cut(bmp - bmp_last), cut(pred[0, :, :, 0] - bmp_last))
                save_bmp(bmp_merged, i, dump_path)

        saver.save(self.sess, model_path)


# we need a better sampling method to fully explore the handwriting world!!!
def sample_strokes(n):
    strokes = 0.1 * (np.random.rand(n, 3) - 0.5)
    return strokes


def uniform_distribution(n):
    return np.argmax(np.random.uniform(0, 1.0, [n, 3, 3]), axis=-1)


def render_with_moves(im_, moves_):
    # do the rendering job


if __name__ == '__main__':
    print(uniform_distribution(10))
    exit(0)

    bmp = np.zeros([h, w], dtype=np.float32)
    pos = np.array([0.5, 0.5, 0.0])
    vel = np.zeros([3])
    moves = sample_strokes(10)
    for i in range( len(moves)):
        print(moves[i])
        update_sheet(bmp, pos, vel, moves[i])
    visualize_bmp(bmp)

    # sim = Simulator()
    # sim.train('models/simulator.ckpt', 'shots')

    # state_predictor = StatePredictor()
    # state_predictor.train('models/state_predictor.ckpt',  'shots')

    # observ_predictor = ObservationPredictor()
    # observ_predictor.train('models/observ_predictor.ckpt', 'shots')
