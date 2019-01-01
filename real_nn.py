import numpy as np
import tensorflow as tf

# the activation function
def g(x):
    assert len(x.shape)==1
    rand = tf.random_uniform([x.shape.as_list()[0]], dtype=tf.float32)
    t = x - rand
    return 0.5*(1 + t / (tf.abs(t) + 1e-8))


def merge(inputs, weights):
    assert len(inputs.shape)==1
    assert len(weights.shape)==2
    inputs = tf.reshape(inputs, [inputs.shape.as_list()[0], 1])
    return tf.reshape(tf.matmul(weights, inputs), [weights.shape.as_list()[0]])


def rand_init(sizes):
    assert len(sizes)<=2
    if len(sizes)==0:
        return np.float32(np.random.rand())
    elif len(sizes)==1:
        return np.float32(np.random.rand(sizes[0]))
    elif len(sizes)==2:
        return np.float32(np.random.rand(sizes[0], sizes[1]))
    else:
        assert False


class RealNN(object):
    def __init__(self, feats):
        # generate weight variables
        self.weights = []
        self.biases = []
        self.in_dim = feats[0]
        self.inputs = tf.placeholder(shape=[self.in_dim], dtype=tf.float32)
        self.layers = [self.inputs]
        for i in range(1,len(feats)):
            w = tf.get_variable(initializer=rand_init([feats[i], feats[i-1]]), name='L%dW' % i)
            self.weights.append(w)
            b = tf.get_variable(initializer=rand_init([feats[i]]), name='L%dB' % i)
            self.biases.append(b)
            self.layers.append(g(merge(self.layers[-1], w)+b))
        self.out_dim = feats[-1]
        self.outputs = self.layers[-1]
        self.truth = tf.placeholder(shape=[self.out_dim], dtype=tf.float32)
    
    def train(self, x, y, max_iter):
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
        self.loss = tf.reduce_mean(tf.abs(self.truth - self.outputs))
        self.minimizer = self.opt.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        _cnt = 0
        while _cnt < max_iter:
            ind = np.random.randint(0, len(x), [])
            _, _loss, _output = self.sess.run([self.minimizer, self.loss, self.weights[-1]], 
                    feed_dict={
                        self.inputs: x[ind],
                        self.truth: y[ind]
                        })
            print('ITR# %d\t LOSS=%.6f' % (_cnt, _loss))
            print(_output)
            _cnt += 1
        saver.save(self.sess, 'models/model.ckpt')
        print('model saved to path: models/....')

    def infer(self, x):
        return None

def int2bins(x):
    x = np.uint8(x)
    op = 0b10000000
    bins = np.array([0.0] * 8)
    for i in range(8):
        if op & x == op:
            bins[i]=1
        else:
            bins[i]=0
        op = op >> 1
    return bins


def concat(bins_1, bins_2):
    return np.concatenate((bins_1, bins_2), axis=0)


def observe(size):
    x = np.random.randint(0,256,[size,2])
    _x = np.zeros([size, 16], dtype=np.float32)
    _y = np.zeros([size, 2], dtype=np.float32)
    for i in range(size):
        _x[i] = concat(int2bins(x[i,0]), int2bins(x[i,1]))
        if x[i,0] > x[i, 1]:
            _y[i, 0] = 0
            _y[i, 1] = 1
        elif x[i, 0] < x[i, 1]:
            _y[i, 0] = 1
            _y[i, 1] = 0
        else:
            _y[i, 0] = 1
            _y[i, 1] = 1
    return _x, _y


def check_acc(y, y_i):
    _score = 0.0
    for i in range(y.shape.as_list()[0]):
        if y[i,0]==y_i[i,0] and y[i,1]==y_i[i,1]:
            _score += 1
    return _score / y.shape.as_list()[0]


if __name__ == '__main__':
    nn = RealNN([16,5,2])
    x, y = observe(1)
    nn.train(x, y, 100)
    #x, y = observe(10)
    #y_i = nn.infer(x)
    #check_acc(y_i, y)

