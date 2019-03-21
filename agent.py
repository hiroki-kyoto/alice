import numpy as np
import struct
import os


def read_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>4I', f.read(16))
        assert rows == 28
        assert cols == 28
        return np.float32(np.fromfile(f, dtype=np.uint8)).reshape(num, rows, cols)/255.0


def read_labels(path):
    with open(path, 'rb') as f:
        _, num = struct.unpack('>2I', f.read(8))
        labels = np.zeros([num, 10])
        ids = np.fromfile(f, dtype=np.uint8)
        for i in range(num):
            labels[i, ids[i]] = 1.0
        return labels


# randomly generate a world for agents to survive
def build_world():
    world = {}
    world['name'] = 'mnist'

    data_dir = 'E:/CodeHub/Datasets/MNIST'
    train_image_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    ims = read_images(train_image_path)
    lbs = read_labels(train_label_path)

    def select_random_images_from_1_to_10():
        demos = [None] * 10
        selected_ = 0
        while selected_ < 10:
            id_ = np.random.randint(0, len(ims), [2])[0]
            num_ = np.argmax(lbs[id_])
            if demos[num_] is None:
                demos[num_] = id_
                selected_ += 1
        return demos

    world['display'] = select_random_images_from_1_to_10

    def reward(input_, response_):
        if np.argmax(response_) == np.argmax(lbs(input_)):
            return +1
        else:
            return -1

    world['interact'] = reward


    return world


if __name__ == '__main__':
    world = build_world()
    ids = world['display']()
    # input_ = agent['observe']()
    # response = agent['act'](input)
    # reward_ = world['interact'](ids[i], response)
    # agent['update'](reward_)
    # agent['die']()
    # agent['recreate']()
    # ...
    print(ids)