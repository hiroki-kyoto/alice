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
    world['order'] = np.arange(0, 10)

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

    def reward(ids, res):
        score_ = 0
        for i in range(len(ids)):
            score_ += int(np.argmax(res[i]) == np.argmax(lbs[ids[i]]))
        return score_

    world['interact'] = reward

    def alter_task():
        world['order'] = np.random.permutation(np.arange(0, 10))

    world['alter'] = alter_task

    return world


def generate_agent():
    agent = {}
    agent['name'] = 'Agent'
    agent['age'] = 10000
    agent['life'] = 100
    agent['states'] = []
    agent['controls'] = []

    def update(reward_):
        agent['life'] += reward_

    def grow_old():
        agent['age'] -= 1
        agent['life'] -= 1
        if agent['age'] <= 0:
            print('Agent dead being too old.')
        if agent['life'] <= 0:
            print('Agent dead being too hungry.')

    def observe(world):
        ids = world['display']()
        # run through controls and states


if __name__ == '__main__':
    world = build_world()
    # input_ = agent['observe'](world)
    # response = agent['act'](input)
    # reward_ = world['interact'](ids[i], response)
    # agent['update'](reward_)
    # agent['die']()
    # agent['recreate']()
    # ...