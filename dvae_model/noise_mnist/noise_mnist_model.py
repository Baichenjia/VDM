import numpy as np
import tensorflow as tf
layers = tf.keras.layers
tf.enable_eager_execution()


class ResBlock(tf.keras.Model):
    """
    Usual full pre-activation ResNet bottleneck block.
    """
    def __init__(self, outer_dim, inner_dim):
        super(ResBlock, self).__init__()
        data_format = 'channels_last'

        self.net = tf.keras.Sequential([
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(inner_dim, (1, 1)),
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(inner_dim, (3, 3), padding='same'),
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(outer_dim, (1, 1))])

    def call(self, x):
        return x + self.net(x)


class MLPBlock(tf.keras.Model):
    def __init__(self, inner_dim):
        super(MLPBlock, self).__init__()
        self.net = tf.keras.Sequential([
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2D(inner_dim, (1, 1))])

    def call(self, x):
        return x + self.net(x)


class MemoryLayer(tf.keras.Model):
    storage = {}

    def __init__(self, idx, output_bool=False, add_bool=False):
        super(MemoryLayer, self).__init__()
        self.idx = idx
        self.output_bool = output_bool
        self.add_bool = add_bool

    def call(self, x):
        if not self.output_bool:
            self.storage[self.idx] = x
            return x
        else:
            if self.idx not in self.storage:
                err = 'MemoryLayer: idx \'%s\' is not initialized. '
                raise ValueError(err)
            stored = self.storage[self.idx]
            if not self.add_bool:
                data = tf.concat([x, stored], axis=-1)
            else:
                data = x + stored
            return data


class ProposalNetwork(tf.keras.Model):
    def __init__(self):
        super(ProposalNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net2 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8)])

        self.net3 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16)])
        self.pad3 = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))

        self.net4 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32)])

        self.net5 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64)])

        self.net6 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(256, 1),
            MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256)])

    def call(self, x):
        # 当输入是 (None, 28, 28, 2),
        x = self.net1(x)           # (b, 28, 28, 8)
        x = self.net2(x)           # (b, 14, 14, 16)
        x = self.net3(x)           # (b, 7, 7, 32)
        x = self.pad3(x)           # (b, 8, 8, 32)
        x = self.net4(x)           # (b, 4, 4, 64)
        x = self.net5(x)           # (b, 2, 2, 128)
        x = self.net6(x)           # (b, 1, 1, 256)
        return x


class PriorNetwork(tf.keras.Model):
    def __init__(self):
        super(PriorNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net2 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8)])

        self.net3 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16)])
        self.pad3 = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))

        self.net4 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32)])

        self.net5 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64)])

        self.net6 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(256, 1),
            MLPBlock(256), MLPBlock(256), MLPBlock(256), MLPBlock(256)])

        # memory layer
        self.mem0 = MemoryLayer(idx="#0", output_bool=False)
        self.mem1 = MemoryLayer(idx="#1", output_bool=False)
        self.mem2 = MemoryLayer(idx="#2", output_bool=False)
        self.mem3 = MemoryLayer(idx="#3", output_bool=False)
        self.mem4 = MemoryLayer(idx="#4", output_bool=False)
        self.mem5 = MemoryLayer(idx="#5", output_bool=False)

    def call(self, x):
        # 当输入是 (None, 28, 28, 2),       各层的维度如下:
        x = self.mem0(x)
        x = self.net1(x)   # (b, 28, 28, 8)
        x = self.mem1(x)
        x = self.net2(x)   # (b, 14, 14, 16)
        x = self.mem2(x)
        x = self.net3(x)
        x = self.pad3(x)   # (b, 8, 8, 32)
        x = self.mem3(x)
        x = self.net4(x)   # (b, 4, 4, 64)
        x = self.mem4(x)
        x = self.net5(x)   # (b, 2, 2, 128)
        x = self.mem5(x)
        x = self.net6(x)   # (b, 1, 1, 256)
        # print([self.mem1.storage[k].shape.as_list() for k in ["#0", "#1", "#2", "#3", "#4", "#5"]])
        return x


class GenerativeNetwork(tf.keras.Model):
    def __init__(self):
        super(GenerativeNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([layers.Conv2D(128, 1),
            MLPBlock(128), MLPBlock(128), MLPBlock(128), MLPBlock(128),
            layers.Conv2D(128, 1), layers.UpSampling2D((2, 2))])

        self.net2 = tf.keras.Sequential([layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64),
            layers.Conv2D(64, 1), layers.UpSampling2D((2, 2))])

        self.net3 = tf.keras.Sequential([layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
            layers.Conv2D(32, 1), layers.UpSampling2D((2, 2))])

        self.net4 = tf.keras.Sequential([layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
            layers.Conv2D(16, 1), layers.UpSampling2D((2, 2))])

        self.net5 = tf.keras.Sequential([layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
            layers.Conv2D(8, 1), layers.UpSampling2D((2, 2))])

        self.net6 = tf.keras.Sequential([layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
            layers.Conv2D(4, 1)])

        self.net7 = tf.keras.Sequential([layers.Conv2D(2, 1),
            ResBlock(2, 2), ResBlock(2, 2), ResBlock(2, 2),
            layers.Conv2D(1, 1)])

        # all memory layers
        self.mem5 = MemoryLayer(idx="#5", output_bool=True)
        self.mem4 = MemoryLayer(idx="#4", output_bool=True)
        self.mem3 = MemoryLayer(idx="#3", output_bool=True)
        self.mem2 = MemoryLayer(idx="#2", output_bool=True)
        self.mem1 = MemoryLayer(idx="#1", output_bool=True)
        self.mem0 = MemoryLayer(idx="#0", output_bool=True)

    def call(self, x):
        # input=(b, 1, 1, 128)
        x = self.net1(x)
        # print("1:", x.shape)
        x = self.mem5(x)
        # print("2:", x.shape)
        x = self.net2(x)
        # print("3:", x.shape)
        x = self.mem4(x)
        # print("4:", x.shape)
        x = self.net3(x)
        # print("5:", x.shape)
        x = self.mem3(x)
        # print("6:", x.shape)
        x = x[:, :-1, :-1, :]
        # print("7:", x.shape)
        x = self.net4(x)
        # print("8:", x.shape)
        x = self.mem2(x)
        # print("9:", x.shape)
        x = self.net5(x)
        # print("10:", x.shape)
        x = self.mem1(x)
        # print("11:", x.shape)
        x = self.net6(x)
        # print("12:", x.shape)
        x = self.mem0(x)
        # print("13:", x.shape)
        x = self.net7(x)
        return x
        # 1: (2, 2, 2, 128)
        # 2: (2, 2, 2, 256)
        # 3: (2, 4, 4, 64)
        # 4: (2, 4, 4, 128)
        # 5: (2, 8, 8, 32)
        # 6: (2, 8, 8, 64)
        # 7: (2, 7, 7, 64)
        # 8: (2, 14, 14, 16)
        # 9: (2, 14, 14, 32)
        # 10: (2, 28, 28, 8)
        # 11: (2, 28, 28, 16)
        # 12: (2, 28, 28, 4)
        # 13: (2, 28, 28, 6)


if __name__ == '__main__':
    proposal_network = ProposalNetwork()
    x1 = tf.convert_to_tensor(np.random.random((2, 28, 28, 2)), tf.float32)
    y1 = proposal_network(x1)

    prior_network = PriorNetwork()
    x1 = tf.convert_to_tensor(np.random.random((2, 28, 28, 2)), tf.float32)
    y2 = prior_network(x1)
    print("output of prior network:", y2.shape)

    generative_network = GenerativeNetwork()
    x2 = tf.convert_to_tensor(np.random.random((2, 1, 1, 128)), tf.float32)
    y3 = generative_network(x2)
    print(y3.shape)

    print("Parameters:", np.sum([np.prod(v.shape.as_list()) for v in prior_network.trainable_variables]))
    print("Parameters:", np.sum([np.prod(v.shape.as_list()) for v in proposal_network.trainable_variables]))
    print("Parameters:", np.sum([np.prod(v.shape.as_list()) for v in generative_network.trainable_variables]))




