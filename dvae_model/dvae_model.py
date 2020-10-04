import numpy as np
import tensorflow as tf
layers = tf.keras.layers


def flatten_two_dims(t):
    return tf.reshape(t, [-1] + t.get_shape().as_list()[2:])


def unflatten_first_dim(t, sh):
    return tf.reshape(t, [sh[0], sh[1]] + t.get_shape().as_list()[1:])


class ResBlock(tf.keras.Model):
    def __init__(self, hidsize):
        super(ResBlock, self).__init__()
        self.hidsize = hidsize
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.dense2 = layers.Dense(hidsize, activation=None)

    def call(self, xs):
        x, a = xs              # 输入包括上一层的 特征 和 动作, 这里连续加入动作是为了增强动作的影响
        res = self.dense1(tf.concat([x, a], axis=-1))
        res = self.dense2(tf.concat([res, a], axis=-1))
        assert x.get_shape().as_list()[-1] == self.hidsize and res.get_shape().as_list()[-1] == self.hidsize
        return x + res


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
    def __init__(self, hidsize=256):
        super(ProposalNetwork, self).__init__()
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.residual_block1 = ResBlock(hidsize)
        self.residual_block2 = ResBlock(hidsize)
        self.residual_block3 = ResBlock(hidsize)
        self.dense2 = layers.Dense(hidsize, activation=None)

    def call(self, xs):
        s, a, s_next = xs
        sh = tf.shape(a)                                   # sh=(None,None,4)
        assert len(s.get_shape().as_list()) == 3 and s.get_shape().as_list()[-1] == 512
        assert len(s_next.get_shape().as_list()) == 3 and s_next.get_shape().as_list()[-1] == 512
        assert len(a.get_shape().as_list()) == 3

        # 先将两个状态连接在一起
        x = tf.concat([s, s_next], axis=-1)                # shape=(None,None,1024)
        x = flatten_two_dims(x)                            # shape=(None,1024)
        a = flatten_two_dims(a)                            # shape=(None,4)

        # 网络
        x = self.dense1(tf.concat([x, a], axis=-1))        # (None, 256), 将状态和动作连接在一起, shape=(None,512)
        x = self.residual_block1([x, a])                   # (None, 256)
        x = self.residual_block2([x, a])                   # (None, 256)
        x = self.residual_block3([x, a])                   # (None, 256)
        x = self.dense2(tf.concat([x, a], axis=-1))        # (None, 256)
        x = unflatten_first_dim(x, sh)                     # shape=(None, None, 256)
        return x


class PriorNetwork(tf.keras.Model):
    def __init__(self, hidsize=256):
        super(PriorNetwork, self).__init__()
        self.hidsize = hidsize
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.residual_block1 = ResBlock(hidsize)
        self.residual_block2 = ResBlock(hidsize)
        self.dense2 = layers.Dense(hidsize, activation=None)
        # Residual connection to Generative network
        self.mem0 = MemoryLayer(idx="#0", output_bool=False)   # (None, 512)
        self.mem1 = MemoryLayer(idx="#1", output_bool=False)   # (None, 256)
        self.mem2 = MemoryLayer(idx="#2", output_bool=False)   # (None, 256)
        self.mem3 = MemoryLayer(idx="#3", output_bool=False)   # (None, 256)

    def call(self, xs):
        s, a = xs
        sh = tf.shape(a)                                   # sh=(None,None,4)
        assert len(s.get_shape().as_list()) == 3 and s.get_shape().as_list()[-1] == 512
        assert len(a.get_shape().as_list()) == 3

        x = flatten_two_dims(s)                            # shape=(None,None,512)
        a = flatten_two_dims(a)                            # shape=(None,4)

        # 网络
        x = self.mem0(x)                                   # (None, 512)
        x = self.dense1(tf.concat([x, a], axis=-1))        # (None, 256), 将状态和动作连接在一起
        x = self.mem1(x)                                   # (None, 256)
        x = self.residual_block1([x, a])                   # (None, 256)
        x = self.mem2(x)                                   # (None, 256)
        x = self.residual_block2([x, a])                   # (None, 256)
        x = self.mem3(x)                                   # (None, 256)
        x = self.dense2(tf.concat([x, a], axis=-1))        # (None, 256)
        x = unflatten_first_dim(x, sh)                     # shape=(None, None, 256)
        return x


class GenerativeNetwork(tf.keras.Model):
    def __init__(self, hidsize=256, outsize=512):
        super(GenerativeNetwork, self).__init__()
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.dense2 = layers.Dense(outsize, activation=tf.nn.leaky_relu)
        self.dense3 = layers.Dense(outsize*2, activation=tf.nn.leaky_relu)

        self.residual_block1 = tf.keras.Sequential([
            layers.Dense(hidsize, activation=tf.nn.leaky_relu),   # 256
            layers.Dense(hidsize, activation=None)
        ])
        self.residual_block2 = tf.keras.Sequential([
            layers.Dense(hidsize, activation=tf.nn.leaky_relu),   # 256
            layers.Dense(hidsize, activation=None)
        ])
        self.residual_block3 = tf.keras.Sequential([
            layers.Dense(outsize, activation=tf.nn.leaky_relu),   # 512
            layers.Dense(outsize, activation=None)
        ])
        # Residual connection to Generative network
        self.mem0 = MemoryLayer(idx="#0", output_bool=True, add_bool=True)
        self.mem1 = MemoryLayer(idx="#1", output_bool=True, add_bool=True)
        self.mem2 = MemoryLayer(idx="#2", output_bool=True, add_bool=True)
        self.mem3 = MemoryLayer(idx="#3", output_bool=True, add_bool=True)

    def call(self, z):
        sh = tf.shape(z)                       # 输入是采样得到的隐变量 z, sh=(None,None,128)
        assert z.get_shape().as_list()[-1] == 128 and len(z.get_shape().as_list()) == 3
        z = flatten_two_dims(z)                # shape=(None,128)

        x = self.dense1(z)                     # (None, 256)
        x = self.mem3(x)                       # (None, 256)
        x = x + self.residual_block1(x)        # (None, 256)
        x = self.mem2(x)                       # (None, 256)
        x = x + self.residual_block2(x)        # (None, 256)
        x = self.mem1(x)                       # (None, 256)
        x = self.dense2(x)                     # (None, 512)
        x = x + self.residual_block3(x)        # (None, 512)
        x = self.mem0(x)                       # (None, 512)
        x = self.dense3(x)                     # (None, 1024)
        x = unflatten_first_dim(x, sh)         # shape=(None, None, 1024)
        return x


if __name__ == "__main__":
    prior_network = PriorNetwork(hidsize=256)
    s = tf.convert_to_tensor(np.random.random((10, 10, 512)), dtype=tf.float32)
    a = tf.convert_to_tensor(np.random.random((10, 10, 4)), dtype=tf.float32)
    mu_sigma = prior_network([s, a])

    print("\n---------\n")

    z = mu_sigma[:, :, :mu_sigma.get_shape().as_list()[-1] // 2]
    generative_network = GenerativeNetwork(hidsize=256)
    y = generative_network(z)
    print(generative_network.trainable_variables)
    print(np.sum([np.prod(v.get_shape().as_list()) for v in generative_network.trainable_variables]))
    print(y.shape)

# ---------------------------------------------------------------
# add_bool = False 时的 Generative network. 需要更多的层
# def call(self, z):
#     sh = tf.shape(z)  # 输入是采样得到的隐变量 z, sh=(None,None,128)
#     assert z.get_shape().as_list()[-1] == 128 and len(z.get_shape().as_list()) == 3
#     z = flatten_two_dims(z)  # shape=(None,128)
#
#     x = self.dense1(z)  # (None, 256)
#     x = self.mem3(x)  # (None, 512)
#     x = self.dense2(x)  # (None, 256)
#     x = x + self.residual_block1(x)  # (None, 256)
#     x = self.mem2(x)  # (None, 512)
#     x = self.dense3(x)  # (None, 256)
#     x = x + self.residual_block2(x)  # (None, 256)
#     x = self.mem1(x)  # (None, 512)
#     x = x + self.residual_block3(x)  # (None, 512)
#     x = self.mem0(x)  # (None, 1024)
#     x = self.dense4(x)  # (None, 1024)
#     x = unflatten_first_dim(x, sh)  # shape=(None, None, 1024)
#     return x