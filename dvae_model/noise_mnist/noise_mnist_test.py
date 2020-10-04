import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from noise_mnist_model import ProposalNetwork, PriorNetwork, GenerativeNetwork
from noise_mnist_utils import normal_parse_params, rec_log_prob
tfd = tfp.distributions
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class MnistTest(tf.keras.Model):
    def __init__(self, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.proposal_network = ProposalNetwork()
        self.prior_network = PriorNetwork()
        self.generative_network = GenerativeNetwork()

    def make_latent_distributions(self, obs, out_obs):
        """
        根据 batch, mask 输出 proposal 网络和 prior 网络的输出
        No no_proposal is True, return None instead of proposal distribution.
        """
        assert obs.numpy().shape[-1] == 1
        # Proposal 网络输入是 原始图像 和 mask
        proposal_params = self.proposal_network(tf.concat([obs, out_obs], axis=-1))   # (None,1,1,32)
        proposal = normal_parse_params(proposal_params, 1e-3)
        # Prior 网络输入是 mask之后的图像 和 mask
        prior_params = self.prior_network(obs)              # 在通道上进行连接
        prior = normal_parse_params(prior_params, 1e-3)
        return proposal, prior

    def reparameterize(self, proposal_dist):
        eps = tf.random.normal(shape=proposal_dist.mean().shape)
        return eps * proposal_dist.stddev() + proposal_dist.mean()

    def batch_vlb(self, obs, out_obs):
        """ 输出 variational lower bound, 训练目标是最大化该值. 输出维度 (batch,)
        """
        proposal, prior = self.make_latent_distributions(obs, out_obs)
        latent = self.reparameterize(proposal)                       # (batch,1,1,16) 重参数化并采样
        _ = self.generative_network(latent)                    # (batch,28,28,1)

    def generate_samples_params(self, obs, k=100):
        """ k 代表采样的个数. 从 prior network 输出分布中采样, 随后输入到 generative network 中采样
        """
        prior_params = self.prior_network(obs)
        prior = normal_parse_params(prior_params, 1e-3)
        #
        samples = []
        for i in range(k):
            latent = self.reparameterize(prior)                # (batch,1,1,16) 重参数化并采样
            sample_params = self.generative_network(latent)    # (batch,28,28,1)
            samples.append(sample_params)
        return samples


def build_test_dataset():
    # data
    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = np.expand_dims((test_images / 255.).astype(np.float32), axis=-1)
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    image_dict = {}
    # dict of image and label
    for idx in range(len(test_labels)):
        label = test_labels[idx]
        if label not in image_dict.keys():
            image_dict[label] = []
        else:
            image_dict[label].append(idx)

    # 随机选择
    idx0_random = np.random.choice(image_dict[0])    # 抽取数字 0 的所有序号
    idx1_random = np.random.choice(image_dict[1])    # 抽取数字 1 的所有序号

    test_x0 = test_images[idx0_random]    # 转为图像
    test_x1 = test_images[idx1_random]    # 转为图像

    return np.expand_dims(test_x0, axis=0), np.expand_dims(test_x1, axis=0)  # shape=(1,28,28,1)


def generate_0(model, k=10):
    test_x, _ = build_test_dataset()   # 取到数字0

    # sample
    samples = model.generate_samples_params(test_x, k=k)

    # plot
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 11, 1)
    plt.imshow(test_x[0, :, :, 0], cmap='gray')
    plt.axis('off')
    idx = 1
    for sample in samples:
        sample = tf.nn.sigmoid(sample).numpy()
        # sample[sample >= 0.0] = 1.
        # sample[sample < 0.0] = 0.
        assert sample.shape == (1, 28, 28, 1)
        plt.subplot(1, 11, idx+1)
        plt.axis('off')
        plt.imshow(sample[0, :, :, 0], cmap='gray')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        idx += 1
    plt.savefig("model/noise_mdp_0.png", dpi=300)
    # plt.show()
    plt.close()


def generate_1(model, k=100):
    # input_num 可选0 或 1
    _, test_x = build_test_dataset()   # 取到数字1

    # sample
    samples = model.generate_samples_params(test_x, k=k)

    # plot
    plt.figure(figsize=(10, 10))
    plt.subplot(10, 11, 1)
    plt.imshow(test_x[0, :, :, 0], cmap='gray')
    plt.axis('off')

    idx = 1
    for sample in samples:
        sample = tf.nn.sigmoid(sample).numpy()
        # sample[sample >= 0.0] = 1.
        # sample[sample < 0.0] = 0.
        assert sample.shape == (1, 28, 28, 1)
        plt.subplot(10, 11, idx+1)
        plt.axis('off')
        plt.imshow(sample[0, :, :, 0], cmap='gray')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        idx += 1
        if idx % 11 == 0:
            idx += 1
    plt.savefig("model/noise_mdp_1.png", dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # initialize model and load weights
    test_x, _ = build_test_dataset()
    dvae_model = MnistTest()
    dvae_model.batch_vlb(tf.convert_to_tensor(test_x), tf.convert_to_tensor(test_x))
    print("load weights...")
    dvae_model.load_weights("model/model_232.h5")
    print("load done")

    # generate 0
    print("Generate number 0")
    generate_0(dvae_model, k=10)

    # generate 1
    print("Generate number 1")
    generate_1(dvae_model, k=100)
