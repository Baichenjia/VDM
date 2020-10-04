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


class MnistDynamics(tf.keras.Model):
    def __init__(self, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.proposal_network = ProposalNetwork()
        self.prior_network = PriorNetwork()
        self.generative_network = GenerativeNetwork()
        self.sigma_mu = sigma_mu
        self.sigma_sigma = sigma_sigma

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

    def prior_regularization(self, prior):
        """
            对 prior network 输出的分布进行约束. 在没有该约束的情况下, 模型一般也不会发散.
            该正则项对原损失函数的影响很小, 几乎不影响学习的过程, 推荐使用. 对应于论文 4.3.2 内容
        """
        # print("先验分布均值 shape=", prior.mean().shape)    # (batch_size, 256)
        num_objects = prior.mean().shape[0]
        mu = tf.reshape(prior.mean(), (num_objects, -1))
        sigma = tf.reshape(prior.stddev(), (num_objects, -1))
        mu_regularise = - tf.reduce_sum(mu ** 2, axis=-1) / (2 * (self.sigma_mu ** 2))
        sigma_regularise = tf.reduce_sum(tf.math.log(sigma)-sigma, axis=-1) * self.sigma_sigma
        return mu_regularise + sigma_regularise     # shape=(batch,)

    def reparameterize(self, proposal_dist):
        eps = tf.random.normal(shape=proposal_dist.mean().shape)
        return eps * proposal_dist.stddev() + proposal_dist.mean()

    def batch_vlb(self, obs, out_obs):
        """ 输出 variational lower bound, 训练目标是最大化该值. 输出维度 (batch,)
        """
        proposal, prior = self.make_latent_distributions(obs, out_obs)
        prior_regularization = self.prior_regularization(prior)      # (batch,)
        latent = self.reparameterize(proposal)                       # (batch,1,1,16) 重参数化并采样
        x_logit = self.generative_network(latent)                    # (batch,28,28,1)
        rec_hood = -1. * rec_log_prob(out_obs, x_logit)
        kl = tfp.distributions.kl_divergence(proposal, prior)        # (batch,1,1,16)
        kl = tf.reduce_sum(tf.squeeze(kl), axis=-1)                  # (batch,)
        info = {"rec_loss": -1*rec_hood, "kl_loss": kl, "prior_reg_loss": -prior_regularization}
        return rec_hood - kl + prior_regularization, info             # (batch,)

    def batch_iwae(self, obs, out_obs, k=10):
        """ 从 proposal 中采样, 计算似然概率, 减去 KL-divergence, 得到 ELBO.
        """
        proposal, prior = self.make_latent_distributions(obs, out_obs)
        estimates = []
        for ix in range(k):
            latent = self.reparameterize(proposal)                 # (batch,1,1,16) 重参数化并采样
            x_logit = self.generative_network(latent)              # (batch,28,28,1)
            rec_hood = -1. * rec_log_prob(out_obs, x_logit)

            prior_log_prob = prior.log_prob(latent)                            # (batch,1,1,16)
            prior_log_prob = tf.reshape(prior_log_prob, (obs.shape[0], -1))    # (batch,16)
            prior_log_prob = tf.reduce_sum(prior_log_prob, axis=-1)            # (batch,)

            proposal_log_prob = proposal.log_prob(latent)                          # (batch,1,1,256)
            proposal_log_prob = tf.reshape(proposal_log_prob, (obs.shape[0], -1))  # (batch,256)
            proposal_log_prob = tf.reduce_sum(proposal_log_prob, axis=-1)          # (batch,)

            # print("**", rec_hood.numpy().mean(), prior_log_prob.numpy().mean(), proposal_log_prob.numpy().mean())
            estimate = rec_hood + prior_log_prob - proposal_log_prob   # (batch_size,) elbo=rec-KL
            estimates.append(estimate)

        estimates_tensor = tf.stack(estimates, axis=1)     # (batch_size, k)
        assert len(estimates_tensor.shape) == 2 and estimates_tensor.shape[1] == k
        # 操作相当于在log内除以k, 输出 shape=(batch_size,)
        return tf.math.reduce_logsumexp(estimates_tensor, axis=1) - tf.math.log(float(k))

    def generate_samples_params(self, obs, out_obs, k=100):
        """ k 代表采样的个数. 从 prior network 输出分布中采样, 随后输入到 generative network 中采样
        """
        _, prior = self.make_latent_distributions(obs, out_obs)
        samples = []
        for i in range(k):
            latent = self.reparameterize(prior)                # (batch,1,1,16) 重参数化并采样
            sample_params = self.generative_network(latent)    # (batch,28,28,1)
            samples.append(sample_params)
        return samples


def build_dataset(train_images, train_labels, storage0=5, storage1=10):
    image_dict = {}
    # dict of image and label
    for idx in range(len(train_labels)):
        label = train_labels[idx]
        if label not in image_dict.keys():
            image_dict[label] = []
        else:
            image_dict[label].append(idx)

    # 构造数字0的样本
    obs_idx0 = image_dict[0]           # 抽取数字 0 的所有序号
    np.random.shuffle(obs_idx0)
    train_x0, train_y0 = [], []
    for idx in obs_idx0:
        for i in range(storage0):
            train_x0.append(idx)
            trans_to_idx = np.random.choice(image_dict[1])
            train_y0.append(trans_to_idx)
    print("training data x0:", len(train_x0))
    print("training data y0:", len(train_y0))

    # 构造数字1的样本
    obs_idx1 = image_dict[1]           # 抽取数字 0 的所有序号
    np.random.shuffle(obs_idx1)
    train_x1, train_y1 = [], []
    for idx in obs_idx1:
        for i in range(storage1):
            train_x1.append(idx)
            trans_to_label = np.random.randint(low=2, high=10)
            trans_to_idx = np.random.choice(image_dict[trans_to_label])
            train_y1.append(trans_to_idx)
    print("training data x1:", len(train_x1))
    print("training data y1:", len(train_y1))

    train_x0_img = train_images[train_x0]
    train_y0_img = train_images[train_y0]
    print("\ntraining data x0:", train_x0_img.shape)
    print("training data y0:", train_y0_img.shape)

    train_x1_img = train_images[train_x1]
    train_y1_img = train_images[train_y1]
    print("\ntraining data x1:", train_x1_img.shape)
    print("training data y1:", train_y1_img.shape)

    train_x_img = np.vstack([train_x0_img, train_x1_img])
    train_y_img = np.vstack([train_y0_img, train_y1_img])
    print("\ntraining data x:", train_x_img.shape)
    print("training data y:", train_y_img.shape)
    return train_x_img, train_y_img


def mnist_data(build_train=True):
    # data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = np.expand_dims((train_images / 255.).astype(np.float32), axis=-1)
    test_images = np.expand_dims((test_images / 255.).astype(np.float32), axis=-1)

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    # train
    if build_train:
        print("Generating training data:")
        train_x, train_y = build_dataset(train_images, train_labels, storage0=5, storage1=50)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(500000)
        train_dataset = train_dataset.batch(512, drop_remainder=True)
    else:
        train_dataset = None

    print("Generating testing data:")
    test_x, test_y = build_dataset(test_images, test_labels, storage0=5, storage1=10)
    test_x = tf.convert_to_tensor(test_x)
    test_y = tf.convert_to_tensor(test_y)
    print("dataset done.")
    return train_dataset, test_x, test_y


def train():
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    dvae_model = MnistDynamics()

    train_dataset, test_x, test_y = mnist_data(build_train=True)
    dvae_model.batch_vlb(test_x[:10], test_y[:10])
    # load weights
    # print("load weights...")
    # dvae_model.load_weights("model/model_99.h5")
    # print("load done")

    # start
    Epochs = 500
    iwae_tests = []
    for epoch in range(Epochs):
        print("Epoch: ", epoch)
        for i, (batch_x, batch_y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:                          # train
                vlb, info = dvae_model.batch_vlb(batch_x, batch_y)
                loss = tf.reduce_mean(-1.0 * vlb)                    # 损失为 -elbo
                if i % 10 == 0:
                    print(i, ":", loss.numpy(),
                          ", rec:", np.mean(info['rec_loss'].numpy()),
                          ", kl:", np.mean(info['kl_loss'].numpy()),
                          ", prior:", np.mean(info['prior_reg_loss'].numpy()))
            gradients = tape.gradient(loss, dvae_model.trainable_variables)
            # gradients, _ = tf.clip_by_global_norm(gradients, 1.)
            optimizer.apply_gradients(zip(gradients, dvae_model.trainable_variables))

        # test IWAE
        iwae_test = dvae_model.batch_iwae(test_x, test_y, k=10)
        iwae_test_mean = iwae_test.numpy().mean()
        print("IWAE test:", iwae_test_mean)
        iwae_tests.append(iwae_test_mean)

        # save
        dvae_model.save_weights("model/model_"+str(epoch)+".h5")
        np.save("model/iwae.npy", np.array(iwae_tests))


if __name__ == '__main__':
    # generate()
    train()
