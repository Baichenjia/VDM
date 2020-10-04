import tensorflow as tf

from utils import small_convnet, fc, activ, flatten_two_dims, unflatten_first_dim, small_deconvnet


class FeatureExtractor(object):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None,
                 layernormalize=None, scope='feature_extractor'):
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy   # False
        self.feat_dim = feat_dim                    # 512
        self.layernormalize = layernormalize        #
        self.policy = policy
        self.hidsize = policy.hidsize               # 512
        self.ob_space = policy.ob_space             # Box(84,84,4)
        self.ac_space = policy.ac_space             # Discrete(4)
        self.obs = self.policy.ph_ob                # shape=(None,None,84,84,4)
        self.ob_mean = self.policy.ob_mean          # shape=(None,None,84,84,4)
        self.ob_std = self.policy.ob_std            # 标量 1.8
        with tf.variable_scope(scope):
            # 观测是连续叠加几帧得到的. 得到下一帧的观测
            self.last_ob = tf.placeholder(dtype=tf.int32,         # (None,1,84,84,4)
                                          shape=(None, 1) + self.ob_space.shape, name='last_ob')
            # 这里在第1维进行 concat, 第1维代表时间步的维度
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1)    # (None,None,84,84,4)

            # 对下一帧的观测 next_ob 提取提取观测特征. 默认 else 分支
            if features_shared_with_policy:            # 使用策略自带的feature或get_features函数
                self.features = self.policy.features
                self.last_features = self.policy.get_features(self.last_ob, reuse=True)
            else:    # features是(t-4,t)时间步特征叠加而成. last_feature是t+1时间步特征. 这里使用本类定义的 get_features 函数
                self.features = self.get_features(self.obs, reuse=False)              # (None,None,512)
                self.last_features = self.get_features(self.last_ob, reuse=True)      # (None,1,512)
            # 由 feature 和 last_feature 构造下一时刻的特征 next_feature.
            self.next_features = tf.concat([self.features[:, 1:], self.last_features], 1)

            self.ac = self.policy.ph_ac    # 根据 (features, next_features, ac) 可以构造辅助任务, 如 inverse model.
            self.scope = scope

            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)    # 转换shape (None,None,84,84,4) -> (None,84,84,4)
        # 对输入先进行规约, 然后通过一个小型卷积神经网络
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)     # 仅提取特征, 自身不包含损失


class InverseDynamics(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None):
        super(InverseDynamics, self).__init__(scope="inverse_dynamics", policy=policy,
                                              features_shared_with_policy=features_shared_with_policy,
                                              feat_dim=feat_dim, layernormalize=layernormalize)

    def get_loss(self):
        # 构造逆环境模型, 流程 输入 [feature(obs), feature(obs_next)] -> 输出动作参数
        # 计算不同动作的高斯或者softmax分布 -> 计算 log_prob 作为 inverse dynamics 的损失.
        with tf.variable_scope(self.scope):
            # features.shape=(None,None,512), next_features.shape=(None,None,512),
            x = tf.concat([self.features, self.next_features], 2)  # x.shape=(None,None,1024)
            sh = tf.shape(x)
            x = flatten_two_dims(x)      # (None, 1024) 融合了 feature 和 next_feature
            x = fc(x, units=self.policy.hidsize, activation=activ)    # (None,512)
            x = fc(x, units=self.ac_space.n, activation=None)         # (None,4)    输出动作logits
            param = unflatten_first_dim(x, sh)                        # (None,None,4)  恢复维度
            idfpd = self.policy.ac_pdtype.pdfromflat(param)           # 根据输出 logits 建立分布
            # 如果是连续动作空间,这里代表高斯-log损失; 如果是离散动作空间, 这里代表 softmax 损失
            return idfpd.neglogp(self.ac)                             # shape等于前2个维度 (None,None)


class VAE(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=False, spherical_obs=False):
        assert not layernormalize, "VAE features should already have reasonable size, no need to layer normalize them"
        self.spherical_obs = spherical_obs
        super(VAE, self).__init__(scope="vae", policy=policy,
                                  features_shared_with_policy=features_shared_with_policy,
                                  feat_dim=feat_dim, layernormalize=False)
        # 将当前状态 s 提取特征的最后一维拆分成2份, 分别代表 mu 和 sigma. 以均值 mu 作为提取的特征.
        self.features = tf.split(self.features, 2, -1)[0]   # features.shape=(None,None,1024), 拆分后为(None,None,512)
        self.next_features = tf.split(self.next_features, 2, -1)[0]    # next_features也取到均值, shape=(None,None,512)

    # 这个函数与基类 FeatureExtractor 的 get_feature 相同
    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        # reuse=True 时重用 FeatureExtractor 的特征提取模块
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=2 * self.feat_dim, last_nl=None, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        with tf.variable_scope(self.scope):
            posterior_mean, posterior_scale = tf.split(self.features, 2, -1)  # mu, sigma 维度分别为 (None,None,512)
            posterior_scale = tf.nn.softplus(posterior_scale)                 # 确保 sigma>0
            posterior_distribution = tf.distributions.Normal(loc=posterior_mean, scale=posterior_scale)  # 后验概率

            sh = tf.shape(posterior_mean)     # (None,None,512)
            prior = tf.distributions.Normal(loc=tf.zeros(sh), scale=tf.ones(sh))   # 先验是标准正态

            posterior_kl = tf.distributions.kl_divergence(posterior_distribution, prior)  # KL约束 (None,None,512)

            posterior_kl = tf.reduce_sum(posterior_kl, [-1])    # 转成 (None,None), 最后一维加和
            assert posterior_kl.get_shape().ndims == 2

            posterior_sample = posterior_distribution.sample()   # 重参数化, 采样. (None,None,512)
            # decoder重建, 输出是一个分布, 分布均值维度是 (None,None,84,84,4)
            reconstruction_distribution = self.decoder(posterior_sample)  # 去噪自编码器, 输入加噪声 (None,None,84,84,4)
            norm_obs = self.add_noise_and_normalize(self.obs)
            # 重建损失, 维度是 (None,None,84,84,4)
            reconstruction_likelihood = reconstruction_distribution.log_prob(norm_obs)  # 重建损失
            assert reconstruction_likelihood.get_shape().as_list()[2:] == [84, 84, 4]
            reconstruction_likelihood = tf.reduce_sum(reconstruction_likelihood, [2, 3, 4])  # (None,None)
            # 计算得到 ELBO
            likelihood_lower_bound = reconstruction_likelihood - posterior_kl   # shape=(batch, T)
            return - likelihood_lower_bound     # 损失为 -ELBO

    def add_noise_and_normalize(self, x):      # x是VAE的最后的输出label,需要加噪声提高鲁棒性,shape=(None,None,84,84,4)
        x = tf.to_float(x) + tf.random_uniform(shape=tf.shape(x), minval=0., maxval=1.)
        x = (x - self.ob_mean) / self.ob_std
        return x

    def decoder(self, z):           # z 是VAE后验分布的均值, shape=(None,None,512)
        nl = tf.nn.leaky_relu
        z_has_timesteps = (z.get_shape().ndims == 3)
        if z_has_timesteps:
            sh = tf.shape(z)
            z = flatten_two_dims(z)         # (None,512)
        with tf.variable_scope(self.scope + "decoder"):
            # 反卷积网络. de-convolution. spherical_obs=True, 输出 z.shape=(None,84,84,4)
            z = small_deconvnet(z, nl=nl, ch=4 if self.spherical_obs else 8, positional_bias=True)
            if z_has_timesteps:
                z = unflatten_first_dim(z, sh)
            if self.spherical_obs:     # 球形损失, scale 在所有维度都是同一个常数, 简化运算
                scale = tf.get_variable(name="scale", shape=(), dtype=tf.float32,
                                        initializer=tf.ones_initializer())
                scale = tf.maximum(scale, -4.)
                scale = tf.nn.softplus(scale)
                scale = scale * tf.ones_like(z)
            else:
                z, scale = tf.split(z, 2, -1)   # 输出 split, 分别作为 mu 和 scale.
                scale = tf.nn.softplus(scale)
            # scale = tf.Print(scale, [scale])
            return tf.distributions.Normal(loc=z, scale=scale)


class JustPixels(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='just_pixels'):
        assert not layernormalize
        assert not features_shared_with_policy
        super(JustPixels, self).__init__(scope=scope, policy=policy,
                                         features_shared_with_policy=False,
                                         feat_dim=None, layernormalize=None)

    def get_features(self, x, reuse):
        # 这里没有卷积，仅对原始的图像输入进行了规约
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)


# if __name__ == '__main__':
#     from functools import partial
#     import gym
#     import numpy as np
#     from functools import partial
#     from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
#     from wrappers import MaxAndSkipEnv, ProcessFrame84, StickyActionEnv
#     from cnn_policy import CnnPolicy
#
#     def make_env_all_params(rank, add_monitor=True):
#         env = gym.make("BreakoutNoFrameskip-v4")
#         assert 'NoFrameskip' in env.spec.id
#         env._max_episode_steps = 4500 * 4
#         env = StickyActionEnv(env)
#         env = MaxAndSkipEnv(env, skip=4)         # 每个动作连续执行4步
#         env = ProcessFrame84(env, crop=False)    # 处理观测
#         env = FrameStack(env, 4)                 # 将连续4帧叠加起来作为输入
#         return env
#     make_env = partial(make_env_all_params, add_monitor=True)
#     # make env
#     env = make_env(0, add_monitor=False)
#     obs = env.reset()
#     print(np.asarray(obs).shape, env.action_space.sample())
#
#     ob_space, ac_space = env.observation_space, env.action_space
#
#     # 一个随机智能体与环境交互, 计算得到的观测的均值和标准差.
#     from utils import random_agent_ob_mean_std
#
#     ob_mean, ob_std = random_agent_ob_mean_std(env)
#     print(ob_mean.shape, np.max(ob_mean), np.min(ob_mean))
#     print(ob_std.shape, np.max(ob_std), np.min(ob_std))
#
#     # 初始化环境
#     envs = [partial(make_env, i) for i in range(5)]
#
#     # CNN policy
#     print("Init Policy.")
#     policy = CnnPolicy(scope='pol',
#                        ob_space=ob_space,
#                        ac_space=ac_space,
#                        hidsize=512,
#                        feat_dim=512,
#                        ob_mean=ob_mean,
#                        ob_std=ob_std,
#                        layernormalize=False,
#                        nl=tf.nn.leaky_relu)
#
#     print("Init Feature Extractor.")
#     feature_extractor = {"none": FeatureExtractor,  # 默认是none
#                          "idf": InverseDynamics,
#                          "vaesph": partial(VAE, spherical_obs=True),
#                          "vaenonsph": partial(VAE, spherical_obs=False),
#                          "pix2pix": JustPixels}['none']
#
#     feature_extractor = feature_extractor(
#         policy=policy, features_shared_with_policy=False, feat_dim=512, layernormalize=False)
#
#     # agent 损失: 包括 actor,critic,entropy 损失; 先在加上 feature 学习时包含的损失
#     print(feature_extractor.loss.shape)
#     # feature_extractor.features.shape=(None,None,512)
#     mean_std = tf.nn.moments(feature_extractor.features, [0, 1])
#     print(len(mean_std))
#     print(mean_std[0].shape)
#     print(mean_std[1].shape)
