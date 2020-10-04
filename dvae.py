import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import flatten_two_dims, unflatten_first_dim, getsess
from dvae_model.dvae_model import ProposalNetwork, PriorNetwork, GenerativeNetwork
from dvae_model.prob_utils import normal_parse_params, rec_log_prob
tfd = tfp.distributions


class DvaeDynamics(object):
    def __init__(self, auxiliary_task, reward_type, sample_seeds, feat_dim=512, scope='dvae'):
        assert reward_type in ["kl", "elbo", "elbo_var", "pred_var", "var_mean"]
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim                            # 512
        self.obs = self.auxiliary_task.obs                  # placeholder, shape=(None,None,84,84,4)
        self.last_ob = self.auxiliary_task.last_ob          # placeholder, shape=(None,1,84,84,4)
        self.ac = self.auxiliary_task.ac                    # (None,None)
        self.ac_space = self.auxiliary_task.ac_space        # Discrete(4)
        self.ob_mean = self.auxiliary_task.ob_mean          # shape=(84,84,4)
        self.ob_std = self.auxiliary_task.ob_std            # 标量 1.8
        self.reward_type = reward_type

        # 直接在像素层面建立环境模型, 输入是图像. 然而并非输出也是图像, 输出是 next_obs 在辅助任务中提取的特征.
        self.features = tf.stop_gradient(self.auxiliary_task.features)   # 辅助任务提取特征. (None,None,512)
        self.out_features = tf.stop_gradient(self.auxiliary_task.next_features)    # obs_next 特征
        # self.out_features = self.auxiliary_task.next_features    # ~~ 这是原来的.  obs_next 特征

        # 将动作扩展为one-hot, 增加1维. 原来为(None,None), 扩展后为(None,None,4). 最后一维是动作的维度
        self.ac_pad = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        self.sh = tf.shape(self.ac_pad)

        with tf.variable_scope(self.scope + "_model"):
            self.proposal_network = ProposalNetwork()
            self.prior_network = PriorNetwork()
            self.generative_network = GenerativeNetwork()

        with tf.variable_scope(self.scope + "_reward"):
            self.reward_kl = self.batch_kl(self.features, self.ac_pad, self.out_features)
            self.reward_elbo = -1.*self.batch_iwae(self.features, self.ac_pad, self.out_features, k=sample_seeds, var=False)
            self.reward_elbo_var = self.batch_iwae(self.features, self.ac_pad, self.out_features, k=sample_seeds, var=True)
            self.reward_pred_var = self.batch_pred_var(self.features, self.ac_pad, self.out_features, k=sample_seeds)
            self.reward_var_mean = self.batch_var_mean(self.features, self.ac_pad, self.out_features, k=sample_seeds)

            # 根据传入的参数，选择其中一个 reward 计算标准
            if self.reward_type == 'kl':
                self.reward = self.reward_kl
            elif self.reward_type == 'elbo':
                self.reward = self.reward_elbo
            elif self.reward_type == 'elbo_var':
                self.reward = self.reward_elbo_var
            elif self.reward_type == 'pred_var':
                self.reward = self.reward_pred_var
            elif self.reward_type == 'var_mean':
                self.reward = self.reward_var_mean
            else:
                raise RuntimeError("Invalid reward_type")

        with tf.variable_scope(self.scope + "_loss"):                  # 用于训练环境模型
            elbo, info = self.batch_vlb(self.features, self.ac_pad, self.out_features)
            self.loss = -1.0 * elbo                                    # (None, None)

            # for log test
            self.rec_loss = info["rec_loss"]
            self.kl_loss = info["kl_loss"]
            self.prior_reg_loss = info['prior_reg_loss']

        # add bai. 这两个变量仅在单独测试 dave.py 时才使用
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # self.train_op = self.optimizer.minimize(tf.reduce_mean(self.loss))

    def make_latent_distributions(self, s, ac, s_next):
        """ 输出 proposal 网络和 prior 网络的输出, 输出均是 tfp 分布
        """
        # proposal 网络
        proposal_params = self.proposal_network([s, ac, s_next])
        proposal = normal_parse_params(proposal_params, 1e-3)
        # prior 网络
        prior_params = self.prior_network([s, ac])
        prior = normal_parse_params(prior_params, 1e-3)
        return proposal, prior

    def prior_regularization(self, prior, sigma_mu=1e4, sigma_sigma=1e-4):
        """
            对 prior network 输出的分布进行约束. 在没有该约束的情况下, 模型一般也不会发散.
            该正则项对原损失函数的影响很小, 几乎不影响学习的过程, 推荐使用. 对应于论文 4.3.2 内容
        """
        mu = flatten_two_dims(prior.mean())                 # (None, 128)
        sigma = flatten_two_dims(prior.stddev())            # (None, 128)
        mu_regularise = - tf.reduce_sum(mu ** 2, axis=-1) / (2 * (sigma_mu ** 2))
        sigma_regularise = tf.reduce_sum(tf.math.log(sigma)-sigma, axis=-1) * sigma_sigma
        reg = mu_regularise + sigma_regularise              # shape=(None,)
        return tf.reshape(reg, (self.sh[0], self.sh[1]))    # shape=(None,None)

    def batch_vlb(self, s, ac, s_next):
        """ 输出 variational lower bound, 训练目标是最大化该值. 输出维度 (batch,)
        """
        proposal, prior = self.make_latent_distributions(s, ac, s_next)   # 返回分布
        prior_regularization = self.prior_regularization(prior)      # (None, None)
        latent = proposal.sample()                                   # (None, None, 128) 重参数化并采样
        rec_params = self.generative_network(latent)                 # (None, None, 1024)
        rec_log = rec_log_prob(rec_params, s_next)                   # 预测值分布 log_prob(s_next), shape=(None, None)
        kl = tfp.distributions.kl_divergence(proposal, prior)        # (None, None, 128)
        kl = tf.reduce_sum(kl, axis=-1)                              # (None, None)
        info = {"rec_params": rec_params, "rec_loss": -rec_log, "kl_loss": kl, "prior_reg_loss": -prior_regularization}
        return rec_log - kl + prior_regularization, info             # (None, None)

    def batch_iwae(self, s, ac, s_next, k, var=False):
        """ 从 proposal 中采样, 计算似然概率, 减去 KL-divergence, 得到 ELBO.
            var=False时, 返回 多次采样的 ELBO 的均值, shape=(batch_size, n_steps)
            var=True时,  返回 多次采样的 ELBO 的方差, shape=(batch_size, n_steps)
        """
        proposal, prior = self.make_latent_distributions(s, ac, s_next)
        estimates = []
        for ix in range(k):
            latent = proposal.sample()                         # (None, None, 128) 重参数化并采样
            rec_params = self.generative_network(latent)       # (None, None, 1024)
            rec_prob = rec_log_prob(rec_params, s_next)        # (None, None)

            prior_log_prob = prior.log_prob(latent)                       # (None, None, 128)
            prior_log_prob = tf.reduce_sum(prior_log_prob, axis=-1)       # (None, None)

            proposal_log_prob = proposal.log_prob(latent)                    # (None, None, 128)
            proposal_log_prob = tf.reduce_sum(proposal_log_prob, axis=-1)    # (None, None)

            estimate = rec_prob + prior_log_prob - proposal_log_prob   # (None, None) ELBO
            estimates.append(estimate)

        estimates_tensor = tf.stack(estimates, axis=-1)                # (None, None, k)
        assert len(estimates_tensor.get_shape().as_list()) == 3 and estimates_tensor.get_shape().as_list()[-1] == k

        if not var:                             # 操作相当于在 log 内除以k, 输出 shape=(None, None)
            return tf.math.reduce_logsumexp(estimates_tensor, axis=-1) - tf.math.log(float(k))
        else:
            return tf.nn.moments(estimates_tensor, axes=-1)[-1]    # variance, shape=(None,None)

    def batch_kl(self, s, ac, s_next):
        # 注意在最后一步取了 reduce_mean, 正常应该是 reduce_sum
        proposal, prior = self.make_latent_distributions(s, ac, s_next)     # 返回分布
        kl = tfp.distributions.kl_divergence(proposal, prior)               # (None, None, 128)
        kl = tf.reduce_mean(kl, axis=-1)                                    # (None, None)
        return kl                                                           # 直接可用作内在激励

    def batch_pred_var(self, s, ac, s_next, k):
        """改成采样, 而不是直接取得均值"""
        proposal, prior = self.make_latent_distributions(s, ac, s_next)  # 返回分布
        mean_estimates = []
        for ix in range(k):
            latent = proposal.sample()                          # (None, None, 128) 重参数化并采样
            rec_params = self.generative_network(latent)        # (None, None, 1024)
            rec_distr = normal_parse_params(rec_params, min_sigma=1e-2)

            #mean_estimates.append(rec_distr.sample())           # 1. 采样. (None, None, 512)
            mean_estimates.append(rec_distr.mean())           # 2. 直接把均值作为采样结果. (None, None, 512)

        mean_estimates_tensor = tf.stack(mean_estimates, axis=-1)   # (None, None, 512, k)
        var = tf.nn.moments(mean_estimates_tensor, axes=-1)[-1]     # variance, shape=(None,None,512)
        var_mean = tf.reduce_mean(var, axis=-1)                     # (None, None)
        return var_mean                                             # 可直接用作内在激励

    def batch_var_mean(self, s, ac, s_next, k):
        proposal, prior = self.make_latent_distributions(s, ac, s_next)  # 返回分布
        var_estimates = []
        for ix in range(k):
            latent = proposal.sample()                          # (None, None, 128) 重参数化并采样
            rec_params = self.generative_network(latent)        # (None, None, 1024)
            rec_distr = normal_parse_params(rec_params, min_sigma=1e-2)
            var_estimates.append(rec_distr.variance())          # 每个元素是 (None, None, 512)

        var_estimates_tensor = tf.stack(var_estimates, axis=-1)        # (None, None, 512, k)
        var_mean = tf.reduce_mean(var_estimates_tensor, axis=[2, 3])   # variance, shape=(None,None,512)
        return var_mean                                                # 可直接用作内在激励

    def calculate_reward(self, ob, last_ob, acs):
        """
            这个将在 rollout l-64 和 l-76 中调用, 根据实际交互过程中遇到的状态和动作来计算内在激励.
            init 中的self.loss定义了计算图, 这里讲真实的 ob, last_ob, acs 作为feed_dict, 返回值
            obs 和 act 预测 last_obs, 计算损失. 这里分为多个 trunk 计算, 猜想是显存有限, 无法一次将批量放入

            输入: ob.shape=(128,128,84,84,4), last_ob.shape=(128,1,84,84,4), acs.shape=(128,128)
            输出: shape=(128,128,512)
        """
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        return np.concatenate([getsess().run(self.reward,
                                             {self.obs: ob[sli(i)],
                                              self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

    # --------------------------------------------------------------------------------------
    # 以下的函数作为日志，在实际中并不使用
    # --------------------------------------------------------------------------------------

    def log_train_loss(self, ob, last_ob, acs, session=None):
        """
            输入: ob.shape=(128,128,84,84,4), last_ob.shape=(128,1,84,84,4), acs.shape=(128,128)
            输出: shape=(128,128,512)
        """
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)

        if session is None:
            session = getsess()

        # 输出损失.   shape=(128, 128)
        loss_np = np.concatenate([session.run(
            self.loss, feed_dict={self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                  self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)
        # print("Train Loss: shape =", loss_np.shape, ", mean=", np.mean(loss_np))

        # 输出损失中的各项
        rec_loss_np = np.concatenate([session.run(self.rec_loss, feed_dict={self.obs: ob[sli(i)],
            self.last_ob: last_ob[sli(i)], self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        kl_loss_np = np.concatenate([session.run(self.kl_loss, feed_dict={self.obs: ob[sli(i)],
            self.last_ob: last_ob[sli(i)], self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        prior_reg_loss_np = np.concatenate([session.run(
            self.prior_reg_loss, feed_dict={self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)], self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)
        # print("Train loss shape: rec_loss: ", rec_loss_np.shape, ", kl_loss shape: ", kl_loss_np.shape, ", prior_reg_loss shape: ", prior_reg_loss_np.shape)
        print("DVAE loss:", np.mean(loss_np), ", rec: ", np.mean(rec_loss_np), ", kl: ",
              np.mean(kl_loss_np), ", prior_reg: ", np.mean(prior_reg_loss_np))
        return np.array([np.mean(loss_np), np.mean(rec_loss_np), np.mean(kl_loss_np), np.mean(prior_reg_loss_np)])

    def log_compute_rewards(self, ob, last_ob, acs, session=None):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)

        if session is None:
            session = getsess()

        # 输出内在激励
        rew_kl_np = np.concatenate([session.run(
            self.reward_kl, feed_dict={self.obs: ob[sli(i)],
                                       self.last_ob: last_ob[sli(i)],
                                       self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        rew_elbo_np = np.concatenate([session.run(
            self.reward_elbo, feed_dict={self.obs: ob[sli(i)],
                                         self.last_ob: last_ob[sli(i)],
                                         self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        rew_elbo_var_np = np.concatenate([session.run(
            self.reward_elbo_var, feed_dict={self.obs: ob[sli(i)],
                                             self.last_ob: last_ob[sli(i)],
                                             self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        rew_pred_var_np = np.concatenate([session.run(
            self.reward_pred_var, feed_dict={self.obs: ob[sli(i)],
                                             self.last_ob: last_ob[sli(i)],
                                             self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        rew_var_mean_np = np.concatenate([session.run(
            self.reward_var_mean, feed_dict={self.obs: ob[sli(i)],
                                             self.last_ob: last_ob[sli(i)],
                                             self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)

        # print("Reward shape:", rew_kl_np.shape, rew_elbo_np.shape, rew_elbo_var_np.shape, rew_pred_var_np.shape)
        print("Reward mean: rew_kl: ", np.mean(rew_kl_np),
              ", rew_elbo:", np.mean(rew_elbo_np),
              ", rew_elbo_var:", np.mean(rew_elbo_var_np),
              ", rew_pred_var:", np.mean(rew_pred_var_np),
              ", rew_var_mean:", np.mean(rew_var_mean_np))

        return np.array([np.mean(rew_kl_np), np.mean(rew_elbo_np), np.mean(rew_elbo_var_np),
                         np.mean(rew_pred_var_np), np.mean(rew_var_mean_np)])

#     def _train(self, ob, last_ob, acs, session=None):
#         # 局部执行训练
#         n_chunks = 8
#         n = ob.shape[0]
#         chunk_size = n // n_chunks
#         assert n % n_chunks == 0
#         sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
#
#         if session is None:
#             session = getsess()
#
#         # 执行训练
#         for i in range(n_chunks):
#             session.run(self.train_op, feed_dict={
#                 self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)], self.ac: acs[sli(i)]})
#
#
# if __name__ == '__main__':
#     # 调用 环境模型 在固定的数据集上训练，保存结果训练的损失和奖励的变化至 dvae_model/pic 下.
#     from functools import partial
#     import gym
#     import numpy as np
#     from functools import partial
#     import os
#     from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
#     from wrappers import MaxAndSkipEnv, ProcessFrame84, StickyActionEnv
#     from cnn_policy import CnnPolicy
#     from auxiliary_tasks import FeatureExtractor
#     from utils import setup_tensorflow_session
#
#     def make_env_all_params(rank, add_monitor=True):
#         env = gym.make("MontezumaRevengeNoFrameskip-v4")
#         assert 'NoFrameskip' in env.spec.id
#         env._max_episode_steps = 4500 * 4
#         env = StickyActionEnv(env)
#         env = MaxAndSkipEnv(env, skip=4)       # 每个动作连续执行4步
#         env = ProcessFrame84(env, crop=False)  # 处理观测
#         env = FrameStack(env, 4)               # 将连续4帧叠加起来作为输入
#         return env
#
#     make_env = partial(make_env_all_params, add_monitor=True)
#     # make env
#     env = make_env(0, add_monitor=False)
#     obs = env.reset()
#     print("obs and action space:", np.asarray(obs).shape, env.action_space.sample())
#
#     ob_space, ac_space = env.observation_space, env.action_space
#
#     # 一个随机智能体与环境交互, 计算得到的观测的均值和标准差.
#     from utils import random_agent_ob_mean_std
#     ob_mean, ob_std = random_agent_ob_mean_std(env)
#     print("obs mean:", ob_mean.shape, np.max(ob_mean), np.min(ob_mean))
#     print("obs std:", ob_std.shape, np.max(ob_std), np.min(ob_std))
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
#     feature_extractor = FeatureExtractor(
#         policy=policy, features_shared_with_policy=False, feat_dim=512, layernormalize=False)
#
#     # agent 损失: 包括 actor,critic,entropy 损失; 先在加上 feature 学习时包含的损失
#     print(feature_extractor.loss.shape)
#     # feature_extractor.features.shape=(None,None,512)
#     mean_std = tf.nn.moments(feature_extractor.features, [0, 1])
#     print(len(mean_std))
#     print(mean_std[0].shape)
#     print(mean_std[1].shape)
#
#     print("\n------------\nInit dynamic.\n")
#     dynamics = DvaeDynamics(auxiliary_task=feature_extractor,
#                             reward_type='elbo_var')
#
#     print("done.")
#
#     # 计算内在激励
#     sess = setup_tensorflow_session()
#     sess.run(tf.global_variables_initializer())
#     # 不变的样本用于在每次训练过后测试内在激励的变化
#     obs_np_log = np.load(os.path.join("dvae_model/data", "obs_"+str(21*128)+".npy"))
#     last_obs_np_log = np.load(os.path.join("dvae_model/data", "last_obs_"+str(21*128)+".npy"))
#     acs_np_log = np.load(os.path.join("dvae_model/data", "acs_"+str(21*128)+".npy"))
#
#     train_logs, reward_logs = [], []
#     for epoch in range(10):
#         print("\nepoch: ", epoch)
#         for i in range(1, 21):
#             obs_np = np.load(os.path.join("dvae_model/data", "obs_"+str(i*128)+".npy"))
#             last_obs_np = np.load(os.path.join("dvae_model/data", "last_obs_"+str(i*128)+".npy"))
#             acs_np = np.load(os.path.join("dvae_model/data", "acs_"+str(i*128)+".npy"))
#             # print("feed data:", obs_np.shape, last_obs_np.shape, acs_np.shape)
#
#             # 训练
#             dynamics._train(obs_np, last_obs_np, acs_np, sess)
#
#             # train log
#             train_log = dynamics.log_train_loss(obs_np, last_obs_np, acs_np, sess)
#             train_logs.append(train_log)
#
#             # 计算奖励
#             reward_log = dynamics.log_compute_rewards(obs_np_log, last_obs_np_log, acs_np_log, sess)
#             reward_logs.append(reward_log)
#             print("----------\n")
#
#     train_logs_np = np.stack(train_logs, axis=0)                  # shape=(2000, 4)
#     reward_logs_np = np.stack(reward_logs, axis=0)                # shape=(2000, 5)
#     print("save to disk:", train_logs_np.shape, reward_logs_np.shape)
#
#     np.save("dvae_model/pic/train_logs_cmp1.npy", train_logs_np)
#     np.save("dvae_model/pic/reward_logs_cmp1.npy", reward_logs_np)
