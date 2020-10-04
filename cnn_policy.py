import tensorflow as tf
from baselines.common.distributions import make_pdtype

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim


class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize, ob_mean, ob_std, feat_dim,
                 layernormalize, nl, scope="policy"):
        """ ob_space: (84,84,4);        ac_space: 4;
            ob_mean.shape=(84,84,4);    ob_std=1.7是标量;            hidsize: 512;
            feat_dim: 512;              layernormalize: False;      nl: tf.nn.leaky_relu.
        """
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)       # 离散动作空间为soft-max分布, 连续状态空间为高斯分布
            self.ph_ob = tf.placeholder(dtype=tf.int32, shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')   # 初始化
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]    # breakout中等于4. 维度, 在soft-max情况下等于动作空间的维度

            sh = tf.shape(self.ph_ob)                 # ph_ob.shape = (None,None,84,84,4)
            x = flatten_two_dims(self.ph_ob)          # x.shape = (None,84,84,4) 将前2维合并
            self.flat_features = self.get_features(x, reuse=False)        # shape=(None,512)
            self.features = unflatten_first_dim(self.flat_features, sh)   # shape=(None,None,512)

            # 定义策略网络和值函数网络. 其输入时已经提取过特征的 feature, 而不是原始的输入.
            with tf.variable_scope(scope, reuse=False):
                x = fc(self.flat_features, units=hidsize, activation=activ)   # activ=tf.nn.relu
                x = fc(x, units=hidsize, activation=activ)                    # 分成 策略和值函数
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)          # 动作logits, shape=(None,4)
                vpred = fc(x, name='value_function_output', units=1, activation=None)   # 值函数, 线性单元, shape=(None,1)
            pdparam = unflatten_first_dim(pdparam, sh)             # shape=(None,None,4)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]   # 值函数, 由于最后一维为1, 因此不要. shape=(None,None)
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)  # 策略输出softmax分布. 有mean,neglogp,kl,entropy,sample等函数
            self.a_samp = pd.sample()           # 采样动作,int型 (None,None), 每个位置是标量
            self.entropy = pd.entropy()         # 熵. (None,None)
            self.nlp_samp = pd.neglogp(self.a_samp)      # -log pi(a|s)  (None,None)

    def get_features(self, x, reuse):                    # 写法同 auxiliary_task 中的 get_features
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob):
        # ob.shape=(128,84,84,1),  在作为 feed_dict 之前增加一个维度 ob[:,None].shape=(128,1,84,84,4)
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None]})
        # 输出 a.shape = vpred.shape = nlp.shape = (128,1)
        return a[:, 0], vpred[:, 0], nlp[:, 0]

#
# if __name__ == '__main__':
#     import gym
#     import numpy as np
#     from functools import partial
#     from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
#     from wrappers import MaxAndSkipEnv, ProcessFrame84, StickyActionEnv
#
#     def make_env_all_params(rank, add_monitor=True):
#         env = gym.make("BreakoutNoFrameskip-v4")
#         assert 'NoFrameskip' in env.spec.id
#         env._max_episode_steps = 4500 * 4
#         env = StickyActionEnv(env)
#         env = MaxAndSkipEnv(env, skip=4)        # 每个动作连续执行4步
#         env = ProcessFrame84(env, crop=False)   # 处理观测
#         env = FrameStack(env, 4)                # 将连续4帧叠加起来作为输入
#         return env
#
#     make_env = partial(make_env_all_params, add_monitor=True)
#     env = make_env(0, add_monitor=False)
#     obs = env.reset()
#     print(np.asarray(obs).shape, env.action_space.sample())
#
#     ob_space, ac_space = env.observation_space, env.action_space
#     # 一个随机智能体与环境交互, 计算得到的观测的均值和标准差.
#     from utils import random_agent_ob_mean_std
#     ob_mean, ob_std = random_agent_ob_mean_std(env)
#     print(ob_mean.shape, np.max(ob_mean), np.min(ob_mean))
#     print(ob_std.shape, np.max(ob_std), np.min(ob_std))
#
#     # 初始化环境
#     envs = [partial(make_env, i) for i in range(5)]
#
#     # CNN policy
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
#     # 调用其中的特征提取模块
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     obs_feed = np.expand_dims(np.expand_dims(np.asarray(obs), axis=0), axis=0)    # (1,1,84,84,4)
#     print(sess.run([policy.features, policy.a_samp, policy.entropy, policy.nlp_samp],
#                    feed_dict={policy.ph_ob: obs_feed}))
#     print("done")
