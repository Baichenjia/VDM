import time

import numpy as np
import tensorflow as tf
from baselines.common import explained_variance
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import RunningMeanStd
from mpi4py import MPI

from mpi_utils import MpiAdamOptimizer
from rollouts import Rollout
from utils import bcast_tf_vars_from_root, get_mean_and_std
from vec_env import ShmemVecEnv as VecEnv

getsess = tf.get_default_session


class PpoOptimizer(object):
    envs = None

    def __init__(self, *, scope, ob_space, ac_space, stochpol, ent_coef, gamma, lam,
                 nepochs, lr, cliprange, nminibatches, normrew, normadv,
                 use_news, ext_coeff, int_coeff, nsteps_per_seg, nsegs_per_env, dynamics, nepochs_dvae):
        self.dynamics = dynamics
        with tf.variable_scope(scope):
            self.use_recorder = True
            self.n_updates = 0
            self.scope = scope
            self.ob_space = ob_space                  # Box(84,84,4)
            self.ac_space = ac_space                  # Discrete(4)
            self.stochpol = stochpol                  # cnn policy 对象
            self.nepochs = nepochs                    # 3
            self.lr = lr                              # 1e-4
            self.cliprange = cliprange                # 0.1
            self.nsteps_per_seg = nsteps_per_seg      # 128
            self.nsegs_per_env = nsegs_per_env        # 1
            self.nminibatches = nminibatches          # 8
            self.gamma = gamma                        # 0.99  ppo中的参数
            self.lam = lam                            # 0.95  ppo中的参数
            self.normrew = normrew                    # 1
            self.normadv = normadv                    # 1
            self.use_news = use_news                  # False
            self.ext_coeff = ext_coeff                # 0.0     完全使用内在激励进行探索
            self.int_coeff = int_coeff                # 1.0
            self.ph_adv = tf.placeholder(tf.float32, [None, None])
            self.ph_ret = tf.placeholder(tf.float32, [None, None])
            self.ph_rews = tf.placeholder(tf.float32, [None, None])
            self.ph_oldnlp = tf.placeholder(tf.float32, [None, None])    # 记录 -log pi(a|s)
            self.ph_oldvpred = tf.placeholder(tf.float32, [None, None])
            self.ph_lr = tf.placeholder(tf.float32, [])
            self.ph_cliprange = tf.placeholder(tf.float32, [])
            neglogpac = self.stochpol.pd.neglogp(self.stochpol.ph_ac)    # 之前选择的动作在当前策略下的-log值
            entropy = tf.reduce_mean(self.stochpol.pd.entropy())
            vpred = self.stochpol.vpred

            # 定义 PPO 中的损失: critic损失, actor损失, entropy损失, 并近似KL和clip-frac
            # 计算 value function 损失
            vf_loss = 0.5 * tf.reduce_mean((vpred - self.ph_ret) ** 2)
            # 计算 critic 损失
            ratio = tf.exp(self.ph_oldnlp - neglogpac)  # p_new / p_old
            negadv = - self.ph_adv
            pg_losses1 = negadv * ratio
            pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange)
            pg_loss_surr = tf.maximum(pg_losses1, pg_losses2)
            pg_loss = tf.reduce_mean(pg_loss_surr)
            ent_loss = (- ent_coef) * entropy            # 熵约束, ent_coef=0.001
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp))   # 近似 KL
            clipfrac = tf.reduce_mean(tf.to_float(tf.abs(pg_losses2 - pg_loss_surr) > 1e-6))

            self.total_loss = pg_loss + ent_loss + vf_loss
            self.to_report = {'tot': self.total_loss, 'pg': pg_loss, 'vf': vf_loss, 'ent': entropy,
                              'approxkl': approxkl, 'clipfrac': clipfrac}

            # add bai.
            self.dynamics_loss = None
            self.nepochs_dvae = nepochs_dvae

    def start_interaction(self, env_fns, dynamics, nlump=2):
        # 在开始与环境交互时定义变量和计算图, 初始化 rollout 类
        self.loss_names, self._losses = zip(*list(self.to_report.items()))

        # 定义损失、梯度和反向传播.  在训练时调用 sess.run(self._train) 进行迭代
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        params_dvae = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dvae_reward")
        print("total params:", np.sum([np.prod(v.get_shape().as_list()) for v in params]))      # 6629459
        print("dvae params:", np.sum([np.prod(v.get_shape().as_list()) for v in params_dvae]))  # 2726144
        if MPI.COMM_WORLD.Get_size() > 1:
            trainer = MpiAdamOptimizer(learning_rate=self.ph_lr, comm=MPI.COMM_WORLD)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr)
        gradsandvars = trainer.compute_gradients(self.total_loss, params)
        self._train = trainer.apply_gradients(gradsandvars)

        # add bai.  单独计算 DVAE 的梯度
        gradsandvars_dvae = trainer.compute_gradients(self.dynamics_loss, params_dvae)
        self._train_dvae = trainer.apply_gradients(gradsandvars_dvae)

        if MPI.COMM_WORLD.Get_rank() == 0:
            getsess().run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        bcast_tf_vars_from_root(getsess(), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self.all_visited_rooms = []
        self.all_scores = []
        self.nenvs = nenvs = len(env_fns)        # 默认 128
        self.nlump = nlump                       # 默认 1
        self.lump_stride = nenvs // self.nlump   # 128/1=128
        self.envs = [
            VecEnv(env_fns[l * self.lump_stride: (l + 1) * self.lump_stride], spaces=[self.ob_space, self.ac_space]) for
            l in range(self.nlump)]

        # 该类在 rollouts.py 中定义
        self.rollout = Rollout(ob_space=self.ob_space, ac_space=self.ac_space, nenvs=nenvs,
                               nsteps_per_seg=self.nsteps_per_seg,
                               nsegs_per_env=self.nsegs_per_env, nlumps=self.nlump,
                               envs=self.envs,
                               policy=self.stochpol,
                               int_rew_coeff=self.int_coeff,
                               ext_rew_coeff=self.ext_coeff,
                               record_rollouts=self.use_recorder,
                               dynamics=dynamics)

        # 环境数(线程数), 周期T
        self.buf_advs = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        for env in self.envs:
            env.close()

    def calculate_advantages(self, rews, use_news, gamma, lam):
        # 这里根据存储的奖励更新 return 和 advantage(GAE), 但写的有点复杂.
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0   从后向前
            nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
            if not use_news:
                nextnew = 0
            nextvals = self.rollout.buf_vpreds[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_last
            nextnotnew = 1 - nextnew
            delta = rews[:, t] + gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t]
            self.buf_advs[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
        self.buf_rets[:] = self.buf_advs + self.rollout.buf_vpreds

    def update(self):
        if self.normrew:         # 规约奖励, 根据 MPI 从其余线程获取的信息
            rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews = np.copy(self.rollout.buf_rews)

        # 调用本类的函数, 根据奖励序列 rews 计算 advantage function
        self.calculate_advantages(rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        # 记录一些统计量进行输出
        info = dict(
            advmean=self.buf_advs.mean(),
            advstd=self.buf_advs.std(),
            retmean=self.buf_rets.mean(),
            retstd=self.buf_rets.std(),
            vpredmean=self.rollout.buf_vpreds.mean(),
            vpredstd=self.rollout.buf_vpreds.std(),
            ev=explained_variance(self.rollout.buf_vpreds.ravel(), self.buf_rets.ravel()),
            rew_mean=np.mean(self.rollout.buf_rews),
            rew_mean_norm=np.mean(rews),
            recent_best_ext_ret=self.rollout.current_max
        )
        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret

        # normalize advantages. 对计算得到的 advantage 由 mean 和 std 进行规约.
        if self.normadv:
            m, s = get_mean_and_std(self.buf_advs)
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        def resh(x):
            if self.nsegs_per_env == 1:
                return x
            sh = x.shape
            return x.reshape((sh[0] * self.nsegs_per_env, self.nsteps_per_seg) + sh[2:])

        # 将本类中定义的 placeholder 与 rollout 类中收集的样本numpy 对应起来, 准备作为 feed-dict
        ph_buf = [
            (self.stochpol.ph_ac, resh(self.rollout.buf_acs)),
            (self.ph_rews, resh(self.rollout.buf_rews)),
            (self.ph_oldvpred, resh(self.rollout.buf_vpreds)),
            (self.ph_oldnlp, resh(self.rollout.buf_nlps)),
            (self.stochpol.ph_ob, resh(self.rollout.buf_obs)),   # 以上是rollout在于环境交互中记录的numpy
            (self.ph_ret, resh(self.buf_rets)),                  # 根据 rollout 记录计算得到的 return
            (self.ph_adv, resh(self.buf_advs)),                  # 根据 rollout 记录计算得到的 advantage.
        ]
        ph_buf.extend([
            (self.dynamics.last_ob,
             self.rollout.buf_obs_last.reshape([self.nenvs * self.nsegs_per_env, 1, *self.ob_space.shape]))
        ])
        mblossvals = []          # 记录训练中的损失

        # 训练 Agent 损失
        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                fd = {ph: buf[mbenvinds] for (ph, buf) in ph_buf}     # 构造 feed_dict
                fd.update({self.ph_lr: self.lr, self.ph_cliprange: self.cliprange})
                mblossvals.append(getsess().run(self._losses + (self._train,), fd)[:-1])    # 计算损失, 同时进行更新

        # add bai.  单独再次训练 DVAE
        for tmp in range(self.nepochs_dvae):
            print("额外训练dvae. ", tmp)
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):     # 循环8次
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                fd = {ph: buf[mbenvinds] for (ph, buf) in ph_buf}                       # 构造 feed_dict
                fd.update({self.ph_lr: self.lr, self.ph_cliprange: self.cliprange})
                d_loss, _ = getsess().run([self.dynamics_loss, self._train_dvae], fd)   # 计算dvae损失, 同时进行更新
                print(d_loss, end=", ")
            print("\n")

        mblossvals = [mblossvals[0]]
        info.update(zip(['opt_' + ln for ln in self.loss_names], np.mean([mblossvals[0]], axis=0)))
        info["rank"] = MPI.COMM_WORLD.Get_rank()
        self.n_updates += 1
        info["n_updates"] = self.n_updates
        info.update({dn: (np.mean(dvs) if len(dvs) > 0 else 0) for (dn, dvs) in self.rollout.statlists.items()})
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        info['tps'] = MPI.COMM_WORLD.Get_size() * self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update)
        self.t_last_update = tnow

        return info

    def step(self):
        self.rollout.collect_rollout()    # 收集样本, 计算内在奖励
        update_info = self.update()       # 更新权重
        return {'update': update_info}

    def get_var_values(self):
        return self.stochpol.get_var_values()

    def set_var_values(self, vv):
        self.stochpol.set_var_values(vv)


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


# if __name__ == '__main__':
#     from functools import partial
#     import gym
#     import numpy as np
#     from functools import partial
#     from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
#     from wrappers import MaxAndSkipEnv, ProcessFrame84, StickyActionEnv
#     from cnn_policy import CnnPolicy
#     from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
#     from dynamics import Dynamics
#
#     def make_env_all_params(rank, add_monitor=True):
#         env = gym.make("BreakoutNoFrameskip-v4")
#         assert 'NoFrameskip' in env.spec.id
#         env._max_episode_steps = 4500 * 4
#         env = StickyActionEnv(env)
#         env = MaxAndSkipEnv(env, skip=4)  # 每个动作连续执行4步
#         env = ProcessFrame84(env, crop=False)  # 处理观测
#         env = FrameStack(env, 4)  # 将连续4帧叠加起来作为输入
#         return env
#
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
#                          "pix2pix": JustPixels}['vaesph']
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
#
#     print("Init dynamic.")
#     dynamics_list = []
#     dynamics = Dynamics(auxiliary_task=feature_extractor,
#                         predict_from_pixels=False,
#                         feat_dim=512)
#
#     print("Init PPO Policy.")
#     agent = PpoOptimizer(
#         scope='ppo',
#         ob_space=ob_space,
#         ac_space=ac_space,
#         stochpol=policy,
#         use_news=False,
#         gamma=0.99,
#         lam=0.95,
#         nepochs=3,
#         nminibatches=8,
#         lr=1e-4,
#         cliprange=0.1,
#         nsteps_per_seg=128,
#         nsegs_per_env=1,
#         ent_coef=0.001,
#         normrew=1,
#         normadv=1,
#         ext_coeff=0.001,
#         int_coeff=1.0,
#         dynamics=dynamics
#     )
#     agent.start_interaction(envs, dynamics)
