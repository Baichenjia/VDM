# #!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import tensorflow as tf
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
import numpy as np

from auxiliary_tasks import FeatureExtractor
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dvae import DvaeDynamics
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_multi_pong, \
    AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit, StickyActionEnv

from utils import save_np_as_mp4


def play_experiment(**args):
    # 该函数在本文件中定义. 对不同的环境进行各种 wrapper. 这是一个函数, 并没有真正初始化环境
    make_env = partial(make_env_all_params, args=args)

    # 初始化
    tester = Tester(make_env=make_env,
                    num_timesteps=args['num_timesteps'],
                    hps=args,
                    envs_per_process=1)

    from utils import setup_tensorflow_session
    tf_sess = setup_tensorflow_session()
    saver = tf.train.Saver()

    # model_path
    model_path = './logs/' + args["env"] + "-" + "elbo_var/model_last.ckpt"

    with tf_sess:
        tester.play(tf_sess, args, saver, model_path)


class Tester(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()    # 初始化 ob_space,ac_space,ob_mean,ob_std, 初始化 self.envs 包含多个环境模型

        self.policy = CnnPolicy(scope='pol',
                                ob_space=self.ob_space,
                                ac_space=self.ac_space,
                                hidsize=512,
                                feat_dim=512,
                                ob_mean=self.ob_mean,
                                ob_std=self.ob_std,
                                layernormalize=False,
                                nl=tf.nn.leaky_relu)

        self.feature_extractor = FeatureExtractor(policy=self.policy,
                                                  features_shared_with_policy=False,
                                                  feat_dim=512,
                                                  layernormalize=hps['layernorm'])

        # 初始化 环境模型 的类. 上述定义的 feature_extractor 将作为一个参数传入
        self.dynamics = DvaeDynamics(auxiliary_task=self.feature_extractor,
                                     reward_type=hps['reward_type'])

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics,
            nepochs_dvae=0
        )

        # agent 损失: 包括 actor,critic,entropy 损失; 先在加上 feature 学习时包含的损失
        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']

        # dynamic 损失,  将所有 dynamic 的损失累加起来
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']

        # 计算状态经过辅助任务提取特征的方差, shape=(512,), 下面取 tf.reduce_mean 后是一个标量
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        """
            该 env 仅是为了初始化 ob_space, ac_space, ob_mean, ob_std. 因此在算完之后 del 掉.
            随后初始化 self.envs_per_process 个 env
        """
        env = self.make_env(0)
        # ob_space.shape=(84, 84, 4)     ac_space.shape=Discrete(4)
        self.ob_space, self.ac_space = env.observation_space, env.action_space

        # 随机智能体与环境交互, 计算观测的均值和标准差. ob_mean.shape=(84,84,4), 是0-255之间的数. ob_std是标量, breakout中为 1.8
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        if self.hps["env_kind"] == "unity":
            env.close()
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def play(self, tf_sess, args_tmp, saver, model_path):
        print("model_path: ", model_path)

        with tf_sess.as_default():
            print("Load wights..")
            saver.restore(tf_sess, model_path)
        print("Load done.")

        # rollout
        env = self.make_env(0)
        obs = env.reset()
        rews, frames = [], []
        while True:
            obs = np.expand_dims(np.squeeze(obs), axis=0)
            assert obs.shape == (1, 84, 84, 4)
            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            obs, rew, done, info = env.step(acs[0])
            rews.append(rew)
            obs = np.array(obs)
            frames.append(env.render(mode='rgb_array'))
            if done:
                break
        print("Total rewards:", np.sum(rews))
        save_np_as_mp4(frames, "logs/video/"+args_tmp['env']+'_video.mp4')


def make_env_all_params(rank, args):
    env = None
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        if args["stickyAtari"]:               # 在智能体执行动作时增加随机性
            env._max_episode_steps = args['max_episode_steps'] * 4
            env = StickyActionEnv(env)
        else:
            env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)         # 每个动作连续执行4步
        env = ProcessFrame84(env, crop=False)    # 处理观测
        env = FrameStack(env, 4)                 # 将连续4帧叠加起来作为输入
        if not args["stickyAtari"]:
            env = ExtraTimeLimit(env, args['max_episode_steps'])   # 限制了一个周期的最大时间步.
        if 'Montezuma' in args['env']:           # 记录智能体的位置, 所在的房间, 已经访问的房间
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':            # 超级马里奥
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":      # 多智能体游戏, Multi-Pong
        env = make_multi_pong()
    return env


""" 
    以下几个函数都是各种参数的初始化
"""


def add_environments_params(parser):
    # Hard Games: [ MontezumaRevenge, Freeway, Gravitar, PrivateEye, Venture, Solaris, Pitfall ]
    # Easy Games: [ Asterix, Seaquest, BeamRider, SpaceInvaders, Hero, Breakout, Pong, Qbert, Riverraid ]
    # Mario: nohup python run.py --env mario --env_kind mario > log_mario.txt 2>&1 &   (level1)
    # Multi Pong: nohup python run.py --env multi-pong --env_kind retro_multi > log_multi.txt 2>&1 &
    parser.add_argument('--env', help='environment ID', default="MontezumaRevenge"+"NoFrameskip-v4", type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)
    parser.add_argument('--stickyAtari', action='store_true', default=False)   # 是否增加环境模型的随机性
    parser.add_argument('--reward_type', type=str, default="elbo", choices=["kl", "elbo", "elbo_var", "pred_var", "var_mean"])


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)         # lambda, gamma 用于计算 GAE advantage
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)           # 规约因子
    parser.add_argument('--norm_rew', type=int, default=1)           # 规约因子
    parser.add_argument('--lr', type=float, default=1e-4)            # 学习率
    parser.add_argument('--ent_coeff', type=float, default=0.001)    # 损失中的熵正则因子
    parser.add_argument('--nepochs', type=int, default=3)            # PPO 每次迭代中会循环训练3次
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)   # 默认是128
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)    # 使用特征, 而不是像素建立环境模型
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",    # 特征提取器的选择
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])

    args = parser.parse_args()

    play_experiment(**args.__dict__)


