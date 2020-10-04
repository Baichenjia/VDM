# #!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import os
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dvae import DvaeDynamics
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit, StickyActionEnv
import datetime


def start_experiment(**args):
    # 该函数在本文件中定义. 对不同的环境进行各种 wrapper. 这是一个函数, 并没有真正初始化环境
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    # 初始化
    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log, tf_sess, saver, logger_dir = get_experiment_environment(**args)
    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train(saver, logger_dir)


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()    # 初始化 ob_space,ac_space,ob_mean,ob_std, 初始化 self.envs 包含多个环境模型

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        # 在建立环境模型之前, 先初始特征提取器. 定义在 auxiliary_task.py 中. 其中 pix2pix 相当于没有提取特征.
        self.feature_extractor = {"none": FeatureExtractor,    # 默认是none
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]  # 通过hps参数选择一个特征提取器
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        # 初始化 环境模型 的类. 上述定义的 feature_extractor 将作为一个参数传入
        self.dynamics = DvaeDynamics(auxiliary_task=self.feature_extractor,
                                     reward_type=hps['reward_type'],
                                     sample_seeds=hps['sample_seeds'])

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
            dynamics=self.dynamics,                # dynamic 对象
            nepochs_dvae=hps["nepochs_dvae"]       # 额外训练 dynamic 的次数
        )

        # agent 损失: 包括 actor,critic,entropy 损失; 先在加上 feature 学习时包含的损失
        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']

        # dynamic 损失
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']

        # add bai. 单独记录DAVE损失，可能要多次训练DAVE
        self.agent.dynamics_loss = self.agent.to_report['dyn_loss']

        # 计算状态经过辅助任务提取特征的方差, shape=(512,), 下面取 tf.reduce_mean 后是一个标量
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        """
            该 env 仅是为了初始化 ob_space, ac_space, ob_mean, ob_std. 因此在算完之后 del 掉.
            随后初始化 self.envs_per_process 个 env
        """
        env = self.make_env(0, add_monitor=False)
        # ob_space.shape=(84, 84, 4)     ac_space.shape=Discrete(4)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        # 随机智能体与环境交互, 计算观测的均值和标准差. ob_mean.shape=(84,84,4), 是0-255之间的数. ob_std是标量, breakout中为 1.8
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self, saver, logger_dir):
        # 初始化计算图, 初始化 rollout 类
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        previous_saved_tcount = 0
        while True:
            info = self.agent.step()      # 与环境交互一个周期, 收集样本, 计算内在激励, 并训练
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.hps["save_period"] and (int(self.agent.rollout.stats['tcount'] / self.hps["save_freq"]) > previous_saved_tcount):
                previous_saved_tcount += 1
                save_path = saver.save(tf.get_default_session(), os.path.join(logger_dir, "model_"+str(previous_saved_tcount)+".ckpt"))
                print("Periodically model saved in path:", save_path)
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                save_path = saver.save(tf.get_default_session(), os.path.join(logger_dir, "model_last.ckpt"))
                print("Model saved in path:", save_path)
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        if args["stickyAtari"]:               # 在智能体执行动作时增加随机性
            env._max_episode_steps = args['max_episode_steps'] * 4
            env = StickyActionEnv(env)
        else:
            env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)            # 每个动作连续执行4步
        env = ProcessFrame84(env, crop=False)       # 处理观测
        env = FrameStack(env, 4)                    # 将连续4帧叠加起来作为输入
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if not args["stickyAtari"]:
            env = ExtraTimeLimit(env, args['max_episode_steps'])  # 限制了一个周期的最大时间步
        if 'Montezuma' in args['env']:        # 记录智能体的位置, 所在的房间, 已经访问的房间
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':           # 超级马里奥
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":     # 多智能体游戏, Multi-Pong
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    # 初始化 MPI 相关的量
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_dir = './logs/' + datetime.datetime.now().strftime(
        args["env"]+"-"+args["reward_type"]+"-"+str(args["nepochs_dvae"])+"-"+str(args["stickyAtari"])+"-%Y-%m-%d-%H-%M-%S-%f")
    logger_context = logger.scoped_configure(
        dir=logger_dir,
        format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()

    # bai.      新增 saver 用于保存权重
    saver = tf.train.Saver()
    return logger_context, tf_context, saver, logger_dir


""" 以下几个函数都是各种参数的初始化
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
    parser.add_argument('--reward_type', type=str, default="elbo", choices=["elbo", "pred_var"])
    parser.add_argument('--nepochs_dvae', type=int, default=3)         # 是否需要额外的训练 dvae model.
    parser.add_argument('--sample_seeds', type=int, default=10)        # dvae model 计算 IWAE 时采样的 seed 数目.


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
    parser.add_argument('--save_period', action='store_true', default=False)  # 是否周期性的保存权重 1e7
    parser.add_argument('--save_freq', type=int, default=int(1e7))            # 周期性的保存权重 1e7


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

    start_experiment(**args.__dict__)


