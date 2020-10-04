from collections import deque, defaultdict
import numpy as np
from mpi4py import MPI
from recorder import Recorder


class Rollout(object):
    def __init__(self, ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, nlumps, envs, policy,
                 int_rew_coeff, ext_rew_coeff, record_rollouts, dynamics):
        # 默认 int_rew_coeff=1.0, ext_rew_coeff=0.0, record_rollouts=True
        self.nenvs = nenvs                                        # 128
        self.nsteps_per_seg = nsteps_per_seg                      # 128
        self.nsegs_per_env = nsegs_per_env                        # 1
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env    # 128
        self.ob_space = ob_space                                  # Box(84,84,4)
        self.ac_space = ac_space                                  # Discrete(4)
        self.nlumps = nlumps                                      # 1
        self.lump_stride = nenvs // self.nlumps                   # 128
        self.envs = envs
        self.policy = policy
        self.dynamics = dynamics

        # 奖励函数的定义, clip(外部奖励) + 权重 * 内部奖励
        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew

        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)    # (128,128)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)      # (128,128)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)      # (128,128)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)  # (128,128)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)   # (128,128)
        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)   # (128,128,84,84,4)
        self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32)

        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        self.env_results = [None] * self.nlumps
        # self.prev_feat = [None for _ in range(self.nlumps)]
        # self.prev_acs = [None for _ in range(self.nlumps)]
        self.int_rew = np.zeros((nenvs,), np.float32)

        self.recorder = Recorder(nenvs=self.nenvs, nlumps=self.nlumps) if record_rollouts else None
        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_ret = None
        self.all_visited_rooms = []
        self.all_scores = []

        self.step_count = 0

    def collect_rollout(self):
        self.ep_infos_new = []
        for t in range(self.nsteps):      # 循环执行 128 步
            self.rollout_step()
        self.calculate_reward()
        self.update_info()

    def calculate_reward(self):
        """  计算内在激励.
        """
        int_rew = self.dynamics.calculate_reward(ob=self.buf_obs,
                                                 last_ob=self.buf_obs_last,
                                                 acs=self.buf_acs)
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)

        # for debug
        # self.dynamics.log_train_loss(ob=self.buf_obs, last_ob=self.buf_obs_last, acs=self.buf_acs)
        # self.dynamics.log_compute_rewards(ob=self.buf_obs, last_ob=self.buf_obs_last, acs=self.buf_acs)

    def rollout_step(self):
        # 该函数每执行一次, self.step_count加1, n_steps总等于128, t=s=self.step_count
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        for l in range(self.nlumps):       # nclumps=1
            # 执行动作. 返回所有线程获得的样本. obs.shape=(128,84,84,4), 第0维是线程数.
            obs, prevrews, news, infos = self.env_get(l)
            # if t > 0:
            #     prev_feat = self.prev_feat[l]
            #     prev_acs = self.prev_acs[l]
            for info in infos:
                epinfo = info.get('episode', {})
                mzepinfo = info.get('mz_episode', {})
                retroepinfo = info.get('retro_episode', {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if epinfo:
                    if "n_states_visited" in info:
                        epinfo["n_states_visited"] = info["n_states_visited"]
                        epinfo["states_visited"] = info["states_visited"]
                    self.ep_infos_new.append((self.step_count, epinfo))

            # slice(0,128) lump_stride=128
            sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)

            # 获得策略网络输出. 输入观测 obs.shape=(128,84,84,1), 返回 acs,vpreds,nlps 维度均为 (128,)
            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            self.env_step(l, acs)

            # 记录周期交互中的变量
            # self.prev_feat[l] = dyn_feat
            # self.prev_acs[l] = acs
            self.buf_obs[sli, t] = obs          # 观测 obs.shape=(128,84,84,4), 放到第t个时间步
            self.buf_news[sli, t] = news        # ?? shape=(128,)  元素都为 True/False, 一般为False
            self.buf_vpreds[sli, t] = vpreds    # 值函数输出 shape=(128,)
            self.buf_nlps[sli, t] = nlps        # -log pi(a|s), shape=(128,)
            self.buf_acs[sli, t] = acs          # 动作输出 shape=(128,)
            if t > 0:
                self.buf_ext_rews[sli, t - 1] = prevrews     # prevrews.shape=(128,) 全0矩阵
            # if t > 0:
            #     dyn_logp = self.policy.call_reward(prev_feat, pol_feat, prev_acs)
            #     int_rew = dyn_logp.reshape(-1, )
            #     self.int_rew[sli] = int_rew
            #     self.buf_rews[sli, t - 1] = self.reward_fun(ext_rew=prevrews, int_rew=int_rew)
            if self.recorder is not None:
                self.recorder.record(timestep=self.step_count, lump=l, acs=acs, infos=infos, int_rew=self.int_rew[sli],
                                     ext_rew=prevrews, news=news)

        """ 每隔一段时间来评价智能体的行为. 训练时用内在奖励学习, 但评价时需要用外在奖励. 因此记录外在奖励
        """
        self.step_count += 1
        if s == self.nsteps_per_seg - 1:      # 每隔128步(周期结尾)才会执行一次.  nsteps_per_seg=128
            for l in range(self.nlumps):      # nclumps=1
                sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
                # nextobs.shape=(128,84,84,4),  ext_rews.shape=(128,)全0矩阵,  nextnews.shape=(128,)全False矩阵
                nextobs, ext_rews, nextnews, _ = self.env_get(l)
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:          # 如果到周期结尾, t=127
                    self.buf_new_last[sli] = nextnews
                    self.buf_ext_rews[sli, t] = ext_rews     # 记录外在奖励
                    _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(nextobs)    # 记录值函数预测
                    # dyn_logp = self.policy.call_reward(self.prev_feat[l], last_pol_feat, prev_acs)
                    # dyn_logp = dyn_logp.reshape(-1, )
                    # int_rew = dyn_logp
                    #
                    # self.int_rew[sli] = int_rew
                    # self.buf_rews[sli, t] = self.reward_fun(ext_rew=ext_rews, int_rew=int_rew)

    def update_info(self):
        all_ep_infos = MPI.COMM_WORLD.allgather(self.ep_infos_new)
        all_ep_infos = sorted(sum(all_ep_infos, []), key=lambda x: x[0])
        if all_ep_infos:
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}
            # all_ep_infos: {'r': [0.0, 0.0, 0.0], 'l': [124, 125, 127], 't': [6.60745, 12.034875, 10.772788]}

            self.statlists['eprew'].extend(all_ep_infos['r'])
            self.stats['eprew_recent'] = np.mean(all_ep_infos['r'])
            self.statlists['eplen'].extend(all_ep_infos['l'])
            self.stats['epcount'] += len(all_ep_infos['l'])
            self.stats['tcount'] += sum(all_ep_infos['l'])
            if 'visited_rooms' in keys_:
                # Montezuma specific logging.
                self.stats['visited_rooms'] = sorted(list(set.union(*all_ep_infos['visited_rooms'])))
                self.stats['pos_count'] = np.mean(all_ep_infos['pos_count'])
                self.all_visited_rooms.extend(self.stats['visited_rooms'])
                self.all_scores.extend(all_ep_infos["r"])
                self.all_scores = sorted(list(set(self.all_scores)))
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited rooms")
                    print(self.all_visited_rooms)
                    print("All scores")
                    print(self.all_scores)
                # 防止记录中出现问题 bai add
                # self.stats['visited_rooms'] = "|".join([str(x) for x in self.stats['visited_rooms']])
            if 'levels' in keys_:
                # Retro logging
                temp = sorted(list(set.union(*all_ep_infos['levels'])))
                self.all_visited_rooms.extend(temp)
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited levels")
                    print(self.all_visited_rooms)

            current_max = np.max(all_ep_infos['r'])
        else:
            current_max = None
        self.ep_infos_new = []

        # 记录最大奖励 best_ext_ret
        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    def env_step(self, l, acs):
        self.envs[l].step_async(acs)
        self.env_results[l] = None

    def env_get(self, l):
        if self.step_count == 0:
            ob = self.envs[l].reset()
            out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        else:
            if self.env_results[l] is None:
                out = self.env_results[l] = self.envs[l].step_wait()
            else:
                out = self.env_results[l]
        return out
