import multiprocessing
import os
import platform
from functools import partial

import numpy as np
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from mpi4py import MPI


def bcast_tf_vars_from_root(sess, vars):
    """
    Send the root node's parameters to every worker.

    Arguments:
      sess: the TensorFlow session.
      vars: all parameter variables including optimizer's
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for var in vars:
        if rank == 0:
            MPI.COMM_WORLD.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))


def get_mean_and_std(array):
    comm = MPI.COMM_WORLD
    task_id, num_tasks = comm.Get_rank(), comm.Get_size()
    local_mean = np.array(np.mean(array))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    mean = sum_of_means / num_tasks

    n_array = array - mean
    sqs = n_array ** 2
    local_mean = np.array(np.mean(sqs))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    var = sum_of_means / num_tasks
    std = var ** 0.5
    return mean, std


def guess_available_gpus(n_gpus=None):
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("Couldn't guess the available gpus on this machine")


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    available_gpus = guess_available_gpus()

    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])


def guess_available_cpus():
    return int(multiprocessing.cpu_count())


def setup_tensorflow_session():
    num_cpu = guess_available_cpus()

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu
    )
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)


def random_agent_ob_mean_std(env, nsteps=10000):
    # 智能体采取随机动作执行一些时间步, 记录 observation 的均值和标准差, 用于后续规约
    ob = np.asarray(env.reset())
    if MPI.COMM_WORLD.Get_rank() == 0:
        obs = [ob]
        for _ in range(nsteps):
            ac = env.action_space.sample()
            ob, _, done, _ = env.step(ac)
            if done:
                ob = env.reset()
            obs.append(np.asarray(ob))
        mean = np.mean(obs, 0).astype(np.float32)
        std = np.std(obs, 0).mean().astype(np.float32)
    else:
        mean = np.empty(shape=ob.shape, dtype=np.float32)
        std = np.empty(shape=(), dtype=np.float32)
    MPI.COMM_WORLD.Bcast(mean, root=0)
    MPI.COMM_WORLD.Bcast(std, root=0)
    return mean, std


def layernorm(x):
    m, v = tf.nn.moments(x, -1, keep_dims=True)
    return (x - m) / (tf.sqrt(v) + 1e-8)


getsess = tf.get_default_session

fc = partial(tf.layers.dense, kernel_initializer=normc_initializer(1.))
activ = tf.nn.relu


def flatten_two_dims(x):
    return tf.reshape(x, [-1] + x.get_shape().as_list()[2:])


def unflatten_first_dim(x, sh):
    return tf.reshape(x, [sh[0], sh[1]] + x.get_shape().as_list()[1:])


def add_pos_bias(x):
    with tf.variable_scope(name_or_scope=None, default_name="pos_bias"):
        b = tf.get_variable(name="pos_bias", shape=[1] + x.get_shape().as_list()[1:], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        return x + b


def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(x, units=feat_dim, activation=None))
    if last_nl is not None:
        x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x


def small_deconvnet(z, nl, ch, positional_bias):
    # z.shape=(None,512) 是VAE后验分布采样的结果; nl=tf.nn.leaky_relu; ch=4; positional_bias=True
    sh = (8, 8, 64)
    z = fc(z, np.prod(sh), activation=nl)     # (None,8*8*64)
    z = tf.reshape(z, (-1, *sh))              # (None,8,8,64)
    z = tf.layers.conv2d_transpose(z, 128, kernel_size=4, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [16, 16]      # (None,16,16,128)
    z = tf.layers.conv2d_transpose(z, 64, kernel_size=8, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [32, 32]      # (None,32,32,64)
    z = tf.layers.conv2d_transpose(z, ch, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert z.get_shape().as_list()[1:3] == [96, 96]      # (None,96,96,4)
    z = z[:, 6:-6, 6:-6]
    assert z.get_shape().as_list()[1:3] == [84, 84]      # (None,84,84,4)
    if positional_bias:
        z = add_pos_bias(z)                              # (None,84,84,4)
    return z


def unet(x, nl, feat_dim, cond, batchnorm=False):
    # 直接在像素层面建立模型. 使用卷积网络而不是全连接
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    layers = []
    x = tf.pad(x, [[0, 0], [6, 6], [6, 6], [0, 0]])      # 在图像的 w 和 h 维度进行 padding.
    x = bn(tf.layers.conv2d(cond(x), filters=32, kernel_size=8, strides=(3, 3), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    layers.append(x)
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [8, 8]

    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = fc(cond(x), units=feat_dim, activation=nl)

    def residual(x):
        res = bn(tf.layers.dense(cond(x), feat_dim, activation=tf.nn.leaky_relu))
        res = tf.layers.dense(cond(res), feat_dim, activation=None)
        return x + res

    for _ in range(4):
        x = residual(x)

    sh = (8, 8, 64)
    x = fc(cond(x), np.prod(sh), activation=nl)
    x = tf.reshape(x, (-1, *sh))
    x += layers.pop()                  # 类似于此处的是 U-net 中 encoder 和 decoder 的残差连接
    x = bn(tf.layers.conv2d_transpose(cond(x), 64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 32, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    x += layers.pop()
    x = tf.layers.conv2d_transpose(cond(x), 4, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert x.get_shape().as_list()[1:3] == [96, 96]
    x = x[:, 6:-6, 6:-6]
    assert x.get_shape().as_list()[1:3] == [84, 84]
    assert layers == []
    return x


def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


# 将一段与环境的交互保存成视频.

import distutils.spawn
import subprocess


def save_np_as_mp4(frames, filename):
    print(filename)
    if distutils.spawn.find_executable('avconv') is not None:
        backend = 'avconv'
    elif distutils.spawn.find_executable('ffmpeg') is not None:
        backend = 'ffmpeg'
    else:
        raise NotImplementedError(
            """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

    frames_per_sec = 30
    h, w = frames[0].shape[:2]
    output_path = filename
    cmdline = (backend,
               '-nostats',
               '-loglevel', 'error',  # suppress warnings
               '-y',
               '-r', '%d' % frames_per_sec,

               # input
               '-f', 'rawvideo',
               '-s:v', '{}x{}'.format(w, h),
               '-pix_fmt', 'rgb24',
               '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly

               # output
               '-vcodec', 'libx264',
               '-pix_fmt', 'yuv420p',
               output_path)

    print('saving ', output_path)
    if hasattr(os, 'setsid'):                            # setsid not present on Windows
        process = subprocess.Popen(cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
    else:
        process = subprocess.Popen(cmdline, stdin=subprocess.PIPE)
    process.stdin.write(np.array(frames).tobytes())
    process.stdin.close()
    ret = process.wait()
    if ret != 0:
        print("VideoRecorder encoder exited with status {}".format(ret))

