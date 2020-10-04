import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def normal_parse_params(params, min_sigma=0.0):
    """
    将输入拆分成两份, 分别代表 mean 和 std.
    min_sigma 是对 sigma 最小值的限制
    """
    n = params.shape[0]
    d = params.shape[-1]                    # channel
    mu = params[..., :d // 2]               # 最后一维的通道分成两份, 分别为均值和标准差
    sigma_params = params[..., d // 2:]
    sigma = tf.math.softplus(sigma_params)
    sigma = tf.clip_by_value(t=sigma, clip_value_min=min_sigma, clip_value_max=1e5)
    distr = tfd.Normal(loc=mu, scale=sigma)
    return distr     # proposal 网络的输出 (None,32), mu.shape=(None,16), sigma.shape=(None,16)


def rec_log_prob(batch, x_logit):
    # 计算重建误差. distr_params 包含了均值和标准差参数. ground_truth为图像. mask掩码
    # ground_truth.shape=(None,28,28,1),  distr_params.shape=(None,28,28,1)
    rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=batch)
    rec_loss = tf.reduce_sum(rec_loss, axis=[1, 2, 3])
    return rec_loss
