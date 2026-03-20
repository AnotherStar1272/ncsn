"""
这个文件主要定义若干 toy distribution / toy density model，主要服务于二维或低维实验。

整体可以分成两类对象：
1. 真实分布（teacher distribution）
   例如 GaussianDist、GMMDist、GMMDistAnneal、Square。
   这些类通常提供 sample() 和 log_prob()/log_pdf()，用于生成训练数据或计算真实密度。
2. 可学习分布模型
   例如 GMM、Gaussian、Gaussian4SVI。
   这些类继承自 nn.Module，内部参数是可训练的，通常输出对数概率或分布参数。

这些模块主要被 toy runner 一类的实验代码使用，用来验证 score matching、Langevin dynamics
等方法在简单分布上的行为是否正确。
"""

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import torch.autograd as autograd


class GaussianDist(object):
    """各向同性或病态协方差的高斯分布。

    作用：
    - 构造一个显式已知的多元高斯分布，便于 toy 实验中直接采样或计算真实 log density。

    输入：
    - dim: 分布维度
    - ill_conditioned: 是否构造病态协方差矩阵

    输出：
    - sample(n): 返回 n 个样本
    - log_pdf(x): 返回输入 x 的对数概率密度
    """
    def __init__(self, dim, ill_conditioned):
        cov = torch.eye(dim)
        # cov = torch.range(1, dim).diag()
        if ill_conditioned:
            cov[dim // 2:, dim // 2:] = 0.0001 * torch.eye(dim // 2)
        # mean = 0 * torch.ones(dim)
        mean = torch.range(1, dim) / 10
        m = MultivariateNormal(mean, cov)
        self.gmm = m

    def sample(self, n):
        return self.gmm.sample(n)

    def log_pdf(self, x):
        return self.gmm.log_prob(x)

class GMMDistAnneal(object):
    """用于 annealed sampling 实验的高斯混合分布。

    作用：
    - 提供一个简单但多峰的目标分布
    - 支持在给定 sigma 下计算 log_prob 和 score
    - 常用于观察不同噪声尺度下的采样轨迹

    输出：
    - sample(n, sigma): 从带噪的混合高斯中采样
    - log_prob(samples, sigma): 计算带噪条件下的对数密度
    - score(samples, sigma): 直接通过自动求导给出真实 score
    """
    def __init__(self, dim):
        self.mix_probs = torch.tensor([0.8, 0.2])
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        self.sigma = 1

    def sample(self, n, sigma=1):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        return torch.randn_like(means) * sigma + means


    def log_prob(self, samples, sigma=1):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * sigma ** 2) - 0.5 * np.log(
                2 * np.pi * sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp

    def score(self, samples, sigma=1):
        with torch.enable_grad():
            samples = samples.detach()
            samples.requires_grad_(True)
            log_probs = self.log_prob(samples, sigma).sum()
            return autograd.grad(log_probs, samples)[0]


class GMMDist(object):
    """标准的高斯混合分布（GMM）teacher。

    作用：
    - 为 toy experiment 提供一个解析形式明确、可采样的多峰数据分布
    - 常被用作训练 score network 的真实数据来源

    输出：
    - sample(n): 采样
    - log_prob(samples): 返回样本在真实分布下的对数概率
    """
    def __init__(self, dim):
        self.mix_probs = torch.tensor([0.8, 0.2])
        # self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        # self.mix_probs = torch.tensor([0.1, 0.1, 0.8])
        # self.means = torch.stack([5 * torch.ones(dim), torch.zeros(dim), -torch.ones(dim) * 5], dim=0)
        self.means = torch.stack([5 * torch.ones(dim), -torch.ones(dim) * 5], dim=0)
        self.sigma = 1
        self.std = torch.stack([torch.ones(dim) * self.sigma for i in range(len(self.mix_probs))], dim=0)

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append((-((samples - self.means[i]) ** 2).sum(dim=-1) / (2 * self.sigma ** 2) - 0.5 * np.log(
                2 * np.pi * self.sigma ** 2)) + self.mix_probs[i].log())
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp


class Square(object):
    """二维均匀方形分布。

    作用：
    - 提供一个边界明确、密度很简单的 toy 分布
    - 可以用来测试模型是否能学到带硬边界的数据分布
    """
    def __init__(self, range=4.):
        self.range = range

    def sample(self, n):
        n = n[0]
        rands = torch.rand(n, 2)
        samples = (rands - 0.5) * self.range * 2
        return samples

    def log_prob(self, samples):
        range_th = torch.tensor(self.range)
        idx = (samples[:, 0] <= range_th) & (samples[:, 0] >= -range_th) & (samples[:, 1] <= range_th) & (samples[:, 1] >= -range_th)
        results = torch.zeros(samples.shape[0])
        results[~idx] = -1e10
        results[idx] = np.log(1 / (self.range * 2) ** 2)

        return results


class GMM(nn.Module):
    """可学习的高斯混合模型。

    作用：
    - 把混合高斯的均值、方差、混合权重都设成可训练参数
    - forward(X) 输出每个输入样本的 log probability

    输出：
    - 形状通常为 [batch]，表示每个样本的对数概率
    """
    def __init__(self, dim):
        super().__init__()
        self.mean = torch.randn(3, dim)
        self.mean[0, :] += 1
        self.mean[2, :] -= 1
        self.mean = nn.Parameter(self.mean)
        self.log_std = nn.Parameter(torch.randn(3, dim))
        self.mix_logits = nn.Parameter(torch.randn(3))

    def forward(self, X):
        # X: [batch, dim]
        # 先把输入扩成 [batch, num_components, dim]，便于同时和每个高斯分量比较。
        energy = (X.unsqueeze(1) - self.mean) ** 2 / (2 * (2 * self.log_std).exp()) + np.log(
            2 * np.pi) / 2. + self.log_std
        # 对最后一个维度求和后得到每个分量各自的 log probability，形状变成 [batch, num_components]。
        log_prob = -energy.sum(dim=-1)
        # 对混合系数做 log_softmax，得到每个分量的对数混合权重，形状是 [num_components]。
        mix_probs = F.log_softmax(self.mix_logits)
        # 把每个分量的 log probability 加上对应的对数混合权重。
        log_prob += mix_probs
        # 在分量维度做 logsumexp，得到最终混合分布对每个样本的 log probability，形状是 [batch]。
        log_prob = torch.logsumexp(log_prob, dim=-1)
        return log_prob


class Gaussian(nn.Module):
    """可学习的单高斯模型。

    作用：
    - 通过可训练的 mean / log_std 拟合一个简单高斯分布
    - 常用于最基础的密度估计 toy 实验

    输出：
    - 对输入 X 的逐维 log probability 项
    """
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))

    def forward(self, X):
        # X: [batch, dim]
        # 逐维计算高斯分布的能量项，每个位置都会得到一个对应维度的值。
        energy = (X - self.mean) ** 2 / (2 * (2 * self.log_std).exp()) + np.log(2 * np.pi) / 2. + self.log_std
        # 对能量取负后得到逐维 log probability；这里没有在 dim 上求和。
        log_prob = -energy
        return log_prob


class Gaussian4SVI(nn.Module):
    """为随机变分推断（SVI）风格实验准备的高斯参数模块。

    作用：
    - 不直接输出 log_prob
    - forward(X) 只返回当前的 mean 和 log_std 参数
    - 方便外部算法自己决定如何采样、如何构造 ELBO
    """
    def __init__(self, batch_size, dim):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(batch_size, dim))
        self.mean = nn.Parameter(torch.zeros(batch_size, dim))

    def forward(self, X):
        # 这个模块不直接根据 X 计算密度。
        # 它只是把当前可学习的 mean 和 log_std 返回给外部算法使用。
        return self.mean, self.log_std
