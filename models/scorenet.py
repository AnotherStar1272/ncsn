"""
这个文件定义了多种 score network / energy network 结构。

它们的共同目标是：
- 输入图像或低维向量
- 输出与输入同形状的 score（即对数密度对输入的梯度近似）
  或者输出一个能量值 / 标量

这些模型会被不同的 runner 选择使用：
- ScoreNetRunner 常用 ResScore
- baseline / anneal 相关 runner 会更多使用 RefineNet 系列，不在本文件中
- toy / 小尺寸实验会用到这里的 MLPScore、SmallScore 等
"""

import torch.nn as nn
import functools
import torch
from torchvision.models import ResNet
import torch.nn.functional as F
from .pix2pix import init_net, UnetSkipConnectionBlock, get_norm_layer, init_weights, ResnetBlock, \
    UnetSkipConnectionBlockWithResNet


class ConvResBlock(nn.Module):
    """下采样或同分辨率的卷积残差块。

    作用：
    - 作为编码器侧的基本 building block
    - 支持保持尺寸不变，或通过 stride=2 做下采样

    输入输出：
    - 输入是四维图像特征 [B, C, H, W]
    - 输出仍是四维图像特征
    """
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)

        if not resize:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            # 不下采样时，主分支输出与输入保持同一空间大小和通道数。
            h = self.main(inputs)
            # 直接与输入做残差相加，形状保持 [B, C, H, W]。
            h += inputs
        else:
            # 下采样时，主分支会把特征图分辨率缩小一半，并调整通道数。
            h = self.main(inputs)
            # shortcut 分支同步做一次下采样，保证可以和主分支相加。
            res = self.residual(inputs)
            h += res
        # 最后再经过一次激活，输出当前残差块的结果。
        return self.final_act(h)


class DeconvResBlock(nn.Module):
    """上采样或同分辨率的反卷积残差块。

    作用：
    - 与 ConvResBlock 对应，主要用于解码器侧
    - 支持保持分辨率，或通过反卷积做上采样
    """
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, True)

        if not resize:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            # 不上采样时，主分支保持输入输出分辨率一致。
            h = self.main(inputs)
            h += inputs
        else:
            # 上采样时，主分支把空间分辨率放大一倍，并调整通道数。
            h = self.main(inputs)
            # shortcut 分支也做同样的上采样，保证残差加法维度一致。
            res = self.residual(inputs)
            h += res
        return self.final_act(h)


class ResScore(nn.Module):
    """残差卷积-反卷积风格的 score network。

    作用：
    - 先编码输入图像，再解码回原分辨率
    - 输出与输入同形状的 score field

    输入：
    - 一般是 [B, 3, H, W] 图像

    输出：
    - 与输入同尺寸的张量，表示每个像素位置的 score 估计
    """
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.ndf = config.model.ndf
        act = 'elu'
        self.convs = nn.Sequential(
            nn.Conv2d(3, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, resize=True, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act),
        )

        self.deconvs = nn.Sequential(
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(4 * self.ndf, 4 * self.ndf, act=act),
            DeconvResBlock(4 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            DeconvResBlock(2 * self.ndf, self.ndf, resize=True, act=act),
            DeconvResBlock(self.ndf, self.ndf, act=act),
            nn.Conv2d(self.ndf, 3, 3, 1, 1)
        )

    def forward(self, x):
        # 输入图像先从 (0, 1) 映射到 (-1, 1)。
        x = 2 * x - 1.
        # 先经过编码器卷积残差块，空间分辨率逐步下降、通道数逐步增加。
        # 例如 32x32 输入大致会变成更小分辨率的高维特征图。
        encoded = self.convs(x)
        # 再经过解码器反卷积残差块，逐步恢复到接近输入的空间尺寸。
        res = self.deconvs(encoded)
        res = self.deconvs(self.convs(x))
        return res


class ResNetScore(nn.Module):
    """基于 ResNet generator 风格改造的 score network。

    结构特点：
    - 前面少量下采样
    - 中间若干个 ResNet block
    - 后面再上采样回原尺寸

    作用：
    - 适合作为图像到图像的残差变换网络
    - 在这里被拿来近似 score function
    """

    def __init__(self, config):
        """根据配置构造一个 ResNet 风格的 score 模型。"""
        super().__init__()

        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf * 2
        n_blocks = 6
        norm_layer = get_norm_layer('instance')
        use_dropout = False
        padding_type = 'reflect'
        assert (n_blocks >= 0)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ELU()]

        n_downsampling = 1
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ELU()]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ELU()]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # 把图像从 (0, 1) 映射到 (-1, 1) 后送入 ResNet generator 主体。
        input = 2 * input - 1.
        # 输出与输入同分辨率、同通道数，用作 score 预测结果。
        return self.model(input)


class UNetResScore(nn.Module):
    """带 ResNet 风格子块的 U-Net score network。

    作用：
    - 通过 encoder-decoder + skip connection 保留多尺度信息
    - 适合做图像级别的 score 预测

    输出：
    - 与输入图像同大小的 score 张量
    """
    def __init__(self, config):
        """构造一个带 skip connection 的 U-Net 结构。"""
        super().__init__()
        # construct unet structure
        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf
        self.config = config
        norm_layer = get_norm_layer('instance')
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
        #                                      innermost=True)  # add the innermost layer

        # for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
        #                                          norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockWithResNet(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlockWithResNet(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

        # init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, input):
        """Standard forward"""
        # 如果训练不是在 logit 空间中进行，这里会先把输入映射到 (-1, 1)。
        if not self.config.data.logit_transform:
            input = 2 * input - 1.
        # U-Net 主体输出与输入同大小的张量，表示每个位置的 score。
        return self.model(input)


class UNetScore(nn.Module):
    """标准 U-Net 风格的 score network。

    作用：
    - 根据输入图像大小动态选择 U-Net 深度
    - 输出与输入同形状的 score
    """
    def __init__(self, config):
        """构造普通版 U-Net score 模型。"""
        super().__init__()
        # construct unet structure
        input_nc = output_nc = config.data.channels
        ngf = config.model.ngf
        self.config = config
        norm_layer = get_norm_layer('instance')
        # unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
        #                                      innermost=True)  # add the innermost layer

        # for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
        #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
        #                                          norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        # unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
        #                                      norm_layer=norm_layer)
        if config.data.image_size == 32:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
        elif config.data.image_size == 16:
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                 norm_layer=norm_layer, innermost=True)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf * 2, input_nc=input_nc, submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

        # init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, input):
        """Standard forward"""
        # 与上面的 UNetResScore 相同，必要时先把输入映射到 (-1, 1)。
        if not self.config.data.logit_transform:
            input = 2 * input - 1.
        # 经过 U-Net 编码器、瓶颈和解码器后，输出与输入同形状的 score 图。
        return self.model(input)


class ResEnergy(nn.Module):
    """卷积式 energy network。

    作用：
    - 输入图像
    - 输出每个样本对应的一个标量能量

    与 score network 的区别：
    - score network 输出与输入同形状的梯度场
    - energy network 先输出一个标量，再可通过对输入求导得到 score
    """
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.ndf = config.model.ndf
        act = 'softplus'
        self.convs = nn.Sequential(
            nn.Conv2d(1, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act)
        )

    def forward(self, x):
        # 先把图像值域映射到 (-1, 1)。
        x = 2 * x - 1.
        # 经过卷积编码器后得到高层特征图，空间尺寸会变小、通道数会增加。
        res = self.convs(x)
        # 展平并在特征维度求平均，最后每个样本输出一个标量能量，形状是 [batch]。
        res = res.view(res.shape[0], -1).mean(dim=-1)
        return res


class MLPScore(nn.Module):
    """适用于小尺寸输入的多层感知机 score network。

    作用：
    - 把输入展平成向量
    - 用全连接网络预测 score
    - 常用于 toy setting 或非常小分辨率输入
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.main = nn.Sequential(
            nn.Linear(10 * 10, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 100),
            nn.LayerNorm(100)
        )

    def forward(self, x):
        # 把输入展平成 [batch, 100] 一类的二维张量，方便送入 MLP。
        x = x.view(x.shape[0], -1)
        if x.is_cuda and self.config.training.ngpu > 1:
            # 多 GPU 时用 data_parallel 在不同卡上并行执行全连接网络。
            score = nn.parallel.data_parallel(
                self.main, x, list(range(self.config.training.ngpu)))
        else:
            score = self.main(x)

        # 再把输出 reshape 回 [batch, 1, 10, 10]，与原输入空间布局一致。
        return score.view(x.shape[0], 1, 10, 10)


class LargeScore(nn.Module):
    """较大容量的卷积 score network。

    作用：
    - 通过 U-Net 风格卷积结构提取空间特征
    - 再通过全连接层做全局整合
    - 适合 28x28 这类较标准的小图像输入
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef
        self.u_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(config.data.channels, nef, 16, stride=2, padding=2),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus()
            nn.ELU()
            # state size. (nc) x 28 x 28
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 28 * 28, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, config.data.channels * 28 * 28)
        )

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            # 多 GPU 时并行跑卷积 U-Net 主体。
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        # 先把卷积输出展平，再通过全连接层整合全局信息。
        # 最终 reshape 成 [batch, channels, 28, 28]，输出和输入图像同大小。
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, 28, 28)
        return score


class Score(nn.Module):
    """中等规模的卷积 score network。

    作用：
    - 与 LargeScore 结构相似，但卷积配置略有不同
    - 常作为基础版 score 模型使用
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef
        self.u_net = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(config.data.channels, nef, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.Conv2d(nef, nef * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.Conv2d(nef * 2, nef * 4, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 4),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*4) x 3 x 3
            nn.ConvTranspose2d(nef * 4, nef * 2, 5, stride=1, padding=0),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 7 x 7
            nn.ConvTranspose2d(nef * 2, nef, 4, stride=2, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef) x 14 x 14
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus()
            nn.ELU()
            # state size. (nc) x 28 x 28
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 28 * 28, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Linear(1024, config.data.channels * 28 * 28)
        )

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            # 多 GPU 并行执行卷积主干。
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        # 卷积特征经全连接层后，恢复成与输入同大小的 score 图。
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, 28, 28)
        return score


class SmallScore(nn.Module):
    """适用于更小输入分辨率的轻量级 score network。

    作用：
    - 面向 10x10 这种更小的图像/网格输入
    - 参数更少，计算也更轻
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        nef = config.model.nef * 4
        self.u_net = nn.Sequential(
            # input is (nc) x 10 x 10
            nn.Conv2d(config.data.channels, nef, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.GroupNorm(4, nef),
            nn.ELU(),
            # state size. (nef) x 6 x 6
            nn.Conv2d(nef, nef * 2, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef * 2),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 6 x 6
            nn.ConvTranspose2d(nef * 2, nef, 3, stride=1, padding=1),
            nn.GroupNorm(4, nef),
            # nn.Softplus(),
            nn.ELU(),
            # state size. (nef*2) x 6 x 6
            nn.ConvTranspose2d(nef, config.data.channels, 4, stride=2, padding=1),
            # nn.Softplus(),
            nn.ELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(config.data.channels * 10 ** 2, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, config.data.channels * 10 ** 2)
        )

    def forward(self, x):
        if x.is_cuda and self.config.training.ngpu > 1:
            # 小模型同样支持多 GPU 并行。
            score = nn.parallel.data_parallel(
                self.u_net, x, list(range(self.config.training.ngpu)))
        else:
            score = self.u_net(x)
        # 最终 reshape 成 [batch, channels, 10, 10]，适配小尺寸输入。
        score = self.fc(score.view(x.shape[0], -1)).view(
            x.shape[0], self.config.data.channels, 10, 10)
        return score
