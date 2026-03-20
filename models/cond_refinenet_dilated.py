"""
这个文件定义条件版 RefineNet Dilated 模型，主要用于 NCSN / AnnealRunner。

和 baseline 版最大的区别：
- forward 需要额外输入 y
- y 通常表示噪声等级类别（不同 sigma 对应的离散标签）
- 因此一个模型就可以学习多个噪声尺度下的 score

输出含义：
- 仍然是与输入图像同尺寸的 score tensor
- 但这是“条件在某个噪声等级 y 下”的 score
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    # 常用 3x3 卷积封装。
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    # 常用 1x1 卷积封装。
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)


class ConditionalBatchNorm2d(nn.Module):
    # 按标签 y 调制缩放与偏移的 BatchNorm。
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        # 先对输入特征做 BatchNorm，形状保持 [B, C, H, W]。
        out = self.bn(x)
        if self.bias:
            # 从标签 y 的嵌入中切出缩放和偏移参数，形状约为 [B, C]。
            gamma, beta = self.embed(y).chunk(2, dim=1)
            # reshape 成 [B, C, 1, 1] 后广播到整张特征图。
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            # 无偏置版本只做缩放调制。
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ConditionalInstanceNorm2d(nn.Module):
    # 按标签 y 调制的 InstanceNorm。
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        # 先做 InstanceNorm，输出形状仍为 [B, C, H, W]。
        h = self.instance_norm(x)
        if self.bias:
            # 按条件标签 y 取出每个样本的缩放和偏移系数。
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class CRPBlock(nn.Module):
    # 无条件版本的 chained residual pooling。
    def __init__(self, features, n_stages, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x):
        # 激活后进入残差池化链，形状保持 [B, C, H, W]。
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            # 池化增强上下文感受野，但不改空间尺寸。
            path = self.maxpool(path)
            # 卷积提炼池化后的局部模式。
            path = self.convs[i](path)
            # 与主分支相加，形成链式残差累积。
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    # 带条件归一化的 chained residual pooling。
    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        # 先激活输入，准备进入条件残差池化链。
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            # 条件归一化让不同标签 y 产生不同的特征调制。
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondRCUBlock(nn.Module):
    # 带条件归一化的 Residual Convolutional Unit。
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1),
                        conv3x3(features, features, stride=1, bias=False))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x, y):
        for i in range(self.n_blocks):
            # 保留 block 输入，最后做残差相加。
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            x += residual
        return x


class CondMSFBlock(nn.Module):
    # Multi-Scale Fusion：
    # 把不同尺度特征在条件 y 下归一化并融合到同一分辨率。
    def __init__(self, in_planes, features, num_classes, normalizer):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        # 创建融合张量，目标尺寸为 [B, features, H_out, W_out]。
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            # 先按条件 y 做归一化，再投影到统一通道数。
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            # 上采样/下采样到统一分辨率，便于后续相加融合。
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class CondRefineBlock(nn.Module):
    # 条件版 RefineBlock，用于解码阶段的多尺度特征融合。
    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act)
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            # 先把每一路输入适配成更适合融合的特征表示。
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            # 多路输入时做多尺度融合，统一输出大小。
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        # 链式池化补充大感受野上下文。
        h = self.crp(h, y)
        # 最后一组 RCU 细化融合结果。
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    # 卷积后平均池化的下采样模块。
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        else:
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            )

    def forward(self, inputs):
        # 先卷积，再做 2 倍平均下采样。
        output = self.conv(inputs)
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output


class MeanPoolConv(nn.Module):
    # 平均池化后卷积的下采样模块。
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        # 先通过步长式切片平均实现 2 倍下采样。
        output = inputs
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        # 再卷积做通道映射与特征抽取。
        return self.conv(output)


class UpsampleConv(nn.Module):
    # 上采样后卷积的恢复分辨率模块。
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        # 先把通道复制 4 份，为 PixelShuffle 的 2 倍上采样准备输入。
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        # 上采样后空间尺寸扩大一倍，通道被重新整理。
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    # 条件残差块：
    # 每一层都会使用 y 做条件归一化，因此可按噪声等级切换行为。
    def __init__(self, input_dim, output_dim, num_classes, resample=None, act=nn.ELU(),
                 normalization=ConditionalBatchNorm2d, adjust_padding=False, dilation=None):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        # 主分支第一次条件归一化。
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        # 第二次条件归一化与卷积，形成残差主分支输出。
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ConditionalInstanceNorm2dPlus(nn.Module):
    # ConditionalInstanceNorm2d 的增强版本。
    # 相比普通条件归一化，还利用了通道均值统计信息做调制。
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        # 先提取每个样本、每个通道的空间均值，得到 [B, C]。
        means = torch.mean(x, dim=(2, 3))
        # 对这些通道均值再做标准化，得到额外的统计调制信号。
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        # 常规 InstanceNorm 后张量仍为 [B, C, H, W]。
        h = self.instance_norm(x)

        if self.bias:
            # 从 y 的嵌入中得到缩放/统计调制/偏移三组参数。
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class CondRefineNetDilated(nn.Module):
    # 条件版 RefineNet 主模型。
    #
    # 输入：
    # - x: 图像张量 [B, C, H, W]
    # - y: 条件标签 [B]，通常表示 sigma 的离散类别
    #
    # 输出：
    # - 与输入同尺寸的 score 张量
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        # self.norm = ConditionalInstanceNorm2d
        self.norm = ConditionalInstanceNorm2dPlus
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = nn.ELU()
        # self.act = act = nn.ReLU(True)

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, start=True)
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            # 顺序执行一个 stage 内的多个条件残差块。
            x = m(x, y)
        return x

    def forward(self, x, y):
        if not self.logit_transform:
            # 若原输入在 [0, 1]，先映射到 [-1, 1]。
            x = 2 * x - 1.

        # 输入图像先映射到 ngf 通道，得到浅层特征 [B, ngf, H, W]。
        output = self.begin_conv(x)

        # 编码阶段逐层提取更深的条件特征。
        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        # 解码/融合阶段，从最深层开始逐步把语义信息传回高分辨率特征。
        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        # 输出头把融合后的高分辨率特征映射回与输入同通道数的 score。
        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output


class CondRefineNetDeeperDilated(nn.Module):
    # 更深层的条件 RefineNet 版本。
    # 相比上面的 CondRefineNetDilated，多了更深的编码层和对应 refine 层。
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.norm = ConditionalInstanceNorm2d
        # self.norm = ConditionalBatchNorm2d
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = nn.ELU()
        # self.act = act = nn.ReLU(True)

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 4 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(4 * self.ngf, 4 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ConditionalResidualBlock(4 * self.ngf, 4 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=4),
            ConditionalResidualBlock(4 * self.ngf, 4 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=4)]
        )

        self.refine1 = CondRefineBlock([4 * self.ngf], 4 * self.ngf, self.num_classes, self.norm, act=act, start=True)
        self.refine2 = CondRefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine5 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            # 依次通过当前 stage 的每个条件残差块。
            x = m(x, y)
        return x

    def forward(self, x, y):
        if not self.logit_transform:
            # 输入归一化到 [-1, 1] 区间。
            x = 2 * x - 1.

        # 初始卷积得到浅层特征 [B, ngf, H, W]。
        output = self.begin_conv(x)

        # 五层编码器逐步降低分辨率、提高语义抽象程度。
        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)
        layer5 = self._compute_cond_module(self.res5, layer4, y)

        # 五层 refine 逐步把深层语义和浅层细节融合回高分辨率表示。
        ref1 = self.refine1([layer5], y, layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], y, layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], y, layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], y, layer2.shape[2:])
        output = self.refine5([layer1, ref4], y, layer1.shape[2:])

        # 末端归一化、激活并映射回输入通道数，得到最终 score 图。
        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output
