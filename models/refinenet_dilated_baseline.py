"""
这个文件定义 BaselineRunner 使用的主模型：RefineNetDilated。

整体结构可以理解为：
1. 编码阶段：多层残差块提取不同尺度的特征
2. 膨胀卷积：扩大感受野，帮助模型感知更大范围上下文
3. refine 阶段：把深层与浅层特征逐步融合回来
4. 输出阶段：生成与输入同尺寸的 score tensor

输出含义：
- 对输入图像中每个像素位置，都输出一个与输入同维度的 score 估计
- 在 score matching 训练里，这个输出会被当作对 log p(x) 梯度的近似
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    # 常用 3x3 卷积封装，自动补齐 padding，保持空间尺寸更方便。
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    # 1x1 卷积封装，常用于通道数调整或 shortcut。
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)


class ConditionalBatchNorm2d(nn.Module):
    # 条件 BatchNorm。
    # 虽然 baseline 版本最后不显式传类别标签，但这里保留了与条件版共享的接口形式。
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
        # 先做 BatchNorm，保持张量形状不变，仍为 [B, C, H, W]。
        out = self.bn(x)
        if self.bias:
            # 从条件嵌入里切出缩放系数和偏移系数，形状约为 [B, C]。
            gamma, beta = self.embed(y).chunk(2, dim=1)
            # 将 [B, C] reshape 成 [B, C, 1, 1] 后广播到整张特征图，输出仍为 [B, C, H, W]。
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            # 只有缩放没有偏移时，同样按通道做条件调制。
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class CRPBlock(nn.Module):
    # Chained Residual Pooling：
    # 通过多次池化和卷积累积上下文信息，再用残差方式加回去。
    def __init__(self, features, n_stages, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x):
        # 先激活，形状保持 [B, C, H, W]。
        x = self.act(x)
        # `path` 是残差池化分支的当前特征。
        path = x
        for i in range(self.n_stages):
            # 池化后空间尺寸不变，因为 stride=1 且做了 padding。
            path = self.maxpool(path)
            # 3x3 卷积进一步混合局部上下文，输出仍为 [B, C, H, W]。
            path = self.convs[i](path)
            # 与主分支相加，累计多层上下文信息。
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    # 带条件归一化的 CRPBlock，接口与条件版 RefineNet 保持一致。
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
        # 先激活输入，形状不变。
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            # 结合条件 y 做归一化调制，输出仍为 [B, C, H, W]。
            path = self.norms[i](path, y)
            # 平均池化扩大感受野，但保持空间大小不变。
            path = self.maxpool(path)
            # 卷积整合池化后的上下文特征。
            path = self.convs[i](path)
            # 残差相加，保留原特征并叠加新上下文。
            x = path + x
        return x


class CondRCUBlock(nn.Module):
    # Residual Convolutional Unit：
    # refine 阶段最基本的卷积细化模块，输入输出大小不变。
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
            # 每个 block 都保留一份输入用于残差连接。
            residual = x
            for j in range(self.n_stages):
                # 条件归一化后形状不变。
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                # 激活后继续保持 [B, C, H, W]。
                x = self.act(x)
                # 3x3 卷积细化当前尺度下的特征。
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)
            # 一个 block 结束后做残差相加。
            x += residual
        return x


class CondMSFBlock(nn.Module):
    # Multi-Resolution Fusion：
    # 把不同尺度的特征归一化、卷积、插值到同一大小后再相加融合。
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
        # 新建一个融合缓冲区，目标形状为 [B, features, H_out, W_out]。
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            # 每一路输入先做条件归一化，保持各自原始分辨率。
            h = self.norms[i](xs[i], y)
            # 再用卷积统一到相同的通道数 `features`。
            h = self.convs[i](h)
            # 插值到统一目标空间尺寸，变成 [B, features, H_out, W_out]。
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            # 把不同分辨率分支累加融合。
            sums += h
        return sums


class CondRefineBlock(nn.Module):
    # RefineBlock 是 RefineNet 的关键模块：
    # 负责把不同层级的特征对齐、融合、上下文增强，再输出更干净的特征图。
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
            # 每一路输入先通过 RCU 做通道/语义适配，输出尺寸通常与该路输入一致。
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            # 多路输入时做多尺度融合，统一到 `output_shape`。
            h = self.msf(hs, y, output_shape)
        else:
            # 单路输入时直接使用这一路特征。
            h = hs[0]

        # CRP 继续扩大感受野并聚合上下文。
        h = self.crp(h, y)
        # 最后再经过一组 RCU 输出更干净的融合特征。
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    # 先卷积后做平均池化的下采样模块。
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
        # 先做卷积变换通道，空间尺寸先保持不变。
        output = self.conv(inputs)
        # 再通过四个棋盘格位置求平均实现 2 倍下采样，输出约为 [B, C_out, H/2, W/2]。
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output


class MeanPoolConv(nn.Module):
    # 先平均池化再卷积的下采样模块。
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        # 先对输入做平均池化式的 2 倍下采样。
        output = inputs
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        # 再通过卷积调整通道并提取局部模式。
        return self.conv(output)


class UpsampleConv(nn.Module):
    # 先上采样再卷积的模块，用于恢复空间分辨率。
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        # 先复制 4 份通道，为 PixelShuffle 做准备，通道数变为原来的 4 倍。
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        # PixelShuffle 做 2 倍上采样，空间尺寸变为 [2H, 2W]，通道回到原始规模。
        output = self.pixelshuffle(output)
        # 最后用卷积整合上采样后的局部信息。
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    # 主干残差块：
    # 支持普通残差、下采样残差、以及带空洞卷积的残差形式。
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
        # 第一次条件归一化，不改变形状。
        output = self.normalize1(x, y)
        # 激活后进入第一层卷积。
        output = self.non_linearity(output)
        # 第一层卷积提取特征；若该块负责下采样，这里或下一层会缩小空间尺寸。
        output = self.conv1(output)
        # 第二次条件归一化。
        output = self.normalize2(output, y)
        # 第二次激活。
        output = self.non_linearity(output)
        # 第二层卷积输出主分支结果。
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            # 通道和分辨率都不变时，shortcut 直接使用原输入。
            shortcut = x
        else:
            # 否则用 shortcut 分支做通道匹配或下采样。
            shortcut = self.shortcut(x)

        # 主分支与 shortcut 相加，得到残差块输出。
        return shortcut + output


class InstanceNorm2dPlus(nn.Module):
    # InstanceNorm2d 的增强版本：
    # 除了 instance normalization，还显式利用通道均值统计做调制。
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, y):
        # 先对每个通道求空间均值，得到 [B, C]。
        means = torch.mean(x, dim=(2, 3))
        # 对通道均值再做样本内标准化，得到一份额外的通道统计量。
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        # 常规 InstanceNorm 输出仍为 [B, C, H, W]。
        h = self.instance_norm(x)

        if self.bias:
            # 用标准化后的通道均值重新调制特征，再做可学习缩放和平移。
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            # 无偏置版本只做缩放。
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out


class RefineNetDilated(nn.Module):
    # BaselineRunner 使用的主模型。
    #
    # 输入：
    # - x: [B, C, H, W]
    #
    # 输出：
    # - 与输入同尺寸的张量，表示 score network 对每个位置的梯度估计
    #
    # 主要阶段：
    # - begin_conv: 输入映射到特征空间
    # - res1-res4: 逐步提取更深层的特征
    # - refine1-refine4: 将深层特征逐步融合回高分辨率表示
    # - end_conv: 输出最终 score
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.norm = InstanceNorm2dPlus
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
            # 依次通过一个 ModuleList 里的残差块，形状按该 stage 的定义更新。
            x = m(x, y)
        return x

    def forward(self, x):
        if not self.logit_transform:
            # 若输入仍在 [0, 1]，这里线性映射到 [-1, 1]，形状保持 [B, C, H, W]。
            x = 2 * x - 1.

        # baseline 版本没有显式条件标签，这里保持为 None。
        y = None
        # 初始 3x3 卷积把输入图像映射到 ngf 通道，输出约为 [B, ngf, H, W]。
        output = self.begin_conv(x)

        # 第一层编码，通常保持分辨率不变，输出约为 [B, ngf, H, W]。
        layer1 = self._compute_cond_module(self.res1, output, y)
        # 第二层编码，通常下采样到一半并增广通道，约为 [B, 2*ngf, H/2, W/2]。
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        # 第三层编码，继续抽取更深层语义特征，分辨率进一步减小。
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        # 第四层编码，获得最深层特征，感受野最大。
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        # 从最深层特征开始逐步 refine，输出尺寸与 layer4 一致。
        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        # 将 layer3 与更深层 refine 结果融合，输出尺寸对齐到 layer3。
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        # 继续把中层特征与更深层语义融合，输出尺寸对齐到 layer2。
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        # 最后与浅层高分辨率特征融合，恢复到接近输入分辨率的表示。
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        # 输出头：归一化 + 激活 + 最后一层卷积。
        output = self.normalizer(output, y)
        output = self.act(output)
        # 最终输出与输入图像通道数一致，形状约为 [B, C, H, W]，表示每个像素位置的 score。
        output = self.end_conv(output)
        return output
