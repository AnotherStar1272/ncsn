"""
这个文件封装了一个常见的 InceptionV3 特征提取器。

主要用途：
- 复用 torchvision 里的预训练 InceptionV3
- 按 block 输出不同层级的特征图
- 通常用于生成模型评价，例如提取 FID / Inception Score 所需的特征

和直接使用 torchvision.models.inception_v3 的区别：
- 这里只保留特征提取相关部分，不关心最终分类头
- 可以灵活指定输出哪些中间 block 的结果
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):
    """返回若干层特征图的预训练 InceptionV3。

    作用：
    - 输入一批 RGB 图像
    - 输出指定 block 的特征张量列表

    常见输出含义：
    - block 0: 低层特征
    - block 1: 更深一层的卷积特征
    - block 2: aux classifier 之前的特征
    - block 3: 最终全局平均池化后的高层语义特征
    """

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """构造预训练 InceptionV3 特征提取器。

        参数说明：
        - output_blocks: 想返回哪些 block 的输出
        - resize_input: 是否先把输入 resize 到 299x299
        - normalize_input: 是否把输入从 (0, 1) 映射到 (-1, 1)
        - requires_grad: 是否允许这个网络参与梯度更新
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """提取 Inception 特征图。

        输入：
        - inp: 形状为 [B, 3, H, W] 的图像张量，默认假定值域在 (0, 1)

        输出：
        - 一个 list
        - list 中每个元素对应一个被选中的 block 输出
        - 顺序与 block 编号从小到大一致
        """
        # outp 用来按顺序收集我们关心的 block 输出。
        outp = []
        # x 初始就是原始输入，形状通常是 [B, 3, H, W]。
        x = inp

        if self.resize_input:
            # 预训练 InceptionV3 通常希望输入接近 299x299。
            # 如果打开该选项，这里会先把输入双线性插值到 [B, 3, 299, 299]。
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            # 把像素值从 (0, 1) 映射到 (-1, 1)，与预训练 Inception 的输入分布保持一致。
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            # 依次通过每个 Inception block，x 的通道数和空间分辨率会逐步变化。
            x = block(x)
            if idx in self.output_blocks:
                # 如果当前 block 是用户指定要输出的层，就把它当前的特征图收集起来。
                outp.append(x)

            if idx == self.last_needed_block:
                # 到达用户需要的最深 block 后就提前退出，避免做多余计算。
                break

        return outp
