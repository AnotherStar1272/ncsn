import numpy as np
import tqdm
from losses.dsm import dsm_score_estimation
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.refinenet_dilated_baseline import RefineNetDilated

__all__ = ['BaselineRunner']


class BaselineRunner():
    def __init__(self, args, config):
        # 保存命令行参数和配置文件内容，后续训练、测试都会反复用到。
        self.args = args
        self.config = config

    def _get_training_artifact_paths(self):
        # 为训练过程中的可视化结果创建单独目录。
        # 这里会把中途采样图和最终 loss 曲线统一放到 run/artifacts/<doc>/ 下。
        artifact_root = os.path.join(self.args.run, 'artifacts', self.args.doc)
        sample_dir = os.path.join(artifact_root, 'train_samples')
        os.makedirs(sample_dir, exist_ok=True)
        return artifact_root, sample_dir

    def _save_checkpoint(self, score, optimizer, step):
        # 保存模型和优化器状态：
        # 1. 保存一个带 step 编号的快照，便于回看中间阶段
        # 2. 同时覆盖最新的 checkpoint.pth，便于恢复训练
        states = [
            score.state_dict(),
            optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def _save_loss_plot(self, train_losses, test_losses, artifact_root):
        # 把训练过程中记录下来的 train/test loss 画成一张总图，
        # 方便训练结束后快速判断是否收敛、是否过拟合。
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        if train_losses:
            train_steps, train_values = zip(*train_losses)
            plt.plot(train_steps, train_values, label='train_loss')
        if test_losses:
            test_steps, test_values = zip(*test_losses)
            plt.plot(test_steps, test_values, label='test_dsm_loss')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('BaselineRunner Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(artifact_root, 'loss_curve.png'))
        plt.close()

    def _save_training_samples(self, score, step, sample_dir):
        # 从随机噪声出发，利用当前的 score network 做一小段 Langevin dynamics，
        # 把当前模型生成出来的样本保存成网格图，便于观察训练中期效果。
        sample_batch_size = getattr(self.config.training, 'sample_batch_size', 16)
        sample_steps = getattr(self.config.training, 'sample_steps', 100)
        sample_step_lr = getattr(self.config.training, 'sample_step_lr', 0.00002)

        samples = torch.rand(
            sample_batch_size,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
            device=self.config.device
        )
        samples = self.Langevin_dynamics(samples, score, n_steps=sample_steps, step_lr=sample_step_lr)[-1]

        if self.config.data.logit_transform:
            samples = torch.sigmoid(samples)

        nrow = int(np.sqrt(sample_batch_size))
        nrow = max(nrow, 1)
        grid = make_grid(samples, nrow=nrow)
        save_image(grid, os.path.join(sample_dir, 'step_{:06d}.png'.format(step)))

    def get_optimizer(self, parameters):
        # 根据配置文件里指定的优化器名称，创建对应的 optimizer。
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        # 如果配置要求在 logit 空间里训练，就先把图像从 (0, 1) 映射到 logit 空间。
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        # 先定义训练集和测试集使用的数据预处理。
        # random_flip 为真时，训练集会额外做随机水平翻转增强。
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        # 根据配置选择具体数据集。
        # 当前 BaselineRunner 支持 CIFAR10、MNIST、CelebA。
        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)


        # 用 DataLoader 把数据集包装成 mini-batch 迭代器。
        # 训练集用于参数更新，test_loader 用于周期性评估 loss。
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        # 为当前实验准备 TensorBoard 输出目录。
        # 如果同名目录已存在，就先删掉，避免新旧实验日志混在一起。
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        artifact_root, sample_dir = self._get_training_artifact_paths()

        # 创建真正要训练的 score network，并把它移动到配置指定的设备上。
        # 这里的模型主体就是 RefineNetDilated。
        score = RefineNetDilated(self.config).to(self.config.device)

        # 用 DataParallel 包一层，这样有多张 GPU 时可以并行。
        score = torch.nn.DataParallel(score)

        # 创建优化器；如果选择继续训练，则从历史 checkpoint 恢复模型和优化器状态。
        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0
        train_losses = []
        test_losses = []
        sample_freq = getattr(self.config.training, 'sample_freq', self.config.training.snapshot_freq)

        # 主训练循环：
        # 外层按 epoch 走，内层按 batch 走；但真正控制训练时长的是 step/n_iters。
        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1

                # 切回训练模式，取出一个 batch 的图像并放到设备上。
                score.train()
                X = X.to(self.config.device)

                # 这一步是常见的 dequantization 处理：
                # 先把离散像素值缩放，再加上一个很小的均匀随机噪声。
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                # 用 denoising score matching 计算当前 batch 的训练损失。
                loss = dsm_score_estimation(score, X, sigma=0.01)

                # 标准的反向传播与参数更新。
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 记录训练 loss，供日志和最终曲线图使用。
                tb_logger.add_scalar('loss', loss, global_step=step)
                train_losses.append((step, loss.item()))
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                # 每隔 100 步在测试集上估计一次 loss，用来监控泛化情况。
                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    with torch.no_grad():
                        test_dsm_loss = dsm_score_estimation(score, test_X, sigma=0.01)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)
                    test_losses.append((step, test_dsm_loss.item()))

                # 按设定频率保存一张当前模型的采样图，
                # 方便直观看到生成结果有没有逐渐变好。
                if step % sample_freq == 0:
                    score.eval()
                    self._save_training_samples(score, step, sample_dir)

                # 按设定频率保存 checkpoint，防止中断时完全丢失进度。
                if step % self.config.training.snapshot_freq == 0:
                    self._save_checkpoint(score, optimizer, step)

                # 到达最大训练步数后，先补存最后一次 checkpoint / 采样图 / loss 曲线，再退出。
                if step >= self.config.training.n_iters:
                    if step % self.config.training.snapshot_freq != 0:
                        self._save_checkpoint(score, optimizer, step)
                    if step % sample_freq != 0:
                        score.eval()
                        self._save_training_samples(score, step, sample_dir)
                    self._save_loss_plot(train_losses, test_losses, artifact_root)
                    tb_logger.close()
                    return 0

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=1000, step_lr=0.00002, log_progress=False):
        # Langevin dynamics 采样过程：
        # 从一个随机初始噪声出发，不断沿着 score 的方向更新，并叠加少量高斯噪声，
        # 逐步把样本推向数据分布附近。
        images = []

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod)
                x_mod = x_mod + step_lr * grad + noise
                if log_progress:
                    print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def test(self):
        # 测试/采样阶段：先加载训练好的 checkpoint，再根据数据集类型生成样本轨迹。
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = RefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        score.eval()

        # MNIST / FashionMNIST 分支：
        # 从随机噪声开始采样，并把每一步的中间结果保存成 .pth 文件。
        if self.config.data.dataset == 'MNIST' or self.config.data.dataset == 'FashionMNIST':
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'MNIST':
                dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                                transform=transform)
            else:
                dataset = FashionMNIST(os.path.join(self.args.run, 'datasets', 'fmnist'), train=True, download=True,
                                       transform=transform)

            dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            samples, _ = next(data_iter)
            samples = samples.cuda()

            samples = torch.rand_like(samples)
            all_samples = self.Langevin_dynamics(samples, score, 1000, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))

        # CelebA 分支：流程与上面类似，只是图像维度变成 3 通道。
        elif self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
            samples, _ = next(iter(dataloader))

            samples = torch.rand(100, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.Langevin_dynamics(samples, score, 1000, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))

        # 其他分支目前主要对应 CIFAR10，依然是从随机噪声出发做采样。
        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)

            dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            samples, _ = next(data_iter)
            samples = samples.cuda()
            samples = torch.rand_like(samples)

            all_samples = self.Langevin_dynamics(samples, score, 1000, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(100, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
