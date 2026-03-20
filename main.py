import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners import *


def parse_args_and_config():
    # 创建命令行参数解析器。
    # 这个函数的任务是：
    # 1. 读取命令行参数
    # 2. 读取配置文件
    # 3. 初始化日志目录和 logger
    # 4. 设置随机种子和运行设备
    # 5. 最终返回 args 和 config
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # --runner: 指定这次运行哪个 runner 类，例如 BaselineRunner / ToyRunner / AnnealRunner。
    parser.add_argument('--runner', type=str, default='AnnealRunner', help='The runner to execute')
    # --config: 指定 configs/ 目录下使用哪个 yml 配置文件。
    parser.add_argument('--config', type=str, default='anneal.yml',  help='Path to the config file')
    # --seed: 指定随机种子，保证实验尽量可复现。
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    # --run: 运行过程中产生的数据（日志、tensorboard、数据集下载等）存放的根目录。
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    # --doc: 当前实验的名字，通常会用来生成日志目录名。
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    # --comment: 给当前实验附加一段说明文字，主要用于日志记录。
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    # --verbose: 控制日志输出等级，例如 info / warning。
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    # --test: 是否进入测试/采样流程；不加时默认执行训练。
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    # --resume_training: 是否从已有 checkpoint 继续训练。
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    # --image_folder: 测试/采样阶段输出图片或样本文件的目录。
    parser.add_argument('-o', '--image_folder', type=str, default='images', help="The directory of image outputs")

    # 真正解析命令行，生成 args 对象。
    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])

    # 当前实验的日志目录，后面会用来保存配置文件、stdout.txt、checkpoint 等。
    args.log = os.path.join(args.run, 'logs', args.doc)

    # 读取配置文件。
    # 训练模式下：从 configs/<配置文件名> 读取；
    # 测试模式下：从对应日志目录里的 config.yml 读取，保证测试用的是训练时同一份配置。
    if not args.test:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        new_config = dict2namespace(config)
    else:
        with open(os.path.join(args.log, 'config.yml'), 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        new_config = config

    # 训练模式下初始化实验目录和日志系统。
    if not args.test:
        # 如果不是断点续训，就删除旧的同名日志目录，重新开始一份新实验。
        if not args.resume_training:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        # 把当前实际使用的配置写一份到日志目录中，便于之后复现实验或做测试。
        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # 配置 logger：
        # handler1 输出到终端；
        # handler2 输出到 run/logs/<doc>/stdout.txt 文件。
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        # 测试模式下只保留终端日志输出，不再额外写 stdout.txt。
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # 自动选择运行设备：
    # 如果检测到 CUDA，就用 GPU；否则退回 CPU。
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 这条日志会输出当前到底使用的是 cpu 还是 cuda。
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # 设置随机种子，尽量让实验结果稳定一些、可复现一些。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 让 cuDNN 自动寻找更快的卷积实现，通常可以提升训练速度。
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    # 这个函数把嵌套字典递归转换成 argparse.Namespace。
    # 这样原本 config['training']['batch_size']
    # 就可以写成 config.training.batch_size，代码里访问会更方便。
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    # 先解析命令行参数并构造完整配置。
    args, config = parse_args_and_config()

    # 以下几条 logging/print 的作用都是：把当前实验的关键信息清楚输出出来，
    # 便于你在终端或日志文件里确认“我到底跑了什么配置、写到了哪里”。

    # 输出日志目录位置。
    logging.info("Writing log file to {}".format(args.log))
    # 输出当前进程 id，方便区分不同运行实例。
    logging.info("Exp instance id = {}".format(os.getpid()))
    # 输出这次实验附带的备注信息。
    logging.info("Exp comment = {}".format(args.comment))
    # 提示下面将输出完整配置对象。
    logging.info("Config =")

    # 用一对分隔线把配置打印包起来，便于在终端里阅读。
    # print(config) 会直接把 Namespace 形式的配置对象完整打印出来。
    print(">" * 80)
    print(config)
    print("<" * 80)

    try:
        # 根据命令行里传入的 runner 名字，动态创建对应的 runner 对象。
        # 例如 args.runner == 'BaselineRunner' 时，这里就等价于：
        # runner = BaselineRunner(args, config)
        runner = eval(args.runner)(args, config)

        # 非 test 模式执行训练；test 模式执行测试/采样。
        if not args.test:
            runner.train()
        else:
            runner.test()
    except:
        # 如果运行过程中抛异常，把完整 traceback 记录到日志里，方便定位问题。
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    # 作为脚本直接运行时，从 main() 开始执行，并把返回值作为程序退出码。
    sys.exit(main())
