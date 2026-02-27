# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import os  # 提供操作系统相关功能
import ruamel_yaml as yaml  # 用于读写YAML配置文件
import numpy as np  # 提供多维数组对象和数学函数
import random  # 生成随机数
import time  # 时间相关操作
import datetime  # 日期时间处理
import json  # JSON数据处理
from pathlib import Path  # 面向对象的路径操作

# PyTorch 相关导入
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式接口
from torch.utils.data import DataLoader  # 数据加载器
import torch.backends.cudnn as cudnn  # 加速卷积计算
import torch.distributed as dist  # 分布式训练支持

# 模型相关组件
from models.EmoBEF import EmoBEF  # 主模型 EmoBEF
from models.vit import interpolate_pos_embed  # 插值位置嵌入
from models.tokenization_bert import BertTokenizer  # BERT分词器

# 工具模块
import utils  # 自定义工具函数和类


# 定义训练函数
def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    """
    执行一个epoch的训练过程。

    参数:
        model: 要训练的模型
        data_loader: 数据加载器
        optimizer: 优化器
        tokenizer: 文本编码器
        epoch: 当前训练轮次
        warmup_steps: 学习率预热步数
        device: 训练设备（CPU/GPU）
        scheduler: 学习率调度器
        config: 配置字典
    """

    model.train()  # 设置模型为训练模式

    # 初始化 MetricLogger 用于记录训练指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50  # 打印频率
    step_size = 100  # 步长（可能用于warmup）
    warmup_iterations = warmup_steps * step_size  # warmup迭代次数

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)  # 在分布式模式下设置sampler的epoch

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()  # 清除梯度

        image = image.to(device, non_blocking=True)  # 将图像数据移动到目标设备

        # 对文本进行编码
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)

        # 动态调整alpha参数
        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        # 前向传播，获取各个损失项
        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)

        # 总损失 = 各子损失之和
        loss = loss_mlm + loss_ita + loss_itm

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 更新metric logger
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 如果是第一个epoch且处于warmup阶段，更新学习率调度器
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # 收集所有进程的统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    # 返回平均后的指标
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# 主函数
def main(args, config):
    """
    主函数，负责初始化、数据准备、模型构建、训练等流程。

    参数:
        args: 命令行参数
        config: 配置字典
    """

    utils.init_distributed_mode(args)  # 初始化分布式模式

    device = torch.device(args.device)  # 设定运行设备

    # 固定随机种子以保证可复现性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']  # 最大训练轮次
    warmup_steps = config['schedular']['warmup_epochs']  # warmup轮次

    #### Dataset ####
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config)]  # 创建预训练数据集

    if args.distributed:
        num_tasks = utils.get_world_size()  # 获取总任务数
        global_rank = utils.get_rank()  # 获取当前rank
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)  # 创建分布式采样器
    else:
        samplers = [None]  # 非分布式无需采样器

    # 创建数据加载器
    data_loader = create_loader(datasets, samplers,
                                batch_size=[config['batch_size']],
                                num_workers=[4],
                                is_trains=[True],
                                collate_fns=[None])[0]

    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = EmoBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)  # 构建模型

    model = model.to(device)  # 移动模型到指定设备

    # 创建优化器
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    # 创建学习率调度器
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    # 如果提供了checkpoint路径，则加载模型权重或继续训练
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        if args.resume:  # 若resume为True，恢复优化器和学习率调度器状态
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        else:  # 否则只加载模型权重，并对pos_embed做插值处理
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        model.load_state_dict(state_dict)  # 加载模型权重
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model  # 保存未封装的模型
    if args.distributed:  # 使用DistributedDataParallel包装模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module  # 获取module属性

    print("Start training")
    start_time = time.time()  # 记录开始时间

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)  # 更新学习率

        # 执行单个epoch的训练
        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

        # 如果是主进程，执行日志记录和保存checkpoint
        if utils.is_main_process():

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }

            # 保存checkpoint
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

            # 写入日志文件
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()  # 同步所有进程

    # 计算并打印总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# 程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 添加各种命令行参数
    parser.add_argument('--config', default='./configs/Pretrain.yaml')  # 配置文件路径
    parser.add_argument('--checkpoint', default='')  # checkpoint路径
    parser.add_argument('--resume', default=False, type=bool)  # 是否从checkpoint恢复训练
    parser.add_argument('--output_dir', default='Pretrain/')  # 输出目录
    parser.add_argument('--text_encoder', default='bert-base-uncased')  # 文本编码器
    parser.add_argument('--device', default='cuda')  # 设备选择
    parser.add_argument('--seed', default=42, type=int)  # 随机种子
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')  # 进程数量
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')  # 分布式URL
    parser.add_argument('--distributed', default=True, type=bool)  # 是否启用分布式训练
    args = parser.parse_args()  # 解析参数

    # 加载配置文件
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 保存配置文件到输出目录
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    # 调用主函数开始训练
    main(args, config)