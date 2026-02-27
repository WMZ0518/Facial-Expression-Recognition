import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from models.Generate_Model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
from clip import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text import *
import random
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=48)

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr-image-encoder', type=float, default=1e-5)
parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)

parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int)

parser.add_argument('--contexts-number', type=int, default=4)
parser.add_argument('--class_token_position', type=str, default="end")
parser.add_argument('--class_specific_contexts', type=str, default='False')
parser.add_argument('--load_and_tune_prompt_learner', type=str, default='True')

parser.add_argument('--text-type', type=str)
parser.add_argument('--exper-name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--temporal-layers', type=int, default=1)

args = parser.parse_args()

random.seed(args.seed)  
np.random.seed(args.seed) 
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

now = datetime.datetime.now()
time_str = now.strftime("%y%m%d%H%M")
time_str = time_str + args.exper_name

print('************************')
for k, v in vars(args).items():
    print(k,'=',v)
print('************************')

if args.dataset == "FERV39K" or args.dataset == "DFEW":
    number_class = 7
    class_names = class_names_7
    class_names_with_context = class_names_with_context_7
    class_descriptor = class_descriptor_7
   
elif args.dataset == "MAFW":
    number_class = 11
    class_names = class_names_11
    class_names_with_context = class_names_with_context_11
    class_descriptor = class_descriptor_11

def contrastive_loss_video(video_features, landmark_features, temperature=0.07):
    """
    计算视频特征和地标特征之间的对比损失。

    参数:
    video_features (torch.Tensor): 视频的特征张量，形状为 [N, D]，其中 N 是视频的数量，D 是特征的维度。
    landmark_features (torch.Tensor): 地标的特征张量，形状为 [N, D]，其中 N 是地标的数量，D 是特征的维度。
    temperature (float): 对比损失中的温度参数，用于控制相似度的集中程度，默认值为 0.07。

    返回:
    torch.Tensor: 视频和地标之间的对比损失。
    """

    # 对视频特征和地标特征进行归一化，以确保它们的模为1
    video_features = F.normalize(video_features, dim=1)  # [N, D]
    landmark_features = F.normalize(landmark_features, dim=1)  # [N, D]

    # 计算视频特征和地标特征之间的相似度矩阵
    similarity_matrix = video_features @ landmark_features.t()  # [N, N]

    # 提取相似度矩阵的对角线元素，这些是正样本对的相似度
    pos_pairs = torch.diag(similarity_matrix)  # [N]

    # 计算负样本对的相似度，通过从相似度矩阵中减去单位矩阵来排除正样本对
    neg_pairs = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(similarity_matrix.device)  # [N, N]

    # 计算正样本对的损失，使用指数函数来放大相似度的差异
    pos_loss = torch.exp(pos_pairs / temperature)  # 正样本对相似度

    # 计算所有负样本对的损失，并在维度1上进行求和
    neg_loss = torch.exp(neg_pairs / temperature).sum(dim=1)  # 所有负样本对相似度

    # 计算最终的对比损失，通过求平均值来汇总所有样本的损失
    loss = -torch.log(pos_loss / (pos_loss + neg_loss)).mean()

    # 返回计算得到的损失
    return loss


def main(set):
    """
    主函数，用于模型的训练和评估。
    
    参数:
    set -- 数据集的索引，用于选择不同的数据集。
    
    返回:
    uar -- 未加权平均召回率。
    war -- 加权平均召回率。
    """
    
    # 根据输入的set值确定数据集和相关的路径
    data_set = set+1
    
    # 根据不同的数据集，设置不同的日志和检查点路径
    if args.dataset == "FERV39K":
        print("*********** FERV39K Dataset ***********")
        log_txt_path = './log/' + 'FER39K-' + time_str + '-log.txt'
        log_curve_path = './log/' + 'FER39K-' + time_str + '-log.png'
        log_confusion_matrix_path = './log/' + 'FER39K-' + time_str + '-cn.png'
        checkpoint_path = '/checkpoint/' + 'FER39K-' + time_str + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'FER39K-' + time_str + '-model_best.pth'
        train_annotation_file_path = "./annotation/FERV39K_train.txt"
        train_annotation_file_path2 = "./annotation_la/FERV39K_train.txt"
        test_annotation_file_path = "./annotation/FERV39K_test.txt"
        test_annotation_file_path2 = "./annotation_la/FERV39K_test.txt"
    
    elif args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log.txt'
        log_curve_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log.png'
        log_confusion_matrix_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-cn.png'
        checkpoint_path = './checkpoint/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-model_best.pth'
        train_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_train.txt"
        train_annotation_file_path2 = "./annotation_la/DFEW_set_"+str(data_set)+"_train.txt"
        test_annotation_file_path = "./annotation/DFEW_set_"+str(data_set)+"_test.txt"
        test_annotation_file_path2 = "./annotation_la/DFEW_set_"+str(data_set)+"_test.txt"
        
    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log.txt'
        log_curve_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log.png'
        log_confusion_matrix_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-cn.png'
        checkpoint_path = './checkpoint/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-model_best.pth'
        train_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_train.txt"
        test_annotation_file_path = "./annotation/MAFW_set_"+str(data_set)+"_test.txt"
    
    # 初始化最佳准确率和记录器
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training name: ' + time_str)

    # 创建模型并加载预训练参数
    CLIP_model, _ = clip.load("ViT-B/32", device='cpu')
    
    # 根据不同的文本类型选择输入文本
    if args.text_type=="class_names":
        input_text = class_names
    elif args.text_type=="class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type=="class_descriptor":
        input_text = class_descriptor

    print("Input Text: ")
    for i in range(len(input_text)):
        print(input_text[i])
        
    # 生成模型
    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    # 只打开可学习的部分
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = True  
        if "temporal_net" in name:
            param.requires_grad = True
        if "prompt_learner" in name:  
            param.requires_grad = True

    model = torch.nn.DataParallel(model).cuda()
    
    # 打印参数   
    print('************************')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('************************')
    
    # 将参数写入日志文件
    with open(log_txt_path, 'a') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    
    # 定义优化器
    optimizer = torch.optim.SGD([{"params": model.module.temporal_net.parameters(), "lr": args.lr},
                                 {"params": model.module.image_encoder.parameters(), "lr": args.lr_image_encoder},
                                 {"params": model.module.prompt_learner.parameters(), "lr": args.lr_prompt_learner}],
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=0.1)
        
    cudnn.benchmark = True

    # 数据加载代码
    # 创建训练数据集，使用 train_data_loader 加载数据
    train_data = train_data_loader(
        # 主标注文件路径，包含训练样本的文件名和标签
        list_file=train_annotation_file_path,
        # 第二个标注文件路径，可能用于加载地标（landmark）相关的信息，辅助对比学习
        list_file2=train_annotation_file_path2,
        # 每个视频被分割成的片段数量，用于捕捉时序信息
        num_segments=16,
        # 每个片段的持续时间（以帧为单位），设为 1 表示每个片段只包含一帧图像
        duration=1,
        # 输入图像的目标尺寸，所有图像将被缩放为 224x224 大小
        image_size=224,
        # 传递命令行参数对象，用于在数据加载时获取相关配置（如 batch size、workers 等）
        args=args
    )
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 list_file2 = test_annotation_file_path2,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # 训练和评估模型
    for epoch in range(0, args.epochs):

        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']
        current_learning_rate_1 = optimizer.state_dict()['param_groups'][1]['lr']
        current_learning_rate_2 = optimizer.state_dict()['param_groups'][2]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + ' ' + str(current_learning_rate_1) + ' ' + str(current_learning_rate_2) + '\n')
            print('Current learning rate: ', current_learning_rate_0, current_learning_rate_1, current_learning_rate_2)         
            
        # 训练一个epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path)

        # 在验证集上评估
        val_acc, val_los = validate(val_loader, model, criterion, args, log_txt_path)
        
        scheduler.step()

        # 记录最佳准确率并保存检查点
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best,
                        checkpoint_path,
                        best_checkpoint_path)

        # 打印和保存日志
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.2f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: ' + str(epoch_time) + 's' + '\n')

    # 计算并返回未加权平均召回率和加权平均召回率
    uar, war = computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set)
    
    return uar, war

def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    """
    训练模型一个完整周期（epoch）

    参数:
        train_loader (DataLoader): 用于加载训练数据
        model (nn.Module): 要训练的神经网络模型
        criterion (Loss Function): 损失函数（如 CrossEntropyLoss）
        optimizer (Optimizer): 优化器（如 SGD 或 Adam）
        epoch (int): 当前训练的轮次编号
        args: 包含训练超参数的命名空间对象
        log_txt_path (str): 日志文件保存路径

    返回:
        top1.avg (float): 平均 Top-1 准确率
        losses.avg (float): 平均损失值
    """

    # 初始化 Loss 和 Accuracy 的统计器
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    # 创建进度条显示工具
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # 切换模型为训练模式
    model.train()

    # 打印当前训练 DataLoader 的基本信息（调试用）
    print("train_loader: ", train_loader)
    print(len(train_loader.dataset))

    # 遍历每个 batch 的数据
    for i, (images, landmarks, target) in enumerate(train_loader):
        print("i: ", i)

        try:
            # 尝试打印当前批次输入数据形状（可用于调试）
            print(f"Batch {i}: images shape = {images.shape}, target shape = {target.shape}")
        except Exception as e:
            # 如果当前 batch 异常，记录异常并跳过该 batch
            print(f"Error in batch {i}: {e}")
            continue

        # 将数据移动到 GPU 上进行加速计算
        images = images.cuda()
        target = target.cuda()
        landmarks = landmarks.cuda()

        # 前向传播：模型输出图像和关键点的特征及分类结果
        im_output, la_output, video_features, landmark_features = model(landmarks, images)

        # 计算对比学习损失（Contrastive Loss）
        con_loss = contrastive_loss_video(video_features, landmark_features)

        # 总损失由两部分构成：图像输出损失 + 关键点输出损失
        loss = criterion(im_output, target) + criterion(la_output, target)+ con_loss

        # 使用图像输出作为主输出进行准确率评估
        output = im_output

        # 计算 Top-1 和 Top-5 准确率
        acc1, _ = accuracy(output, target, topk=(1, 5))

        # 更新损失和准确率统计器
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # 梯度清零、反向传播、更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 定期打印训练进度
        if i % args.print_freq == 0:
            progress.display(i)

    # 返回本轮训练的平均准确率和平均损失
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    """
    在验证集上评估模型性能

    参数:
        val_loader (DataLoader): 验证数据加载器
        model (nn.Module): 已训练好的模型
        criterion (Loss Function): 损失函数
        args: 包含训练参数的对象
        log_txt_path (str): 日志文件保存路径

    返回:
        top1.avg (float): 平均 Top-1 准确率
        losses.avg (float): 平均损失值
    """

    # 初始化损失和准确率的统计器
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')

    # 创建进度条显示工具
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ',
                             log_txt_path=log_txt_path)

    # 切换模型为评估模式
    model.eval()

    # 不计算梯度，节省内存和计算资源
    with torch.no_grad():
        for i, (images, landmarks, target) in enumerate(val_loader):
            # 将数据移到 GPU
            images = images.cuda()
            target = target.cuda()
            landmarks = landmarks.cuda()

            # 前向传播获取输出
            im_output, la_output, video_features, landmark_features = model(landmarks, images)

            # 计算对比损失
            con_loss = contrastive_loss_video(video_features, landmark_features)

            # 总损失 = 图像输出损失 + 关键点输出损失
            loss = criterion(im_output, target) + criterion(la_output, target)

            # 使用图像分支输出作为最终预测结果
            output = im_output

            # 计算 Top-1 准确率
            acc1, _ = accuracy(output, target, topk=(1, 5))

            # 更新统计信息
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # 定期显示进度
            if i % args.print_freq == 0:
                progress.display(i)

        # 显示本轮验证的最终准确率
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')

    # 返回准确率和损失
    return top1.avg, losses.avg
def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """
    计算指定 k 值的 Top-k 准确率，即预测结果中前 k 个预测值包含正确标签的比例。

    参数:
        output (torch.Tensor): 模型输出的结果，形状为 [batch_size, num_classes]。
        target (torch.Tensor): 真实标签，形状为 [batch_size]。
        topk (tuple): 需要计算准确率的 k 值列表，例如 (1, 5) 表示计算 Top-1 和 Top-5 准确率。

    返回:
        res (list): 包含每个 k 值对应的准确率，例如 [top1_accuracy, top5_accuracy]。
    """
    
    # 禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        # 获取最大的 k 值，用于确定需要取多少个预测结果
        maxk = max(topk)
        
        # 获取当前 batch 的大小
        batch_size = target.size(0)
        
        # 获取模型输出中概率最高的 maxk 个类别索引，并将结果转置以便后续处理
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # 比较预测结果与真实标签，生成一个布尔矩阵，表示预测是否正确
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        # 初始化结果列表
        res = []
        
        # 遍历每个 k 值，计算对应的准确率
        for k in topk:
            # 取出前 k 个预测结果，并将其展平成一维向量
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            
            # 计算当前 k 值的准确率，并添加到结果列表中
            res.append(correct_k.mul_(100.0 / batch_size))
        
        # 返回结果列表
        return res

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, best_checkpoint_path, log_confusion_matrix_path, log_txt_path, data_set):
    
    pre_trained_dict = torch.load(best_checkpoint_path)['state_dict']
    model.load_state_dict(pre_trained_dict)
    
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, landmarks, target) in enumerate(tqdm.tqdm(val_loader)):
            
            images = images.cuda()
            target = target.cuda()
            landmarks = landmarks.cuda()
            #output = model(landmarks,images)        
            im_output,la_output,video_features,landmark_features= model(landmarks,images)      
            #con_loss = contrastive_loss_video(video_features, landmark_features)
            #loss = criterion(im_output, target)+ (0.25*criterion(la_output, target)) + con_loss
            output = im_output
            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)
    
    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()
        
    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "FERV39K":
        title_ = "Confusion Matrix on FERV39k"
    elif args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold "+str(data_set)
    elif args.dataset == "MAFW":
        title_ = "Confusion Matrix on MAFW fold "+str(data_set)

    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()
    
    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')        
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')
    
    return uar, war


if __name__ == '__main__':
    
    UAR = 0.0
    WAR = 0.0

    if args.dataset == "FERV39K":
        all_fold = 1
    elif args.dataset == "DFEW":
        all_fold = 5
    elif args.dataset == "MAFW":
        all_fold = 5
    print('args: ',args)
    for set in range(all_fold):
        uar, war = main(set)
        UAR += float(uar)
        WAR += float(war)
        
    print('********* Final Results *********')   
    print("UAR: %0.2f" % (UAR/all_fold))
    print("WAR: %0.2f" % (WAR/all_fold))
    print('*********************************')
