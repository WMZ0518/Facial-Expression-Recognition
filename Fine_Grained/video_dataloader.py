# 导入必要的库
import os.path
from numpy.random import randint  # 随机整数生成器
from torch.utils import data      # PyTorch 数据加载工具
import glob                       # 文件路径匹配
import os                         # 系统操作接口
from dataloader.video_transform import *  # 自定义视频帧变换工具
import numpy as np                # 数值计算库
from PIL import Image, ImageDraw  # 图像处理库

class VideoRecord(object):
    """
    封装单个视频样本的信息
    
    每个记录包含：
        - 视频路径
        - 帧数量
        - 类别标签
    """

    def __init__(self, row):
        self._data = row  # 存储原始行数据

    @property
    def path(self):
        return self._data[0]  # 返回视频目录路径

    @property
    def num_frames(self):
        return int(self._data[1])  # 返回帧数

    @property
    def label(self):
        return int(self._data[2])  # 返回类别标签


class VideoDataset(data.Dataset):
    """
    视频数据集类：支持图像 + 关键点图像的双路输入

    支持训练时随机采样、测试时均匀采样
    支持数据增强、帧堆叠等预处理操作
    """

    def __init__(self, list_file, list_file2, num_segments, duration, mode, transform, image_size):
        """
        初始化视频数据集
        
        参数:
            list_file (str): 主视频列表文件路径（每行：path n_frames class_idx）
            list_file2 (str): 对应的关键点视频列表文件路径
            num_segments (int): 分割段数（将视频分成多少段来采样）
            duration (int): 每个片段抽取多少连续帧
            mode (str): 'train' 或 'test'
            transform (Compose): 图像变换操作序列
            image_size (int): 输入图像大小（如 224）
        """
        self.list_file = list_file
        self.list_file2 = list_file2
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()  # 解析数据列表

    def _parse_list(self):
        """
        解析视频列表文件，构造视频记录列表
        
        Data Form: [video_id, num_frames, class_idx]
        """
        tmp = [x.strip().split(' ') for x in open(self.list_file)]  # 读取主视频列表
        tmp = [item for item in tmp]
        tmp2 = [x.strip().split(' ') for x in open(self.list_file2)]  # 读取对应的关键点视频列表
        tmp2 = [item2 for item2 in tmp2]

        assert len(tmp) == len(tmp2), "list_file 和 list_file2 长度不匹配"

        # 构建视频记录对象列表
        self.video_list = [VideoRecord(item) for item in tmp]
        # 构建关键点视频记录对象列表
        self.landmark_list = [VideoRecord(item) for item in tmp2]

        print(('video number:%d' % (len(self.video_list))))  # 打印视频总数

    def _get_train_indices(self, record):
        """
        训练模式下：从每个分段中随机选取一帧
        
        参数:
            record (VideoRecord): 单个视频样本信息

        返回:
            offsets (np.array): 抽取帧的索引数组
        """
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            # 每段内随机选择起始帧
            offsets = np.multiply(list(range(self.num_segments)), average_duration) \
                      + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            # 若总帧数大于分段数，则随机选择帧并排序
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            # 否则使用边缘填充补充缺失帧
            offsets = np.pad(np.array(list(range(record.num_frames))),
                             (0, self.num_segments - record.num_frames),
                             'edge')
        return offsets

    def _get_test_indices(self, record):
        """
        测试模式下：从每个分段中间选取一帧
        
        参数:
            record (VideoRecord): 单个视频样本信息

        返回:
            offsets (np.array): 抽取帧的索引数组
        """
        if record.num_frames > self.num_segments + self.duration - 1:
            # 多帧情况下均分视频段，取中间帧
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            # 帧不足时用边缘填充
            offsets = np.pad(np.array(list(range(record.num_frames))),
                             (0, self.num_segments - record.num_frames),
                             'edge')
        return offsets

    def __getitem__(self, index):
        """
        获取一个样本（图像 + 关键点图像 + 标签）

        参数:
            index (int): 样本索引

        返回:
            images (Tensor): 图像张量
            landmarks (Tensor): 关键点图像张量
            label (int): 标签
        """
        record = self.video_list[index]
        la_record = self.landmark_list[index]

        # 根据训练/测试模式获取帧索引
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        # 固定为 16 帧（调试用）
        segment_indices = np.arange(16)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        """
        加载指定索引的帧，并应用变换
        
        参数:
            record (VideoRecord): 当前视频信息
            indices (array): 要加载的帧索引数组

        返回:
            images (Tensor): 经过处理的图像张量
            landmarks (Tensor): 经过处理的关键点图像张量
            label (int): 标签
        """
        # 获取当前视频所有帧的路径
        video_frames_path = glob.glob(os.path.join(record.path, '*'))
        video_frames_path.sort()

        # 获取对应关键点帧路径（与图像路径一致）
        la_video_frames_path = glob.glob(os.path.join(record.path, '*'))
        la_video_frames_path.sort()

        images = list()
        landmarks = list()

        # 遍历所有目标帧索引
        for seg_ind in indices:
            p = int(seg_ind)

            # 读取连续 duration 帧
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                la_seg_imgs = [Image.open(os.path.join(la_video_frames_path[p])).convert('RGB')]

                images.extend(seg_imgs)
                landmarks.extend(la_seg_imgs)

                if p < record.num_frames - 1:
                    p += 1  # 下一帧

        # 应用变换（如 Resize、Flip、Normalize 等）
        images = self.transform(images)
        # reshape 成 batch × channel × height × width 形式
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        # 同样处理关键点图像
        landmarks = self.transform(landmarks)
        landmarks = torch.reshape(landmarks, (-1, 3, self.image_size, self.image_size))

        return images, landmarks, record.label

    def __len__(self):
        """返回数据集中样本总数"""
        return len(self.video_list)


def train_data_loader(list_file, list_file2, num_segments, duration, image_size, args):
    """
    构建训练数据加载器
    
    参数:
        list_file (str): 主视频列表路径
        list_file2 (str): 关键点视频列表路径
        num_segments (int): 分段数
        duration (int): 每段持续帧数
        image_size (int): 图像尺寸
        args: 包含 dataset 的命名空间

    返回:
        train_data (DataLoader): 训练数据加载器
    """
    # 不同数据集采用不同的数据增强策略
    if args.dataset == "DFEW":
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.5),
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    elif args.dataset == "FERV39K":
        train_transforms = torchvision.transforms.Compose([
            RandomRotation(4),
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])  
    elif args.dataset == "MAFW":
        train_transforms = torchvision.transforms.Compose([
            GroupRandomSizedCrop(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()]) 

    # 创建训练数据集
    train_data = VideoDataset(list_file=list_file,
                              list_file2=list_file2,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(list_file, list_file2, num_segments, duration, image_size):
    """
    构建测试数据加载器
    
    参数:
        list_file (str): 主视频列表路径
        list_file2 (str): 关键点视频列表路径
        num_segments (int): 分段数
        duration (int): 每段帧数
        image_size (int): 图像尺寸

    返回:
        test_data (DataLoader): 测试数据加载器
    """
    # 测试阶段仅做归一化和缩放
    test_transform = torchvision.transforms.Compose([
        GroupResize(image_size),
        Stack(),
        ToTorchFormatTensor()
    ])

    # 创建测试数据集
    test_data = VideoDataset(list_file=list_file,
                             list_file2=list_file2,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data
''' def _parse_list(self):
         #
        # Data Form: [video_id, num_frames, class_idx]
        #
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))'''