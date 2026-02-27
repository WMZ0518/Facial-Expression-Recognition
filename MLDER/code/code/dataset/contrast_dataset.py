import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    """重排序训练数据集类，用于加载和处理训练数据"""
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        # 初始化数据集
        self.ann = []  # 存储所有标注信息
        for f in ann_file:
            # 加载多个JSON文件中的标注数据，并合并到self.ann中
            self.ann += json.load(open(f,'r'))
        self.transform = transform  # 图像变换操作（如归一化）
        self.image_root = image_root  # 图像文件根目录
        self.max_words = max_words  # 最大文本长度
        self.img_ids = {}  # 映射图像ID到索引编号
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']  # 获取当前标注对应的图像ID
            if img_id not in self.img_ids.keys():
                # 如果图像ID未被记录，则为其分配一个新编号
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.ann)
    
    def __getitem__(self, index):    
        # 根据索引获取一个样本
        ann = self.ann[index]
        
        # 构建图像路径并加载图像
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)  # 应用图像变换
        
        # 对描述文本进行预处理
        caption = pre_caption(ann['caption'], self.max_words) 

        # 返回图像、文本和对应的图像ID索引
        return image, caption, self.img_ids[ann['image_id']]

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        # 加载单个JSON文件中的标注数据
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform  # 图像变换操作
        self.image_root = image_root  # 图像文件根目录
        self.max_words = max_words  # 最大文本长度
        
        # 存储文本、图像路径及相关映射关系
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            # 每张图像存储其路径
            self.image.append(ann['image'])
            # 每张图像对应多个文本，初始化空列表
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                # 预处理每条文本并保存
                self.text.append(pre_caption(caption, self.max_words))
                # 建立图像到文本ID的映射
                self.img2txt[img_id].append(txt_id)
                # 建立文本ID到图像ID的映射
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        # 返回图像数量
        return len(self.image)
    
    def __getitem__(self, index):    
        # 根据索引加载图像
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  # 应用图像变换

        # 返回图像及其索引
        return image, index
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []  # 存储所有标注信息
        for f in ann_file:
            # 加载多个JSON文件中的标注数据，并合并到self.ann中
            self.ann += json.load(open(f,'r'))
        self.transform = transform  # 图像变换操作
        self.max_words = max_words  # 最大文本长度
        
        
    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.ann)
    

    def __getitem__(self, index):    
        # 根据索引获取一个样本
        ann = self.ann[index]
        
        # 如果有多个文本描述，则随机选择一个进行训练
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        # 加载图像并转换为RGB格式
        image = Image.open(ann['image']).convert('RGB')   
        # 应用图像变换
        image = self.transform(image)
                
        # 返回图像和文本
        return image, caption
    