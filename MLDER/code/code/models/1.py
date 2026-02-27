import torch
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from clip import clip
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
save_dir = "/home/et24-liax/new_fer/pretrain/ckp"
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

checkpoint, _ = clip.load("ViT-B/16", device=device, jit=False)
checkpoint.float()  # 强制转换为 float32

'''
# 加载 .pth 文件
ckpt_path = '/home/et24-liax/new_fer/MAE/job_dir/checkpoint-360.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')  # 加载到 CPU，避免 GPU 不一致问题
clipmodel, _ = clip.load("ViT-B/16", device=device, jit=False)
'''

# 查看 keys（判断是 state_dict 还是其他结构）
print("Checkpoint keys:", checkpoint.keys())

# 常见情况：模型权重在 'model' 或 'state_dict' key 下
if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # 直接就是 state_dict

# 打印参数名和形状
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
