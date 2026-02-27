import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageTextDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform):
        self.root_dir = root_dir
        self.transform = transform
        #self.image_size = image_size

        self.annotations = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().strip('"')  # 去掉前后引号
                if not line:
                    continue
                img_name, description = line.split(' ', 1)
                img_name = img_name.strip()  # 保守起见再 strip 一次
                description = description.strip()
                self.annotations.append((img_name, description))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, description = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, description
# 图像预处理：插值扩展到 224x224 + 转 tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)),
])

# 使用数据集
dataset = ImageTextDataset(
    root_dir='/home/et24-liax/new_fer/pretrain/static_dataset/affectnet/trainnew',
    csv_file='/home/et24-liax/new_fer/pretrain/code/AU_handle/affectnet_AU_descriptions.csv',
    transform=transform
)

# 构建 DataLoader
Pretrain_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 示例查看一批数据
'''
if __name__ == "__main__":
    for images, texts in dataloader:
        print("Image batch shape:", images.shape)  # [B, 3, 224, 224]
        print("Text batch example:", texts[0]

'''

        
