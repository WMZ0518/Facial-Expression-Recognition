from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    """
    ResNet中的Bottleneck块，用于减少通道数以提高效率。
    
    属性:
        expansion (int): 扩展比例，用于决定输出通道数。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        初始化Bottleneck模块。

        参数:
            inplanes (int): 输入通道数。
            planes (int): 中间层通道数。
            stride (int): 卷积步长，默认为1。
        """
        super().__init__()

        # 第一个1x1卷积层，用于降低通道数
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化层
        self.relu1 = nn.ReLU(inplace=True)  # ReLU激活函数

        # 第二个3x3卷积层，保持通道数不变
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化层
        self.relu2 = nn.ReLU(inplace=True)  # ReLU激活函数

        # 如果stride > 1，则使用AvgPool2d进行下采样
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个1x1卷积层，用于恢复通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # 批归一化层
        self.relu3 = nn.ReLU(inplace=True)  # ReLU激活函数

        self.downsample = None
        self.stride = stride

        # 如果需要下采样，则构建下采样层
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),  # 使用平均池化进行下采样
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),  # 1x1卷积调整通道数
                ("1", nn.BatchNorm2d(planes * self.expansion))  # 批归一化层
            ]))

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        identity = x  # 保存输入作为残差连接

        # 第一个卷积块
        out = self.relu1(self.bn1(self.conv1(x)))
        # 第二个卷积块
        out = self.relu2(self.bn2(self.conv2(out)))
        # 根据stride进行池化操作
        out = self.avgpool(out)
        # 第三个卷积块
        out = self.bn3(self.conv3(out))

        # 如果存在下采样层，则对输入进行处理
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        # 最终ReLU激活
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """
    二维注意力池化层，将空间维度压缩并使用多头注意力机制。
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        初始化AttentionPool2d模块。

        参数:
            spacial_dim (int): 空间维度大小（如图像的高或宽）。
            embed_dim (int): 嵌入维度。
            num_heads (int): 多头注意力中的头数。
            output_dim (int, optional): 输出维度。默认为None，表示与嵌入维度相同。
        """
        super().__init__()
        # 可学习的位置编码
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 查询、键、值投影线性层
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出投影线性层
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads  # 注意力头数

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为[N, C, H, W]。

        返回:
            torch.Tensor: 输出张量。
        """
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # 转换为(HW, N, C)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # 添加全局平均池化结果
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # 添加位置编码
        # 多头注意力计算
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    修改后的ResNet结构，适用于CLIP模型。
    
    特点包括：
    - 使用3层stem卷积代替传统的1层
    - 使用平均池化代替最大池化
    - 使用注意力池化代替传统平均池化
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        初始化ModifiedResNet模块。

        参数:
            layers (List[int]): 每个阶段的残差块数量。
            output_dim (int): 输出维度。
            heads (int): 注意力头的数量。
            input_resolution (int): 输入分辨率，默认为224。
            width (int): 初始宽度，默认为64。
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # Stem部分：3层卷积+平均池化
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # 残差层
        self._inplanes = width  # 当前通道数
        self.layer1 = self._make_layer(width, layers[0])  # 第一层
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # 第二层，步长为2
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # 第三层，步长为2
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # 第四层，步长为2

        # 注意力池化层
        embed_dim = width * 32  # 计算特征维度
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        构建残差层。

        参数:
            planes (int): 输出通道数。
            blocks (int): 块的数量。
            stride (int): 步长，默认为1。

        返回:
            nn.Sequential: 包含多个残差块的序列。
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]  # 添加第一个带下采样的块

        self._inplanes = planes * Bottleneck.expansion  # 更新通道数
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))  # 添加后续的块

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))  # 第一个卷积块
            x = self.relu2(self.bn2(self.conv2(x)))  # 第二个卷积块
            x = self.relu3(self.bn3(self.conv3(x)))  # 第三个卷积块
            x = self.avgpool(x)  # 平均池化
            return x

        x = x.type(self.conv1.weight.dtype)  # 类型转换
        x = stem(x)  # Stem部分
        x = self.layer1(x)  # 第一层
        x = self.layer2(x)  # 第二层
        x = self.layer3(x)  # 第三层
        x = self.layer4(x)  # 第四层
        x = self.attnpool(x)  # 注意力池化

        return x


class LayerNorm(nn.LayerNorm):
    """支持半精度浮点数(fp16)的LayerNorm。"""

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        orig_type = x.dtype  # 保存原始类型
        ret = super().forward(x.type(torch.float32))  # 转换为float32进行计算
        return ret.type(orig_type)  # 恢复原始类型


class QuickGELU(nn.Module):
    """快速GELU激活函数。"""

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        return x * torch.sigmoid(1.702 * x)  # 快速近似GELU激活


class ResidualAttentionBlock(nn.Module):
    """
    带有残差连接的注意力块。
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        初始化ResidualAttentionBlock模块。

        参数:
            d_model (int): 模型维度。
            n_head (int): 注意力头数。
            attn_mask (torch.Tensor, optional): 注意力掩码。默认为None。
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # 多头注意力
        self.ln_1 = LayerNorm(d_model)  # 第一个LayerNorm
        # 多层感知机(MLP)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),  # 全连接层
            ("gelu", QuickGELU()),  # GELU激活
            ("c_proj", nn.Linear(d_model * 4, d_model))  # 投影层
        ]))
        self.ln_2 = LayerNorm(d_model)  # 第二个LayerNorm
        self.attn_mask = attn_mask  # 注意力掩码

    def attention(self, x: torch.Tensor):
        """
        注意力计算。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 注意力输出。
        """
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        x = x + self.attention(self.ln_1(x))  # 残差连接+注意力
        x = x + self.mlp(self.ln_2(x))  # 残差连接+MLP
        return x


class Transformer(nn.Module):
    """
    Transformer模块，由多个ResidualAttentionBlock组成。
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        初始化Transformer模块。

        参数:
            width (int): 模型宽度（维度）。
            layers (int): 层数。
            heads (int): 注意力头数。
            attn_mask (torch.Tensor, optional): 注意力掩码。默认为None。
        """
        super().__init__()
        self.width = width  # 模型宽度
        self.layers = layers  # 层数
        # 创建多个ResidualAttentionBlock
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        return self.resblocks(x)  # 序列前向传播

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 模块，用于将图像转换为向量表示。
    
    参数:
        input_resolution (int): 输入图像的分辨率（例如 224）。
        patch_size (int): 图像分块的大小（例如 16）。
        width (int): 嵌入维度（模型宽度）。
        layers (int): Transformer层数。
        heads (int): 注意力头的数量。
        output_dim (int): 输出维度（通常与CLIP嵌入维度一致）。
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution  # 输入图像分辨率
        self.output_dim = output_dim  # 输出向量维度

        # 将图像分割为patches，并线性投影到width维
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5  # 缩放因子，用于初始化参数
        # 类别嵌入（class token）
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # 位置编码，包含所有patches + class token
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)  # 预处理层归一化

        # Transformer主干网络
        self.transformer = Transformer(width, layers, heads)

        # 后处理归一化层
        self.ln_post = LayerNorm(width)
        # 投影矩阵，将Transformer输出映射到最终输出空间
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为[N, C, H, W]。

        返回:
            torch.Tensor: 输出张量，形状为[N, output_dim]。
        """
        x = self.conv1(x)  # [N, C, H, W] -> [N, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [N, width, grid^2]
        x = x.permute(0, 2, 1)  # [N, grid^2, width]

        # 添加类别token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x], dim=1)  # [N, grid^2 + 1, width]

        # 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)  # 预归一化

        # 调整维度以适配Transformer输入格式(NLD -> LND)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  # Transformer前向传播
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 对[class]标记应用LayerNorm
        x = self.ln_post(x[:, 0, :])  # 只取[class]标记的输出

        # 最终投影
        if self.proj is not None:
            x = x @ self.proj  # 应用投影矩阵

        return x


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pre-training (CLIP) 模型。
    
    结合视觉和文本编码器，学习跨模态表示。
    """

    def __init__(self,
                 embed_dim: int,
                 # 视觉部分参数
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # 文本部分参数
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length  # 文本最大长度

        # 根据vision_layers类型选择不同的视觉编码器
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64  # 计算注意力头数
            # 使用ModifiedResNet作为视觉编码器
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64  # 计算注意力头数
            # 使用VisionTransformer作为视觉编码器
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # 文本Transformer模块
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 构建因果注意力掩码
        )

        self.vocab_size = vocab_size  # 词汇表大小
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # 位置编码
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)  # 最终归一化层

        # 文本特征投影矩阵
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # logits缩放因子
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()  # 初始化参数

    def initialize_parameters(self):
        """初始化模型参数"""
        # 初始化词嵌入权重
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # 初始化位置编码
        nn.init.normal_(self.positional_embedding, std=0.01)

        # 如果使用ModifiedResNet作为视觉编码器
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                # 初始化注意力相关权重
                for component in [self.visual.attnpool.q_proj, self.visual.attnpool.k_proj, 
                                 self.visual.attnpool.v_proj, self.visual.attnpool.c_proj]:
                    nn.init.normal_(component.weight, std=std)

            # 对ResNet的BN层进行特殊初始化
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 初始化Transformer参数
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)  # 注意力输入投影
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)  # 注意力输出投影
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)  # MLP第一层
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)  # MLP第二层

        # 初始化文本投影矩阵
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        构建因果注意力掩码，确保每个位置只能注意到之前的token
        
        返回:
            torch.Tensor: 注意力掩码
        """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))  # 初始化为负无穷
        mask.triu_(1)  # 上三角矩阵置为0（允许注意）
        return mask

    @property
    def dtype(self):
        """获取模型参数的数据类型"""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """编码图像"""
        return self.visual(image.type(self.dtype))  # 确保输入数据类型匹配

    def encode_text(self, text):
        """
        编码文本

        参数:
            text (torch.Tensor): 文本输入，形状为[N, context_length]

        返回:
            torch.Tensor: 文本特征向量
        """
        # 获取词嵌入
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # 添加位置编码
        x = x + self.positional_embedding.type(self.dtype)
        # 调整维度以适配Transformer输入格式(NLD -> LND)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Transformer前向传播
        x = self.transformer(x)
        # 恢复原始维度(LND -> NLD)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 应用最终归一化
        x = self.ln_final(x).type(self.dtype)

        # 提取EOS（end of sequence）标记的特征
        # 这里通过argmax找到最后一个非填充token的位置
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        """
        前向传播函数，计算图像和文本的相似度

        参数:
            image (torch.Tensor): 图像输入
            text (torch.Tensor): 文本输入

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像和文本的logits
        """
        # 分别编码图像和文本
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 特征归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """
    将模型中适用的参数转换为fp16半精度浮点数
    
    参数:
        model (nn.Module): 要转换的模型
    """

    def _convert_weights_to_fp16(l):
        """内部函数：将单个模块的权重转换为fp16"""
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()  # 权重转为half
            if l.bias is not None:
                l.bias.data = l.bias.data.half()  # 偏置转为half

        if isinstance(l, nn.MultiheadAttention):
            # 多头注意力相关参数
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()  # 转为half

        # 特定层的投影参数
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()  # 转为half

    model.apply(_convert_weights_to_fp16)  # 对整个模型应用转换
def build_model(state_dict: dict):
    """
    从预训练权重构建CLIP模型
    
    参数:
        state_dict (dict): 包含预训练权重的字典
        
    返回:
        CLIP: 构建好的CLIP模型
    """
    
    # 判断是使用Vision Transformer还是ResNet作为视觉编码器
    vit = "visual.proj" in state_dict

    if vit:
        # 如果是Vision Transformer
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # 获取视觉模型宽度
        # 计算视觉Transformer层数
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # 获取patch大小
        # 计算网格大小（基于位置编码）
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size  # 计算图像分辨率
    else:
        # 如果是ResNet
        # 计算每层的块数
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)  # 将counts转换为元组作为vision_layers
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # 获取视觉模型宽度
        # 计算输出宽度
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None  # ResNet不使用patch size
        # 验证输出宽度与位置编码匹配
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32  # 计算图像分辨率

    # 从权重中提取CLIP参数
    embed_dim = state_dict["text_projection"].shape[1]  # 嵌入维度
    context_length = state_dict["positional_embedding"].shape[0]  # 文本上下文长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 词汇表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 文本Transformer宽度
    transformer_heads = transformer_width // 64  # 注意力头数
    # 计算Transformer层数
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 创建CLIP模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 移除不需要的键
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 将模型参数转换为fp16
    convert_weights(model)
    
    # 加载状态字典
    model.load_state_dict(state_dict)
    
    # 返回评估模式的模型
    return model.eval()