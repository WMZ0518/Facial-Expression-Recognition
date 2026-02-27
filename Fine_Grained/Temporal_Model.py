# 导入必要的库
import torch
from einops import rearrange, repeat  # 张量操作工具
from torch import nn, einsum  # 神经网络模块和爱因斯坦求和
import math  # 数学函数


class GELU(nn.Module):
    """
    高斯误差线性单元激活函数（GELU）
    
    参考论文：Gaussian Error Linear Units (GELUs)
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    """
    残差连接模块
    
    功能：
        - 对任意函数 fn 添加残差连接（skip connection）
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x  # 输入 + 函数输出（残差连接）


class PreNorm(nn.Module):
    """
    LayerNorm + 函数调用组合模块（前置归一化）

    功能：
        - 在调用函数前进行 LayerNorm 归一化
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # LayerNorm 层
        self.fn = fn  # 要执行的函数

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 先归一化再调用函数


class FeedForward(nn.Module):
    """
    前馈神经网络模块（FFN）

    结构：
        - Linear -> GELU -> Dropout -> Linear -> Dropout
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）

    参数:
        dim: 输入维度
        heads: 注意力头数
        dim_head: 每个头的维度
        dropout: dropout 概率
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，防止内积过大

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 生成 QKV
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 生成 QKV
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # 计算 attention scores
        attn = dots.softmax(dim=-1)  # softmax 得到 attention 权重
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 加权聚合 values
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """
    标准 Transformer 编码器模块

    参数:
        dim: 输入特征维度
        depth: Transformer 层数
        heads: 注意力头数量
        dim_head: 每个头的维度
        mlp_dim: FFN 中间层维度
        dropout: dropout 概率
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)  # 多头注意力
            x = ff(x)    # 前馈网络
        return x


###########################################################
############# output = mean of the all tokens #############
###########################################################
class Temporal_Transformer_Mean(nn.Module):
    """
    时间建模模块（Transformer）+ 所有 token 平均池化输出

    参数:
        num_patches: 视频帧数量（token 数）
        input_dim: 输入特征维度
        depth: Transformer 层数
        heads: 注意力头数量
        mlp_dim: FFN 中间层维度
        dim_head: 每个头的维度
    """

    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout = 0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))  # 位置编码
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)  # reshape
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]  # 添加位置编码
        x = self.temporal_transformer(x)  # Transformer 编码
        x = x.mean(dim=1)  # 所有 token 取平均
        return x


###########################################################
#############      output = class tokens      #############
###########################################################
class Temporal_Transformer_Cls(nn.Module):
    """
    时间建模模块（Transformer）+ 使用 CLS token 作为输出表示

    参数:
        num_patches: 视频帧数量
        input_dim: 输入特征维度
        depth: Transformer 层数
        heads: 注意力头数量
        mlp_dim: FFN 中间层维度
        dim_head: 每个头的维度
    """

    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout = 0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))  # CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, input_dim))  # 位置编码（含 CLS）
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # 扩展为 batch 维度
        x = torch.cat((cls_tokens, x), dim=1)  # 拼接 CLS token
        x = x + self.pos_embedding[:, :(n + 1)]  # 添加位置编码
        x = self.temporal_transformer(x)  # Transformer 编码
        x = x[:, 0]  # 提取 CLS token 输出
        return x


###########################################################
#############        output = all tokens      #############
###########################################################
class Temporal_Transformer_All(nn.Module):
    """
    时间建模模块（Transformer）+ 返回所有 token 的输出

    参数:
        num_patches: 视频帧数量
        input_dim: 输入特征维度
        depth: Transformer 层数
        heads: 注意力头数量
        mlp_dim: FFN 中间层维度
        dim_head: 每个头的维度
    """

    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout = 0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))  # 位置编码
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)  # reshape
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]  # 添加位置编码
        x = self.temporal_transformer(x)  # Transformer 编码
        return x  # 返回所有 token 的输出