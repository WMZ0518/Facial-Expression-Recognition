import torch
from torch import nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# 初始化一个简单的文本分词器
_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    """
    CLIP模型中的文本编码器模块，用于将提示（prompt）和对应的tokenized prompts编码为特征向量。

    参数:
        clip_model (CLIP): 一个完整的CLIP模型实例。
    """

    def __init__(self, clip_model):
        super().__init__()
        # 提取CLIP模型中的Transformer部分
        self.transformer = clip_model.transformer
        # 提取位置嵌入层（Positional Embedding）
        self.positional_embedding = clip_model.positional_embedding
        # 提取最终的LayerNorm层
        self.ln_final = clip_model.ln_final
        # 提取文本投影矩阵（Text Projection Matrix）
        self.text_projection = clip_model.text_projection
        # 获取当前模型的数据类型（如float32或bfloat16等）
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        前向传播函数，用于处理输入的prompts并生成对应的文本特征。

        参数:
            prompts (torch.Tensor): 输入的提示张量，形状为[N, n_ctx, dim]。
            tokenized_prompts (torch.Tensor): 已经token化的提示张量，形状为[N, n_ctx]。

        返回:
            torch.Tensor: 编码后的文本特征向量。
        """
        # 将位置嵌入加到prompts上，并指定数据类型
        x = prompts + self.positional_embedding.type(self.dtype)
        # 调整维度顺序：从NLD（Batch Size, Length, Dimension）转换为LND（Length, Batch Size, Dimension）
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 经过Transformer进行处理
        x = self.transformer(x)
        # 恢复维度顺序：从LND（Length, Batch Size, Dimension）恢复为NLD（Batch Size, Length, Dimension）
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 应用最终的LayerNorm层，并设置数据类型
        x = self.ln_final(x).type(self.dtype)

        # 提取每个序列中"End of Text"标记的特征，并与文本投影矩阵相乘以生成最终的文本特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    """
    可学习的Prompt模块，用于动态生成特定类别的文本提示（Prompts）。

    参数:
        class_names (List[str]): 类别名称列表。
        clip_model (CLIP): 一个完整的CLIP模型实例。
        args (argparse.Namespace): 包含训练参数的对象。
    """

    def __init__(self, class_names, clip_model, args):
        super().__init__()
        # 类别数量
        n_cls = len(class_names)
        # 上下文长度（即提示中的额外上下文tokens数）
        n_ctx = args.contexts_number
        # 数据类型（如float32等）
        dtype = clip_model.dtype
        # 上下文维度（由CLIP模型的最终LayerNorm层权重决定）
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 随机初始化上下文向量
        if args.class_specific_contexts == 'True':
            # 如果使用类别相关的上下文
            if args.load_and_tune_prompt_learner == 'True':
                print("Initializing class-specific contexts")
            # 创建形状为[n_cls, n_ctx, ctx_dim]的空张量，用于存储不同类别的上下文向量
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            # 如果使用通用的上下文
            if args.load_and_tune_prompt_learner == 'True':
                print("Initializing a generic context")
            # 创建形状为[n_ctx, ctx_dim]的空张量，用于存储通用上下文向量
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        # 使用正态分布对上下文向量进行初始化（标准差为0.02）
        nn.init.normal_(ctx_vectors, std=0.02)
        # 构造提示前缀，例如“X X X”，表示n_ctx个占位符
        prompt_prefix = " ".join(["X"] * n_ctx)

        # 将上下文向量包装为可优化的参数
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # 计算每个类别名称的token长度（用于后续构建提示模板）
        name_lens = [len(_tokenizer.encode(name)) for name in class_names]


        ####### 构建每个类别的提示（格式为：prefix + " " + class_name）
        prompts = [prompt_prefix + " " + name for name in class_names]
        if args.load_and_tune_prompt_learner == 'True':
            print(f'Initial context: "{prompt_prefix}"')  # 打印初始上下文
            print(f"Number of context words (tokens): {n_ctx}")  # 打印上下文token数量
            print('Prompts format: ', prompts[0])  # 打印第一个提示格式

        """
        Initializing a generic context
        Initial context: "X X X X"
        Number of context words (tokens): 4
        Prompts format:  X X X X angry
        Generated Prompts Shape: torch.Size([7, 6, 512])
        Tokenized Prompts Shape: torch.Size([7, 6])
        Encoded Text Features Shape: torch.Size([7, 512])
        """

        # 对所有提示进行tokenize处理，并拼接成一个tensor
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # 在不计算梯度的情况下获取token嵌入
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # 注册缓冲区，用于保存SOS（Start of Sentence）和EOS（End of Sentence）的token嵌入
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        # 存储其他相关信息
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # tokenized prompts张量
        self.name_lens = name_lens  # 每个类别的名称长度
        self.class_token_position = args.class_token_position  # 类别token的位置（end/middle/front）

    def forward(self):
        """
        前向传播函数，用于生成最终的提示（Prompts）张量。

        返回:
            torch.Tensor: 形状为[n_cls, *, dim]的提示张量。
        """
        # 获取当前的上下文向量
        ctx = self.ctx
        # 如果上下文是二维的，则扩展为三维（添加批次维度）
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        # 获取SOS和EOS token嵌入
        prefix = self.token_prefix
        suffix = self.token_suffix

        # 根据类别token的位置构造提示
        if self.class_token_position == "end":
            # 类别token放在末尾
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            # 类别token放在中间
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            # 类别token放在最前面
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            # 如果指定的位置无效，抛出异常
            raise ValueError

        return prompts
    
