# 导入必要的库和模块
from torch import nn  # PyTorch 核心模块
from models.Temporal_Model import *  # 自定义时间建模模块（如 Temporal_Transformer_Cls）
from models.Prompt_Learner import *  # 可学习提示模块，用于文本增强

class GenerateModel(nn.Module):
    """
    主模型类：结合图像、关键点、文本特征进行视频情感识别
    
    功能：
        - 提取视觉特征（图像和关键点）
        - 使用 CLIP 模型编码文本提示
        - 利用时序 Transformer 建模视频帧间关系
        - 最终输出分类 logits 和归一化特征向量
    """

    def __init__(self, input_text, clip_model, args):
        """
        初始化模型组件
        
        参数:
            input_text (list[str]): 输入文本描述（如类别名或描述）
            clip_model (CLIP): 预训练的 CLIP 模型
            args: 包含各种超参数的命名空间对象
        """
        super().__init__()
        self.args = args
        self.input_text = input_text

        # 可学习提示生成器：根据输入文本生成可微调的 prompts
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        # 已分词的原始 prompt，用于文本编码器输入
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # 文本编码器：使用 CLIP 的文本分支对 prompt 进行编码
        self.text_encoder = TextEncoder(clip_model)

        # 获取数据类型（通常是 float16 或 float32）
        self.dtype = clip_model.dtype

        # 图像编码器：使用 CLIP 的视觉分支提取每帧图像特征
        self.image_encoder = clip_model.visual

        # 时间建模网络：Transformer 架构，用于建模视频帧之间的时序关系
        self.temporal_net = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,  # 层数由 args 控制
            heads=8,                     # 多头注意力头数
            mlp_dim=1024,                # MLP 中间层维度
            dim_head=64                  # 每个 head 的维度
        )

        ####### 新增部分：landmark 编码相关 #######
        # landmark 编码器：复用 image_encoder（共享权重）
        self.landmark_encoder = clip_model.visual

        '''
        # 另一种设计：为 landmark 单独使用一个 temporal_net（被注释掉）
        self.temporal_net2 = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=512,
            depth=args.temporal_layers,
            heads=8,
            mlp_dim=1024,
            dim_head=64
        )
        '''
        #########################################

        # 存储原始 CLIP 模型（可能在其它地方使用）
        self.clip_model_ = clip_model

def forward(self, landmarks, image):
    """
    前向传播函数

    参数:
        landmarks: [N, T, C, H, W]   → 视频关键点图像序列
        image:     [N, T, C, H, W]   → 视频帧图像序列

    返回:
        im_output:        [N, num_classes]
        la_output:        [N, num_classes]
        video_features:   [N, D]    → 图像路径的全局表示（D为特征维度）
        text_features:    [num_classes, D] → 各类别文本向量（已归一化）
    """

    ################# Visual Part（图像分支） #################
    n, t, c, h, w = image.shape       # image: [N, T, C, H, W]
    image = image.contiguous().view(-1, c, h, w)
    # reshape → image: [N*T, C, H, W]，展平为帧级批次处理

    image_features = self.image_encoder(image.type(self.dtype))
    # image_features: [N*T, D]，D为CLIP视觉编码器输出维度（如512）

    image_features = image_features.contiguous().view(n, t, -1)
    # reshape → image_features: [N, T, D]，恢复视频结构

    video_features = self.temporal_net(image_features)
    # video_features: [N, D]，CLS token 输出表示整段视频（时序建模）

    video_features = video_features / video_features.norm(dim=-1, keepdim=True)
    # 归一化 → video_features: [N, D]，每个样本为单位向量

    ###########################################################

    ################# Landmark Part（关键点分支） #################
    n, t, c, h, w = landmarks.shape         # landmarks: [N, T, C, H, W]
    landmarks = landmarks.contiguous().view(-1, c, h, w)
    # reshape → [N*T, C, H, W]

    landmark_features = self.image_encoder(landmarks.type(self.dtype))
    # landmark_features: [N*T, D]

    landmark_features = landmark_features.contiguous().view(n, t, -1)
    # reshape → [N, T, D]

    la_video_features = self.temporal_net(landmark_features)
    # la_video_features: [N, D]，关键点视频路径的全局特征

    la_video_features = la_video_features / la_video_features.norm(dim=-1, keepdim=True)
    # la_video_features: [N, D]，归一化

    ###########################################################

    ################## Text Part（文本编码） ##################
    prompts = self.prompt_learner()
    # prompts: [num_classes, L, dim_text]，每类一个 prompt，L为token数

    tokenized_prompts = self.tokenized_prompts
    # tokenized_prompts: 预处理后的 tokenizer 输出，用于 mask 或位置信息

    text_features = self.text_encoder(prompts, tokenized_prompts)
    # text_features: [num_classes, D]，文本向量和图像特征对齐在同一空间

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    # 归一化 → [num_classes, D]

    ###########################################################

    ###### 分类得分计算：视频特征与文本特征之间的余弦相似度 ######
    im_output = video_features @ text_features.t() / 0.01
    # im_output: [N, D] × [D, num_classes] → [N, num_classes]

    la_output = la_video_features @ text_features.t() / 0.01
    # la_output: [N, num_classes]

    ###########################################################

    return im_output, la_output, video_features, la_video_features
