from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class EmoBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        # 存储分词器，用于文本处理
        self.tokenizer = tokenizer 

        # MLM（掩码语言模型）任务中被掩盖 token 的概率
        self.mlm_probability = config['mlm_probability']

        # 嵌入维度，即最终特征向量的维度
        embed_dim = config['embed_dim']
     
        # 创建视觉编码器：使用 VisionTransformer 构建图像特征提取器
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'],   # 图像输入尺寸，如224x224
            patch_size=16,                  # 每个图像块大小为16x16
            embed_dim=768,                  # 图像块嵌入维度
            depth=12,                       # Transformer 编码层数
            num_heads=12,                   # 注意力头数量
            mlp_ratio=4,                    # MLP 隐藏层扩展比例
            qkv_bias=True,                  # 使用查询/键/值偏置
            norm_layer=partial(nn.LayerNorm, eps=1e-6)  # 层归一化配置
        )   
        
        # 如果启用 DeiT 初始化，则加载预训练权重并调整位置嵌入
        if init_deit:
            # 从指定 URL 加载 DeiT 预训练权重
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            
            # 调整位置嵌入以适配当前模型设置
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            
            # 加载状态字典到视觉编码器中，允许不完全匹配
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)          
            
        # 获取视觉编码器输出维度
        vision_width = config['vision_width']       
        
        # 加载 BERT 配置文件
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        # 加载预训练的 BERT 文本编码器，用于 MLM 任务
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        # 获取文本编码器隐藏层维度
        text_width = self.text_encoder.config.hidden_size
        
        # 将视觉特征投影到共享的嵌入空间
        self.vision_proj = nn.Linear(vision_width, embed_dim)

        # 将文本特征投影到共享的嵌入空间
        self.text_proj = nn.Linear(text_width, embed_dim)         

        # 温度参数，控制对比学习中 softmax 分布的平滑程度
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        
        # 队列大小，用于存储历史负样本
        self.queue_size = config['queue_size']

        # 动量系数，用于更新动量模型参数
        self.momentum = config['momentum']  

        # ITM（图像文本匹配）分类头，输出是否匹配的二分类结果
        self.itm_head = nn.Linear(text_width, 2)     

        # 创建动量模型（Momentum Models）
        # 视觉编码器动量模型
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        
        # 视觉特征投影动量模型
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)

        # 文本编码器动量模型
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)       
        
        # 文本特征投影动量模型
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        # 配对模型列表，便于统一进行参数复制和更新
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        
        # 复制动量模型的初始参数
        self.copy_params()

        # 创建队列，用于存储历史批次的图像和文本特征
        # 注册为 buffer，不参与梯度计算
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        # 对队列中的特征进行 L2 归一化
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def forward(self, image, text, alpha=0):
        """前向传播函数"""
        
        # 锁定温度参数范围
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        
        # 提取图像特征
        image_embeds = self.visual_encoder(image)  # shape: [B, N+1, D]

        # 生成注意力掩码（所有 token 有效）
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)  # shape: [B, N+1]

        # 提取图像 CLS token 并投影 + 归一化
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  # shape: [B, embed_dim]

        # 提取文本特征（BERT 输出）
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,                      
                                        return_dict=True, mode='text')            
        text_embeds = text_output.last_hidden_state  # shape: [B, T, H]
        
        # 提取文本 CLS token 并投影 + 归一化
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)  # shape: [B, embed_dim]                 

        # 获取动量特征
        with torch.no_grad():
            self._momentum_update()  # 更新动量模型参数

            # 使用动量模型提取图像特征
            image_embeds_m = self.visual_encoder_m(image)  # shape: [B, N+1, D]
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  # shape: [B, embed_dim]

            # 合并动量图像特征与队列中的历史特征
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # shape: [embed_dim, queue_size+B]

            # 使用动量模型提取文本特征
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,                      
                                                return_dict=True, mode='text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)  # shape: [B, embed_dim]

            # 合并动量文本特征与队列中的历史特征
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)  # shape: [embed_dim, queue_size+B]

            # 计算相似度矩阵（动量图像 vs 所有文本）
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # shape: [B, queue_size+B]
            # 计算相似度矩阵（动量文本 vs 所有图像）
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # shape: [B, queue_size+B]

            # 构造目标标签（正样本在对角线上）
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)  # shape: [B, queue_size+B]

            # 混合硬标签和软标签
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        # 计算当前模型下的图像-文本相似度
        sim_i2t = image_feat @ text_feat_all / self.temp  # shape: [B, queue_size+B]
        sim_t2i = text_feat @ image_feat_all / self.temp  # shape: [B, queue_size+B]
                             
        # 计算 InfoNCE 损失
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        # 总的对比损失
        loss_ita = (loss_i2t + loss_t2i) / 2

        # 更新队列中的图像和文本特征
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # 正样本对前向传播（图像-文本融合）

        output_pos = self.text_encoder.bert(encoder_embeds=text_embeds, 
                                        attention_mask=text.attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,      
                                        return_dict=True,
                                        mode='fusion',
                                       )            
        
        with torch.no_grad():
            bs = image.size(0)  # 获取 batch size
            
            # 计算图像到文本和文本到图像的 softmax 权重
            weights_i2t = F.softmax(sim_i2t[:,:bs], dim=1)   # 只选当前 batch 内的文本
            weights_t2i = F.softmax(sim_t2i[:,:bs], dim=1)

            # 排除对角线上的正样本（避免抽到自己作为负样本）
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # 为每个文本选择一个负图像
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # 根据权重采样
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)  # shape: [B, N+1, D]

        # 为每个图像选择一个负文本
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)   
        text_atts_neg = torch.stack(text_atts_neg, dim=0)      

        # 合并正负样本的文本和注意力掩码
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)     

        # 合并负图像和正图像
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)   

        # 负样本对前向传播
        output_neg = self.text_encoder.bert(encoder_embeds=text_embeds_all, 
                                        attention_mask=text_atts_all,
                                        encoder_hidden_states=image_embeds_all,
                                        encoder_attention_mask=image_atts_all,      
                                        return_dict=True,
                                        mode='fusion',
                                       )                         

        # 合并正负样本的 [CLS] 特征
        vl_embeddings = torch.cat([
            output_pos.last_hidden_state[:,0,:], 
            output_neg.last_hidden_state[:,0,:]
        ], dim=0)
        
        # 通过 ITM 分类头
        vl_output = self.itm_head(vl_embeddings)            

        # 构造 ITM 标签（1表示正样本，0表示负样本）
        itm_labels = torch.cat([
            torch.ones(bs, dtype=torch.long),
            torch.zeros(2*bs, dtype=torch.long)
        ], dim=0).to(image.device)
        
        # 计算 ITM 损失
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
        
        ##================= MLM ========================##                
        # 克隆输入 ID 和标签
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        # 构造 MLM 掩码矩阵
        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix=probability_matrix) 
        
        # 使用动量模型预测 soft label
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids, 
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,      
                                           return_dict=True,
                                           return_logits=True,   
                                          )    
        
        # 当前模型执行 MLM 任务
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,      
                                       return_dict=True,
                                       labels=labels,   
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha
                                      )                           
        loss_mlm = mlm_output.loss        

        # 返回三个任务的损失：MLM、ITA、ITM
        return loss_mlm, loss_ita, loss_itm  

    @torch.no_grad()    
    def copy_params(self):
        """
        将主模型参数复制给对应的动量模型。
        动量模型用于对比学习中的特征提取，保持稳定更新。
        """
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # 复制参数值
                param_m.data.copy_(param.data)
                # 设置为不参与梯度更新
                param_m.requires_grad = False


    @torch.no_grad()        
    def _momentum_update(self):
        """
        使用动量机制更新动量模型参数：
        param_m = momentum * param_m + (1 - momentum) * param
        这样可以保证动量模型缓慢更新，提升对比学习的稳定性。
        """
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                # 应用动量更新公式
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        """
        更新图像和文本特征队列（用于负样本采样）。
        先跨 GPU 收集当前批次特征，再将它们加入队列并移除旧特征。
        """
        # 跨 GPU 收集特征
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # 确保队列大小是 batch size 的整数倍

        # 替换队列中的特征（先进先出）
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # 移动指针

        self.queue_ptr[0] = ptr  # 更新指针位置


    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        """
        对输入 token 序列进行掩码处理，用于 MLM（掩码语言建模）任务。
        根据概率矩阵决定哪些 token 被掩盖，并按比例替换为 [MASK]、随机词或保留原词。
        """
        if masked_indices is None:                                       
            # 根据概率矩阵生成掩码
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        # 忽略 PAD 和 CLS token 的掩码
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            # 只计算被掩盖 token 的损失，未掩盖位置设为 -100
            targets[~masked_indices] = -100             

        # 80% 情况下：替换为 [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% 情况下：替换为随机词
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # 剩余 10%：保留原始 token（不做任何修改）

        # 返回处理后的输入 ID 和标签（如果有）
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    在分布式训练中跨 GPU 收集张量。
    *** 注意 ***: 此操作没有梯度传播。
    """
    # 获取世界大小（GPU 数量）
    world_size = torch.distributed.get_world_size()

    # 初始化收集列表
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]

    # 执行 all_gather 操作
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # 合并所有张量
    output = torch.cat(tensors_gather, dim=0)
    return output