# model.py
# -*- coding: utf-8 -*-
"""
Hand-to-Gripper (2-finger) Mapping Model (Left/Right Only)
Without Contact (HOI) - Ablation Study

本模型将手部关键点映射到夹爪的左右指尖，不再预测 base 点。
**此版本去除了contact模态输入，用于消融实验。**

输入:
- color:        [B, 3, H, W]    RGB图像
- bbox:         [B, 4]          手部边界框 [x1,y1,x2,y2]
- keypoints_3d: [B, 21, 3]      手部3D关键点
- contact:      [B, 21]         接触概率 (保留参数但不使用)
- is_right:     [B]             1=右手, 0=左手 (作为特征输入)

输出:
- logits_left:  [B,21]          左指尖的边际得分
- logits_right: [B,21]          右指尖的边际得分
- S_lr:         [B,21,21]       (left=i, right=j) 成对相容性得分
- pred_pair:    [B,2]           (i_left*, j_right*) 由联合打分最大化得到
- img_emb:      [B,D]           图像全局嵌入
- node_emb:     [B,21,D]        节点嵌入

关键点归一化流程:
1. 将掌根(joint 0)移动到原点
2. 用掌根(0)和四指根部(5,9,13,17)拟合平面，计算掌心法向量
3. 将法向量旋转到x轴正方向（掌心朝x正向）
4. 全局尺度归一化（所有点到原点距离的均值为1）

模型架构:
- DINOv2Backbone: 使用预训练DINOv2提取图像特征（保留空间维度）
- HandNodeEncoder: 使用图注意力网络建模手指骨骼连接结构 + Cross-Attention融合图像
- PairDecoder: 预测 left/right 的边际得分和成对相容性
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ------------------------------
# 手部骨骼图结构定义
# ------------------------------
# 21个关键点的连接关系（骨骼边）
# 0: Wrist
# 1-4: Thumb (CMC, MCP, IP, TIP)
# 5-8: Index (MCP, PIP, DIP, TIP)
# 9-12: Middle (MCP, PIP, DIP, TIP)
# 13-16: Ring (MCP, PIP, DIP, TIP)
# 17-20: Pinky (MCP, PIP, DIP, TIP)

HAND_EDGES = [
    # 拇指
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 掌心横向连接（可选，增强掌心结构）
    (5, 9), (9, 13), (13, 17),
]


def build_hand_adjacency_matrix(num_joints: int = 21, edges: list = HAND_EDGES, 
                                  self_loop: bool = True) -> torch.Tensor:
    """构建手部骨骼的邻接矩阵"""
    adj = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0  # 无向图
    if self_loop:
        adj = adj + torch.eye(num_joints)
    # 归一化（对称归一化）
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    adj = adj / deg.sqrt() / deg.sqrt().T
    return adj


# ------------------------------
# DINOv2 视觉骨干
# ------------------------------
class DINOv2Backbone(nn.Module):
    """
    使用预训练DINOv2提取图像特征
    输出: 保留空间维度的feature map [B, H', W', D] 和 全局特征 [B, D]
    """
    def __init__(self, model_name: str = "dinov2_vits14", out_dim: int = 256, 
                 freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        
        # 加载DINOv2模型
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        print(f"Loaded DINOv2 model: {model_name}")
        
        # DINOv2特征维度
        if 'vits' in model_name:
            dino_dim = 384
        elif 'vitb' in model_name:
            dino_dim = 768
        elif 'vitl' in model_name:
            dino_dim = 1024
        elif 'vitg' in model_name:
            dino_dim = 1536
        else:
            dino_dim = 384  # default
        
        self.dino_dim = dino_dim
        self.out_dim = out_dim
        
        # 投影层
        self.proj = nn.Linear(dino_dim, out_dim)
        self.proj_spatial = nn.Conv2d(dino_dim, out_dim, 1)
        
        # 冻结DINOv2参数
        if freeze_backbone:
            for param in self.dino.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            feat_map: [B, H', W', D] 空间特征图
            feat_global: [B, D] 全局特征
        """
        B, C, H, W = x.shape
        
        # DINOv2期望224x224或能被14整除的尺寸
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.dino.forward_features(x)
            
            if isinstance(features, dict):
                patch_tokens = features['x_norm_patchtokens']  # [B, N, D]
                cls_token = features['x_norm_clstoken']  # [B, D]
            else:
                # 旧版本API
                patch_tokens = features[:, 1:, :]  # 去掉CLS token
                cls_token = features[:, 0, :]
        
        # 重塑为空间维度 (patch_size=14, 224/14=16)
        h = w = 224 // 14  # = 16
        feat_map = patch_tokens.view(B, h, w, self.dino_dim)  # [B, 16, 16, D]
        
        # 投影
        feat_map_proj = self.proj_spatial(feat_map.permute(0, 3, 1, 2))  # [B, out_dim, 16, 16]
        feat_map_proj = feat_map_proj.permute(0, 2, 3, 1)  # [B, 16, 16, out_dim]
        
        feat_global = self.proj(cls_token)  # [B, out_dim]
        
        return feat_map_proj, feat_global


# ------------------------------
# 图注意力层 (Graph Attention)
# ------------------------------
class GraphAttentionLayer(nn.Module):
    """单头图注意力层"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, N, in_dim] 节点特征
            adj: [N, N] 邻接矩阵
        Returns:
            h_out: [B, N, out_dim]
        """
        B, N, _ = h.shape
        
        # 线性变换
        Wh = self.W(h)  # [B, N, out_dim]
        
        # 计算注意力系数
        # 拼接每对节点的特征
        Wh_repeat_i = Wh.unsqueeze(2).expand(B, N, N, self.out_dim)  # [B, N, N, D]
        Wh_repeat_j = Wh.unsqueeze(1).expand(B, N, N, self.out_dim)  # [B, N, N, D]
        concat = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=-1)  # [B, N, N, 2D]
        
        e = self.leaky_relu(self.a(concat).squeeze(-1))  # [B, N, N]
        
        # 只在邻接位置计算注意力
        adj = adj.to(h.device)
        mask = (adj == 0)
        e = e.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        alpha = self.dropout(alpha)
        
        # 加权聚合
        h_out = torch.bmm(alpha, Wh)  # [B, N, out_dim]
        
        return h_out


class MultiHeadGraphAttention(nn.Module):
    """多头图注意力"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.head_dim = out_dim // num_heads
        self.num_heads = num_heads
        
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, self.head_dim, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 多头注意力
        head_outputs = [head(h, adj) for head in self.heads]
        h_multi = torch.cat(head_outputs, dim=-1)  # [B, N, out_dim]
        
        # 投影 + 残差
        h_out = self.proj(h_multi)
        h_out = self.dropout(h_out)
        
        # 如果维度匹配，加残差
        if h.shape[-1] == h_out.shape[-1]:
            h_out = h_out + h
        return self.norm(h_out)


# ------------------------------
# Cross-Attention 模块
# ------------------------------
class CrossAttention(nn.Module):
    """
    Cross-Attention: Query来自关节特征，Key/Value来自图像特征图
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D] 关节特征 (N_q=21)
            kv: [B, N_kv, D] 图像特征 (N_kv=H'*W')
        Returns:
            out: [B, N_q, D]
        """
        B, N_q, D = query.shape
        _, N_kv, _ = kv.shape
        
        # 投影
        Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_kv]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)  # [B, H, N_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)
        
        # 残差 + LayerNorm
        query = query + self.dropout(out)
        query = self.norm1(query)
        
        # FFN
        query = query + self.ffn(query)
        query = self.norm2(query)
        
        return query


# ------------------------------
# 节点特征编码 + Graph Attention + Cross-Attention
# ------------------------------
class HandNodeEncoder(nn.Module):
    """
    节点编码器：
    1. MLP编码节点特征
    2. Graph Attention建模骨骼结构
    3. Cross-Attention融合图像特征
    4. Transformer自注意力
    """
    def __init__(self, in_dim: int = 25, hidden: int = 256, out_dim: int = 256,
                 num_gat_layers: int = 2, num_cross_attn_layers: int = 2,
                 num_transformer_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # 1. 节点特征MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        
        # 2. 图注意力层（建模骨骼连接）
        self.gat_layers = nn.ModuleList([
            MultiHeadGraphAttention(out_dim, out_dim, num_heads=4, dropout=dropout)
            for _ in range(num_gat_layers)
        ])
        
        # 3. Cross-Attention层（融合图像特征）
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model=out_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])
        
        # 4. Transformer自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=num_heads, dim_feedforward=out_dim * 2,
            batch_first=True, dropout=dropout, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 预计算邻接矩阵（注册为buffer）
        adj = build_hand_adjacency_matrix(21, HAND_EDGES, self_loop=True)
        self.register_buffer('adj', adj)
    
    def forward(self, node_feats: torch.Tensor, img_feat_map: torch.Tensor,
                img_feat_global: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_feats: [B, 21, in_dim] 节点特征
            img_feat_map: [B, H', W', D] 图像空间特征
            img_feat_global: [B, D] 全局图像特征（可选）
        Returns:
            H: [B, 21, out_dim]
        """
        B = node_feats.shape[0]
        
        # 1. MLP编码
        H = self.node_mlp(node_feats)  # [B, 21, D]
        
        # 2. 图注意力（建模骨骼结构）
        for gat in self.gat_layers:
            H = gat(H, self.adj)  # [B, 21, D]
        
        # 3. Cross-Attention（融合图像特征）
        # 将图像特征展平为序列
        H_img, W_img, D = img_feat_map.shape[1], img_feat_map.shape[2], img_feat_map.shape[3]
        img_seq = img_feat_map.view(B, H_img * W_img, D)  # [B, H'*W', D]
        
        for cross_attn in self.cross_attn_layers:
            H = cross_attn(H, img_seq)  # [B, 21, D]
        
        # 4. Transformer自注意力
        H = self.transformer(H)  # [B, 21, D]
        
        return H


# ------------------------------
# 成对解码器：left/right 两个查询 + 成对相容性
# ------------------------------
class PairDecoder(nn.Module):
    """
    成对解码器 (Pair Decoder)
    
    功能：
    - 预测夹爪左指尖 (left) 和右指尖 (right) 对应的手部关键点
    - 不再预测 base 点
    
    输出：
    - logits_left:  [B, 21]      每个关键点作为左指尖的边际得分
    - logits_right: [B, 21]      每个关键点作为右指尖的边际得分  
    - S_lr:         [B, 21, 21]  成对相容性矩阵 S_lr[i,j] 表示 (left=i, right=j) 的相容性
    - pred_pair:    [B, 2]       最终预测的 (left_idx, right_idx)
    
    预测方式：
    - 联合得分 = logits_left[i] + logits_right[j] + S_lr[i,j]
    - 找到使联合得分最大的 (i, j) 对
    """
    def __init__(self, d_model: int = 256):
        """
        初始化成对解码器
        
        Args:
            d_model: 模型隐藏维度，与节点嵌入维度一致
        """
        super().__init__()
        
        # ====== 两个语义查询向量 ======
        # q_left: 用于计算每个关键点作为"左指尖"的边际得分
        # q_right: 用于计算每个关键点作为"右指尖"的边际得分
        self.q_left = nn.Parameter(torch.randn(d_model))
        self.q_right = nn.Parameter(torch.randn(d_model))
        
        # 初始化查询向量
        for q in [self.q_left, self.q_right]:
            nn.init.normal_(q, mean=0.0, std=0.02)

        # ====== 成对相容性双线性矩阵 ======
        # W_lr: 用于计算 left-right 成对相容性
        # S_lr[i,j] = H[i] @ W_lr @ H[j].T  表示关键点i作为left、j作为right的相容程度
        self.W_lr = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W_lr)

    def forward(self, H: torch.Tensor):
        """
        前向传播
        
        Args:
            H: [B, 21, D] 节点嵌入，由 HandNodeEncoder 输出
            
        Returns:
            logits_left:  [B, 21]     左指尖边际得分
            logits_right: [B, 21]     右指尖边际得分
            S_lr:         [B, 21, 21] 成对相容性矩阵
            pred_pair:    [B, 2]      预测的 (left_idx, right_idx)
        """
        B, N, D = H.shape  # B=batch, N=21关键点, D=隐藏维度

        # ====== Step 1: 计算边际得分 ======
        # 每个关键点与查询向量做点积，得到该点作为 left/right 的得分
        logits_left  = torch.einsum('bnd,d->bn', H, self.q_left)   # [B, N]
        logits_right = torch.einsum('bnd,d->bn', H, self.q_right)  # [B, N]

        # ====== Step 2: 计算成对相容性 ======
        # S_lr[b,i,j] = H[b,i,:] @ W_lr @ H[b,j,:].T
        # 表示在第b个样本中，关键点i作为left、关键点j作为right的相容程度
        S_lr = torch.einsum('bnd,de,bme->bnm', H, self.W_lr, H)  # [B, N, N]

        # ====== Step 3: 联合打分找最优配对 ======
        # comb[i,j] = logits_left[i] + logits_right[j] + S_lr[i,j]
        # 允许 i=j (同一个关键点同时作为左右指尖)
        comb = (
            logits_left[:, :, None] +     # [B, N, 1]
            logits_right[:, None, :] +    # [B, 1, N]
            S_lr                          # [B, N, N]
        )  # [B, N, N]

        # 展平后找最大值索引
        comb_flat = comb.view(B, -1)             # [B, N^2]
        idx = torch.argmax(comb_flat, dim=1)     # [B]
        
        # 从展平索引恢复 (i_left, j_right)
        i_left = idx // N      # [B]
        j_right = idx % N      # [B]
        pred_pair = torch.stack([i_left, j_right], dim=1)  # [B, 2]

        return logits_left, logits_right, S_lr, pred_pair


# ------------------------------
# 顶层模型
# ------------------------------
class Hand2GripperModel(nn.Module):
    """
    Hand2Gripper 主模型 (Left/Right Only)
    
    功能：
    - 将手部3D关键点映射到夹爪的左右指尖
    - 使用 DINOv2 预训练骨干提取图像特征
    - 使用图注意力网络建模手指骨骼结构
    - 使用 Cross-Attention 融合图像特征
    
    架构：
    1. DINOv2Backbone: 提取图像特征 [B,H',W',D] 和全局特征 [B,D]
    2. HandNodeEncoder: 编码关节特征，融合图像，输出节点嵌入 [B,21,D]
    3. PairDecoder: 预测 left/right 的关键点索引
    
    输入：
    - img_crop:     [B, 3, S, S]   裁剪后的手部图像
    - keypoints_3d: [B, 21, 3]     3D关键点坐标
    - contact:      [B, 21]        接触概率 (保留但不使用)
    - is_right:     [B]            是否右手标志
    
    输出：
    - logits_left:  [B, 21]        左指尖边际得分
    - logits_right: [B, 21]        右指尖边际得分
    - S_lr:         [B, 21, 21]    成对相容性矩阵
    - pred_pair:    [B, 2]         预测的 (left_idx, right_idx)
    - img_emb:      [B, D]         图像全局嵌入
    - node_emb:     [B, 21, D]     节点嵌入
    """
    def __init__(self, d_model: int = 256, img_size: int = 256,
                 backbone: str = "dinov2_vits14", freeze_backbone: bool = True):
        """
        初始化模型
        
        Args:
            d_model: 模型隐藏维度
            img_size: 裁剪后的图像尺寸
            backbone: DINOv2骨干模型名称 ("dinov2_vits14", "dinov2_vitb14", etc.)
            freeze_backbone: 是否冻结 DINOv2 参数
        """
        super().__init__()
        self.img_size = img_size
        self.crop_scale = 1.2  # bbox扩展比例
        
        # ====== 视觉骨干 ======
        # 使用预训练 DINOv2 提取图像特征
        self.backbone = DINOv2Backbone(
            model_name=backbone, out_dim=d_model, freeze_backbone=freeze_backbone
        )
        
        # ====== 节点编码器 ======
        # 融合3D关键点特征和图像特征
        self.encoder = HandNodeEncoder(
            in_dim=25,                    # 3(xyz) + 21(onehot) + 1(is_right) NO CONTACT (HOI)
            hidden=d_model, 
            out_dim=d_model,
            num_gat_layers=2,             # 图注意力层数
            num_cross_attn_layers=2,      # Cross-Attention层数  
            num_transformer_layers=2,     # Transformer自注意力层数
            num_heads=8, 
            dropout=0.1
        )
        
        # ====== 成对解码器 ======
        # 预测 left/right 关键点
        self.decoder = PairDecoder(d_model=d_model)
    
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)

    @staticmethod
    def _normalize_keypoints_xyz(kp3d: torch.Tensor) -> torch.Tensor:
        """
        手掌姿态归一化：
        1. 将掌根(0)移动到原点
        2. 用掌根(0)和四指根部(5,9,13,17)拟合平面，计算法向量
        3. 将法向量旋转到x轴正方向（掌心朝x正向）
        4. 全局尺度归一化
        """
        kp = kp3d.clone()
        B = kp.shape[0]
        device = kp.device
        dtype = kp.dtype

        # Step 1: 掌根居中
        wrist = kp[:, 0:1, :]  # [B,1,3]
        kp = kp - wrist

        # Step 2: 提取5个关键点拟合平面
        palm_indices = [0, 5, 9, 13, 17]  # 掌根 + 四指MCP关节
        palm_pts = kp[:, palm_indices, :]  # [B,5,3]

        # 计算中心并去中心
        palm_center = palm_pts.mean(dim=1, keepdim=True)  # [B,1,3]
        palm_centered = palm_pts - palm_center  # [B,5,3]

        # SVD 找最小特征值对应的特征向量作为法向量
        U, S, Vh = torch.linalg.svd(palm_centered, full_matrices=False)  # Vh: [B,3,3]
        normal = Vh[:, 2, :]  # 最小奇异值对应的右奇异向量，即法向量 [B,3]

        # Step 3: 确保法向量朝向掌心
        finger_dir = kp[:, 9, :] - kp[:, 0, :]  # [B,3]
        finger_dir = finger_dir / (finger_dir.norm(dim=1, keepdim=True) + 1e-8)

        up_ref = kp[:, 10, :] - kp[:, 9, :]  # [B,3]
        up_ref = up_ref / (up_ref.norm(dim=1, keepdim=True) + 1e-8)

        expected_normal = torch.cross(finger_dir, up_ref, dim=1)  # [B,3]
        dot = (normal * expected_normal).sum(dim=1, keepdim=True)  # [B,1]
        normal = normal * torch.sign(dot + 1e-8)

        # Step 4: 构建旋转矩阵
        target = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        target = target.unsqueeze(0).expand(B, 3)

        normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-8)

        axis = torch.cross(normal, target, dim=1)
        axis_norm = axis.norm(dim=1, keepdim=True) + 1e-8
        axis = axis / axis_norm

        cos_theta = (normal * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        sin_theta = axis_norm.squeeze(-1).unsqueeze(-1)

        is_parallel = (axis_norm.squeeze(-1) < 1e-6)
        is_same_dir = (cos_theta.squeeze(-1) > 0)

        K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
        R = I + sin_theta.unsqueeze(-1) * K + (1 - cos_theta).unsqueeze(-1) * (K @ K)

        for b in range(B):
            if is_parallel[b]:
                if is_same_dir[b]:
                    R[b] = torch.eye(3, device=device, dtype=dtype)
                else:
                    R[b] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))

        kp = torch.bmm(kp, R.transpose(1, 2))

        dist = torch.norm(kp, dim=-1)
        scale = dist.mean(dim=1, keepdim=True).clamp(min=1e-6)
        kp = kp / scale.unsqueeze(-1)

        return kp

    @staticmethod
    def _build_node_features(kp_xyz_norm: torch.Tensor, contact: torch.Tensor, 
                              is_right: torch.Tensor) -> torch.Tensor:
        """拼接 [xyz | onehot | is_right] -> [B,21,25] (No Contact)"""
        B, N, _ = kp_xyz_norm.shape
        onehot = torch.eye(N, device=kp_xyz_norm.device).unsqueeze(0).repeat(B, 1, 1)
        # contact_f = contact.unsqueeze(-1) # 删除contact
        isr = is_right.view(B, 1, 1).repeat(1, N, 1).float()
        feats = torch.cat([kp_xyz_norm, onehot, isr], dim=-1) # Removed contact_f
        return feats

    @staticmethod
    def _expand_bbox(bbox: torch.Tensor, H: int, W: int, scale: float = 1.2) -> torch.Tensor:
        x1, y1, x2, y2 = bbox.unbind(dim=1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        w2 = w * scale / 2.0
        h2 = h * scale / 2.0
        nx1 = (cx - w2).clamp(min=0.0, max=W - 1.0)
        ny1 = (cy - h2).clamp(min=0.0, max=H - 1.0)
        nx2 = (cx + w2).clamp(min=0.0, max=W - 1.0)
        ny2 = (cy + h2).clamp(min=0.0, max=H - 1.0)
        return torch.stack([nx1, ny1, nx2, ny2], dim=1)
    
    def _read_color(self, color: np.ndarray) -> torch.Tensor:
        """
        入: color 可为 [H,W,3] (HWC) 或 [3,H,W] (CHW), uint8 或 float32/64
        出: [1,3,H,W], float32, 值域 [0,1]
        """
        if color.ndim != 3:
            raise ValueError(f"color ndim={color.ndim}, 需要 3 维")
        # 转 float32 且归一化
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        else:
            color = color.astype(np.float32, copy=False)

        if color.shape[0] == 3 and color.ndim == 3:       # CHW
            chw = color
        elif color.shape[-1] == 3:                        # HWC -> CHW
            chw = np.transpose(color, (2, 0, 1))
        else:
            raise ValueError(f"color 形状不合法: {color.shape}, 期望 CHW 或 HWC 且含 3 通道")

        chw = np.ascontiguousarray(chw)
        t = torch.from_numpy(chw).unsqueeze(0)            # [1,3,H,W]
        return t.float()


    def _read_bbox(self, bbox: np.ndarray) -> torch.Tensor:
        """
        入: bbox [4] (x1,y1,x2,y2)，任意数值类型
        出: [1,4] float32
        """
        bbox = np.asarray(bbox).astype(np.float32, copy=False)
        if bbox.shape != (4,):
            raise ValueError(f"bbox 形状应为 (4,), 实际 {bbox.shape}")
        return torch.from_numpy(bbox).unsqueeze(0).float()  # [1,4]


    def _read_keypoints_3d(self, keypoints_3d: np.ndarray) -> torch.Tensor:
        """
        入: [21,3] 或 [1,21,3]
        出: [1,21,3] float32
        """
        kp = np.asarray(keypoints_3d)
        if kp.ndim == 2:
            if kp.shape != (21, 3):
                raise ValueError(f"keypoints_3d 形状应为 (21,3)，实际 {kp.shape}")
            kp = kp[None, ...]                             # -> [1,21,3]
        elif kp.ndim == 3:
            if kp.shape[1:] != (21, 3) and not (kp.shape[0:3] == (1, 21, 3)):
                raise ValueError(f"不支持的 keypoints_3d 形状: {kp.shape}")
        else:
            raise ValueError(f"keypoints_3d 维度应为 2 或 3，实际 {kp.ndim}")

        kp = kp.astype(np.float32, copy=False)
        return torch.from_numpy(kp).float()               # [1,21,3]


    def _read_contact(self, contact: np.ndarray) -> torch.Tensor:
        """
        入: [21] 或 [1,21]
        出: [1,21] float32
        """
        c = np.asarray(contact)
        if c.ndim == 1:
            if c.shape != (21,):
                raise ValueError(f"contact 形状应为 (21,), 实际 {c.shape}")
            c = c[None, ...]                               # -> [1,21]
        elif c.ndim == 2 and c.shape[0] == 1 and c.shape[1] == 21:
            pass
        else:
            raise ValueError(f"不支持的 contact 形状: {c.shape}")
        c = c.astype(np.float32, copy=False)
        return torch.from_numpy(c).float()                # [1,21]


    def _read_is_right(self, is_right: np.ndarray) -> torch.Tensor:
        """
        入: 标量、[1]、或 [B] 的 0/1
        出: [1] long（模型内部会转 float 拼到特征）
        """
        ir = np.asarray(is_right)
        if ir.ndim == 0:
            ir = ir[None]                                  # -> [1]
        elif ir.ndim == 1 and ir.shape[0] == 1:
            pass
        else:
            raise ValueError(f"is_right 形状应为标量或(1,), 实际 {ir.shape}")
        ir = ir.astype(np.int64, copy=False)
        return torch.from_numpy(ir)                        # [1], long


    def _crop_and_resize(self, color: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        B, C, H, W = color.shape
        bbox = self._expand_bbox(bbox, H, W, self.crop_scale)
        crops = []
        for b in range(B):
            x1, y1, x2, y2 = bbox[b]
            x1i = int(torch.floor(x1).item())
            y1i = int(torch.floor(y1).item())
            x2i = int(torch.ceil(x2).item())
            y2i = int(torch.ceil(y2).item())
            x2i = max(x2i, x1i + 1)
            y2i = max(y2i, y1i + 1)
            patch = color[b:b+1, :, y1i:y2i, x1i:x2i]
            patch = F.interpolate(patch, size=(self.img_size, self.img_size), 
                                   mode="bilinear", align_corners=False)
            crops.append(patch)
        crop_img = torch.cat(crops, dim=0)
        return crop_img  # [B,3,S,S]

    def forward(self, img_crop: torch.Tensor, keypoints_3d: torch.Tensor,
                contact: torch.Tensor, is_right: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        模型前向传播
        
        Args:
            img_crop:     [B, 3, S, S]   裁剪并resize后的手部图像
            keypoints_3d: [B, 21, 3]     手部3D关键点坐标
            contact:      [B, 21]        接触概率 (保留接口但不使用)
            is_right:     [B]            是否右手 (1=右手, 0=左手)
            
        Returns:
            dict: 包含以下键值:
                - logits_left:  [B, 21]      左指尖边际得分
                - logits_right: [B, 21]      右指尖边际得分
                - S_lr:         [B, 21, 21]  成对相容性矩阵
                - pred_pair:    [B, 2]       预测的 (left_idx, right_idx)
                - img_emb:      [B, D]       图像全局嵌入
                - node_emb:     [B, 21, D]   节点嵌入
        """
        # ====== Step 1: 提取图像特征 ======
        # img_feat_map: [B, H', W', D] 空间特征图，用于 Cross-Attention
        # img_emb: [B, D] 全局特征，用于输出
        img_feat_map, img_emb = self.backbone(img_crop)
        
        # ====== Step 2: 关键点归一化与节点特征构建 ======
        # 将3D关键点归一化到统一坐标系
        kp_xyz_norm = self._normalize_keypoints_xyz(keypoints_3d)
        # 拼接节点特征: [xyz | (contact-removed) | onehot | is_right]
        node_feats = self._build_node_features(kp_xyz_norm, contact, is_right)

        # ====== Step 3: 编码 ======
        # 通过图注意力 + Cross-Attention + Transformer 得到节点嵌入
        H = self.encoder(node_feats, img_feat_map, img_emb)  # [B, 21, D]

        # ====== Step 4: 解码 ======
        # 预测 left/right 的关键点索引
        logits_left, logits_right, S_lr, pred_pair = self.decoder(H)

        return {
            "logits_left": logits_left,     # [B, 21] 左指尖边际得分
            "logits_right": logits_right,   # [B, 21] 右指尖边际得分
            "S_lr": S_lr,                   # [B, 21, 21] 成对相容性
            "pred_pair": pred_pair,         # [B, 2] 预测的 (left, right)
            "img_emb": img_emb,             # [B, D] 图像嵌入
            "node_emb": H,                  # [B, 21, D] 节点嵌入
        }


# ------------------------------
# 可视化工具
# ------------------------------
def visualize_hand_keypoints(kp_before: np.ndarray, kp_after: np.ndarray, 
                              title: str = "Hand Keypoints Normalization",
                              save_path: str = None):
    """
    可视化手部关键点归一化前后的对比
    
    Args:
        kp_before: [21, 3] 归一化前的关键点
        kp_after:  [21, 3] 归一化后的关键点
        title: 图表标题
        save_path: 保存路径，None则显示
    """
    import matplotlib.pyplot as plt
    
    finger_links = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    finger_colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Before Normalization')
    for finger_idx, links in enumerate(finger_links):
        pts = kp_before[links]
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', 
                 color=finger_colors[finger_idx], linewidth=2, markersize=4)
    ax1.scatter(*kp_before[0], color='black', s=100, marker='*', label='Wrist')
    palm_idx = [0, 5, 9, 13, 17]
    palm_pts = kp_before[palm_idx]
    ax1.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], 
                color='cyan', s=60, marker='s', label='Palm plane')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()
    _set_axes_equal(ax1)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('After Normalization (Palm normal → +X)')
    for finger_idx, links in enumerate(finger_links):
        pts = kp_after[links]
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', 
                 color=finger_colors[finger_idx], linewidth=2, markersize=4)
    ax2.scatter(*kp_after[0], color='black', s=100, marker='*', label='Wrist (origin)')
    palm_pts = kp_after[palm_idx]
    ax2.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], 
                color='cyan', s=60, marker='s', label='Palm plane')
    ax2.quiver(0, 0, 0, 0.5, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0.5, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0, 0.5, color='b', arrow_length_ratio=0.1, linewidth=2)
    ax2.text(0.55, 0, 0, '+X (palm normal)', fontsize=8)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.legend()
    _set_axes_equal(ax2)
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def _set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


# ------------------------------
# 演示
# ------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand2Gripper Model Demo")
    parser.add_argument("--npz", type=str, default="", help="Path to .npz sample file")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--save_vis", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--use_random", action="store_true", help="Use random data instead of npz file")
    parser.add_argument("--no_freeze_backbone", action="store_true", help="不冻结DINOv2参数")
    args = parser.parse_args()
    
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 初始化模型（默认使用DINOv2并冻结参数）
    model = Hand2GripperModel(
        d_model=256, img_size=256, 
        freeze_backbone=not args.no_freeze_backbone
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 加载checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model._load_checkpoint(args.checkpoint)
        print("Checkpoint loaded successfully")
    
    model.eval()

    if args.npz and os.path.exists(args.npz) and not args.use_random:
        print(f"Loading data from {args.npz}")
        data = np.load(args.npz, allow_pickle=True)
        
        img_rgb = data["img_rgb"]
        bbox = data["bbox"]
        kpts_3d = data["kpts_3d"]
        contact_logits = data["contact_logits"]
        is_right = data["is_right"]
        
        has_gt = "selected_gripper_blr_ids" in data
        if has_gt:
            gt_blr = data["selected_gripper_blr_ids"]
            print(f"Ground truth (base, left, right): {gt_blr}")
        
        img_rgb_t = model._read_color(img_rgb).to(device)
        bbox_t = model._read_bbox(bbox).to(device)
        kpts_3d_t = model._read_keypoints_3d(kpts_3d).to(device)
        contact_t = model._read_contact(contact_logits).to(device)
        is_right_t = model._read_is_right(is_right).to(device)
        
    else:
        print("Using random data for demo...")
        H, W = 480, 640
        
        img_rgb = np.random.rand(H, W, 3).astype(np.float32)
        bbox = np.array([120, 80, 320, 360], dtype=np.int32)
        kpts_3d = np.random.randn(21, 3).astype(np.float32) * 0.05
        contact_logits = np.random.rand(21).astype(np.float32)
        is_right = np.array([1], dtype=np.int64)
        
        has_gt = False
        
        img_rgb_t = model._read_color(img_rgb).to(device)
        bbox_t = model._read_bbox(bbox).to(device)
        kpts_3d_t = model._read_keypoints_3d(kpts_3d).to(device)
        contact_t = model._read_contact(contact_logits).to(device)
        is_right_t = model._read_is_right(is_right).to(device)

    # ===== 可视化归一化效果 =====
    print("="*80)
    print("Visualizing keypoints normalization...")
    kp_before = kpts_3d if kpts_3d.ndim == 2 else kpts_3d[0]  # [21,3]
    kp_after = model._normalize_keypoints_xyz(kpts_3d_t).cpu().numpy()[0]  # [21,3]
    visualize_hand_keypoints(
        kp_before, kp_after,
        title="Hand Keypoints Normalization",
        save_path=args.save_vis
    )

    # ===== 模型推理 =====
    print("="*80)
    print("Running model inference...")
    with torch.no_grad():
        crop_t = model._crop_and_resize(img_rgb_t, bbox_t)  # [1,3,256,256]
        out = model(crop_t, kpts_3d_t, contact_t, is_right_t)

    pred_pair = out["pred_pair"].cpu().numpy()[0]  # [2]
    print(f"Predicted (left, right): {pred_pair}")
    
    if has_gt:
        gt_lr = gt_blr[1:]  # 取 left, right 部分
        print(f"Ground truth (left, right): {gt_lr}")
        match = np.array_equal(pred_pair, gt_lr)
        print(f"Match: {match}")

    # 打印各logits的top-3
    print("-"*80)
    print("Top-3 predictions for each role:")
    for role, key in [("Left", "logits_left"), ("Right", "logits_right")]:
        logits = out[key].cpu().numpy()[0]  # [21]
        top3_idx = np.argsort(logits)[::-1][:3]
        top3_scores = logits[top3_idx]
        print(f"  {role}: {list(zip(top3_idx.tolist(), top3_scores.tolist()))}")

    print("-"*80)
    print(f"img_emb shape: {out['img_emb'].shape}")
    print(f"node_emb shape: {out['node_emb'].shape}")
    print("Done.")