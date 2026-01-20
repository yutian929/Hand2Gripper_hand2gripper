# model.py
# -*- coding: utf-8 -*-
"""
Hand-to-Gripper (2-finger) Mapping Model (Ordered Base/Left/Right)

输入:
- color:        [B, 3, H, W]
- bbox:         [B, 4]  [x1,y1,x2,y2]
- keypoints_3d: [B, 21, 3]
- contact:      [B, 21]
- is_right:     [B]     1=右手, 0=左手 (不镜像，仅作为特征)

输出:
- logits_base:  [B,21]
- logits_left:  [B,21]
- logits_right: [B,21]
- S_bl:         [B,21,21]  (base=i, left=j)   成对相容性
- S_br:         [B,21,21]  (base=i, right=k)  成对相容性
- S_lr:         [B,21,21]  (left=j, right=k)  成对相容性
- pred_triple:  [B,3]  (i_base*, j_left*, k_right*)  由三元联合打分 comb 最大化得到
- img_emb:      [B,D]
- node_emb:     [B,21,D]

关键点归一化流程:
1. 将掌根(joint 0)移动到原点
2. 用掌根(0)和四指根部(5,9,13,17)拟合平面，计算掌心法向量
3. 将法向量旋转到x轴正方向（掌心朝x正向）
4. 全局尺度归一化（所有点到原点距离的均值为1）
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ------------------------------
# 小型视觉骨干
# ------------------------------
class TinyCNN(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 3, 2, 1), nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], 3, 2, 1), nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], 3, 2, 1), nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], ch[4], 3, 2, 1), nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[4], ch[4], 3, 1, 1), nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(ch[4], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)   # B,32,128,128
        x = self.conv2(x)   # B,64,64,64
        x = self.conv3(x)   # B,128,32,32
        x = self.conv4(x)   # B,256,16,16
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # B,256
        x = self.proj(x)    # B,D
        return x


# ------------------------------
# 节点特征编码 + Transformer
# ------------------------------
class HandNodeEncoder(nn.Module):
    """
    节点输入 = [xyz(3) | contact(1) | onehot_joint(21) | is_right(1)] 共 26 维
    """
    def __init__(self, in_dim: int = 26, hidden: int = 256, n_layers: int = 2, out_dim: int = 256):
        super().__init__()
        mlp = []
        dim = in_dim
        for _ in range(n_layers):
            mlp += [nn.Linear(dim, hidden), nn.ReLU(inplace=True)]
            dim = hidden
        mlp += [nn.Linear(hidden, out_dim)]
        self.mlp = nn.Sequential(*mlp)

        enc = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=8, dim_feedforward=out_dim * 2, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=4)

        # FiLM 调制
        self.film_gamma = nn.Linear(out_dim, out_dim)
        self.film_beta = nn.Linear(out_dim, out_dim)

    def forward(self, node_feats: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        H = self.mlp(node_feats)  # [B,21,D]
        gamma = self.film_gamma(img_emb).unsqueeze(1)  # [B,1,D]
        beta = self.film_beta(img_emb).unsqueeze(1)    # [B,1,D]
        H = gamma * H + beta
        H = self.tr(H)  # [B,21,D]
        return H


# ------------------------------
# 三元解码器：三个查询 + 三个成对相容性
# ------------------------------
class TripleDecoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        # 三个有语义的查询
        self.q_base = nn.Parameter(torch.randn(d_model))
        self.q_left = nn.Parameter(torch.randn(d_model))
        self.q_right = nn.Parameter(torch.randn(d_model))
        for q in [self.q_base, self.q_left, self.q_right]:
            nn.init.normal_(q, mean=0.0, std=0.02)

        # 三个成对相容性的双线性矩阵
        self.W_bl = nn.Parameter(torch.empty(d_model, d_model))  # base-left
        self.W_br = nn.Parameter(torch.empty(d_model, d_model))  # base-right
        self.W_lr = nn.Parameter(torch.empty(d_model, d_model))  # left-right
        nn.init.xavier_uniform_(self.W_bl)
        nn.init.xavier_uniform_(self.W_br)
        nn.init.xavier_uniform_(self.W_lr)

    def forward(self, H: torch.Tensor):
        """
        H: [B,21,D]
        返回:
          logits_base/left/right: [B,21]
          S_bl/S_br/S_lr: [B,21,21]
          pred_triple: [B,3]
        """
        B, N, D = H.shape

        # 边际打分
        logits_base  = torch.einsum('bnd,d->bn', H, self.q_base)   # [B,N]
        logits_left  = torch.einsum('bnd,d->bn', H, self.q_left)   # [B,N]
        logits_right = torch.einsum('bnd,d->bn', H, self.q_right)  # [B,N]

        # 成对相容性
        S_bl = torch.einsum('bnd,de,bme->bnm', H, self.W_bl, H)  # (base=i, left=j)
        S_br = torch.einsum('bnd,de,bme->bnm', H, self.W_br, H)  # (base=i, right=k)
        S_lr = torch.einsum('bnd,de,bme->bnm', H, self.W_lr, H)  # (left=j, right=k)

        # 三元联合打分 (允许 i=j=k)
        # comb[b, i, j, k] = logits_base[i] + logits_left[j] + logits_right[k]
        #                   + S_bl[i,j] + S_br[i,k] + S_lr[j,k]
        comb = (
            logits_base[:, :, None, None] +
            logits_left[:, None, :, None] +
            logits_right[:, None, None, :] +
            S_bl[:, :, :, None] +
            S_br[:, :, None, :] +
            S_lr[:, None, :, :]
        )  # [B,21,21,21]

        comb_flat = comb.view(B, -1)             # [B, 21^3]
        idx = torch.argmax(comb_flat, dim=1)     # [B]
        i_base = idx // (N * N)
        rem = idx % (N * N)
        j_left = rem // N
        k_right = rem % N
        pred_triple = torch.stack([i_base, j_left, k_right], dim=1)  # [B,3]

        return logits_base, logits_left, logits_right, S_bl, S_br, S_lr, pred_triple


# ------------------------------
# 顶层模型
# ------------------------------
class Hand2GripperModel(nn.Module):
    def __init__(self, d_model: int = 256, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.crop_scale = 1.2
        self.backbone = TinyCNN(out_dim=d_model)
        self.encoder = HandNodeEncoder(in_dim=26, hidden=d_model, n_layers=2, out_dim=d_model)
        self.decoder = TripleDecoder(d_model=d_model)
    
    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.load_state_dict(checkpoint)


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
        # palm_centered: [B,5,3] -> 对每个batch做SVD
        U, S, Vh = torch.linalg.svd(palm_centered, full_matrices=False)  # Vh: [B,3,3]
        normal = Vh[:, 2, :]  # 最小奇异值对应的右奇异向量，即法向量 [B,3]

        # Step 3: 确保法向量朝向掌心（与中指方向叉积判断）
        # 中指方向：从掌根(0)到中指MCP(9)
        finger_dir = kp[:, 9, :] - kp[:, 0, :]  # [B,3]
        finger_dir = finger_dir / (finger_dir.norm(dim=1, keepdim=True) + 1e-8)

        # 从掌根到中指tip的方向用于判断
        # 使用中指MCP到中指PIP的方向作为"向上"参考
        up_ref = kp[:, 10, :] - kp[:, 9, :]  # [B,3]
        up_ref = up_ref / (up_ref.norm(dim=1, keepdim=True) + 1e-8)

        # 计算预期的法向量方向：finger_dir × up_ref 应该大致指向掌心
        expected_normal = torch.cross(finger_dir, up_ref, dim=1)  # [B,3]

        # 如果 normal 与 expected_normal 方向相反，则翻转
        dot = (normal * expected_normal).sum(dim=1, keepdim=True)  # [B,1]
        normal = normal * torch.sign(dot + 1e-8)  # 确保同向

        # Step 4: 构建旋转矩阵，将 normal 旋转到 x 轴正方向
        # 目标：normal -> [1, 0, 0]
        target = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        target = target.unsqueeze(0).expand(B, 3)  # [B,3]

        # 使用 Rodrigues 公式计算旋转矩阵
        normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-8)  # 归一化

        # 旋转轴 = normal × target
        axis = torch.cross(normal, target, dim=1)  # [B,3]
        axis_norm = axis.norm(dim=1, keepdim=True) + 1e-8
        axis = axis / axis_norm  # [B,3]

        # 旋转角 cos(theta) = normal · target
        cos_theta = (normal * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # [B,1]
        sin_theta = axis_norm.squeeze(-1).unsqueeze(-1)  # [B,1]

        # 处理 normal ≈ target 或 normal ≈ -target 的特殊情况
        # 当 axis_norm 很小时，说明两向量平行
        is_parallel = (axis_norm.squeeze(-1) < 1e-6)  # [B]
        is_same_dir = (cos_theta.squeeze(-1) > 0)  # [B]

        # Rodrigues: R = I + sin(θ)[K] + (1-cos(θ))[K]^2
        # 其中 [K] 是 axis 的反对称矩阵
        K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
        R = I + sin_theta.unsqueeze(-1) * K + (1 - cos_theta).unsqueeze(-1) * (K @ K)

        # 处理特殊情况
        for b in range(B):
            if is_parallel[b]:
                if is_same_dir[b]:
                    R[b] = torch.eye(3, device=device, dtype=dtype)
                else:
                    # 180度旋转，绕y轴或z轴
                    R[b] = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))

        # Step 5: 应用旋转
        kp = torch.bmm(kp, R.transpose(1, 2))  # [B,21,3] @ [B,3,3] -> [B,21,3]

        # Step 6: 全局尺度归一化
        dist = torch.norm(kp, dim=-1)  # [B,21]
        scale = dist.mean(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]
        kp = kp / scale.unsqueeze(-1)

        return kp

    @staticmethod
    def _build_node_features(kp_xyz_norm: torch.Tensor, contact: torch.Tensor, is_right: torch.Tensor) -> torch.Tensor:
        """拼接 [xyz | contact | onehot | is_right] -> [B,21,26]"""
        B, N, _ = kp_xyz_norm.shape
        onehot = torch.eye(N, device=kp_xyz_norm.device).unsqueeze(0).repeat(B, 1, 1)  # [B,21,21]
        contact_f = contact.unsqueeze(-1)                                              # [B,21,1]
        isr = is_right.view(B, 1, 1).repeat(1, N, 1).float()                           # [B,21,1]
        feats = torch.cat([kp_xyz_norm, contact_f, onehot, isr], dim=-1)               # [B,21,26]
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
            patch = F.interpolate(patch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            crops.append(patch)
        crop_img = torch.cat(crops, dim=0)
        return crop_img  # [B,3,S,S]

    def forward(self, img_crop: torch.Tensor, keypoints_3d: torch.Tensor,
                contact: torch.Tensor, is_right: torch.Tensor) -> Dict[str, torch.Tensor]:

        img_emb = self.backbone(img_crop)                              # [B,D]

        kp_xyz_norm = self._normalize_keypoints_xyz(keypoints_3d)  # [B,21,3]
        node_feats = self._build_node_features(kp_xyz_norm, contact, is_right)  # [B,21,26]
        H = self.encoder(node_feats, img_emb)                          # [B,21,D]

        logits_base, logits_left, logits_right, S_bl, S_br, S_lr, pred_triple = self.decoder(H)

        return {
            "logits_base": logits_base,
            "logits_left": logits_left,
            "logits_right": logits_right,
            "S_bl": S_bl,
            "S_br": S_br,
            "S_lr": S_lr,
            "pred_triple": pred_triple,
            "img_emb": img_emb,
            "node_emb": H,
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
    from mpl_toolkits.mplot3d import Axes3D
    
    # 手指连接关系：每根手指从掌根或MCP连接到指尖
    # 0: 掌根, 1-4: 拇指, 5-8: 食指, 9-12: 中指, 13-16: 无名指, 17-20: 小指
    finger_links = [
        [0, 1, 2, 3, 4],      # 拇指
        [0, 5, 6, 7, 8],      # 食指
        [0, 9, 10, 11, 12],   # 中指
        [0, 13, 14, 15, 16],  # 无名指
        [0, 17, 18, 19, 20],  # 小指
    ]
    finger_colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    fig = plt.figure(figsize=(14, 6))
    
    # 归一化前
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Before Normalization')
    for finger_idx, links in enumerate(finger_links):
        pts = kp_before[links]
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', 
                 color=finger_colors[finger_idx], linewidth=2, markersize=4)
    ax1.scatter(*kp_before[0], color='black', s=100, marker='*', label='Wrist')
    # 绘制掌心平面点
    palm_idx = [0, 5, 9, 13, 17]
    palm_pts = kp_before[palm_idx]
    ax1.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], 
                color='cyan', s=60, marker='s', label='Palm plane')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    _set_axes_equal(ax1)
    
    # 归一化后
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('After Normalization (Palm normal → +X)')
    for finger_idx, links in enumerate(finger_links):
        pts = kp_after[links]
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', 
                 color=finger_colors[finger_idx], linewidth=2, markersize=4)
    ax2.scatter(*kp_after[0], color='black', s=100, marker='*', label='Wrist (origin)')
    # 绘制掌心平面点
    palm_pts = kp_after[palm_idx]
    ax2.scatter(palm_pts[:, 0], palm_pts[:, 1], palm_pts[:, 2], 
                color='cyan', s=60, marker='s', label='Palm plane')
    # 绘制坐标轴
    ax2.quiver(0, 0, 0, 0.5, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0.5, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0, 0.5, color='b', arrow_length_ratio=0.1, linewidth=2)
    ax2.text(0.55, 0, 0, '+X (palm normal)', fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
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
    """设置3D坐标轴等比例"""
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
# 演示: 从npz文件读取数据并可视化
# ------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand2Gripper Model Demo")
    parser.add_argument("--npz", type=str, default="", help="Path to .npz sample file")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--save_vis", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--use_random", action="store_true", help="Use random data instead of npz file")
    args = parser.parse_args()
    
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 初始化模型
    model = Hand2GripperModel(d_model=256, img_size=256).to(device)
    
    # 加载checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        model._load_checkpoint(args.checkpoint)
        print("Checkpoint loaded successfully")
    
    model.eval()

    if args.npz and os.path.exists(args.npz) and not args.use_random:
        # ===== 从npz文件读取数据 =====
        print(f"Loading data from {args.npz}")
        data = np.load(args.npz, allow_pickle=True)
        
        # 读取数据（参考train.py中的格式）
        # img_rgb: (H, W, 3), uint8 或 float
        # bbox: (4,), [x1, y1, x2, y2]
        # kpts_3d: (21, 3), float
        # contact_logits: (21,), float
        # is_right: (1,) 或标量, int
        # selected_gripper_blr_ids: (3,), int (可选，用于对比GT)
        
        img_rgb = data["img_rgb"]  # (H, W, 3)
        bbox = data["bbox"]  # (4,)
        kpts_3d = data["kpts_3d"]  # (21, 3)
        contact_logits = data["contact_logits"]  # (21,)
        is_right = data["is_right"]  # (1,) 或标量
        
        # 检查是否有GT标签
        has_gt = "selected_gripper_blr_ids" in data
        if has_gt:
            gt_blr = data["selected_gripper_blr_ids"]  # (3,)
            print(f"Ground truth (base, left, right): {gt_blr}")
        
        # 转换为tensor
        img_rgb_t = model._read_color(img_rgb).to(device)  # [1,3,H,W]
        bbox_t = model._read_bbox(bbox).to(device)  # [1,4]
        kpts_3d_t = model._read_keypoints_3d(kpts_3d).to(device)  # [1,21,3]
        contact_t = model._read_contact(contact_logits).to(device)  # [1,21]
        is_right_t = model._read_is_right(is_right).to(device)  # [1]
        
        print(f"Image shape: {img_rgb.shape}")
        print(f"Bbox: {bbox}")
        print(f"Keypoints 3D shape: {kpts_3d.shape}")
        print(f"Is right hand: {is_right}")
        
    else:
        # ===== 使用随机数据 =====
        print("Using random data for demo...")
        H, W = 480, 640
        
        img_rgb = np.random.rand(H, W, 3).astype(np.float32)  # (H,W,3)
        bbox = np.array([120, 80, 320, 360], dtype=np.int32)  # (4,)
        kpts_3d = np.random.randn(21, 3).astype(np.float32) * 0.05  # (21,3)
        contact_logits = np.random.rand(21).astype(np.float32)  # (21,)
        is_right = np.array([1], dtype=np.int64)  # (1,)
        
        has_gt = False
        
        # 转换为tensor
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

    pred_triple = out["pred_triple"].cpu().numpy()[0]  # [3]
    print(f"Predicted (base, left, right): {pred_triple}")
    
    if has_gt:
        print(f"Ground truth (base, left, right): {gt_blr}")
        match = np.array_equal(pred_triple, gt_blr)
        print(f"Match: {match}")

    # 打印各logits的top-3
    print("-"*80)
    print("Top-3 predictions for each role:")
    for role, key in [("Base", "logits_base"), ("Left", "logits_left"), ("Right", "logits_right")]:
        logits = out[key].cpu().numpy()[0]  # [21]
        top3_idx = np.argsort(logits)[::-1][:3]
        top3_scores = logits[top3_idx]
        print(f"  {role}: {list(zip(top3_idx.tolist(), top3_scores.tolist()))}")

    print("-"*80)
    print(f"img_emb shape: {out['img_emb'].shape}")
    print(f"node_emb shape: {out['node_emb'].shape}")
    print("Done.")