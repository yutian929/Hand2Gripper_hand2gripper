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
"""

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
        """不镜像，只做 wrist(0) 居中 + 全局尺度归一"""
        kp = kp3d.clone()
        wrist = kp[:, 0:1, :]
        kp = kp - wrist
        dist = torch.norm(kp, dim=-1)
        scale = dist.mean(dim=1, keepdim=True).clamp(min=1e-6)
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
        bbox = self._expand_bbox(bbox, H, W, scale=1.2)
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
# 演示: 随机数据跑一次 forward
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, H, W = 2, 480, 640
    model = Hand2GripperModel(d_model=256, img_size=256).to(device)
    model.eval()

    # 使用 numpy 生成随机样本
    color = np.random.rand(B, 3, H, W).astype(np.float32)  # [B, 3, H, W]
    bbox = np.array([[120, 80, 320, 360], [50, 60, 300, 420]], dtype=np.int32)  # [B, 4]
    keypoints_3d = np.random.randn(B, 21, 3).astype(np.float32) * 0.05  # [B, 21, 3]
    keypoints_3d[:, 0, :] = 0.0  # wrist set to 0
    contact = np.random.rand(B, 21).astype(np.float32)  # [B, 21]
    is_right = np.array([1, 0], dtype=np.int64)  # [B]

    # 转换为 torch tensor
    color = torch.tensor(color).to(device)  # [B, 3, H, W]
    bbox = torch.tensor(bbox).to(device)  # [B, 4]
    keypoints_3d = torch.tensor(keypoints_3d).to(device)  # [B, 21, 3]
    contact = torch.tensor(contact).to(device)  # [B, 21]
    is_right = torch.tensor(is_right).to(device)  # [B]

    with torch.no_grad():
        crop = model._crop_and_resize(color, bbox)
        out = model(crop, keypoints_3d, contact, is_right)
    print("-"*100)
    print("pred_triple:", out["pred_triple"])
    print("logits_base/left/right:", out["logits_base"].shape, out["logits_left"].shape, out["logits_right"].shape)
    print("S_bl/S_br/S_lr:", out["S_bl"].shape, out["S_br"].shape, out["S_lr"].shape)

    color      = np.random.rand(3, H, W).astype(np.float32)            # (3,H,W)
    bbox       = np.array([120, 80, 320, 360], np.int32)     # (4,)
    keypoints_3d = np.random.randn(21, 3).astype(np.float32) * 0.05    # (21,3)
    contact    = np.random.rand(21).astype(np.float32)                 # (21,)
    is_right   = np.array([1], dtype=np.int64)                         # (1,)

    # ===== 读入并扩成 batch =====
    color_t    = model._read_color(color)          # [1,3,H,W]
    bbox_t     = model._read_bbox(bbox)            # [1,4]
    kp3d_t     = model._read_keypoints_3d(keypoints_3d)  # [1,21,3]
    contact_t  = model._read_contact(contact)      # [1,21]
    isright_t  = model._read_is_right(is_right)    # [1]

    # ===== 裁剪并前向 =====
    crop_t     = model._crop_and_resize(color_t, bbox_t)   # [1,3,S,S]
    out        = model(crop_t.to(device),
                    kp3d_t.to(device),
                    contact_t.to(device),
                    isright_t.to(device))
    print("-"*100)
    print("pred_triple:", out["pred_triple"])
    print("logits_base/left/right:", out["logits_base"].shape, out["logits_left"].shape, out["logits_right"].shape)
    print("S_bl/S_br/S_lr:", out["S_bl"].shape, out["S_br"].shape, out["S_lr"].shape)
    print("img_emb:", out["img_emb"].shape)
    print("node_emb:", out["node_emb"].shape)