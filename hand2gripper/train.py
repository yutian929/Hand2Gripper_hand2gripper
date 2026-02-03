# train.py
# -*- coding: utf-8 -*-
"""
二元输出 (left, right) 的训练脚本。
- 数据集的真值仍为3个 (base, left, right)，但训练时只使用 left/right
- 模型: from models.simple_pair import Hand2GripperModel
- 损失:
  * 两个有序CE (left/right)
  * 二元联合分类 (基于联合打分 comb 的 21^2 类)
  * 接触一致 (soft-target)
  * 距离上界先验 (限制过远，允许闭合)

用法示例：
  python train.py --dataset_root /path/to/raw --epochs 10 --batch_size 64
"""
import os
import random
import argparse
from dataclasses import dataclass

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

# from models.simple_pair import Hand2GripperModel
from hand2gripper.models.simple_pair_del_hoi import Hand2GripperModel


# ------------------------------
# 实用函数
# ------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# 早停和稳定性检测
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, metric='triple_acc'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.history = []
        
    def __call__(self, score):
        self.history.append(score)
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop
    
    def is_stable(self, window_size=5, stability_threshold=0.001):
        """检查最近window_size个epoch的准确率是否稳定"""
        if len(self.history) < window_size:
            return False
        
        recent_scores = self.history[-window_size:]
        variance = np.var(recent_scores)
        return variance < stability_threshold

# ------------------------------
# 真实数据集
# ------------------------------
class Hand2GripperDataset(Dataset):
    """
    """
    def __init__(self, root_dir: str):
        super().__init__()
        self.all_sample_paths = self._get_all_valid_sample_paths(root_dir)
    
    def _get_all_valid_sample_paths(self, root_dir):
        """
        遍历root_dir下所有的目录和文件，找出.npz作为尾缀的，且包含如下内容的：
        img_rgb  # (H, W, 3)
        bbox  # (4,)
        crop_img_rgb  # (256, 256, 3)
        kpts_2d  # (21, 2)
        kpts_3d  # (21, 3)
        is_right  # (1,)
        contact_logits  # (21,)
        selected_gripper_blr_ids  # (3,)
        """
        all_sample_paths = []

        if not root_dir:
            print(f"[Hand2GripperDataset] Warning: empty root_dir provided")
            return all_sample_paths

        if not os.path.exists(root_dir):
            print(f"[Hand2GripperDataset] Warning: root_dir does not exist: {root_dir}")
            return all_sample_paths

        # Walk the directory tree and collect all .npz files that pass validity check
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fn in filenames:
                if not fn.lower().endswith('.npz'):
                    continue
                full_path = os.path.join(dirpath, fn)
                try:
                    if self._check_valid_sample(full_path):
                        all_sample_paths.append(full_path)
                except Exception:
                    # _check_valid_sample already prints errors for bad files
                    continue

        # sort for determinism
        all_sample_paths.sort()
        print(f"[Hand2GripperDataset] Loaded {len(all_sample_paths)} samples from {root_dir}")

        return all_sample_paths
    
    def _check_valid_sample(self, sample_path):
        """检查样本文件是否有效（存在且包含必要字段）"""
        if not os.path.exists(sample_path):
            return False
        if not sample_path.endswith('.npz'):
            return False
        required_fields = [
            "img_rgb",
            "bbox",
            "crop_img_rgb",
            "kpts_2d",
            "kpts_3d",
            "is_right",
            "contact_logits",
            "selected_gripper_blr_ids"
        ]
        try:
            data = np.load(sample_path)
            for field in required_fields:
                if field not in data:
                    return False
            return True
        except Exception as e:
            print(f"Error loading sample {sample_path}: {e}")
            return False

    def __len__(self):
        return len(self.all_sample_paths)

    def __getitem__(self, idx):
        sample_path = self.all_sample_paths[idx]
        # Load file
        try:
            data = np.load(sample_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load sample {sample_path}: {e}")

        img_rgb_t = torch.from_numpy(data["img_rgb"]).permute(2, 0, 1).float() / 255.0  # [3,H,W], float32
        bbox_t = torch.from_numpy(data["bbox"]).long()  # [4], int64
        kpts_3d_t = torch.from_numpy(data["kpts_3d"]).float()  # [21,3], float32
        contact_logits_t = torch.from_numpy(data["contact_logits"]).float()  # [21], float32
        is_right_t = torch.from_numpy(data["is_right"]).long()  # [1] or scalar, int64
        selected_gripper_blr_ids_t = torch.from_numpy(data["selected_gripper_blr_ids"]).long()  # [3], int64

        return {
            "img_rgb_t": img_rgb_t,
            "bbox_t": bbox_t,
            "kpts_3d_t": kpts_3d_t,
            "contact_logits_t": contact_logits_t,
            "is_right_t": is_right_t,
            "selected_gripper_blr_ids_t": selected_gripper_blr_ids_t,
        }


# ------------------------------
# 损失 / 指标
# ------------------------------
@dataclass
class LossWeights:
    """Loss 权重配置"""
    pair_ce: float = 1.0           # 二元联合分类损失权重
    contact_align: float = 0.2     # 接触对齐损失权重
    dist_upper: float = 0.1        # 距离上界先验损失权重（仅限制过远，允许闭合）


def build_comb_logits(out):
    """
    构造二元联合打分 comb: [B, 21, 21]
    comb[i,j] = logits_left[i] + logits_right[j] + S_lr[i,j]
    """
    ll = out["logits_left"]   # [B, 21]
    lr = out["logits_right"]  # [B, 21]
    S_lr = out["S_lr"]        # [B, 21, 21]
    comb = (
        ll[:, :, None] +      # [B, 21, 1]
        lr[:, None, :] +      # [B, 1, 21]
        S_lr                  # [B, 21, 21]
    )
    return comb  # [B, 21, 21]


def contact_align_loss_two(prob_left, prob_right, contact, eps=1e-6):
    """
    把两路概率平均后与 contact 对齐（soft-target CE）。
    
    Args:
        prob_left:  [B, 21] 左指尖概率分布
        prob_right: [B, 21] 右指尖概率分布
        contact:    [B, 21] 接触概率
        eps:        防止log(0)的小常数
    """
    tgt = contact.clamp(min=0.0)
    s = tgt.sum(dim=1, keepdim=True)
    tgt = torch.where(s > 0, tgt / (s + eps), torch.full_like(tgt, 1.0 / tgt.shape[1]))
    avg_prob = (prob_left + prob_right) / 2.0
    loss = -(tgt * (avg_prob + eps).log()).sum(dim=1).mean()
    return loss


def distance_upper_prior(prob_left, prob_right, kp_xyz_norm, d_max=3.5):
    """只限制过远抓取（允许闭合/相等）。"""
    dist = torch.cdist(kp_xyz_norm, kp_xyz_norm, p=2)  # [B,21,21]
    E = (prob_left.unsqueeze(2) * prob_right.unsqueeze(1) * dist).sum(dim=(1, 2))  # [B]
    return F.relu(E - d_max).mean()


def compute_losses(out, gt_left, gt_right, kp_xyz_norm, contact, lw: LossWeights):
    """
    计算训练损失
    
    Args:
        out: 模型输出字典
        gt_left:  [B] left 真值索引
        gt_right: [B] right 真值索引
        kp_xyz_norm: [B, 21, 3] 归一化后的关键点
        contact: [B, 21] 接触概率
        lw: 损失权重配置
    """
    # 两个有序 CE
    ce_left = F.cross_entropy(out["logits_left"], gt_left)
    ce_right = F.cross_entropy(out["logits_right"], gt_right)

    # 二元联合分类: 21^2 类
    comb = build_comb_logits(out)                    # [B, 21, 21]
    B, N, _ = comb.shape
    comb_flat = comb.view(B, -1)                     # [B, 441]
    idx_pos = gt_left * N + gt_right                 # [B]
    ce_pair = F.cross_entropy(comb_flat, idx_pos)

    # 接触一致
    pl = F.softmax(out["logits_left"], dim=-1)
    pr = F.softmax(out["logits_right"], dim=-1)
    contact_loss = contact_align_loss_two(pl, pr, contact)

    # 距离上界先验（仅限制过远）
    dist_loss = distance_upper_prior(pl, pr, kp_xyz_norm)

    loss = (ce_left + ce_right) + lw.pair_ce * ce_pair \
           + lw.contact_align * contact_loss + lw.dist_upper * dist_loss

    return loss, {
        "loss": loss.item(),
        "ce_left": ce_left.item(),
        "ce_right": ce_right.item(),
        "ce_pair": ce_pair.item(),
        "contact": contact_loss.item(),
        "dist_upper": dist_loss.item(),
    }


@torch.no_grad()
def eval_metrics(out, gt_left, gt_right):
    """
    计算验证指标：left/right 单独准确率和二元联合准确率
    
    Args:
        out: 模型输出字典
        gt_left:  [B] left 真值索引
        gt_right: [B] right 真值索引
    """
    pl = out["logits_left"].argmax(dim=-1)   # [B]
    pr = out["logits_right"].argmax(dim=-1)  # [B]
    pred_pair = out["pred_pair"]             # [B, 2]

    left_acc = (pl == gt_left).float().mean().item()
    right_acc = (pr == gt_right).float().mean().item()
    pair_acc = ((pred_pair[:, 0] == gt_left) & (pred_pair[:, 1] == gt_right)).float().mean().item()
    
    return {"left_acc": left_acc, "right_acc": right_acc, "pair_acc": pair_acc}


# ------------------------------
# 训练主函数
# ------------------------------
def main(args):

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 数据
    train_dir = os.path.join(args.dataset_root, 'train')
    val_dir = os.path.join(args.dataset_root, 'val')
    test_dir = os.path.join(args.dataset_root, 'test')

    print(f"Loading training data from {train_dir}")
    train_ds = Hand2GripperDataset(train_dir)
    print(f"Loading validation data from {val_dir}")
    val_ds = Hand2GripperDataset(val_dir)
    print(f"Loading test data from {test_dir}")
    test_ds = Hand2GripperDataset(test_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型/优化器（默认使用DINOv2并冻结参数）

    model = Hand2GripperModel(
        d_model=256, img_size=256, 
        freeze_backbone=not args.no_freeze_backbone
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    lw = LossWeights()
    
    # 加载预训练模型
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading pretrained model from {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
            print("Model loaded successfully")
            
            if args.resume_optimizer:
                optimizer_path = args.resume.replace('.pt', '_optimizer.pt')
                if os.path.exists(optimizer_path):
                    print(f"Loading optimizer state from {optimizer_path}")
                    opt.load_state_dict(torch.load(optimizer_path, map_location=device))
                    print("Optimizer state loaded successfully")
                else:
                    print(f"Warning: Optimizer state file not found: {optimizer_path}")
        else:
            print(f"Warning: Pretrained model file not found: {args.resume}")
            print("Starting training from scratch")
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=args.patience, 
        min_delta=0.001, 
        metric='pair_acc'  # 改为 pair_acc
    )

    best_val = -1.0
    ep = 0
    
    # 训练循环
    while True:
        ep += 1
        
        # 如果指定了epochs，检查是否达到
        if args.epochs is not None and ep > args.epochs:
            print(f"Reached specified epochs: {args.epochs}")
            break
        model.train()
        meter = {k: 0.0 for k in ["loss", "ce_left", "ce_right", "ce_pair", "contact", "dist_upper"]}
        for step, batch in enumerate(train_loader, 1):
            img_rgb_t = batch["img_rgb_t"].to(device)
            bbox_t = batch["bbox_t"].to(device).float()
            kpts_3d_t = batch["kpts_3d_t"].to(device)
            contact_logits_t = batch["contact_logits_t"].to(device)
            is_right_t = batch["is_right_t"].to(device)
            
            # 数据集仍然返回 BLR 三元组，只用 L/R
            selected_gripper_blr_ids_t = batch["selected_gripper_blr_ids_t"].to(device)
            gt_left_t = selected_gripper_blr_ids_t[:, 1].view(-1)
            gt_right_t = selected_gripper_blr_ids_t[:, 2].view(-1)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                preprocessed_crop_img_rgb = model._crop_and_resize(img_rgb_t, bbox_t)
                out = model.forward(preprocessed_crop_img_rgb, kpts_3d_t, contact_logits_t, is_right_t.view(-1))
                kp = kpts_3d_t.clone()
                wrist = kp[:, 0:1, :]
                kp = kp - wrist
                scale = kp.norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
                kp_norm = kp / scale.unsqueeze(-1)

                loss, loss_items = compute_losses(out, gt_left_t, gt_right_t, kp_norm, contact_logits_t, lw)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for k, v in loss_items.items():
                meter[k] += v

        n_batch = len(train_loader)
        train_log = {k: v / n_batch for k, v in meter.items()}

        # 验证
        model.eval()
        eval_meter = {"left_acc": 0.0, "right_acc": 0.0, "pair_acc": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                img_rgb_t = batch["img_rgb_t"].to(device)
                bbox_t = batch["bbox_t"].to(device).float()
                kpts_3d_t = batch["kpts_3d_t"].to(device)
                contact_logits_t = batch["contact_logits_t"].to(device)
                is_right_t = batch["is_right_t"].to(device)
                
                # 数据集仍然返回 BLR 三元组，只用 L/R
                selected_gripper_blr_ids_t = batch["selected_gripper_blr_ids_t"].to(device)
                gt_left_t = selected_gripper_blr_ids_t[:, 1].view(-1)
                gt_right_t = selected_gripper_blr_ids_t[:, 2].view(-1)

                preprocessed_crop_img_rgb = model._crop_and_resize(img_rgb_t, bbox_t)
                out = model.forward(preprocessed_crop_img_rgb, kpts_3d_t, contact_logits_t, is_right_t.view(-1))
                m = eval_metrics(out, gt_left_t, gt_right_t)
                for k in eval_meter:
                    eval_meter[k] += m[k]

        n_val = len(val_loader)
        eval_log = {k: v / n_val for k, v in eval_meter.items()}

        print(f"[Epoch {ep}] "
              f"loss={train_log['loss']:.4f} "
              f"CE(L/R/Pair)={train_log['ce_left']:.3f}/{train_log['ce_right']:.3f}/{train_log['ce_pair']:.3f} "
              f"contact={train_log['contact']:.3f} dist_upper={train_log['dist_upper']:.3f} | "
              f"val_acc(L/R/Pair)={eval_log['left_acc']:.3f}/{eval_log['right_acc']:.3f}/{eval_log['pair_acc']:.3f}")

        # 保存最好（二元联合准确率）
        if eval_log["pair_acc"] > best_val:
            best_val = eval_log["pair_acc"]
            torch.save(model.state_dict(), args.save)
            # 如果启用了优化器状态保存，同时保存优化器状态
            if args.resume_optimizer:
                optimizer_save_path = args.save.replace('.pt', '_optimizer.pt')
                torch.save(opt.state_dict(), optimizer_save_path)
                print(f"  >> Saved best to {args.save} and optimizer to {optimizer_save_path} (pair_acc={best_val:.3f})")
            else:
                print(f"  >> Saved best to {args.save} (pair_acc={best_val:.3f})")

        # 早停检查
        if early_stopping(eval_log["pair_acc"]):
            print(f"Early stopping at epoch {ep} (patience={args.patience})")
            break
            
        # 稳定性检查（仅在未指定epochs时）
        if args.epochs is None and ep >= args.stability_window:
            if early_stopping.is_stable(args.stability_window, args.stability_threshold):
                print(f"Training converged at epoch {ep} (accuracy stabilized)")
                break

    # 测试
    print("\nStarting evaluation on test set...")
    model.eval()
    test_meter = {"left_acc": 0.0, "right_acc": 0.0, "pair_acc": 0.0}
    with torch.no_grad():
        for batch in test_loader:
            img_rgb_t = batch["img_rgb_t"].to(device)
            bbox_t = batch["bbox_t"].to(device).float()
            kpts_3d_t = batch["kpts_3d_t"].to(device)
            contact_logits_t = batch["contact_logits_t"].to(device)
            is_right_t = batch["is_right_t"].to(device)
            
            selected_gripper_blr_ids_t = batch["selected_gripper_blr_ids_t"].to(device)
            gt_left_t = selected_gripper_blr_ids_t[:, 1].view(-1)
            gt_right_t = selected_gripper_blr_ids_t[:, 2].view(-1)

            preprocessed_crop_img_rgb = model._crop_and_resize(img_rgb_t, bbox_t)
            out = model.forward(preprocessed_crop_img_rgb, kpts_3d_t, contact_logits_t, is_right_t.view(-1))
            m = eval_metrics(out, gt_left_t, gt_right_t)
            for k in test_meter:
                test_meter[k] += m[k]

    n_test = len(test_loader)
    test_log = {k: v / n_test for k, v in test_meter.items()}
    print(f"Test Set Results: L/R/Pair Acc = {test_log['left_acc']:.3f}/{test_log['right_acc']:.3f}/{test_log['pair_acc']:.3f}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="", help="真实数据根目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save", type=str, default="hand2gripper.pt", help="模型保存路径")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--stability_window", type=int, default=10, help="稳定性检测窗口大小")
    parser.add_argument("--stability_threshold", type=float, default=0.005, help="稳定性阈值")
    parser.add_argument("--resume", type=str, default="", help="从预训练模型继续训练")
    parser.add_argument("--resume_optimizer", action="store_true", help="同时加载优化器状态")
    # DINOv2参数（默认使用DINOv2并冻结）
    parser.add_argument("--no_freeze_backbone", action="store_true", help="不冻结DINOv2参数（默认冻结）")
    args = parser.parse_args()
    main(args)