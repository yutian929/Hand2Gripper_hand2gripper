# train.py
# -*- coding: utf-8 -*-
"""
三元输出 (base, left, right) 的训练脚本。
- 支持两种数据源：
  1) 真实数据: Hand2GripperDataset(需 processor_config.DataManager)
  2) 玩具数据: ToyTripleDataset (--use_toy 开关)
- 模型: from models.model import Hand2GripperModel  (你的三元输出版 model.py)
- 损失:
  * 三个有序CE (base/left/right)
  * 三元联合分类 (基于联合打分 comb 的 21^3 类)
  * 接触一致 (soft-target)
  * 距离上界先验 (限制过远，允许闭合)

用法示例：
  python train.py --dataset_root /path/to/raw --epochs 10 --batch_size 64
  python train.py --use_toy --epochs 3
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

from models.simple import Hand2GripperModel


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
    triple_ce: float = 1.0
    contact_align: float = 0.2
    dist_upper: float = 0.1  # 仅限制过远，允许闭合/相等


def build_comb_logits(out):
    """按模型三元联合打分构造 comb: [B,21,21,21]"""
    lb = out["logits_base"]
    ll = out["logits_left"]
    lr = out["logits_right"]
    S_bl = out["S_bl"]
    S_br = out["S_br"]
    S_lr = out["S_lr"]
    comb = (
        lb[:, :, None, None] +
        ll[:, None, :, None] +
        lr[:, None, None, :] +
        S_bl[:, :, :, None] +
        S_br[:, :, None, :] +
        S_lr[:, None, :, :]
    )
    return comb


def contact_align_loss_three(prob_base, prob_left, prob_right, contact, eps=1e-6):
    """把三路概率平均后与 contact 对齐（soft-target CE）。"""
    tgt = contact.clamp(min=0.0)
    s = tgt.sum(dim=1, keepdim=True)
    tgt = torch.where(s > 0, tgt / (s + eps), torch.full_like(tgt, 1.0 / tgt.shape[1]))
    avg_prob = (prob_base + prob_left + prob_right) / 3.0
    loss = -(tgt * (avg_prob + eps).log()).sum(dim=1).mean()
    return loss


def distance_upper_prior(prob_left, prob_right, kp_xyz_norm, d_max=3.5):
    """只限制过远抓取（允许闭合/相等）。"""
    dist = torch.cdist(kp_xyz_norm, kp_xyz_norm, p=2)  # [B,21,21]
    E = (prob_left.unsqueeze(2) * prob_right.unsqueeze(1) * dist).sum(dim=(1, 2))  # [B]
    return F.relu(E - d_max).mean()


def compute_losses(out, gt_base, gt_left, gt_right, kp_xyz_norm, contact, lw: LossWeights):
    # 三个有序CE
    ce_base = F.cross_entropy(out["logits_base"], gt_base)
    ce_left = F.cross_entropy(out["logits_left"], gt_left)
    ce_right = F.cross_entropy(out["logits_right"], gt_right)

    # 三元联合分类: 21^3 类
    comb = build_comb_logits(out)                    # [B,21,21,21]
    B, N, _, _ = comb.shape
    comb_flat = comb.view(B, -1)                     # [B, 9261]
    idx_pos = gt_base * (N * N) + gt_left * N + gt_right
    ce_triple = F.cross_entropy(comb_flat, idx_pos)

    # 接触一致
    pb = F.softmax(out["logits_base"], dim=-1)
    pl = F.softmax(out["logits_left"], dim=-1)
    pr = F.softmax(out["logits_right"], dim=-1)
    contact_loss = contact_align_loss_three(pb, pl, pr, contact)

    # 距离上界先验（仅限制过远）
    dist_loss = distance_upper_prior(pl, pr, kp_xyz_norm)

    loss = (ce_base + ce_left + ce_right) + lw.triple_ce * ce_triple \
           + lw.contact_align * contact_loss + lw.dist_upper * dist_loss

    return loss, {
        "loss": loss.item(),
        "ce_base": ce_base.item(),
        "ce_left": ce_left.item(),
        "ce_right": ce_right.item(),
        "ce_triple": ce_triple.item(),
        "contact": contact_loss.item(),
        "dist_upper": dist_loss.item(),
    }


@torch.no_grad()
def eval_metrics(out, gt_base, gt_left, gt_right):
    """有序 Top-1 与三元联合准确率。"""
    pb = out["logits_base"].argmax(dim=-1)
    pl = out["logits_left"].argmax(dim=-1)
    pr = out["logits_right"].argmax(dim=-1)
    triple = out["pred_triple"]  # [B,3]

    base_acc = (pb == gt_base).float().mean().item()
    left_acc = (pl == gt_left).float().mean().item()
    right_acc = (pr == gt_right).float().mean().item()
    triple_acc = ((triple[:, 0] == gt_base) & (triple[:, 1] == gt_left) & (triple[:, 2] == gt_right)).float().mean().item()
    return {"base_acc": base_acc, "left_acc": left_acc, "right_acc": right_acc, "triple_acc": triple_acc}


# ------------------------------
# 训练主函数
# ------------------------------
def main(args):

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 数据
    full_ds = Hand2GripperDataset(args.dataset_root)
    total = len(full_ds)
    tsize = int(total * args.train_ratio)
    vsize = total - tsize
    train_ds, val_ds = random_split(full_ds, [tsize, vsize])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 模型/优化器
    model = Hand2GripperModel(d_model=256, img_size=256).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    lw = LossWeights()
    
    # 加载预训练模型
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading pretrained model from {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print("Model loaded successfully")
            
            # 如果指定了加载优化器状态
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
        metric='triple_acc'
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
        meter = {k: 0.0 for k in ["loss", "ce_base", "ce_left", "ce_right", "ce_triple", "contact", "dist_upper"]}
        for step, batch in enumerate(train_loader, 1):
            img_rgb_t = batch["img_rgb_t"].to(device)
            bbox_t = batch["bbox_t"].to(device).float()
            kpts_3d_t = batch["kpts_3d_t"].to(device)
            contact_logits_t = batch["contact_logits_t"].to(device)
            is_right_t = batch["is_right_t"].to(device)
            
            selected_gripper_blr_ids_t = batch["selected_gripper_blr_ids_t"].to(device)
            gt_base_t = selected_gripper_blr_ids_t[:, 0].view(-1)
            gt_left_t = selected_gripper_blr_ids_t[:, 1].view(-1)
            gt_right_t = selected_gripper_blr_ids_t[:, 2].view(-1)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                preprocessed_crop_img_rgb = model._crop_and_resize(img_rgb_t, bbox_t)
                out = model.forward(preprocessed_crop_img_rgb, kpts_3d_t, contact_logits_t, is_right_t.view(-1))
                # 关键点规范化（与模型内部一致）用于距离先验
                kp = kpts_3d_t.clone()
                wrist = kp[:, 0:1, :]
                kp = kp - wrist
                scale = kp.norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
                kp_norm = kp / scale.unsqueeze(-1)

                loss, loss_items = compute_losses(out, gt_base_t, gt_left_t, gt_right_t, kp_norm, contact_logits_t, lw)

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
        eval_meter = {"base_acc": 0.0, "left_acc": 0.0, "right_acc": 0.0, "triple_acc": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                img_rgb_t = batch["img_rgb_t"].to(device)
                bbox_t = batch["bbox_t"].to(device).float()
                kpts_3d_t = batch["kpts_3d_t"].to(device)
                contact_logits_t = batch["contact_logits_t"].to(device)
                is_right_t = batch["is_right_t"].to(device)
                
                selected_gripper_blr_ids_t = batch["selected_gripper_blr_ids_t"].to(device)
                gt_base_t = selected_gripper_blr_ids_t[:, 0].view(-1)
                gt_left_t = selected_gripper_blr_ids_t[:, 1].view(-1)
                gt_right_t = selected_gripper_blr_ids_t[:, 2].view(-1)

                preprocessed_crop_img_rgb = model._crop_and_resize(img_rgb_t, bbox_t)
                out = model.forward(preprocessed_crop_img_rgb, kpts_3d_t, contact_logits_t, is_right_t.view(-1))
                m = eval_metrics(out, gt_base_t, gt_left_t, gt_right_t)
                for k in eval_meter:
                    eval_meter[k] += m[k]

        n_val = len(val_loader)
        eval_log = {k: v / n_val for k, v in eval_meter.items()}

        print(f"[Epoch {ep}] "
              f"loss={train_log['loss']:.4f} "
              f"CE(B/L/R/T)={train_log['ce_base']:.3f}/{train_log['ce_left']:.3f}/{train_log['ce_right']:.3f}/{train_log['ce_triple']:.3f} "
              f"contact={train_log['contact']:.3f} dist_upper={train_log['dist_upper']:.3f} | "
              f"val_acc(B/L/R/T)={eval_log['base_acc']:.3f}/{eval_log['left_acc']:.3f}/{eval_log['right_acc']:.3f}/{eval_log['triple_acc']:.3f}")

        # 保存最好（三元联合准确率）
        if eval_log["triple_acc"] > best_val:
            best_val = eval_log["triple_acc"]
            torch.save(model.state_dict(), args.save)
            # 如果启用了优化器状态保存，同时保存优化器状态
            if args.resume_optimizer:
                optimizer_save_path = args.save.replace('.pt', '_optimizer.pt')
                torch.save(opt.state_dict(), optimizer_save_path)
                print(f"  >> Saved best to {args.save} and optimizer to {optimizer_save_path} (triple_acc={best_val:.3f})")
            else:
                print(f"  >> Saved best to {args.save} (triple_acc={best_val:.3f})")

        # 早停检查
        if early_stopping(eval_log["triple_acc"]):
            print(f"Early stopping at epoch {ep} (patience={args.patience})")
            break
            
        # 稳定性检查（仅在未指定epochs时）
        if args.epochs is None and ep >= args.stability_window:
            if early_stopping.is_stable(args.stability_window, args.stability_threshold):
                print(f"Training converged at epoch {ep} (accuracy stabilized)")
                break

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="", help="真实数据根目录（含各 sample.npz 文件）")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，None表示自动训练直到收敛")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例（真实数据时生效）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save", type=str, default="hand2gripper.pt", help="模型保存路径")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--stability_window", type=int, default=10, help="稳定性检测窗口大小")
    parser.add_argument("--stability_threshold", type=float, default=0.005, help="稳定性阈值")
    parser.add_argument("--resume", type=str, default="", help="从预训练模型继续训练（模型文件路径）")
    parser.add_argument("--resume_optimizer", action="store_true", help="同时加载优化器状态（需要保存的优化器状态文件）")
    args = parser.parse_args()
    main(args)