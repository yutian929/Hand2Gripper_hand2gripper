# inference.py
# -*- coding: utf-8 -*-
"""
Hand-to-Gripper 模型推理脚本。

功能:
- 加载一个训练好的 Hand2GripperModel。
- 读取一个 .npz 格式的样本数据。
- 执行完整的预处理、模型推理和后处理。
- 打印预测结果并可选地生成可视化图像。

用法示例:
  # 对 sample.npz 进行推理，并保存可视化结果到 output.png
  python inference.py --checkpoint hand2gripper.pt --input sample.npz --output output.png

  # 仅打印结果，不生成可视化
  python inference.py --checkpoint hand2gripper.pt --input sample.npz
"""
import os
import argparse
import numpy as np
import torch
import cv2
from typing import Dict

# 假设此脚本与 models 目录在同一级别
from models.simple import Hand2GripperModel

# ------------------------------
# 可视化工具函数 (从 phantom.utils.hand2gripper_visualize.py 简化而来)
# ------------------------------
def vis_selected_gripper(image: np.ndarray, kpts_2d: np.ndarray, gripper_joints_seq: np.ndarray) -> np.ndarray:
    """在图像上绘制选定的抓手关节点和连线。"""
    img_vis = image.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Base: Green, Left: Blue, Right: Red
    labels = ["B", "L", "R"]
    
    # 绘制连线
    base_pt = tuple(kpts_2d[gripper_joints_seq[0]].astype(int))
    left_pt = tuple(kpts_2d[gripper_joints_seq[1]].astype(int))
    right_pt = tuple(kpts_2d[gripper_joints_seq[2]].astype(int))
    
    cv2.line(img_vis, base_pt, left_pt, (255, 255, 0), 2)  # Cyan
    cv2.line(img_vis, base_pt, right_pt, (255, 0, 255), 2) # Magenta

    # 绘制关节点
    for i, joint_id in enumerate(gripper_joints_seq):
        pt = tuple(kpts_2d[joint_id].astype(int))
        cv2.circle(img_vis, pt, 5, colors[i], -1)
        cv2.putText(img_vis, labels[i], (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        
    return img_vis

# ------------------------------
# 推理器类
# ------------------------------
class Hand2GripperInference:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        """
        初始化模型并加载权重。
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.device = torch.device(device)
        self.model = Hand2GripperModel(d_model=256, img_size=256)
        
        # 使用模型自带的加载函数
        self.model._load_checkpoint(checkpoint_path)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {checkpoint_path} and set to evaluation mode on {self.device}.")

    @torch.no_grad()
    def predict(self, color: np.ndarray, bbox: np.ndarray, keypoints_3d: np.ndarray,
                contact: np.ndarray, is_right: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        对单个样本执行完整的推理流程。

        Args:
            color (np.ndarray): 原始图像, [H,W,3] 或 [3,H,W], uint8/float
            bbox (np.ndarray): 边界框, [4]
            keypoints_3d (np.ndarray): 3D关节点, [21,3]
            contact (np.ndarray): 接触概率/logits, [21]
            is_right (np.ndarray): 是否为右手, 标量或 [1]

        Returns:
            Dict[str, torch.Tensor]: 模型的原始输出字典。
        """
        # 1. 使用模型内部的读取函数将 numpy 数组转换为 batched tensor
        color_t = self.model._read_color(color).to(self.device)
        bbox_t = self.model._read_bbox(bbox).to(self.device)
        kp3d_t = self.model._read_keypoints_3d(keypoints_3d).to(self.device)
        contact_t = self.model._read_contact(contact).to(self.device)
        isright_t = self.model._read_is_right(is_right).to(self.device)

        # 2. 预处理：裁剪和缩放图像
        # 注意：这一步是在模型外部完成的，与训练脚本保持一致
        crop_t = self.model._crop_and_resize(color_t, bbox_t)

        # 3. 模型前向传播
        outputs = self.model(crop_t, kp3d_t, contact_t, isright_t)
        
        return outputs

# ------------------------------
# 主函数
# ------------------------------
def main(args):
    # 检查输入文件
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input data file not found: {args.input}")

    # 设置设备
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # 初始化推理器
    try:
        inference_engine = Hand2GripperInference(args.checkpoint, device=device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 加载数据
    print(f"Loading data from {args.input}...")
    try:
        data = np.load(args.input, allow_pickle=True)
        # 从 .npz 文件中提取所需数据
        color_np = data["img_rgb"]
        bbox_np = data["bbox"]
        kpts_3d_np = data["kpts_3d"]
        contact_np = data["contact_logits"]
        is_right_np = data["is_right"]
        kpts_2d_np = data["kpts_2d"] # 用于可视化
    except Exception as e:
        print(f"Error loading data from {args.input}. Ensure it's a valid .npz file with required keys. Error: {e}")
        return

    # 执行推理
    outputs = inference_engine.predict(
        color=color_np,
        bbox=bbox_np,
        keypoints_3d=kpts_3d_np,
        contact=contact_np,
        is_right=is_right_np
    )

    # 后处理和打印结果
    pred_triple = outputs['pred_triple'].squeeze().cpu().numpy()
    print("\n" + "="*30)
    print("      Inference Result")
    print("="*30)
    print(f"Predicted Gripper Triple (Base, Left, Right): {pred_triple}")
    print(f"  - Base Joint ID:  {pred_triple[0]}")
    print(f"  - Left Joint ID:  {pred_triple[1]}")
    print(f"  - Right Joint ID: {pred_triple[2]}")
    print("="*30)

    # 可视化
    if args.output:
        print(f"\nGenerating visualization and saving to {args.output}...")
        # 确保图像是 HWC, uint8, BGR 格式给 OpenCV
        if color_np.dtype != np.uint8:
            vis_img = (color_np * 255).astype(np.uint8)
        else:
            vis_img = color_np.copy()
        
        if vis_img.shape[0] == 3: # CHW -> HWC
            vis_img = np.transpose(vis_img, (1, 2, 0))
        
        # RGB -> BGR for OpenCV
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        # 绘制结果
        vis_img_result = vis_selected_gripper(vis_img, kpts_2d_np, pred_triple)
        
        # 保存图像
        try:
            cv2.imwrite(args.output, vis_img_result)
            print(f"Visualization saved successfully.")
        except Exception as e:
            print(f"Error saving visualization: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand2Gripper Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data sample (.npz file).")
    parser.add_argument("--output", type=str, default=None, help="Optional: Path to save the output visualization image.")
    parser.add_argument("--cpu", action="store_true", help="Force use CPU even if CUDA is available.")
    
    args = parser.parse_args()
    main(args)

# python inference.py --checkpoint hand2gripper.pt --input /home/yutian/projs/Hand2Gripper_phantom/data/processed/epic/1/hand2gripper_annotator_processor/right/110.npz --output vis.png