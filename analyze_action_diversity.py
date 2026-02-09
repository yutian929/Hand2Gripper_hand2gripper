"""
统计数据集中 GT 的 L/R index 组合多样性。

selected_gripper_blr_ids: (3,)
  - [0]: gripper type id (B)
  - [1]: L index
  - [2]: R index

将 (L, R) 视为无序对，即 (4,8) 和 (8,4) 归为同一类。
按动作类型（目录名）分组统计，并输出总体统计。
"""

import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def analyze_action_diversity(dataset_dir):
    """
    遍历 dataset_dir 下所有 .npz 文件,
    统计每个动作类型下 (L, R) 无序对的出现次数.
    """
    # 按动作类型分组: action_type -> { (min, max) pair: count }
    action_pair_counts = defaultdict(lambda: defaultdict(int))
    # 总体统计
    overall_pair_counts = defaultdict(int)
    # 总样本数
    total_samples = 0
    # 失败文件
    failed_files = []

    # 收集所有 .npz 文件
    all_npz_files = []
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for fn in filenames:
            if fn.lower().endswith('.npz'):
                all_npz_files.append(os.path.join(dirpath, fn))

    all_npz_files.sort()
    print(f"找到 {len(all_npz_files)} 个 .npz 文件")

    for fpath in tqdm(all_npz_files, desc="分析中"):
        try:
            data = np.load(fpath, allow_pickle=True)
            if "selected_gripper_blr_ids" not in data:
                continue

            blr = data["selected_gripper_blr_ids"]  # (3,)
            l_idx = int(blr[1])
            r_idx = int(blr[2])

            # 无序对: 用 frozenset 或 sorted tuple
            pair = tuple(sorted((l_idx, r_idx)))

            # 推断动作类型: 从路径中提取
            # 路径格式可能是: .../action_type/object_XX/left_or_right/xxx.npz
            # 或者:           .../train/xxx.npz (flat)
            rel_path = os.path.relpath(fpath, dataset_dir)
            parts = rel_path.split(os.sep)

            if len(parts) >= 2:
                action_type = parts[0]  # 第一级目录作为动作类型
            else:
                action_type = "unknown"

            action_pair_counts[action_type][pair] += 1
            overall_pair_counts[pair] += 1
            total_samples += 1

        except Exception as e:
            failed_files.append((fpath, str(e)))

    # ========== 打印结果 ==========
    print("\n" + "=" * 70)
    print(f"总样本数: {total_samples}")
    print(f"加载失败: {len(failed_files)}")
    print("=" * 70)

    # 按动作类型打印
    for action_type in sorted(action_pair_counts.keys()):
        pairs = action_pair_counts[action_type]
        action_total = sum(pairs.values())
        print(f"\n{'─' * 50}")
        print(f"动作类型: {action_type}  (共 {action_total} 个样本)")
        print(f"{'─' * 50}")
        print(f"  {'(L, R) 无序对':<20} {'数量':>8} {'占比':>10}")
        for pair, count in sorted(pairs.items(), key=lambda x: -x[1]):
            pct = count / action_total * 100
            print(f"  {str(pair):<20} {count:>8} {pct:>9.1f}%")
        print(f"  不同对数: {len(pairs)}")

    # 总体统计
    print(f"\n{'=' * 70}")
    print(f"总体 (L, R) 无序对统计  (共 {total_samples} 个样本)")
    print(f"{'=' * 70}")
    print(f"  {'(L, R) 无序对':<20} {'数量':>8} {'占比':>10}")
    for pair, count in sorted(overall_pair_counts.items(), key=lambda x: -x[1]):
        pct = count / total_samples * 100
        print(f"  {str(pair):<20} {count:>8} {pct:>9.1f}%")
    print(f"\n  总共不同的 (L, R) 对数: {len(overall_pair_counts)}")

    # 返回字典
    result = {
        "per_action": {k: dict(v) for k, v in action_pair_counts.items()},
        "overall": dict(overall_pair_counts),
        "total_samples": total_samples,
    }
    return result


if __name__ == "__main__":
    dataset_path = "/data0/Hand2GripperDatasets/Hand2Gripper_TrainValTest_Dataset"
    result = analyze_action_diversity(dataset_path)

    print("\n\n========== 最终字典 ==========")
    print("per_action:")
    for action, pairs in result["per_action"].items():
        print(f"  {action}: {pairs}")
    print(f"overall: {result['overall']}")
    print(f"total_samples: {result['total_samples']}")
