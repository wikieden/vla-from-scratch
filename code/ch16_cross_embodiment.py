"""跨本体：RDT 128 维统一动作空间、掩码损失、X-VLA 软提示、6D 旋转。

讲义：第 16 章 跨本体；运行：uv run python ch16_cross_embodiment.py
统一空间布局取自 rdt-1b-2410.07864.txt 附录 C 表 4；软提示尺寸取自 LeRobot xvla/configuration_xvla.py。
"""

import numpy as np

UNIFIED_DIM = 128
# RDT 表 4：物理量 -> 格子区间 [start, end)
LAYOUT = {
    "right_arm_joint_pos": (0, 10),
    "right_gripper_pos": (10, 15),
    "right_arm_joint_vel": (15, 25),
    "right_gripper_vel": (25, 30),
    "right_eef_pos": (30, 33),
    "right_eef_6d": (33, 39),
    "left_arm_joint_pos": (50, 60),
    "left_gripper_pos": (60, 65),
    "base_lin_vel": (100, 102),
    "base_ang_vel": (102, 103),
}
# 每种机器人：物理量 -> 用了几个数（6 自由度的臂只填 10 格里的前 6 格）
ROBOTS = {
    "6 自由度单臂": {"right_arm_joint_pos": 6, "right_gripper_pos": 1},
    "ALOHA 双臂": {"right_arm_joint_pos": 6, "right_gripper_pos": 1, "left_arm_joint_pos": 6, "left_gripper_pos": 1},
}


def unified_slots(robot):
    """一种机器人的动作按物理意义填进 128 格时，用到哪些格子。"""
    slots = []
    for key, n in ROBOTS[robot].items():
        start, end = LAYOUT[key]
        assert n <= end - start
        slots += list(range(start, start + n))
    return slots


def fill_unified(values, robot):
    """把原始动作向量按含义填进 128 格，返回 (值, 掩码)。"""
    slots = unified_slots(robot)
    assert len(values) == len(slots)
    vec, mask = np.zeros(UNIFIED_DIM), np.zeros(UNIFIED_DIM)
    vec[slots], mask[slots] = values, 1.0
    return vec, mask


def masked_mse(pred, target, mask):
    """只在有数的格子上算均方误差。"""
    pred, target, mask = (np.asarray(a, float) for a in (pred, target, mask))
    return float((mask * (pred - target) ** 2).sum() / mask.sum())


def soft_prompt_params(num_domains, len_prompts, hidden):
    """X-VLA 软提示的参数量：一张 num_domains × (len_prompts · hidden) 的查表。"""
    return num_domains * len_prompts * hidden


def rot6d_to_matrix(x):
    """6D 旋转（两列）→ 旋转矩阵：Gram–Schmidt 正交化，再叉乘补第三列。"""
    a1, a2 = np.asarray(x[:3], float), np.asarray(x[3:6], float)
    b1 = a1 / np.linalg.norm(a1)
    u2 = a2 - (b1 @ a2) * b1
    b2 = u2 / np.linalg.norm(u2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def main():
    print("\n第二步 按物理意义填 128 格")
    for robot in ROBOTS:
        s = unified_slots(robot)
        print(f"  {robot}：用 {len(s)} / {UNIFIED_DIM} 格（{len(s) / UNIFIED_DIM:.1%}），格号 {s}")
    vec, mask = fill_unified([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 1.0], "6 自由度单臂")
    pred = vec + 0.1 * mask  # 有数的格子各差 0.1
    pred[127] = 9.0  # 空格子上乱猜也不扣分
    print(f"  掩码损失 = {masked_mse(pred, vec, mask):.4f}（空格第 127 格猜 9.0 不算）")

    print("\n第三步 X-VLA 软提示与 6D 旋转")
    print(f"  30 个域 × 32 个提示 × 1024 维 = {soft_prompt_params(30, 32, 1024):,} 个参数")
    r = rot6d_to_matrix([1, 0.2, 0, 0.1, 1, 0.3])
    print(f"  6D → 矩阵，RᵀR ≈ I：{np.allclose(r.T @ r, np.eye(3))}，det = {np.linalg.det(r):.3f}")


if __name__ == "__main__":
    main()
