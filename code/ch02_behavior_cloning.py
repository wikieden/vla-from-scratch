"""行为克隆：看图答题对答案，以及复合误差。

讲义：第 2 章 行为克隆；运行：uv run python ch02_behavior_cloning.py
复合误差用一维小车模拟：状态是偏离车道中线多少；示范只覆盖了 |偏离| ≤ 覆盖宽度 的状态。
"""

import numpy as np


def l1_loss(pred, target):
    """行为克隆的 L1 损失：模型动作和人的动作差多少（ACT 用 L1）。"""
    return float(np.abs(np.asarray(pred, float) - np.asarray(target, float)).mean())


def rollout(coverage, steps=40, bump_at=12, bump=0.3, seed=0):
    """一条轨迹：见过的状态里会往回拉；没见过的状态只能乱猜，偏差越滚越大。"""
    rng = np.random.default_rng(seed)
    x, path = 0.0, []
    for t in range(steps):
        if t == bump_at:
            x += bump  # 被碰歪一下
        if abs(x) <= coverage:
            act = -0.5 * x  # 训练数据里见过：照着示范往中线拉回
        else:
            act = np.sign(x) * 0.15 + rng.normal(0, 0.05)  # 没见过：乱猜，还往外偏
        x += act
        path.append(x)
    return np.array(path)


def main():
    print("\n第一步 看图答题：L1 损失")
    print(f"  模型 [0, 0, -0.02, 0, 0, 0, 0] vs 人 [0, 0, -0.01, 0, 0, 0, 0]：L1 = "
          f"{l1_loss([0, 0, -0.02, 0, 0, 0, 0], [0, 0, -0.01, 0, 0, 0, 0]):.4f}")

    print("\n第二步 复合误差：被碰歪后见没见过")
    for name, cov in (("纯 BC（只录正中间）", 0.1), ("加了偏了再拉回的示范", 0.5)):
        p = rollout(cov)
        print(f"  {name}：最大偏离 {np.abs(p).max():.2f}，最后偏离 {abs(p[-1]):.2f}")


if __name__ == "__main__":
    main()
