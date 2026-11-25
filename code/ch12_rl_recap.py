"""Recap：奖励、价值、优势，以及按 30% 分位数贴 positive / negative 标签。

讲义：第 12 章 RL 后训练 Recap；运行：uv run python ch12_rl_recap.py
数字是讲义里编的例子：每段 N = 10 步，每步奖励 −1，价值 = −（还剩几步）。
"""

import numpy as np


def episode_return(steps, success, c_fail=100):
    """Recap 的奖励：每步 −1；最后成功得 0，失败得 −C_fail。"""
    return -(steps - 1) + (0 if success else -c_fail)


def advantage(remaining_before, remaining_after, n=10):
    """A = N 步奖励之和 + V(后) − V(前)，V = −还剩几步。"""
    return -n + (-remaining_after) - (-remaining_before)


def label(advs, q=0.3, corrections=()):
    """优势高于本任务 30% 分位数的贴 positive；人纠正的那些一律 positive。"""
    thr = float(np.quantile(advs, q))
    return thr, [a > thr or i in corrections for i, a in enumerate(advs)]


def main():
    print("\n第一步 奖励：越快成功越好")
    for steps, ok in ((30, True), (50, True), (50, False)):
        print(f"  {steps} 步、{'成功' if ok else '失败'}：总奖励 {episode_return(steps, ok)}")

    print("\n第二步 优势：这一段比平时好多少")
    for name, before, after in (("顺利", 40, 25), ("一般", 40, 30), ("衣服掉了", 40, 38)):
        print(f"  {name}：还剩 {before} → {after}，A = {advantage(before, after):+d}")

    print("\n第三步 按分位数贴标签")
    advs = [5, 0, -8, 3, -4, 2, -6, 1, -5, 4]  # 一个任务里 10 段的优势（编的）
    thr, tags = label(advs)
    print(f"  30% 分位数门槛 {thr:+.1f}；positive {sum(tags)} / {len(tags)} 段")


if __name__ == "__main__":
    main()
