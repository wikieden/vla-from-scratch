"""多峰：求平均会撞墙，抽一个才对，抽定了别变心。

讲义：第 4 章 多峰；运行：uv run python ch04_multimodality.py
"""

import numpy as np


def l2_answer(values, counts):
    """L2 训练的模型在这个画面下会输出的值：平均。"""
    v, c = np.asarray(values, float), np.asarray(counts, float)
    return float((v * c).sum() / c.sum())


def sample_answer(values, counts, rng):
    """生成模型：按示范里的比例抽一个。"""
    p = np.asarray(counts, float) / sum(counts)
    return float(rng.choice(values, p=p))


def switches(n_replans, commit, rng):
    """每次重规划都抽一次方向；commit=True 时画面会提醒"已经选了哪边"（只剩那一座峰）。"""
    first = rng.choice([-1, 1])
    picks = [first]
    for _ in range(n_replans - 1):
        picks.append(first if commit else rng.choice([-1, 1]))
    return sum(a != b for a, b in zip(picks, picks[1:]))


def main():
    print("\n第一步 两种都对，平均起来是错的")
    print(f"  50 条 Δy = +0.05、50 条 Δy = −0.05：L2 输出 {l2_answer([0.05, -0.05], [50, 50]):+.2f}（撞上石头）")

    print("\n第二步 不求平均，改成抽一个")
    rng = np.random.default_rng(0)
    draws = [sample_answer([0.05, -0.05], [50, 50], rng) for _ in range(10)]
    print(f"  抽 10 次：{draws}（只会是 ±0.05，不会是 0）")

    print("\n第三步 抽定了别变心")
    rng = np.random.default_rng(1)
    print(f"  每 5 步独立抽签、重抽 8 次：换峰 {switches(8, False, rng)} 次")
    print(f"  画面提醒、重抽 8 次：换峰 {switches(8, True, rng)} 次")


if __name__ == "__main__":
    main()
