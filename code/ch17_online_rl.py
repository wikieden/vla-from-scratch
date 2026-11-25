"""在线 RL（SimpleVLA-RL）：组内相对优势、全同组概率、Clip Higher、温度。

讲义：第 17 章 在线 RL；运行：uv run python ch17_online_rl.py
数据取自 simplevla-rl-2509.09674.txt 表 5（LIBERO，OpenVLA-OFT 骨干）。
"""

import math

# 表 5：平均成功率 (只做 SFT, SFT + RL)
TABLE5 = {"每任务 1 条示范": (48.9, 96.9), "每任务全部 500 条": (91.0, 99.1)}
LIBERO_LONG_ONE_TRAJ = (17.3, 91.7)


def group_advantages(rewards):
    """GRPO 式 8：(R_i − 组平均) / 组标准差（总体标准差）；全同时返回全 0。"""
    g = len(rewards)
    mean = sum(rewards) / g
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / g)
    if std == 0:
        return [0.0] * g
    return [(r - mean) / std for r in rewards]


def p_all_same(p, g):
    """G 条独立轨迹全成功或全失败（没有梯度）的概率。"""
    return p**g + (1 - p) ** g


def clip_ratio_ceiling(p_old, eps_high):
    """一次更新里，一个动作的概率最多涨到多少：p_old × (1 + ε_high)。"""
    return p_old * (1 + eps_high)


def softmax_t(logits, t):
    """温度为 t 的 softmax。"""
    m = max(logits)
    e = [math.exp((z - m) / t) for z in logits]
    s = sum(e)
    return [x / s for x in e]


def main():
    print("\n第一步 只做 SFT vs 加 RL（LIBERO 平均）")
    for k, (sft, rl) in TABLE5.items():
        print(f"  {k}：{sft}% → {rl}%")
    print(f"  LIBERO-Long（1 条示范）：{LIBERO_LONG_ONE_TRAJ[0]}% → {LIBERO_LONG_ONE_TRAJ[1]}%")

    print("\n第二步 组内相对优势（G = 8，成功 3 条）")
    adv = group_advantages([1, 1, 1, 0, 0, 0, 0, 0])
    print(f"  成功：{adv[0]:+.2f}，失败：{adv[-1]:+.2f}")

    print("\n第三步 全同组概率与两个探索技巧")
    for p in (0.1, 0.9):
        print(f"  p = {p}，G = 8：一组白跑的概率 {p_all_same(p, 8):.2f}")
    print(f"  概率 0.10 的动作一次最多涨到 {clip_ratio_ceiling(0.10, 0.20):.3f}（上界 1.2）/ "
          f"{clip_ratio_ceiling(0.10, 0.28):.3f}（上界 1.28）")
    for t in (1.0, 1.6):
        print(f"  温度 {t}：{[round(x, 3) for x in softmax_t([2, 1, 0], t)]}")


if __name__ == "__main__":
    main()
