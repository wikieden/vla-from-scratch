"""动作当词：每一维切成 256 格；逐词念为什么慢。

讲义：第 6 章 动作当词；运行：uv run python ch06_action_tokens.py
"""

import numpy as np


def bin_width(lo, hi, n_bins=256):
    """一格多宽。"""
    return (hi - lo) / n_bins


def to_bin(x, lo, hi, n_bins=256):
    """连续值 → 格子编号（0 到 n_bins − 1）。"""
    return int(np.clip(np.floor(round((x - lo) / bin_width(lo, hi, n_bins), 9)), 0, n_bins - 1))


def quantile_range(values, q_lo=0.01, q_hi=0.99):
    """OpenVLA 的做法：用 1%–99% 分位数定范围，异常值撑不大格子。"""
    return float(np.quantile(values, q_lo)), float(np.quantile(values, q_hi))


def decode_time(chunk, dims, tokens_per_s):
    """逐词念完一个 chunk 要多少秒。"""
    return chunk * dims / tokens_per_s


def main():
    print("\n第一步 连续的数 → 格子编号")
    lo, hi = -0.128, 0.128
    print(f"  范围 [−0.128, +0.128] 切 256 格：每格 {bin_width(lo, hi) * 1000:.1f} 毫米")
    for x in (0.0, 0.02, -0.05, -0.1):
        print(f"  Δx = {x:+.2f} → 第 {to_bin(x, lo, hi)} 格")
    w = bin_width(-0.128, 1.0)
    print(f"  混进一个 +1.0 的异常值：每格 {w * 1000:.1f} 毫米，正常动作只用得上 {0.256 / w:.0f} 格")
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.uniform(-0.128, 0.128, 999), [1.0]])
    q = quantile_range(data)
    print(f"  改用 1%–99% 分位数：范围 [{q[0]:+.3f}, {q[1]:+.3f}]，异常值不再撑大格子")

    print("\n第二步 逐词念：50 步 × 7 维")
    print(f"  {50 * 7} 个词，每秒 42 个：{decode_time(50, 7, 42):.1f} 秒，而这段只管 1 秒")


if __name__ == "__main__":
    main()
