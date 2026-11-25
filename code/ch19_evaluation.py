"""评测：LIBERO 回合数、Wilson 区间、两比例 z 检验、协议差异。

讲义：第 19 章 评测；运行：uv run python ch19_evaluation.py
数据取自 openvla-oft-2502.19645.txt 表 I、openvla-2406.09246.txt 表 2、FastWAM README 复现表。
"""

import math

LIBERO = {"suites": 4, "tasks_per_suite": 10, "episodes_per_task": 50}
OFT_PROTOCOLS = {"① 过滤数据、只有第三人称图": 95.3, "② 原始数据、加腕部图和状态": 94.5, "③ 过滤数据、加腕部图和状态": 97.1}


def wilson_ci(k, n, z=1.96):
    """成功 k 次、共 n 回合的 Wilson 区间。"""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def two_prop_z(k1, n1, k2, n2):
    """两比例 z 检验（合并成功率），返回 (z, 双侧 p 值)。"""
    p1, p2, pp = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return z, math.erfc(abs(z) / math.sqrt(2))


def successes(rate_pct, n):
    """论文只给百分比时，反推成功次数。"""
    return round(rate_pct / 100 * n)


def main():
    print("\n第一步 LIBERO 一次评测多少回合")
    total = LIBERO["suites"] * LIBERO["tasks_per_suite"] * LIBERO["episodes_per_task"]
    print(f"  4 套件 × 10 任务 × 50 回合 = {total}")

    print("\n第二步 Wilson 区间")
    for k, n in ((7, 10), (successes(71.3, 80), 80), (successes(97.1, 2000), 2000)):
        lo, hi = wilson_ci(k, n)
        print(f"  {k}/{n}：[{lo:.3f}, {hi:.3f}]，宽 {100 * (hi - lo):.1f} 个点")

    print("\n第三步 两比例 z 检验")
    for name, (a, b, n) in {"OFT vs FastWAM": (97.1, 96.4, 2000), "OpenVLA bf16 vs int8": (71.3, 58.1, 80)}.items():
        z, p = two_prop_z(successes(a, n), n, successes(b, n), n)
        print(f"  {name}（各 {n} 回合）：z = {z:.2f}，p = {p:.2f}")
    for k, v in OFT_PROTOCOLS.items():
        print(f"  OFT 协议 {k}：{v}%")


if __name__ == "__main__":
    main()
