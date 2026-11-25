"""OpenVLA-OFT：并行解码、L1 回归、FiLM 的数字。

讲义：第 15 章 OpenVLA-OFT；运行：uv run python ch15_oft.py
数据取自 openvla-oft-2502.19645.txt 表 I、表 II（A100，一张 224 px 图 + LIBERO 指令）。
"""

import numpy as np

# 表 II：(变体, 每次出几个动作 K, 延迟 s)
TABLE_II = [
    ("OpenVLA（逐词念）", 1, 0.2396),
    ("+ PD", 1, 0.0629),
    ("+ PD&AC", 8, 0.0735),
    ("+ PD&AC, Cont-L1", 8, 0.0729),
    ("+ 腕部相机 + 本体状态", 8, 0.1120),
]
# 表 II：扩散头测试时去噪步数 -> (延迟 s, LIBERO-Long SR %)
DIFFUSION_STEPS = {50: (1.9070, 91.1), 10: (0.4145, 91.0), 5: (0.2279, 90.0), 2: (0.0996, 85.7), 1: (0.0731, 0.0)}
# 表 I：四个 LIBERO 套件平均成功率
TABLE_I = {"OpenVLA 微调": 76.5, "+ PD&AC": 90.2, "+ PD&AC + 扩散": 95.4, "OFT（L1）": 95.3, "OFT + 腕部 + 状态": 97.1}


def throughput(k, latency):
    """每次前向出 k 个动作、花 latency 秒，一秒能出几个动作。"""
    return k / latency


def l2_fit(values, weights):
    """L2 损失的最优常数解：加权平均。"""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    return float((v * w).sum() / w.sum())


def l1_fit(values, weights):
    """L1 损失的最优常数解：加权中位数（在候选点上直接比总误差）。"""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    costs = [float((w * np.abs(v - c)).sum()) for c in v]
    return float(v[int(np.argmin(costs))])


def film(f, gamma, beta):
    """FiLM：(1 + γ) ⊙ F + β，γ、β 对所有图块共享。"""
    return (1 + np.asarray(gamma, float)) * np.asarray(f, float) + np.asarray(beta, float)


def main():
    print("\n第一步 并行解码：吞吐 = K / 延迟")
    for name, k, t in TABLE_II:
        print(f"  {name:<16} K = {k}  延迟 {t:.4f} s  吞吐 {throughput(k, t):6.1f} Hz")
    speedup = throughput(8, 0.0735) / throughput(1, 0.2396)
    print(f"  PD&AC 比原版快 {speedup:.1f} 倍（论文写 26×）")

    print("\n第二步 L1 取中位数，L2 取平均")
    for split in ((0.5, 0.5), (0.6, 0.4)):
        vals, w = [0.05, -0.05], list(split)
        print(f"  左 {split[0]:.0%} / 右 {split[1]:.0%}：L2 → {l2_fit(vals, w):+.3f}，L1 → {l1_fit(vals, w):+.3f}")
    for name, sr in TABLE_I.items():
        print(f"  {name:<14} {sr:.1f}%")
    for steps, (t, sr) in DIFFUSION_STEPS.items():
        print(f"  扩散 {steps:>2} 步：延迟 {t:.4f} s、吞吐 {throughput(8, t):6.1f} Hz、LIBERO-Long {sr:.1f}%")

    print("\n第三步 FiLM")
    out = film([0.5, -0.2, 1.0], [0.2, 0, -0.5], [0, 0.1, 0])
    print(f"  (1 + γ) ⊙ F + β = {np.round(out, 3).tolist()}")
    print(f"  γ = β = 0 时不变：{film([0.5, -0.2, 1.0], [0, 0, 0], [0, 0, 0]).tolist()}")


if __name__ == "__main__":
    main()
