"""π0：推理时间账、flow matching、三块注意力、Q 读 K、τ 的 Beta 抽样。

讲义：第 8 章 π0；运行：uv run python ch08_pi0.py
时间取自 pi0-2410.24164.txt 附录 D 表 I（RTX 4090、3 个相机）；τ 抽样公式取自附录 B。
"""

import math

import numpy as np

LATENCY_MS = {"图像编码": 14, "大脑前缀前向（1 次）": 32, "小脑 flow 10 步（合计）": 27}


def mix(a, eps, tau):
    """训练时造"半雾半清"的动作：A^τ = τA + (1 − τ)ε。"""
    return tau * a + (1 - tau) * eps


def euler_path(start, velocity, steps=10):
    """推理：从噪声出发，每步走 δ = 1/steps；velocity(x, τ) 给出速度。"""
    x, delta, path = start, 1 / steps, [start]
    for k in range(steps):
        x = x + delta * velocity(x, k * delta)
        path.append(x)
    return path


def block_mask(n_prefix, n_state, n_action):
    """π0 的三块注意力：前缀只看前缀；状态看前缀 + 自己；动作看全部。"""
    n = n_prefix + n_state + n_action
    m = np.zeros((n, n), dtype=int)
    m[:n_prefix, :n_prefix] = 1
    m[n_prefix:n_prefix + n_state, :n_prefix + n_state] = 1
    m[n_prefix + n_state:, :] = 1
    return m


def attention_weights(q_deg, k_degs, q_len=3.0):
    """2 维玩具：Q 和各个 K 点积（Q 长 3、K 长 1），再 softmax。"""
    scores = np.array([q_len * math.cos(math.radians(q_deg - k)) for k in k_degs])
    e = np.exp(scores - scores.max())
    return e / e.sum()


def tau_cdf(tau, s=0.999):
    """π0 的 τ 抽样：x = (s − τ)/s ~ Beta(1.5, 1)，P(τ ≤ t) = 1 − ((s − t)/s)^1.5。"""
    x = max(0.0, (s - tau) / s)
    return 1 - x**1.5


def main():
    print("\n第一步 推理时间账")
    for k, v in LATENCY_MS.items():
        print(f"  {k}：{v} ms")
    print(f"  合计 {sum(LATENCY_MS.values())} ms")

    print("\n第二步 flow matching（一维：ε = −1，A = +1）")
    a, eps = 1.0, -1.0
    print(f"  τ = 0.3：A^τ = {mix(a, eps, 0.3):+.1f}，目标速度 u = A − ε = {a - eps:+.0f}")
    path = euler_path(eps, lambda x, t: a - eps)
    print(f"  每步都猜对速度：{[round(p, 1) for p in path]}")

    print("\n第三步 三块注意力（2 图 + 1 指令 | 1 状态 | 3 动作）")
    m = block_mask(3, 1, 3)
    print(m)
    print(f"  前两块看得到动作吗：{bool(m[:4, 4:].any())} → K/V 可以缓存，大脑算 1 次、小脑算 10 次")

    print("\n插问 2 动作专家的 Q 怎么学会查大脑的 K")
    ks = [25, 115, 205, 295]  # 杯子、碗、桌面、"放进"
    for name, q in (("训练前（指向桌面）", 205), ("训练后（指向杯子）", 25)):
        w = attention_weights(q, ks)
        print(f"  {name}：杯子权重 {w[0]:.0%}")

    print("\n第四步 τ 怎么抽：多练雾大的地方")
    for lo, hi in ((0, 0.1), (0, 0.5), (0.9, 1.0)):
        print(f"  τ 落在 [{lo}, {hi}]：均匀 {hi - lo:.0%}，π0 {tau_cdf(hi) - tau_cdf(lo):.0%}")


if __name__ == "__main__":
    main()
