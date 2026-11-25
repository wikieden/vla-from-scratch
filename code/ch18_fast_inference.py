"""跑得快：异步推理阈值、动作队列模拟、前缀计算量、量化显存。

讲义：第 18 章 跑得快；运行：uv run python ch18_fast_inference.py
数据取自 smolvla-2506.01844.txt 图 5、openvla-2406.09246.txt §5.4 表 2。
"""

import math

SMOLVLA_FIG5 = {"同步": {"time_s": 13.75, "cubes_60s": 9, "sr": 78.3}, "异步": {"time_s": 9.70, "cubes_60s": 19, "sr": 73.3}}
OPENVLA_QUANT = {"bf16": (16.8, 71.3), "int8": (10.2, 58.1), "int4": (7.0, 71.9)}  # 显存 GB、成功率 %


def min_threshold(latency, dt, n):
    """不断粮的最小 g：往返期间要走 latency/dt 步，占一段 n 步的比例。"""
    return (latency / dt) / n


def simulate_queue(g, latency, dt, n, seconds):
    """按控制步模拟动作队列；返回 (停着的秒数, 前向次数)。

    每一步：① 新一段到了就换上（跳过发出观测后已经走掉的步数）；② 队列剩余比例 ≤ g 或为空时发观测；
    ③ 队列有动作就执行一个，没有就停一步。
    """
    lag = math.ceil(round(latency / dt, 9))
    queue, arrive, executed_since_send = n, None, 0
    idle = calls = 0
    for step in range(round(seconds / dt)):
        if arrive is not None and step >= arrive:
            queue, arrive = n - executed_since_send, None
        if arrive is None and (queue == 0 or queue / n <= g):
            arrive, executed_since_send, calls = step + lag, 0, calls + 1
        if queue > 0:
            queue -= 1
            executed_since_send += 1
        else:
            idle += 1
    return idle * dt, calls


def prefix_cost(tokens_per_frame, cams, layers):
    """前缀计算量的粗估：token 数 × 层数。"""
    return tokens_per_frame * cams * layers


def weight_gb(params, bits):
    """只算权重的显存（1 GB = 1e9 字节）。"""
    return params * bits / 8 / 1e9


def main():
    print("\n第一步 异步推理：阈值 g 与动作队列")
    g_min = min_threshold(0.3, 1 / 30, 50)
    print(f"  Δt = 1/30 s、n = 50、往返 0.3 s：最小 g = {g_min:.2f}")
    for g in (0.0, g_min, 0.7, 1.0):
        idle, calls = simulate_queue(g, 0.3, 1 / 30, 50, 10)
        print(f"  g = {g:.2f}：10 秒内停了 {idle:.2f} s，前向 {calls} 次")
    s, a = SMOLVLA_FIG5["同步"], SMOLVLA_FIG5["异步"]
    print(f"  SmolVLA：{s['time_s']} s → {a['time_s']} s（快 {1 - a['time_s'] / s['time_s']:.1%}），"
          f"60 秒 {s['cubes_60s']} → {a['cubes_60s']} 个")

    print("\n第二步 前缀计算量 ∝ token × 层")
    big, small = prefix_cost(256, 3, 18), prefix_cost(64, 3, 9)
    print(f"  π0 式 {big:,}，SmolVLA 式 {small:,}，少 {big / small:.0f} 倍")

    print("\n第三步 量化显存（7B 只算权重）")
    for bits in (16, 8, 4):
        print(f"  {bits:>2} 位：{weight_gb(7e9, bits):.1f} GB")
    for k, (gb, sr) in OPENVLA_QUANT.items():
        print(f"  OpenVLA 实测 {k}：{gb} GB、成功率 {sr}%")


if __name__ == "__main__":
    main()
