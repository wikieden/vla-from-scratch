"""实时分块 RTC：推理延迟折成几步，以及软遮罩权重（式 5）。

讲义：第 13 章 实时执行 RTC；运行：uv run python ch13_rtc.py
H = 16、d = 4、s = 5 取自 real-time-chunking-2506.07339.txt 图 3。
"""

import math


def delay_steps(latency_ms, period_ms):
    """d = ⌊δ / Δt⌋：想一次的工夫，机器人本该走几步。"""
    return math.floor(latency_ms / period_ms)


def soft_mask(h, d, s):
    """式 5：前 d 步权重 1；中间区 c(e^c − 1)/(e − 1)；最后 s 步 0。"""
    w = []
    for i in range(h):
        if i < d:
            w.append(1.0)
        elif i < h - s:
            c = (h - s - i) / (h - s - d + 1)
            w.append(c * (math.exp(c) - 1) / (math.e - 1))
        else:
            w.append(0.0)
    return w


def main():
    print("\n第一步 推理延迟折成控制步")
    print(f"  50 Hz（Δt = 20 ms）、想一次 80 ms：d = {delay_steps(80, 20)}")

    print("\n第三步 软遮罩权重（H = 16、d = 4、s = 5）")
    w = soft_mask(16, 4, 5)
    print("  " + " ".join(f"{x:.3f}" for x in w))


if __name__ == "__main__":
    main()
