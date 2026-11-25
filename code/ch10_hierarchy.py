"""慢想快做：高层每秒一次（插话立刻重想），低层每秒 50 个动作。

讲义：第 10 章 慢想快做；运行：uv run python ch10_hierarchy.py
任务和插话时刻是编的；节奏按 Hi Robot、GR00T N1 原文。
"""

import math

SYSTEMS = {"GR00T N1": {"高层 Hz": 10, "低层 Hz": 120, "16 步 chunk 用时 ms": 63.9}}


def high_level_calls(task_s, period_s, interrupts):
    """高层想几次：从 0 秒起每 period 一次；插话立刻想一次，之后从插话时刻重新计时。"""
    times, t, pending = [], 0.0, sorted(interrupts)
    while t < task_s - 1e-9:
        times.append(round(t, 3))
        nxt = t + period_s
        if pending and t < pending[0] < nxt - 1e-9:
            nxt = pending.pop(0)  # 用户插话：不等满一个周期
        t = nxt
    return times


def low_level_actions(task_s, hz):
    """低层一共出多少个动作。"""
    return math.floor(task_s * hz)


def main():
    print("\n第一步 两只钟")
    no_int = high_level_calls(8, 1.0, [])
    with_int = high_level_calls(8, 1.0, [4.3])
    print(f"  8 秒任务、每秒一次：高层想 {len(no_int)} 次")
    print(f"  第 4.3 秒插话：高层想 {len(with_int)} 次，时刻 {with_int}")
    print(f"  低层 50 Hz × 8 秒 = {low_level_actions(8, 50)} 个动作")
    g = SYSTEMS["GR00T N1"]
    print(f"  GR00T N1：高层 {g['高层 Hz']} Hz、低层 {g['低层 Hz']} Hz，16 步 {g['16 步 chunk 用时 ms']} ms")


if __name__ == "__main__":
    main()
