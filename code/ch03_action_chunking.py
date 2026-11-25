"""动作分块：决定次数、盲走时间、滚动重规划。

讲义：第 3 章 动作分块；运行：uv run python ch03_action_chunking.py
"""

import math


def decisions(task_s, hz, k):
    """一个 task_s 秒、hz 控制频率的任务，每次出 k 步，要做几次决定。"""
    return math.ceil(task_s * hz / k)


def reaction_delay(event_s, ta, hz):
    """事件发生在 event_s 秒；每走 ta 步重看一次，多久后才会发现。"""
    period = ta / hz
    next_look = math.ceil(round(event_s / period, 9)) * period
    return next_look - event_s


def main():
    print("\n第一步 一次出一段：决定次数")
    print(f"  10 秒 × 50 Hz：一步一想 {decisions(10, 50, 1)} 次，chunk k = 50 只要 {decisions(10, 50, 50)} 次")

    print("\n第三步 滚动重规划：T_p = 50、T_a = 50 vs 5")
    for ta in (50, 5):
        print(f"  T_a = {ta}：每 {ta / 50:.1f} s 重看一次，杯子 0.44 s 被碰走，{reaction_delay(0.44, ta, 50):.2f} s 后才发现")
    print(f"  每次想 50 步只走 5 步：{50 - 5} 步是备用计划")


if __name__ == "__main__":
    main()
