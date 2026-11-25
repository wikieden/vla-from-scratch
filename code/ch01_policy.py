"""机器人策略：看一眼、听一句、吐一串数。

讲义：第 1 章 机器人策略是什么；运行：uv run python ch01_policy.py
动作是 OpenVLA 那种 7 维末端增量：Δx, Δy, Δz, Δroll, Δpitch, Δyaw, 夹爪（0 张开 / 1 合上）。
"""

import numpy as np

NAMES = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "夹爪"]


def describe(action):
    """把一个 7 维动作翻译成人话，只说非零的位置。"""
    parts = []
    for name, v in zip(NAMES, action):
        if name == "夹爪":
            parts.append("夹爪合上" if v >= 0.5 else "夹爪张开")
        elif v != 0:
            parts.append(f"{name} {v * 100:+.0f} 厘米" if name in ("Δx", "Δy", "Δz") else f"{name} {v:+.2f}")
    return "，".join(parts)


def toy_policy(height_cm, gripper_closed):
    """玩具策略 π(o, ℓ)：指令"抓起杯子"。离杯子还有高度就往下 1 厘米，到了就合上夹爪。"""
    if height_cm > 0:
        return np.array([0, 0, -0.01, 0, 0, 0, float(gripper_closed)])
    return np.array([0, 0, 0, 0, 0, 0, 1.0])


def run_loop(height_cm):
    """策略循环：看 → 想 → 做，做完世界就变了，再看。"""
    actions, closed = [], False
    while not closed:
        a = toy_policy(height_cm, closed)
        actions.append(a)
        height_cm = round(height_cm + a[2] * 100)  # 做：手往下走，下一眼看到的高度变了
        closed = a[6] >= 0.5
    return actions


def main():
    print("\n第一步 动作就是一个向量")
    a = np.array([0.02, 0, -0.01, 0, 0, 0, 1])
    print(f"  {a.tolist()} = {describe(a)}")

    print("\n第二步 策略循环：一个时刻只输出一个动作")
    for t, act in enumerate(run_loop(3), 1):
        print(f"  时刻 {t}：{act.tolist()}（{describe(act)}）")


if __name__ == "__main__":
    main()
