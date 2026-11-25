"""隐动作：从两帧画面里猜"发生了什么"，码本只有 8 个方向。

讲义：第 11 章 隐动作与世界模型；运行：uv run python ch11_latent_actions.py
玩具设定是编的：画面 = 一个小球的位置；码本 = 8 个方向，每个码走 1.55。
"""

import math

import numpy as np

ARROWS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
CODEBOOK = np.array([[math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)] for k in range(8)])
STEP = 1.55
LAPA = {"比真实动作标签训练的最强 VLA 高": "6.22%", "预训练效率": "30 倍以上"}


def encode(x_t, x_next):
    """编码器（逆动力学）：两帧之间的变化，选最接近的那个方向码。"""
    d = np.asarray(x_next, float) - np.asarray(x_t, float)
    return int(np.argmax(CODEBOOK @ (d / np.linalg.norm(d))))


def decode(x_t, z):
    """解码器（世界模型）：这一帧 + 码 → 还原下一帧。"""
    return np.asarray(x_t, float) + STEP * CODEBOOK[z]


def main():
    print("\n第二步 编码器 + 解码器")
    x_t, x_next = (0.0, 0.0), (1.6, -2.0)
    z = encode(x_t, x_next)
    rec = decode(x_t, z)
    err = float(np.linalg.norm(rec - np.asarray(x_next)))
    print(f"  变化 (1.6, −2.0) → 码 z{z} {ARROWS[z]}，还原到 ({rec[0]:.1f}, {rec[1]:.1f})，误差 {err:.2f}")

    print("\n第三步 LAPA 的效果")
    for k, v in LAPA.items():
        print(f"  {k}：{v}")


if __name__ == "__main__":
    main()
