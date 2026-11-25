"""FAST：相邻 token 太像（抄上一个就能得高分）；DCT 先说形状再补细节。

讲义：第 7 章 FAST；运行：uv run python ch07_fast.py
DCT 用正交的 DCT-II，自己用 NumPy 写，不依赖 scipy。
"""

import numpy as np


def copy_last_accuracy(seq):
    """一个只会"抄上一个"的模型，从第 2 个开始猜，正确率多少。"""
    hits = sum(a == b for a, b in zip(seq, seq[1:]))
    return hits, len(seq) - 1


def dct_matrix(n):
    """正交 DCT-II 矩阵：第 k 行是第 k 个频率的余弦。"""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (i + 0.5) * k / n) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


def dct(x):
    """一段动作 → 频率系数（低频在前）。"""
    x = np.asarray(x, float)
    return dct_matrix(len(x)) @ x


def idct(c):
    """频率系数 → 一段动作（正交矩阵的逆就是转置）。"""
    c = np.asarray(c, float)
    return dct_matrix(len(c)).T @ c


def keep_low(x, k):
    """只保留前 k 个低频系数再还原，返回重建结果和最大误差。"""
    c = dct(x)
    c[k:] = 0
    r = idct(c)
    return r, float(np.abs(r - x).max())


def fast_tokens(x, gamma):
    """FAST 的第 ③ 步：系数乘 γ 再取整，数一数还剩几个非零（越少越好压）。"""
    q = np.round(dct(x) * gamma).astype(int)
    return q, int(np.count_nonzero(q))


def main():
    print("\n第一步 相邻的词太像")
    seq = [128, 128, 129, 129, 129, 130, 130, 130, 131, 131]
    hits, n = copy_last_accuracy(seq)
    print(f"  {seq}：抄上一个猜对 {hits} / {n}")

    print("\n第二步 DCT：先说形状，再补细节")
    flat = np.full(50, 0.01)
    c = dct(flat)
    print(f"  匀速 50 步：非零系数 {np.count_nonzero(np.round(c, 12))} 个，第 0 个 = {c[0]:.4f}")
    t = np.linspace(0, 1, 50)
    smooth = 0.01 * np.sin(np.pi * t) ** 2  # 加速 → 匀速 → 减速（编的）
    for k in (1, 3, 6):
        _, err = keep_low(smooth, k)
        print(f"  加减速：保留 {k} 个低频，最大误差 {err:.5f}")
    jitter = 0.01 * (-1.0) ** np.arange(50)  # 来回抖
    _, err = keep_low(jitter, 6)
    print(f"  来回抖：保留 6 个低频，最大误差 {err:.4f}（能量全在高频）")
    _, nz = fast_tokens(smooth / 0.01, gamma=10)
    print(f"  归一化后乘 γ = 10 取整：50 个系数里非零 {nz} 个")


if __name__ == "__main__":
    main()
