"""VLM：图片切块成 token，和文字一起排队；prefix-LM 遮挡。

讲义：第 5 章 VLM；运行：uv run python ch05_vlm.py
"""

import numpy as np


def num_patches(image, patch):
    """一张 image × image 的图，切成 patch × patch 的块，一共多少块。"""
    return (image // patch) ** 2


def prefix_lm_mask(n_prefix, n_suffix):
    """prefix-LM 遮挡矩阵：行 = 谁在看，列 = 被看的；前缀互相全看，回答只能看前面。"""
    n = n_prefix + n_suffix
    m = np.tril(np.ones((n, n), dtype=int))
    m[:n_prefix, :n_prefix] = 1
    return m


def main():
    print("\n第一步 图片切成小块，每块一个 token")
    n = num_patches(224, 14)
    print(f"  224 / 14 = {224 // 14}，{224 // 14} × {224 // 14} = {n} 个图像 token；3 个相机 = {3 * n} 个")

    print("\n第二步 prefix-LM：前缀互相看，回答往前看")
    m = prefix_lm_mask(7, 3)  # 4 个图像 token + 3 个指令 token = 前缀 7 个；回答 3 个
    print(m)
    print(f"  第 0 个图像 token 能看到第 6 个（指令）吗：{bool(m[0, 6])}；能看到回答吗：{bool(m[0, 7])}")


if __name__ == "__main__":
    main()
