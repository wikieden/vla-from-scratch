"""潜空间世界模型：像素损失被背景淹没、潜空间损失只看物体、信息泄露。

讲义：第 20 章 潜空间世界模型；运行：uv run python ch20_latent_world_model.py
玩具设定是编的：一帧 = 1 个物体位置 + n_bg 个背景像素；表 1、表 3 的数取自 vla-jepa-2602.10098.txt。
"""

import numpy as np

JEPA_TABLES = {"LIBERO 平均": (97.2, 96.1), "LIBERO-Plus 平均": (79.5, 62.9)}  # (有人手视频, 没有)


def expected_pixel_loss(move, n_bg, sigma):
    """像素损失的期望：物体部分 move²，背景部分 n_bg·σ²（模型猜不到相机抖动）。"""
    return move**2 + n_bg * sigma**2, move**2, n_bg * sigma**2


def latent_loss(move):
    """理想编码器只留物体位置：潜空间里只剩物体那一项。"""
    return move**2


def simulate_losses(move, n_bg, sigma, trials=20000, seed=0):
    """抽样验证：预测"下一帧 = 这一帧"，真实下一帧物体移动 move、背景随机抖动。"""
    rng = np.random.default_rng(seed)
    obj_err = np.full(trials, float(move))  # 最朴素的预测"下一帧和这一帧一样"：物体误差就是移动量
    bg_err = rng.normal(0, sigma, size=(trials, n_bg))
    pixel = obj_err**2 + (bg_err**2).sum(axis=1)
    return float(pixel.mean()), float((obj_err**2).mean())


def leaky_reconstruction(frames_next, actions):
    """隐变量直接等于下一帧（看得到未来）：不管动作是什么，重建误差都是 0。"""
    z = np.asarray(frames_next, float)
    recon = z.copy()
    del actions  # 动作根本没用上
    return float(((recon - frames_next) ** 2).mean())


def main():
    print("\n第二步 像素损失 vs 潜空间损失（物体移动 1、背景 100 像素、抖动 σ = 0.5）")
    total, obj, bg = expected_pixel_loss(1, 100, 0.5)
    print(f"  像素损失期望 {total:.0f}（物体 {obj:.0f} + 背景 {bg:.0f}，背景占 {bg / total:.0%}）")
    print(f"  潜空间损失 {latent_loss(1):.0f}")
    sim_pixel, sim_latent = simulate_losses(1, 100, 0.5)
    print(f"  抽样：像素 {sim_pixel:.2f}，潜空间 {sim_latent:.2f}")

    print("\n第三步 信息泄露：隐变量抄下一帧")
    rng = np.random.default_rng(1)
    nxt = rng.normal(size=(4, 101))
    for acts in (np.zeros(4), rng.normal(size=4)):
        print(f"  动作 {np.round(acts, 2).tolist()}：重建误差 {leaky_reconstruction(nxt, acts):.1f}")

    print("\n第四步 人手视频的作用")
    for k, (w, wo) in JEPA_TABLES.items():
        print(f"  {k}：有 {w}% / 无 {wo}%，差 {w - wo:.1f}")


if __name__ == "__main__":
    main()
