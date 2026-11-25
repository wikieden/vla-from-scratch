"""知识隔离：stop-gradient 让小脑只读不改；π0.5 的数据配比。

讲义：第 9 章 知识隔离与 π0.5；运行：uv run python ch09_knowledge_insulation.py
梯度用一个两参数的玩具模型手算：大脑参数 w_b 产生 K/V，小脑参数 w_a 读它出动作。
"""

PI05 = {"预训练步数": 280_000, "后训练步数": 80_000, "非目标场景占比": 0.976}
KI_SPEEDUP = 7.5  # π0 要多 7.5 倍训练步数（KI 图 6）


def grads(w_b, w_a, x, target, stop_gradient):
    """前向：kv = w_b·x（大脑写白板），action = w_a·kv（小脑读白板）；损失 (action − target)²。

    返回 (∂L/∂w_b, ∂L/∂w_a)。stop_gradient=True 时，kv 在反向里当常数，梯度传不回大脑。
    """
    kv = w_b * x
    err = w_a * kv - target
    g_a = 2 * err * kv
    g_b = 0.0 if stop_gradient else 2 * err * w_a * x
    return g_b, g_a


def main():
    print("\n第一步 一起训练：小脑的误差顺着 attention 流进大脑")
    g_b, g_a = grads(w_b=1.0, w_a=0.3, x=2.0, target=1.5, stop_gradient=False)
    print(f"  随机小脑 w_a = 0.3：大脑收到梯度 {g_b:+.2f}，小脑 {g_a:+.2f}")

    print("\n第二步 stop-gradient：前向一样，反向停在 sg")
    g_b, g_a = grads(w_b=1.0, w_a=0.3, x=2.0, target=1.5, stop_gradient=True)
    print(f"  加 sg：大脑收到梯度 {g_b:+.2f}（大脑另由 FAST 词和图文数据教），小脑 {g_a:+.2f}")
    print(f"  π0 要多 {KI_SPEEDUP} 倍训练步数才追上")

    print("\n第三步 π0.5 的配方")
    print(f"  预训练 {PI05['预训练步数']:,} 步 + 后训练 {PI05['后训练步数']:,} 步")
    print(f"  {PI05['非目标场景占比']:.1%} 的预训练样本不是目标场景，目标场景只占 {1 - PI05['非目标场景占比']:.1%}")


if __name__ == "__main__":
    main()
