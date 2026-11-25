"""2026 开源 VLA：config 五个字段、LingBot 55 维、q01/q99 归一化、稀疏 MoE、微调显存。

讲义：第 21 章 2026 开源 VLA；运行：uv run python ch21_open_models_2026.py
数字取自 LeRobot policies/<名字>/configuration_*.py 的默认值；None 表示 config 里没有直接写。
"""

# 模型 -> (主干, chunk, 动作维, 去噪步)
MODELS = {
    "SmolVLA": ("SmolVLM2-500M（前 16 层）", 50, 32, 10),
    "π0.5": ("PaliGemma", 50, 32, 10),
    "X-VLA": ("Florence-2", 32, 20, 10),
    "Wall-OSS-0.5": ("Qwen2.5-VL", 32, 26, 10),
    "LingBot-VLA 2.0": ("Qwen3-VL-4B", 50, 55, 10),
    "EO-1": ("Qwen2.5-VL-3B", 16, 32, None),
    "EVO1": ("InternVL3-1B（14 层）", 50, 24, None),
    "DM0.5": ("OpenDM", 50, 32, None),
    "MolmoAct2": ("AI2 VLM", 30, None, 8),
    "VLA-JEPA": ("Qwen3-VL", 7, 7, None),
    "FastWAM": ("Wan2.2-TI2V-5B", 32, 7, 10),
}
LINGBOT_CANONICAL = {"arm.position": 14, "end.position": 14, "effector.position": 2, "hand.position": 12,
                     "waist.position": 4, "head.position": 2, "base.velocity": 3, "reserved.slots": 4}
BYTES_PER_PARAM = {"全参数": 16, "LoRA": 2, "只推理": 2}


def canonical_dim(layout):
    """统一空间的总格数。"""
    return sum(layout.values())


def quantile_normalize(x, q01, q99):
    """把 [q01, q99] 线性映到 [−1, 1]，超出的截到 ±1。"""
    y = 2 * (x - q01) / (q99 - q01) - 1
    return max(-1.0, min(1.0, y))


def moe_active_fraction(num_experts, top_k):
    """每个 token 用到的路由专家比例。"""
    return top_k / num_experts


def moe_active_width(top_k, expert_width, shared_width):
    """每个 token 实际走过的 FFN 中间维：k 个专家 + 共享专家。"""
    return top_k * expert_width + shared_width


def finetune_gb(params_b, mode):
    """只算权重和优化器状态的显存（GB）；params_b 以十亿参数计。"""
    return params_b * BYTES_PER_PARAM[mode]


def main():
    print("\n第一步 config 里的五个字段")
    for name, (vlm, chunk, dim, steps) in MODELS.items():
        print(f"  {name:<16} 主干 {vlm:<24} chunk {chunk:>3}  动作维 {str(dim or '—'):>3}  去噪 {steps or '—'}")
    flow10 = sum(1 for *_, s in MODELS.values() if s == 10)
    print(f"  写明 10 步去噪的：{flow10} / {len(MODELS)}")

    print("\n第二步 LingBot-VLA 2.0 的统一空间与 q01/q99 归一化")
    print(f"  {' + '.join(str(v) for v in LINGBOT_CANONICAL.values())} = {canonical_dim(LINGBOT_CANONICAL)}")
    print(f"  q01 = −0.2、q99 = 0.6：x = 0.2 → {quantile_normalize(0.2, -0.2, 0.6):+.2f}，"
          f"x = 0.9 → {quantile_normalize(0.9, -0.2, 0.6):+.2f}（截断）")

    print("\n第三步 稀疏 MoE 动作专家")
    print(f"  32 选 4：每个 token 用 {moe_active_fraction(32, 4):.1%} 的路由专家")
    print(f"  4 × 512 + 704 = {moe_active_width(4, 512, 704)}（等于不用 MoE 时的 2752）")

    print("\n第四步 微调显存（只算权重和优化器）")
    for p in (0.45, 3, 4):
        print(f"  {p}B：全参数 {finetune_gb(p, '全参数'):.1f} GB，LoRA {finetune_gb(p, 'LoRA'):.1f} GB")


if __name__ == "__main__":
    main()
