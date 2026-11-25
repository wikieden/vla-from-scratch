"""主流 VLA 全景：用三个问题给模型分类。

讲义：第 14 章 主流 VLA 模型全景；运行：uv run python ch14_vla_landscape.py
"""

# 模型 -> (动作怎么出, 大脑小脑怎么连, 年份)
MODELS = {
    "ACT": ("一次出一段（CVAE）", "没有 VLM", 2023),
    "Diffusion Policy": ("扩散头", "没有 VLM", 2023),
    "RT-2": ("离散 token", "同一个模型念词", 2023),
    "OpenVLA": ("离散 token", "同一个模型念词", 2024),
    "π0-FAST": ("离散 token（FAST）", "同一个模型念词", 2025),
    "OpenVLA-OFT": ("连续回归（L1）", "同一个模型，并行解码", 2025),
    "Octo": ("扩散头", "小 Transformer 主干", 2024),
    "RDT-1B": ("扩散头", "扩散 Transformer", 2024),
    "π0": ("flow 头", "两套权重共享 attention", 2024),
    "CogACT": ("扩散头", "VLM 认知特征 → 扩散模块", 2024),
    "GR00T N1": ("flow 头", "DiT cross-attention 读 VLM 第 12 层", 2025),
    "SmolVLA": ("flow 头", "小 VLM 前一半层 + 动作专家", 2025),
}


def group_by_head(models):
    """按"动作怎么出"把模型分组。"""
    groups = {}
    for name, (head, _, _) in models.items():
        key = "离散 token" if head.startswith("离散") else head
        groups.setdefault(key, []).append(name)
    return groups


def main():
    print("\n第二步 按动作怎么出排一排")
    for head, names in group_by_head(MODELS).items():
        print(f"  {head}：{', '.join(names)}")
    flows = [n for n, (h, *_) in MODELS.items() if h in ("flow 头", "扩散头")]
    print(f"  扩散 / flow 头一共 {len(flows)} 个")


if __name__ == "__main__":
    main()
