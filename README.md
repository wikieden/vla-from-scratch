# VLA 从零到 π0

**在线阅读：<https://wikieden.github.io/vla-from-scratch/>** · [讲义目录](https://wikieden.github.io/vla-from-scratch/lessons/index.html) · [全部动画](https://wikieden.github.io/vla-from-scratch/animations/index.html) · [复习卡片](https://wikieden.github.io/vla-from-scratch/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)

> 在线阅读：<https://wikieden.github.io/vla-from-scratch/>
> 21 章 · 26 个动画（GIF + 可交互页面）· 42 张静态图 · 50 张复习卡片 · 覆盖 VLA 论文主线（RT-1 → π0 → π*0.6）与 2026 年的开源模型

## 这本书讲什么

从"机器人策略到底输入什么、输出什么"讲起：先讲行为克隆、动作分块、多峰这些老问题，再讲 VLM 怎么变成 VLA（动作当词、FAST、π0 的动作专家），然后是知识隔离、分层、隐动作、RL 后训练、实时执行；最后七章讲 2025–2026 的工程主题：微调配方（OFT）、跨本体、在线 RL、推理加速、评测、潜空间世界模型，以及怎么读懂一个新开源模型的 config。

- **适合谁**：会一点 Transformer（向量、点积、attention），不需要机器人背景；用到的 RL、统计、旋转、系统知识，在那一章现场补，或链接到对应的书。
- **读完能做什么**：拿到一个新 VLA，能说出它动作怎么出、大脑小脑怎么连、吃了什么数据；能估它跑多快、要多少显存、论文分数可不可信；能为自己的机器人选一个开源模型起步（对应里程碑 V-M0–V-M8）。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。

全书反复问三个问题：**动作怎么出、大脑小脑怎么连、吃了什么数据**；再加两个工程问题：**多快、能不能复现**。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍"讲。
- **关键机制都有图和动画**：结构和数据流画静态图（三块注意力、知识隔离的梯度路线、统一动作空间），数字之间的关系画图表（τ 的抽样分布、Wilson 区间、量化显存），要拖参数才看得出规律的做成可交互动画（chunk 长度、异步阈值、组内优势）。
- **对照原文**：讲到论文的地方标本地文本行号（如 OFT 表 II L319–L360）；开源模型的细节对照 LeRobot、openpi、openvla-oft 的源码行号；编的数会写明"编的"。
- **保留真实的学习过程**：练习与作答记录里是学习时的原话、对错和点评。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释，全书缩写见 [缩写与术语表](https://wikieden.github.io/vla-from-scratch/lessons/appendix-glossary.html)，符号读法见 [符号表](https://wikieden.github.io/vla-from-scratch/lessons/appendix-symbols.html)。
- **每章都有代码对照**：`solutions/chapter-code/` 一章一个 Python + NumPy 模块，配测试核对讲义里的数字和一本 notebook；只算讲过的例子，不提前给练习答案。
- **每章收尾有口诀**：全部口诀汇总在本页末尾。

## 示例动画

| | |
|---|---|
| ![π0 推理](https://wikieden.github.io/vla-from-scratch/animations/08-pi0-inference.gif)<br>**第 8 章 · π0 推理**：大脑算一次存 K/V，小脑推 10 步<br>[讲义](https://wikieden.github.io/vla-from-scratch/lessons/08-pi0.html) · [▶ 交互](https://wikieden.github.io/vla-from-scratch/animations/08-pi0-inference.html) | ![知识隔离](https://wikieden.github.io/vla-from-scratch/animations/09b-knowledge-insulation.gif)<br>**第 9 章 · 知识隔离**：小脑只读不改，大脑另有老师<br>[讲义](https://wikieden.github.io/vla-from-scratch/lessons/09-knowledge-insulation.html) · [▶ 交互](https://wikieden.github.io/vla-from-scratch/animations/09b-knowledge-insulation.html) |
| ![RTC](https://wikieden.github.io/vla-from-scratch/animations/13-rtc.gif)<br>**第 13 章 · RTC**：同步会停、直接换会跳、RTC 平滑<br>[讲义](https://wikieden.github.io/vla-from-scratch/lessons/13-rtc.html) · [▶ 交互](https://wikieden.github.io/vla-from-scratch/animations/13-rtc.html) | ![并行解码](https://wikieden.github.io/vla-from-scratch/animations/15-oft.gif)<br>**第 15 章 · 并行解码**：逐词念 vs 一次填满，吞吐 4.2 → 108.8 Hz<br>[讲义](https://wikieden.github.io/vla-from-scratch/lessons/15-oft.html) · [▶ 交互](https://wikieden.github.io/vla-from-scratch/animations/15-oft.html) |
| ![组内优势](https://wikieden.github.io/vla-from-scratch/animations/17-online-rl.gif)<br>**第 17 章 · GRPO**：一组轨迹谁加分谁减分，全同组白跑<br>[讲义](https://wikieden.github.io/vla-from-scratch/lessons/17-online-rl.html) · [▶ 交互](https://wikieden.github.io/vla-from-scratch/animations/17-online-rl.html) | ![异步推理](https://wikieden.github.io/vla-from-scratch/animations/18-fast-inference.gif)<br>**第 18 章 · 异步推理**：队列剩到 g 就开始想<br>[讲义](https://wikieden.github.io/vla-from-scratch/lessons/18-fast-inference.html) · [▶ 交互](https://wikieden.github.io/vla-from-scratch/animations/18-fast-inference.html) |

全部动画和本地打开方法见 [animations/README.md](https://wikieden.github.io/vla-from-scratch/animations/index.html)。

## 目录

| 章 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|
| [1](https://wikieden.github.io/vla-from-scratch/lessons/01-what-is-a-policy.html) | 机器人策略是什么：看一眼、听一句、吐一串数 | VLA 的输入是图 + 指令，输出是一个 7 维动作向量，一秒重复很多次 | [▶](https://wikieden.github.io/vla-from-scratch/animations/01-policy-loop.html) | V01–V02 | 完成 |
| [2](https://wikieden.github.io/vla-from-scratch/lessons/02-behavior-cloning.html) | 行为克隆：照着人类示范抄作业 | 训练就是看图答题；只见过人走的路，一走偏就越错越远 | [▶](https://wikieden.github.io/vla-from-scratch/animations/02-compounding-error.html) | V03–V04、V17 | 完成 |
| [3](https://wikieden.github.io/vla-from-scratch/lessons/03-action-chunking.html) | 动作分块：看一眼，规划一整段 | 一次出 k 步，决定次数降到 1/k；想 50 步只走 5 步 | [▶](https://wikieden.github.io/vla-from-scratch/animations/03-receding-horizon.html) | V05–V07 | 完成 |
| [4](https://wikieden.github.io/vla-from-scratch/lessons/04-multimodality.html) | 多峰：两种都对的做法，平均起来是错的 | L2 求平均会撞墙；生成模型抽一个，抽定了别变心 | [▶](https://wikieden.github.io/vla-from-scratch/animations/04-multimodality.html) | V08–V09 | 完成 |
| [5](https://wikieden.github.io/vla-from-scratch/lessons/05-vlm.html) | VLM：图片切块和文字一起排队 | 224 的图切成 256 个 token；PaliGemma = SigLIP + 线性投影 + Gemma | [▶](https://wikieden.github.io/vla-from-scratch/animations/05-vlm-tokens.html) | V10–V11 | 完成 |
| [6](https://wikieden.github.io/vla-from-scratch/lessons/06-action-tokens.html) | 动作当词：每个数切成 256 格 | 每维切 256 格当词念；50 × 7 = 350 个词，念 8.3 秒 | [▶](https://wikieden.github.io/vla-from-scratch/animations/06-binning-autoregressive.html) | V12–V13 | 完成 |
| [7](https://wikieden.github.io/vla-from-scratch/lessons/07-fast.html) | FAST：先压缩再变词 | 相邻 token 太像，模型学会抄上一个；DCT + BPE 先说形状再补细节 | [▶](https://wikieden.github.io/vla-from-scratch/animations/07-fast-dct.html) | V14–V15 | 完成 |
| [8](https://wikieden.github.io/vla-from-scratch/lessons/08-pi0.html) | π0：大脑 + 小脑，动作专家 | VLM 前缀算一次存 K/V，300M 动作专家 flow matching 推 10 步，73 ms | [▶](https://wikieden.github.io/vla-from-scratch/animations/08-pi0-inference.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/08b-pi0-block-mask.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/08c-tau-sampling.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/08d-cross-attention.html) | V16、V18–V24 | 完成 |
| [9](https://wikieden.github.io/vla-from-scratch/lessons/09-knowledge-insulation.html) | 知识隔离与 π0.5：别让小脑冲掉大脑的常识 | 小脑读 K/V 但梯度不回传；大脑另学 FAST 词和图文；π0.5 五路数据两段训练 | [▶](https://wikieden.github.io/vla-from-scratch/animations/09-gradient-interference.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/09b-knowledge-insulation.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/09c-pi05-recipe.html) | V25 | 已写，练习待作答 |
| [10](https://wikieden.github.io/vla-from-scratch/lessons/10-hierarchy.html) | 慢想快做：高层 1 秒想一次，低层每秒 50 个动作 | 高层写子任务、低层出动作；文字好懂，向量信息多 | [▶](https://wikieden.github.io/vla-from-scratch/animations/10-two-clocks.html) | V27–V28 | 已写，练习待作答 |
| [11](https://wikieden.github.io/vla-from-scratch/lessons/11-latent-actions.html) | 没有动作标签：隐动作与世界模型 | 从两帧里猜一个小码当"动作词"，再用少量机器人数据学词 → 关节 | [▶](https://wikieden.github.io/vla-from-scratch/animations/11-latent-action.html) | V29–V30 | 已写，练习待作答 |
| [12](https://wikieden.github.io/vla-from-scratch/lessons/12-rl-recap.html) | RL 后训练 Recap：告诉模型"这段做得比平时好" | 价值函数估还要多久，优势贴 positive / negative，上场只说 positive | [▶](https://wikieden.github.io/vla-from-scratch/animations/12-recap-advantage.html) | V31–V32 | 已写，练习待作答 |
| [13](https://wikieden.github.io/vla-from-scratch/lessons/13-rtc.html) | 实时执行 RTC：边想下一段，边把这一段走完 | 走掉的 d 步冻住，剩下的按软遮罩照着补，不用重训 | [▶](https://wikieden.github.io/vla-from-scratch/animations/13-rtc.html) | V33–V34 | 已写，练习待作答 |
| [14](https://wikieden.github.io/vla-from-scratch/lessons/14-vla-landscape.html) | 主流 VLA 模型全景：架构、关键改进、优劣势 | 三个问题看懂任何 VLA；五条主线、四类动作头 | [▶](https://wikieden.github.io/vla-from-scratch/animations/14-vla-landscape.html) | V35–V36 | 已写，练习待作答 |
| [15](https://wikieden.github.io/vla-from-scratch/lessons/15-oft.html) | OpenVLA-OFT：一次吐完所有动作，直接回归连续值 | 并行解码 + chunk 快 26 倍；L1 取中位数；FiLM 按指令调色 | [▶](https://wikieden.github.io/vla-from-scratch/animations/15-oft.html) | V37–V38 | 已写，练习待作答 |
| [16](https://wikieden.github.io/vla-from-scratch/lessons/16-cross-embodiment.html) | 跨本体：不同的机器人，怎么排进同一张动作表 | RDT 128 格按物理意义填；X-VLA 每个域 32 个软提示 | [▶](https://wikieden.github.io/vla-from-scratch/animations/16-cross-embodiment.html) | V39–V40 | 已写，练习待作答 |
| [17](https://wikieden.github.io/vla-from-scratch/lessons/17-online-rl.html) | 在线 RL：SimpleVLA-RL，只看成没成也能练 | GRPO 组内比较算优势；全同组白跑；pushcut | [▶](https://wikieden.github.io/vla-from-scratch/animations/17-online-rl.html) | V41–V42 | 已写，练习待作答 |
| [18](https://wikieden.github.io/vla-from-scratch/lessons/18-fast-inference.html) | 跑得快：异步推理、少看几眼、压小权重 | g ≥ (E[ℓ]/Δt)/n；计算量 ∝ token × 层；int8 慢反而掉点 | [▶](https://wikieden.github.io/vla-from-scratch/animations/18-fast-inference.html) | V43–V44 | 已写，练习待作答 |
| [19](https://wikieden.github.io/vla-from-scratch/lessons/19-evaluation.html) | 评测：97.1% 和 96.4%，到底谁更强 | Wilson 区间、两比例检验；分数饱和看难子项和扰动 | [▶](https://wikieden.github.io/vla-from-scratch/animations/19-evaluation.html) | V45–V46 | 已写，练习待作答 |
| [20](https://wikieden.github.io/vla-from-scratch/lessons/20-latent-world-model.html) | 潜空间世界模型：VLA-JEPA | 在潜空间里预测未来，未来帧只当答案；推理不用世界模型 | [▶](https://wikieden.github.io/vla-from-scratch/animations/20-latent-world-model.html) | V47–V48 | 已写，练习待作答 |
| [21](https://wikieden.github.io/vla-from-scratch/lessons/21-open-models-2026.html) | 2026 开源 VLA：读一份 config，选一个能微调的模型 | 五个字段读 config；20 / 26 / 55 / 128 格；全参 16 字节、LoRA 2 字节 | [▶](https://wikieden.github.io/vla-from-scratch/animations/21-open-models-2026.html) | V49–V50 | 已写，练习待作答 |

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
