# VLA 从零到 π0

**在线阅读：<https://wikieden.github.io/vla-from-scratch/>** · [讲义目录](https://wikieden.github.io/vla-from-scratch/lessons/index.html) · [全部动画](https://wikieden.github.io/vla-from-scratch/animations/index.html) · [复习卡片](https://wikieden.github.io/vla-from-scratch/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)



## 目录

| 章 | 主题 | 卡片 | 动画 | 状态 |
|---|---|---|---|---|
| [1](https://wikieden.github.io/vla-from-scratch/lessons/01-what-is-a-policy.html) | 机器人策略是什么：看一眼、听一句、吐一串数 | V01–V02 | [▶](https://wikieden.github.io/vla-from-scratch/animations/01-policy-loop.html) | 完成 |
| [2](https://wikieden.github.io/vla-from-scratch/lessons/02-behavior-cloning.html) | 行为克隆：照着人类示范抄作业 | V03–V04、V17 | [▶](https://wikieden.github.io/vla-from-scratch/animations/02-compounding-error.html) | 完成 |
| [3](https://wikieden.github.io/vla-from-scratch/lessons/03-action-chunking.html) | 动作分块：看一眼，规划一整段 | V05–V07 | [▶](https://wikieden.github.io/vla-from-scratch/animations/03-receding-horizon.html) | 完成 |
| [4](https://wikieden.github.io/vla-from-scratch/lessons/04-multimodality.html) | 多峰：两种都对的做法，平均起来是错的 | V08–V09 | [▶](https://wikieden.github.io/vla-from-scratch/animations/04-multimodality.html) | 完成 |
| [5](https://wikieden.github.io/vla-from-scratch/lessons/05-vlm.html) | VLM：图片切块和文字一起排队 | V10–V11 | [▶](https://wikieden.github.io/vla-from-scratch/animations/05-vlm-tokens.html) | 完成 |
| [6](https://wikieden.github.io/vla-from-scratch/lessons/06-action-tokens.html) | 动作当词：每个数切成 256 格 | V12–V13 | [▶](https://wikieden.github.io/vla-from-scratch/animations/06-binning-autoregressive.html) | 完成 |
| [7](https://wikieden.github.io/vla-from-scratch/lessons/07-fast.html) | FAST：先压缩再变词 | V14–V15 | [▶](https://wikieden.github.io/vla-from-scratch/animations/07-fast-dct.html) | 完成 |
| [8](https://wikieden.github.io/vla-from-scratch/lessons/08-pi0.html) | π0：大脑 + 小脑，动作专家 | V16、V18、V20–V24 | [▶](https://wikieden.github.io/vla-from-scratch/animations/08-pi0-inference.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/08b-pi0-block-mask.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/08c-tau-sampling.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/08d-cross-attention.html) | 完成 |
| [9](https://wikieden.github.io/vla-from-scratch/lessons/09-knowledge-insulation.html) | 知识隔离与 π0.5：别让小脑冲掉大脑的常识 | V25 | [▶](https://wikieden.github.io/vla-from-scratch/animations/09-gradient-interference.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/09b-knowledge-insulation.html) [▶](https://wikieden.github.io/vla-from-scratch/animations/09c-pi05-recipe.html) | 已写，练习待作答 |
| [10](https://wikieden.github.io/vla-from-scratch/lessons/10-hierarchy.html) | 慢想快做：高层 1 秒想一次，低层每秒 50 个动作 | — | [▶](https://wikieden.github.io/vla-from-scratch/animations/10-two-clocks.html) | 已写，练习待作答 |
| [11](https://wikieden.github.io/vla-from-scratch/lessons/11-latent-actions.html) | 没有动作标签：隐动作与世界模型 | — | [▶](https://wikieden.github.io/vla-from-scratch/animations/11-latent-action.html) | 已写，练习待作答 |
| [12](https://wikieden.github.io/vla-from-scratch/lessons/12-rl-recap.html) | RL 后训练 Recap：告诉模型"这段做得比平时好" | — | [▶](https://wikieden.github.io/vla-from-scratch/animations/12-recap-advantage.html) | 已写，练习待作答 |
| [13](https://wikieden.github.io/vla-from-scratch/lessons/13-rtc.html) | 实时执行 RTC：边想下一段，边把这一段走完 | — | [▶](https://wikieden.github.io/vla-from-scratch/animations/13-rtc.html) | 已写，练习待作答 |
| [14](https://wikieden.github.io/vla-from-scratch/lessons/14-vla-landscape.html) | 主流 VLA 模型全景：架构、关键改进、优劣势 | — | [▶](https://wikieden.github.io/vla-from-scratch/animations/14-vla-landscape.html) | 已写，练习待作答 |

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
