# iKAT 2026 交互赛道 LLM 评估结果 — 完整解读

> 2026-07-17。数据:`results/ikat26-results-RALI/`(Marcel Gohsen 2026-07-16 邮件)。
> 分析脚本:session scratchpad `analyze_ikat26.py`;本文所有数字可由其复现。
> 相关:邮件草稿 `ikat_2026_report/docs/email_draft_user_types.md`(未发送)。

## 0. 结果包结构速查

| 文件 | 内容 | 规模 |
|---|---|---|
| `dialogue-level-results-RALI.json` | 每 (run × user_type × 对话) 一条,5 个对话级分数(1–5) | 264 条 = 6 run × 44 对 |
| `rubric-level-results-RALI.json` | 逐轮记录:rubric、user_utterance、我们的 response 原文、4 个分数 | 4022 条(**每条重复×2**,真实 2011) |
| `retrieval-results-RALI.json` | run → u1–u4 → {total_results, 每 rubric 指标, 检索段落} | 6 run × 4 user |
| `rubric-based-qrels.txt` | `rubric_id \t docid:passage \t rel(0–4)`,3 列无 Q0 | 27474 行,92 个 rubric |

6 个 run 与我们提交记录(`results/ClueWeb_ikat/ikat_26_sim_run/`)完全对应:
`pers-conv-qwen-only`(R6)、`conv-pers-qwen3-fuse-norerank-extract`(R3)、`splade-pers-conv-qwen`(R5)、
`pers-conv-dense-BM25-rankllama-gpt5mini-extract-new`(R4)、`…-gpt5mini`(X1 无画像抽取)、`…-gpt5mini-extract`(X2 最小推理档)。

## 1. 四个 user type(u1–u4)是什么

- **同一 topic 下 u1–u4 的 persona/PTKB 完全相同**(driver 日志验证:`ikat2023-test-10-1` 四个 user 均 ptkb_n=12;`ikat2024-test-13` 均 16)。区别只在模拟器的**聊天行为**。
- **设计目的**(trecikat.com 2026 guidelines,原文已验证):Year 4 考察系统对"行为各异、信息需求相同"的用户的**鲁棒性**。公开的行为维度举例:语言水平低 → 模糊/歧义/拼错的 query(如 u2 的 "besrt fruits acid reflux");耐心差 → 情绪化回复;不配合的用户也必须服务好。
- **每个 u 具体是什么行为组合:官方故意保密**("a secret set of user simulator implementations")。2025 overview、SIGIR'26 资源论文、trecikat.com 全站、Sim.API GitHub、SCAI'26 页面均无枚举 → 已写邮件草稿询问。
- 2025 年只有一个模拟器(无变体):debug 用 Gemma-3-4B,正式跑 GPT-4.1;Sim.API 中间件支持每 topic 挂多个模拟器实现——u1–u4 的机制载体。

### 覆盖矩阵(6 个 run 相同;u1:9 / u2:10 / u3:10 / u4:15,不对称原因未公开)

| conv_id | u1 | u2 | u3 | u4 |
|---|---|---|---|---|
| ikat2023-test-10-1 | . | x | x | x |
| ikat2023-test-15-1 | x | x | x | x |
| ikat2023-test-15-2 | x | x | x | x |
| ikat2023-test-17-1 | x | . | . | x |
| ikat2023-test-17-2 | x | . | . | x |
| ikat2024-test-5 | x | x | x | x |
| ikat2024-test-11 | . | . | x | x |
| ikat2024-test-12 | . | x | . | x |
| ikat2024-test-13 | x | . | . | x |
| ikat2024-test-14 | . | x | x | x |
| ikat2025-test-1-1 | x | x | x | x |
| ikat2025-test-2-1 | x | x | x | x |
| ikat2025-test-2-2 | x | x | x | x |
| ikat2025-test-7-1 | . | x | . | x |
| ikat2025-test-9-1 | . | . | x | x |

## 2. 官方文档里有没有 user type 设计阐述

- **2026 guidelines**:有设计理念(见上),**无 per-type 定义**;评估细节写 "More details will follow later"。
- **2025 overview**:整套 rubric/模拟器机制的出处(今年沿用),但当年无多 user type。
- → **邮件草稿**已按要求写好:`ikat_2026_report/docs/email_draft_user_types.md`。问 3 件事:u1–u4 行为定义与设计理念(是否会写进 2026 overview、可否引用)、topic×user 分配不对称是否有意、total_results 聚合口径确认;附两个数据 bug 反馈。

## 3. topics:15 个;与原版的区别

**2023 / 2024 / 2025 各 5 个**(不止 23/24):

| conv_id | 原版标题 | #ptkb | 原版轮数 | #rubric | rubric~原utt | 模拟utt~原utt |
|---|---|---|---|---|---|---|
| ikat2023-test-10-1 | Recipes and food | 12 | 21 | 6 | 0.48 | 0.31 |
| ikat2023-test-15-1 | Find a movie | 7 | 15 | 6 | 0.38 | 0.22 |
| ikat2023-test-15-2 | Find a movie | 8 | 11 | 9 | 0.36 | 0.24 |
| ikat2023-test-17-1 | Buying a phone | 9 | 12 | 6 | 0.49 | 0.31 |
| ikat2023-test-17-2 | Buying a phone | 9 | 16 | 8 | 0.58 | 0.33 |
| ikat2024-test-5 | Health condition (hair loss) | 16 | 16 | 7 | 0.46 | 0.29 |
| ikat2024-test-11 | Explore home safety improvements | 22 | 13 | 5 | 0.09 | 0.13 |
| ikat2024-test-12 | Select a football team | 18 | 10 | 5 | 0.27 | 0.30 |
| ikat2024-test-13 | Buying a pc | 16 | 11 | 6 | 0.41 | 0.27 |
| ikat2024-test-14 | Growing plants indoors | 17 | 11 | 6 | 0.64 | 0.37 |
| ikat2025-test-1-1 | Acid reflux | 21 | 12 | 6 | 0.58 | 0.35 |
| ikat2025-test-2-1 | Making good coffee | 20 | 14 | 6 | 0.57 | 0.29 |
| ikat2025-test-2-2 | Select a wine | 20 | 11 | 5 | 0.48 | 0.21 |
| ikat2025-test-7-1 | Baking bread at home | 21 | 13 | 5 | 0.48 | 0.48 |
| ikat2025-test-9-1 | Second hand bike repair | 19 | 11 | 6 | 0.60 | 0.48 |

**区别**:2026 只复用**persona/PTKB + 话题场景**;原版的写死剧本(turns/resolved_utterance/金标 response)不用了,
**对话由 LLM 模拟器按 rubric 序列现场生成**。上表两列重叠度佐证:rubric 与原版 turn 的词汇重叠(~0.4–0.6)明显高于模拟器实际说的话与原版的重叠(~0.2–0.5)→ rubric 从原版对话的信息需求提炼,模拟器围绕 rubric 重新展开。
官方印证(SIGIR'26 资源论文 §3.6):"we manually derived 92 rubric questions from the … conversations";"our simulator generates user utterances according to the current rubric and progresses to the next one once the previous rubric has been satisfied."

## 4. mixed_initiative_strategies 是什么、为什么低

官方定义(2025 overview,原文验证):**"How well does the system employ a portfolio of different proactive actions (e.g., asking clarifying questions, asking follow-up questions, asking for feedback, etc.) throughout the conversation?"** —— 考察系统"反客为主"的能力。

低的原因:我们的系统每轮直接检索+回答,**从不主动提澄清/追问**。2025 年人评时几乎所有队伍该项垫底(最好的 proactive 系统归一化才 0.45–0.46)。我们的分布(264 条):1 分×20、2 分×54、3 分×108、4 分×82。提分方向:让 RAG 回答在合适时机主动追问/给后续建议。

## 5. retrieval-results 结构与 rubric 语义

- **rubric ≠ user utterance**。rubric = organizer 手写的**子话题考察问题**(`rub_<conv>_<n>`,全集 92 个),同时是模拟器的对话大纲和打分标准;user_utterance 是模拟器围绕当前 rubric 说出的口语化 query。
- **模拟器协议**(2025 overview,原文验证,2026 沿用):按当前 rubric 生成 utterance(采样多个、取与 rubric SBERT 相似度最高者;prompt 含对话历史+全部 PTKB+当前 rubric)→ LLM 按 0–5 给你的回答打 `rubric_score` → **>3 进入下一个 rubric;≤3 给反馈换说法重问,最多 3 次**后强制前进 → 全部 rubric 走完说再见。这解释了:同一 rubric 连续多轮出现、"can you summarize that" 类追问 = 模拟器给第二次机会。
- `total_results` = 8 项 trec 指标(AP、nDCG、nDCG@5、RR、P/R(rel≥2)@5/10);qrels 按 rubric 池化(0–4 档)。
- **⚠ 聚合陷阱(pytrec_eval 逆向验证 8/8 精确命中)**:`total_results` **对全部 92 个 rubric 求平均,该 user type 未触达的 rubric 记 0 分**(实测 ours×59/92=theirs)。**u4 检索总分高(nDCG@5≈0.72–0.79 vs 其他≈0.48–0.55)纯属覆盖率假象**(u4 跑满 15 个对话)。跨 user type 比较必须只对覆盖到的 rubric 求均值。

### retrieval total_results(nDCG@5 / P(rel=2)@5 / RR)

| run | u1 | u2 | u3 | u4 |
|---|---|---|---|---|
| R3 +convQwen | 0.510/0.580/0.630 | 0.477/0.530/0.648 | 0.521/0.583/0.661 | 0.725/0.815/0.967 |
| X1 no-extract | 0.507/0.554/0.629 | 0.545/0.596/0.663 | 0.506/0.559/0.674 | 0.789/0.889/0.987 |
| X2 min-reason | 0.494/0.528/0.620 | 0.529/0.580/0.650 | 0.510/0.570/0.654 | 0.775/0.850/0.985 |
| R4 +BM25+RL | 0.528/0.578/0.641 | 0.518/0.567/0.663 | 0.494/0.565/0.665 | 0.780/0.865/0.989 |
| R6 qwen-only | 0.486/0.548/0.641 | 0.498/0.567/0.658 | 0.511/0.583/0.674 | 0.733/0.822/0.995 |
| R5 +SPLADE | 0.527/0.578/0.641 | 0.513/0.572/0.658 | 0.552/0.604/0.674 | 0.716/0.796/0.973 |

## 6. rubric-level-results 与我们记录的对账

- 每条 = (run, conv, user_type, turn) + rubric + user_utterance + **我们 response 原文** + 4 个分数。
- **数据 bug ①**:4022 条 = 每条逐字节重复 ×2(真实 2011)。**bug ②**:2 条的 `relevance_and_usefulnesst-score` 是未解析的 judge 输出字符串而非整数。**bug ③**(注意):字段名本身带拼写错误(`…usefulnesst-score`、`…qualityt-score`)。
- **对账:去重后 2011/2011(100%)与我们 264 个 session JSON 的 utterance+response 逐字一致**——评的确实是我们提交的原文。
- 我们共 2275 轮,2011 轮被评;**未评的 264 轮 = 每个 session 的最后一轮告别语**("Okay, thank you!"×203、"all right, thanks"×60)。
- **坑**:`turn_id` 是跨对话的全局流水号(某对话从 24 起跳),对齐须按内容匹配。

## 7. 各分数官方定义(均已对照原文验证)

**逐轮(rubric-level)**:
| 字段 | 量表 | 官方定义 |
|---|---|---|
| `rubric_score` | 0–5 | 回答是否答到当前 rubric:5=高度相关、完整、准确;0=完全没答上。模拟器实时打分,>3 过关 |
| `engagement-score` | 1–5 | "To what extent does the system encourage the user to engage with it?" |
| `relevance_and_usefulnesst-score` | 1–5 | "Given the conversation history, user's latest request, and user's PTKB, how would you assess the relevance and usefulness of the generated response?" |
| `overall_subtopic_qualityt-score` | 1–5 | "Considering all factors (relevance, engagement, and other factors …), what is the overall quality and utility of the system's performance?" |

**逐对话(dialogue-level,1–5)**:
| 字段 | 官方定义 |
|---|---|
| `mixed_initiative_strategies` | 主动行为组合:澄清问题、追问、征求反馈等 |
| `personalization` | "How well does the system tailor the dialogue to the specific user persona provided?" |
| `information_flow_and_coherence` | "How well does the system maintain a coherent and logical flow of information?" |
| `trustworthiness` | "How likely would you be to believe that past and future answers of this system are factually correct?" |
| `overall_user_satisfaction` | "Considering the entire dialogue, how satisfied would you be as the user …?" |

问卷逐字沿用 2025 年 NIST 人评题目;本轮为 LLM-as-judge 代打,人工评估仍在进行(组织者邮件)。
2025 年聚合方式(overview):rubric 级 run 分 = overall-quality 按信心加权;对话级 = satisfaction 按信心加权;系统总分 = 两者调和平均。

## 8. 六 run 成绩速览

**对话级(均值,n=44)**:
| run | personalization | trustworthiness | info_flow | mixed_init | satisfaction |
|---|---|---|---|---|---|
| X1 no-extract | **3.98** | 3.55 | **4.09** | **3.39** | 4.05 |
| X2 min-reason | 3.95 | 3.50 | 4.02 | 3.30 | **4.07** |
| R4 +BM25+RL | 3.95 | 3.41 | 4.07 | 2.52 | 3.93 |
| R6 qwen-only | 3.93 | 3.68 | 3.98 | 3.00 | 3.93 |
| R3 +convQwen | 3.82 | **3.70** | 3.98 | 2.70 | 4.02 |
| R5 +SPLADE | 3.80 | 3.39 | 3.86 | 2.82 | 3.84 |

**逐轮(均值)**:
| run | n | rubric_score | engagement | relevance&usefulness | subtopic_quality |
|---|---|---|---|---|---|
| X2 min-reason | 642 | **4.13** | 3.42 | **4.40** | **3.95** |
| X1 no-extract | 646 | 4.06 | **3.55** | 4.33 | 3.87 |
| R5 +SPLADE | 682 | 3.97 | 3.20 | 4.11 | 3.69 |
| R4 +BM25+RL | 680 | 3.93 | 3.14 | 4.24 | 3.79 |
| R6 qwen-only | 682 | 3.85 | 3.32 | 4.23 | 3.75 |
| R3 +convQwen | 690 | 3.74 | 3.17 | 4.12 | 3.65 |

意外发现:两个"附赠" run(X1 无画像抽取、X2 最小推理档)在对话质量维度普遍领先,主打的 R3/R4 未占优——待 per-user-type 修正聚合(只算覆盖 rubric)后细看,写 report 时可深挖。

## 9. 来源

- TREC iKAT 2025 overview:https://trec.nist.gov/pubs/trec34/papers/Overview_ikat.pdf(本地文本:session scratchpad `ikat25_overview.txt`)
- iKAT 2025 资源论文(SIGIR'26, Abbasiantaeb et al.):https://downloads.webis.de/publications/papers/abbasiantaeb_2026.pdf(`ikat25_sigir.txt`)
- iKAT 2026 guidelines:https://www.trecikat.com/guidelines/(user simulators 保密声明、行为维度)
- Sim.API 中间件:https://github.com/marcel-gohsen/user-simulation-api(多模拟器/topic;2025 rubrics.csv)
- RUBRIC 方法论:Farzi & Dietz, "Pencils Down! Automatic Rubric-based Evaluation of Retrieve/Generate Systems"(ICTIR 2024)
