# PTKB 逐句分析 + 非可加性实验:与 Mo et al. (CIKM'24) 的方法对齐文档

> 2026-07-18。目标实验:每 turn 逐句注入 PTKB → per-statement 检索增量 → 经验 noyau →
> **检验"单独正向的语句合并后是否仍正向"(非可加性)——该问题 Mo et al. 未讨论,是我们的增量**。
> 依据:arXiv:2407.16192 全文 + 官方代码 fengranMark/PersonalizedCIR(add_ptkb_one_by_one.py /
> automatically_select_ptkb.py,本地存档于 session scratchpad)。

## 1. 他的做法(代码级事实,论文正文多处未写)

1. **Prompt**:联合生成 rewrite+response——"please help reformulate the question into rewrite …
   but also generate an informative response … based on user's personal information";system 消息
   "you are a query rewriter and knowledge selector";单条语句作为 `User's personal information:`;
   对话历史 Question/Response 交错;输出 `Rewrite: $Rewrite\nResponse: $Response`。
2. **gpt-3.5-turbo-16k,未设 temperature(=默认 1.0)** → per-statement Δ 含大量生成噪声。
3. **检索 query 形式**:BM25 用 rewrite+response 拼接,ANCE 只用 rewrite(§4.2.2)。
4. **逐句注入 = 只给该句**(其余语句一概不给)。
5. **判定**:per-turn NDCG@3(单句) − NDCG@3(无 PTKB) > 0(threshold=0),一 turn 可多句;
   ≥1 句正向 ⇒ 该 turn "need PTKB"(67/176)。1818 个 (turn, 句) 检索 run,qrels 按伪 qid 复制。
6. **代码噪点**(复现要小心):首轮 conv==[] 时把字符串当序列 zip(被空 history 掩盖);
   choose_ptkb 的 start 未初始化(发布脱漏);Table 2/3 有两处数字矛盾(见精读报告)。
7. **集合**:他硬编码的 176 qid 与我们的 per_query 评测集**逐一相同**(已验证)。
   论文 Table 1 写 "11.6M collection" 疑为 116M 笔误(他 BM25 绝对数与我们同量级)。

## 2. 我们的对齐与偏离(his | ours | 理由)

| 项 | Mo et al. | 我们 | 理由 |
|---|---|---|---|
| 改写器 | gpt-3.5-16k, T=1 | **qwen3-32b-AWQ, T=0** | 免费、确定性;他的 Δ 混生成噪声(他 "Automatic 全集反而低于 None" 的矛盾嫌疑人) |
| 噪声控制 | 无 | 30 个 (turn,句) × 3 次重复 → 噪声地板 ε | Δ 的可归因性 |
| prompt | 联合 rw+rs | **MQ4CS 极简 rewrite-only**(用户钦定) | 模板恒定、只变语句;代价:BM25 无 response 可拼,绝对数与他不可比(我们只比同管线内 Δ) |
| ∅ 基线 | 未见脚本 | MQ4CS + 空 ptkb(header 保留) | 与其他条件唯一差异 = 语句本身 |
| 检索器 | BM25 + ANCE | **BM25 + qwen dense(精确流式)** | 生产检索器;noyau 按检索器分开标(复刻其 Fig.1 retriever-specific 现象) |
| 判定 | Δ>0 | Δ>0 主 + Δ>ε 敏感性 | 复刻 + 稳健 |
| 评测 | 176 turns, NDCG@3, pytrec_eval | 同 qrels 同 176 | 完全对齐 |
| **新增** | — | noyau 联合、full、非可加性检验、分层报告 | 论文空白 |

## 3. 条件矩阵(iKAT-23,176 turns)

∅(1)→ 逐句(1818)→ noyau_BM25 / noyau_qwen(各≤176,由单句 Δ>0 得出)→ full(176)
+ 噪声重复(90)。改写总计 ≈ 2.4k 次(本地 vllm T=0);检索:BM25 pyserini batch +
qwen dense 一趟流式精确扫描(全部查询一起,~分钟级)。

## 4. 预注册分析(动手前写死)

- **主指标**:单句均正向(该 turn noyau 内全部 Δ>ε)但 **noyau 合并 Δ<0** 的 turn 占比;
- 次指标:Δ(noyau) vs Σ单句Δ(sub-additivity 缺口)、vs max 单句 Δ;Wilcoxon;
- 分层:有正向句的 turn(预计 ~90-100 个,参照他 Fig.1)vs 无;BM25 与 qwen 分开报告;
- 复刻检查:我们的 "need PTKB" turn 数与他 ~98(BM25)/~89(ANCE) 的量级对照;
- **定位声明**:noyau 用 test qrels 选出,属 oracle 分析,不是可部署方法(与他同性质)。
