# iKAT 2025 数据报告（格式、与 23/24 的区别、track 关系）

> 生成脚本：[`data_preprocessing_scripts/preprocess_ikat25.py`](data_preprocessing_scripts/preprocess_ikat25.py)
> 原始数据：`github.com/irlabamsterdam/ikat2025`（`offline/` 目录）
> 产出：`data/topics/ikat25/ikat_2025_test.json`（处理后，与 23/24 同 schema）、`data/qrels/ikat_2025_qrel.txt`

---

## 1. 一句话结论

iKAT 2025 的 **passage ranking（官方叫 offline / Cranfield-style）** 评测资源齐全（topics + NIST qrels），可以**完全照 23/24 的方式离线评测**。我们已把它处理成与 23/24 一模一样的内部格式，**45 个被判过的 turn 的 qid 100% 对齐 qrel**，可直接用于任务四。

---

## 2. 数据规模

| 项 | 值 |
|---|---|
| conversations（topics） | **17** |
| 总 turn 数 | **188** |
| 被 NIST 判过相关性的 turn（offline 评测覆盖） | **45**（分布在 12 个 conversation） |
| qrel 行数（passage-level，0–4 级） | 5650 |
| PTKB 条目 | 每个 persona ~19–21 条 |

**topic 编号 `X-Y` 的含义**（实测）：`X` = 用户/persona（**PTKB 在同一个 X 内共享**），`Y` = 该 persona 下的某个对话主题。例如 `3-1`(Finger injuries) 与 `3-2`(New activities) 共享同一份 PTKB（"I work at a desk job."）。共 9 个 persona、17 个对话。

---

## 3. 原始数据格式（2025 offline）

`offline/2025_test_topics.json` 是一个 list，每个元素是一个 conversation：

```json
{
  "number": "1-1",
  "title": "Acid reflux",
  "ptkb": ["I want to stop doom scrolling.", "I like my local pizzeria.", ...],   // list[str]
  "responses": [                                                                   // 轮次数组
    {
      "turn_id": 1,                                  // int
      "user_utterance": "Hi there! Can you tell me some food good for acid reflux?",
      "resolved_utterance": "...",                   // 人工改写（oracle）
      "response": "...",                             // 系统参考回答
      "relevant_ptkbs": ["I enjoy apple, kiwi..."],  // list[str]，是 PTKB 原文不是索引
      "citations": ["clueweb22-en0022-52-08268:1", ...]  // 回答出处 passage id
    },
    ...
  ]
}
```

`offline/qrel/qrels-nist.trec`（标准 TREC qrel，空格分隔）：
```
9-1_10 0 clueweb22-en0000-91-01807:2 3
qid    Q0 docid:passage                 rel(0-4)
```
注意 **qid = `{number}_{turn_id}`，用下划线**（`9-1_10`）。

---

## 4. 与 2023/2024 的显著区别

### 4.1 字段重命名（raw 层）
| 概念 | 2023/2024 | 2025 |
|---|---|---|
| 轮次数组 | `turns` | **`responses`** |
| 用户话语 | `utterance` | **`user_utterance`** |
| 人工改写 | `resolved_utterance` | `resolved_utterance`（同名）|
| PTKB 容器 | `ptkb`：**dict** `{"1":..,"2":..}` | `ptkb`：**list[str]** |
| PTKB 相关性 | `ptkb_provenance`：**索引** list[int] | `relevant_ptkbs`：**PTKB 原文** list[str] |
| 回答出处 | `response_provenance` | **`citations`** |
| qrel qid 分隔符 | `-`（如 `0-10`） | **`_`**（如 `9-1_10`）|

### 4.2 处理脚本如何对齐（已实现）
- `responses→turns`、`user_utterance→current_utterance`、`resolved_utterance→oracle_utterance`、`citations→response_provenance` 字段映射；
- `ptkb` list → 1-indexed dict（与 23/24 一致）；`relevant_ptkbs`（字符串）→ 反查得到 `ptkb_provenance`（索引）；
- **turn_id 统一成 23/24 的连字符格式** `f"{number}-{turn_id}"`（`9-1-10`），因为 `Turn.get_turn_order()` 用 `int(turn_id.split("-")[-1])`，下划线会崩；
- **qrel qid `9-1_10` → `9-1-10`**（`replace("_","-")`），从而与 processed turn_id 对齐 → 实测 45/45 命中。

### 4.3 评测层面的区别（重要）
- **判过的 turn 很少**：23/24 大部分 turn 有判；2025 offline 只有 **45/188** 个 turn 被 NIST 判过（pooling），其余 turn 在 trec_eval 里不计分。任务四在 2025 上实际打分的是这 45 个 turn。
- 2025 额外提供 `qrels-llm.txt`（GPT-4.1 判，38809 行）可作 **LLM-judge 对照**；以及 PTKB 相关性 qrel（`qrels-ptkb-nist.trec`）。

---

## 4bis. PTKB 从"静态选择"变"动态可编辑"（2025 显著新增）

iKAT 2025 强调系统应能在对话中**发现/新增 PTKB**（不只是从给定列表里选）。guidelines 原文：*"New PTKB statements can be extracted from the previous conversations and mentioned in the list of relevant PTKB statements"*；评测会同时评 *"pre-defined and newly extracted PTKB statements"*（P / R / F1）。

**topics 文件如何体现？分两层：**
1. **静态层（主 topics 文件内）**：conv 级 `ptkb`（21 条初始 PTKB）+ 每轮 `relevant_ptkbs`（这轮相关的 PTKB **原文**）。这对应 "PTKB 识别/选择" 子任务。实测：134 处 `relevant_ptkbs` 全部能在 conv 级 ptkb 列表里找到（我的预处理把它们转成 `ptkb_provenance` 索引，无遗漏）。
2. **动态层（单独文件 `offline/ptkb/ptkb-update.json`）**：体现"**编辑/新增**"。结构：
   ```json
   {"number": "1-2", "title": "Healthier lifestyle",
    "new_ptkb": [{"statement": "I have acid reflux.", "turn_dependence": [5, 10]}, ...]}
   ```
   - 编号 `X-Y`：**`X`=persona/用户**，**`Y`=第几段对话**。同一 persona 的 `X-1`/`X-2` **共享同一份 base PTKB**（实测全同；唯一例外 6-2 一个 typo `"I I am..."`，已修）。
   - **为什么 new_ptkb 全挂在 `X-2`（`1-2…9-2`，无 7-2）**：overview §2.2 *"users reveal additional information about themselves **in the first conversation, which becomes relevant in the second conversation**"*。即新事实在**第一段(X-1)被吐露/accumulate**，但挂在 **X-2** 下，因为那是它**需要被"记起来"并用上**的地方；X-1 自己没有（首次出现、无前文可记）；persona 7 只有 7-1（单对话）故无条目。
   - **`turn_dependence` = 第二段对话(X-2)里"需要用到这条记忆"的 turn_id**（结合对话内容判定）。铁证：persona 6 `"I have an oily scalp and hair"` td=`[1,2,3]`，而 `6-2` 前三轮全在问"洗发水/油性头皮"；反证：persona 1 `deadline` 在 `1-1` turn 8 才被说出但 td=`[4,5]`、`kidneys` 在 `1-1` turn 10 被说出但 td=`[8]` → td **不是**"X-1 里被 reveal 的轮次"。
   - **`ptkb-update.json` 是 organizer 手写的 oracle**（overview: *"All the topics, conversations, and PTKB statements were manually created by the organizers"*），**不是 pool 系统提交**。

**怎么评测（已核实，诚实版——更正我前面"新 statement 也在 offline 打分"的说法）**：
- **offline PTKB 子任务 = "PTKB Statement Classification"**：每个 turn 对 **static PTKB（那 ~21 条）** 逐条做二分类"是否相关"，用 `ikat_2025_ptkb_qrel_{organizers,nist}.txt` 评 P/R/F1。**实测：两个 PTKB qrel 里的 ptkb-id 全部落在 static `0..N-1`，没有任何"新 PTKB"的独立 id** → **新发现的事实在 offline qrel 里并不单独打分**。
- 真正考"动态/edit 能力"的是 **interactive track**：Sim.API 每轮回传 `relevant_ptkbs`（**自由文本**，可含新发现的），由 NIST rubric/dialogue 评测（guidelines: *"assess both pre-defined and newly extracted PTKB statements"*）。
- 所以 `ptkb-update.json` 是**描述跨对话依赖的 oracle 资源**，不是 offline 的打分 qrel。

**与 23/24 区别**：23/24 的 PTKB 静态，`ptkb_provenance` 只选已知 PTKB；2025 同一 user 多段对话 + 单独的 `ptkb-update.json`（跨对话新事实，oracle）。

**2025 结果里有没有评"新提取"？→ 没有**：Overview §5.2 PTKB Provenance 结果只报对 judged（=static）PTKB 的 set-based **P/R/F1**（turn + conversation 级；Table 5 NIST / Table 6 organizers；Fig 10/11 按对话/按深度），**没有任何单独的"newly-extracted PTKB performance"**。"extract additional PTKB" 只出现在个别系统（如 `ucsc-dynamicPTKB`）的方法描述里，是策略不是被测指标 → 公开 offline 评测里"系统提取到有意义新 PTKB"这件事**实际上没被有效衡量**。

**2026（Year 4）guideline 的变化**：显式新增独立小节 **"Extracted Relevant PTKB Statements Assessment"**：*"...Precision, Recall, and F1 ... We will **assess both pre-defined and newly extracted PTKB statements** by the teams."* 即 2026 把"评新提取"写成了独立 assessment（2025 只是说了没在结果兑现）。多对话 + *"track persona statements of reoccurring users"* 框架延续；但 2025 那句 "conv1 埋点→conv2 用上" 的精确表述，当前 2026 页面未见明文。

**对本项目的影响**：任务四（oracle passage ranking）**不碰 PTKB**，无影响；只跟任务五（Sim.API 每轮回传 `relevant_ptkbs`）相关。

**本地已放好**：`data/qrels/ikat_2025_ptkb_qrel_organizers.txt`、`ikat_2025_ptkb_qrel_nist.txt`、`ikat_2025_ptkb_update.json`（qid 已 `_`→`-` 对齐 turn_id）。

---

## 5. 你最关心的问题：interactive track 和 passage ranking 怎么区分？

**先纠正我自己之前的口误**：我一度说"两个 track 共享一份 topics"，当时没有依据。现在按 `ikat2025` 仓库的实际结构核实如下。

### 5.1 2025 确实是两套并行评测，资源分目录存放
```
ikat2025/
├── offline/        ← Cranfield 式 passage ranking（+ 生成）
│   ├── 2025_test_topics.json          (17 conv, 多轮, 含 resolved_utterance)
│   ├── qrel/qrels-nist.trec           (NIST passage 判, 0-4)   ← 任务四用这个
│   ├── qrel/qrels-llm.txt             (LLM 判)
│   ├── nuggets/ gold-response-nist.json (生成子任务的评测)
│   └── runs/auto|gen/...              (官方 baseline run，含 orga-ance-norerank、mq4cs-bm25/splade 等)
└── interactive/    ← 用户模拟
    ├── assessements/NIST/topic{X-Y}.final  (对话级人工评测)
    ├── rubrics/rubrics.json                (rubric 评测标准)
    └── runs/trec-ikat25-runs/*.jsonl       (各队 interactive run)
```

### 5.2 关键观察：两套底层用的是**同一批 17 个对话**
- interactive 的 NIST 评测文件命名是 `topic1-1.final … topic9-2.final`，**恰好就是 offline topics 里的那 17 个 `number`**。
- 所以**不是"不同 topic 集分给两个 track"**，而是：**同一批对话**，**两种评测方法 + 两套评测资源**：
  - **offline（passage ranking）**：把每个 turn 的 query（raw / oracle resolved）→ 检索 ClueWeb → 用 **passage qrel** 打分（nDCG@10、MRR…）。**这是经典可复现的离线评测，本项目任务四走这条线。**
  - **interactive（用户模拟）**：系统只拿**首句** utterance，之后通过 **Sim.API** 一来一回（系统回 response+citations+relevant_ptkbs → API 给下一句），最后用**对话级 / rubric 人工或 LLM 评测**打分。

### 5.3 "只给第一问、后面自由"的理解对吗？
**基本对，但要修正一点**：interactive 里后续 utterance 不是系统自己编，而是 **simulated user 根据该 topic 的 persona + 参考对话轨迹生成**的——所以是"沿着这个 topic 的对话走，但由模拟器实时产出后续 user turn"，不是完全自由。offline 那套则直接把**完整多轮 + 人工改写**都给你。

### 5.4 官方今年只收 interactive 提交
guidelines 原文：*"This year, we only accept interactive **submissions**"*。但这只是**提交形式**的要求；**offline 评测资源（topics+qrel）照样公开**，所以我们做 passage ranking 的离线实验完全没问题（也正是大量 `offline/runs/...` baseline 的来源）。

---

## 6. 对任务四的直接收益
- 2025 可出**完整指标**（不再"只出 ranking"）：qrel = `data/qrels/ikat_2025_qrel.txt`。
- 仓库里有官方 **ANCE / BM25 / SPLADE** 的 2025 offline run（如 `orga-ance-norerank`、`mq4cs-gpt41-bm25/splade`），后续可作为额外对照（注意它们的 pipeline 与我们的不完全一致，仅供参考）。
- ⚠️ 待办（任务四接线时）：在 `evaluation.py` 的 `--topics` choices 加 `ikat_25_test`，并新增 `filter_ikat_25_evaluated_turns`（按 45 个 judged qid 过滤），在新 yaml 的 `param_mapping` 配 `input_query_path` / `qrel_file_path` / `passage_block_num`。
