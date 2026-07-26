# 分支对话、本体、与可解释的会话式查询改写:一份文献调研

**日期**:2026 年 7 月 26 日

**这份报告回答三个问题**:第一,除了 TREC CAsT 2022,还有哪些数据集或基准是**分支结构**的对话,以及有没有人专门研究这种分支本身;第二,把**分支结构和本体结合起来做可解释的会话式查询改写**——这个想法在文献里处于什么位置,是全新的还是已有人做;第三,如果要做这个方向,现实的机会和陷阱在哪里。

调研方式是四路并行的关键词检索加引文图谱遍历,加上我自己对官方数据和 overview 原文的直接核实。报告里出现的每一篇论文都经过抓取页面确认;凡是没能核实的地方都明确标注。

---

## 一、三句话结论

**第一,在信息检索领域,CAsT 2022 几乎是唯一的分支对话测试集,而且很小。** 18 棵话题树、205 个用户轮次、50 条根到叶路径。它是唯一同时具备"任意深度的分支树 + 每个分支各自的相关性判定 + 专门为路径设计的评价指标"的公开资源。它的后继者 iKAT 放弃了树结构。别处被称为"分支"的数据,基本都是更弱的形态。

**第二,"本体 + 会话式查询改写 + 可解释性"这个组合,据我核实的范围,几乎不存在。** 这个判断有四条独立证据支撑,详见第三节。但周边并不空:结构化的会话需求表示有人做(CONVINSE),把会话结构画成图用于解释有人提过(但只是一张示意图),显式分解成语言学子问题的改写有人做(ZeQR),知识图谱路径当解释在**推荐**侧已经很成熟。所以这是一块有边界、有邻居、但中心确实空着的地。

**第三,这个领域最大的问题不是没人做,而是几乎没人评。** 大量论文在标题或摘要里宣称"可解释",证据却只是一两个案例展示。整个调研里真正做过可解释性人工评测的只有个位数几篇。而且更麻烦的是,邻近领域里做得最严谨的那几个用户实验,结论都是**泼冷水的**:解释会同时增加用户对错误答案的信任,而在会话搜索里,用户甚至给"完全不给解释"的回答打了最高的有用性分。

---

## 二、分支对话的数据版图

### 2.1 一个有用的分类:什么才算"真的分支"

调研中最有价值的分析性产出,是把被称为"分支"的东西分成了五种形态。这个区分很实用,因为很多看起来是命中的数据集其实不是。

**第一种,一层扇出**。一个查询或问题分裂成若干独立的侧面(facet)或解释,每个侧面有自己的答案或相关性判定,但**不会二次分叉**。Qulac、ClariQ、AmbigQA、ASQA、MIMICS 都属于这一类。

**第二种,平行的独立线性对话**。若干条完整的线性对话共享一个话题或人物设定,但分化在第一轮之前就固定了。TREC iKAT 2023 和 2025 属于这一类。

**第三种,发布时被压平的树**。原始数据本来是树,发布时被拆成一条条根到叶的路径,兄弟之间的链接被丢弃。DialoGPT、PolyAI 的 Reddit 数据、ProCIS,以及**所有大规模的人机对话语料**都是这样。

**第四种,树只存在于模型里而不在数据里**。例如 Context-Agent 的 NTM 基准,它的"非线性"指的是单条线性转录里的话题跳转,树是智能体的内部记忆结构。

**第五种,真正的分支**,但存在于开放域对话、游戏与互动小说、论坛回帖树、以及众包撰写的替代回复里——**不在会话搜索里**。

### 2.2 CAsT 2022:唯一的那一个

官方 overview 的原话是:"a major difference from previous years is that topics in CAsT 2022 follow a 'tree' structure with distinct conversational paths",而且"User and System utterances are decoupled to support branches with multiple different system responses to a single user utterance. **The turn structure is encoded by the parent relationship.**"

规模:18 个话题,平均每个话题 11.39 条用户话语、2.7 个子话题,共 205 条用户话语;每个话题最多 9 条、最少 1 条不同的会话路径。判定方面,深度 20 的池子里有 49,878 个段落、判定了 43,027 个;18 个话题里只有 17 个被评估(话题 134 没评);205 个用户轮次里有 163 个拿到了判定。

**一个复用时必须知道的坑**:用户回答系统澄清问题的那些轮次**被刻意排除在判定之外**,overview 原话是"vague and un-specified turns were not included, only the clarified versions"。也就是说,**开启一条分支的那个回答本身没有 qrels**,尽管它开启的分支有。

数据发布了两种序列化:`2022_evaluation_topics_tree_v1.0.json`(18 棵树,带 `parent` 链接)和 `2022_evaluation_topics_flattened_duplicated_v1.0.json`(50 条路径,共享前缀被复制)。这个"同时发布树形和展开形"的做法本身就是发布分支对话数据的一个好先例。有一个小坑:仓库的 README 写的字段名是 `parent_turn`,而实际 JSON 里的键是 `parent`。

我自己对这份数据做过统计,独立复现的结果是:18 棵树、205 个用户轮、50 条路径,而且**分叉高度集中在对话早期**——按分叉父节点的深度统计,大约三分之二的分叉发生在前四层。

### 2.3 一个独立佐证:分支确实倾向于发生在早期

《The Branch Not Taken: Predicting Branching in Online Conversations》(Meital、Rokach、Vainshtein、Grinberg,arXiv:2404.13613)在 Reddit 的三个子版块上研究"分支预测"任务,即判断一条新评论会不会回复到一个**中间节点**而不是叶子。规模是 CMV 7,000 段对话约 52 万条评论、ELI5 9,000 段约 38 万条、AskScience 7,000 段约 52 万条,平均每段对话 28 到 34 条分支。

他们的核心发现之一是:**分支与更多参与者相关,并且倾向于发生在树的较早层级。** 这与我在 CAsT 2022 上算出的分布完全一致,而且是在完全不同的数据、完全不同的领域上得到的。如果要论证"对话分支的早期集中性是一个普遍现象而非 CAsT 的构造产物",这是最好的外部证据。

### 2.4 其他真正的分支数据(按可用性排序)

**ShARC**(EMNLP 2018)是规则理解领域的分支数据集,**它本身就是一组对话树**:948 条规则文本,每条配一棵对话树,树在"是/否"回答处分叉。发布的是**按节点展平**的 32,436 条实例,但树是**完全可还原的**——每条实例带 `tree_id` 和 `history`(到达该节点的问答路径),按 `tree_id` 分组、把 `history` 读成路径前缀就能重建整棵树。许可 CC BY-SA 3.0,测试集 2024 年 2 月已公开。它的后续 **EXtrA-ShaRC**(UMAP 2024)更值得注意,后面第四节会讲。

**MultiTalk**(AAAI 2021)是最纯粹的分支对话树:分支因子 10、延续因子 3、最大深度 6,120 个起始提示、32 万句话语。收集方式是**同一个标注者在看到完整上下文后,自己写出全部十条回复**,所以这十条是同一上下文的真正平行延续。数据是 CSV,带显式的 `parent` 和 `depth` 字段,许可 CC BY-SA 4.0。

**CausalDialogue**(Findings of ACL 2023)是一个有向无环图而不只是树:2,322 个图、4,866 条分支、46,109 条话语。它的特别之处在于**既捕捉分叉(一句话引出多种不同回应)也捕捉汇合(不同的中间轮次收敛到同一个回应)**。如果要研究"分支之后还能不能合回来",这是唯一的数据。

**FloDial**(EMNLP 2021)把流程图当作对话的骨架:2,738 段对话、12 张排障流程图,众包时**系统性地遍历流程图的各条路径**,所以语料在构造上就是分支的一个覆盖集;每条话语带 `grounded_doc_id` 指向具体的流程图节点。

**Conversational Tree Search / REIMBURSE**(EACL 2023)是领域专家用图形化工具手写的对话图:123 个节点、最大深度 32、最大出度 14,配 452 条自由提问和 408 条回答改写。规模很小,但它是"专家定义的树 = 系统行为的约束"这个思路最明确的一篇。

### 2.5 大模型聊天日志:分支信息被系统性地丢弃了

这一节的结论是一个**经过核实的否定结果**,而且我认为它本身就值得写进论文。

**ChatGPT 自己的导出格式是一棵真正的树**:`conversations.json` 里存的是一个 `mapping` 字典,每个节点带 `id`、`parent`、`children`、`message`。用户点"重新生成"会产生兄弟助手节点,编辑之前的某轮用户消息会从那里分叉。HuggingFace 的开源 `chat-ui` 实现了同样的结构。

但是,**所有公开的研究语料都是线性化之后的**:WildChat 和 LMSYS-Chat-1M 是通过一个网关收集的,那个网关从来没见过分叉;ShareGPT 和 ShareChat 抓的是渲染后的单路径视图;ShareLM 在入库时归一化成了扁平的 role/content 列表。我核实了这些数据集的实际字段结构——**没有 `parent`、没有 `children`、没有 `branch`、没有兄弟字段**。

也就是说:**一个来自真实部署产品的、包含用户重新生成和提示编辑的分支树语料,目前不存在。**

现实的替代品有三个。**OpenAssistant(OASST1/OASST2)** 是最好的一个:66,497 棵树、161,443 条消息,字段里有 `parent_id` 和 `message_tree_id`,而且——这点很关键——**兄弟回复带 `rank` 字段,是逐分支的人类偏好信号**。这在所有分支数据里几乎是独一份。**PRISM**(NeurIPS 2024)是"用户选择分支"的最好代理:1,500 名参与者,**第一轮由最多四个模型同时回应同一条用户消息**,带评分和文字反馈,然后选一条继续。**Chatbot Arena 的 `arena-human-preference-140k`** 里 `full_conversation` 字段是显式的双路分叉,每个分叉带 `winner` 标签,但分支是系统诱导的、永远是二元对称的。

### 2.6 论坛数据:规模和结构成反比

Reddit 天然是回帖树。Pushshift 数据集保存了完整的森林(6.5 亿条提交、56 亿条评论,每条带 `link_id` 和 `parent_id`),但它不是对话语料,没有轮次定义也没有标注。把它变成对话的两个大规模工作——DialoGPT(1.47 亿条实例)和 PolyAI 的 Reddit 集(7.27 亿条)——**都明确地把树拆成了路径**。DialoGPT 论文的原话是:"We extract each path from the root node to the leaf node as a training instance."

而**保留了树并且带标注**的语料都小三到六个数量级:Coarse Discourse 9,473 个帖子链、Winning Arguments/ChangeMyView 3,051 段对话、RumourEval 2019 只有 446 个谣言。ConvoKit 这个框架值得一提,因为它**不压平**——它的 `Conversation` 对象直接提供 `get_root_to_leaf_paths()`、`traverse()`、`get_subtree()` 这些方法,压平是显式的可选操作而不是入库时的默认损耗。

还有一个警示性的数据点:Kummerfeld 等人(ACL 2019)发现,被广泛使用的 Ubuntu Dialogue Corpus 里**89% 的对话要么缺消息、要么多消息**——压平过的语料经常是悄悄坏掉的。

### 2.7 游戏与叙事:分支结构最成熟的地方,而且已经有本体

这一块不是会话搜索,但**它是现存最大的真正分支对话结构,而且是"分支 + 本体"这个组合唯一已经落地的地方**。

**KNUDGE**(EMNLP 2024,《Ontologically Faithful Generation of Non-Player Character Dialogues》)是对你们那个想法最直接相关的先例。它取自游戏 *The Outer Worlds*:45 个支线任务、**159 棵对话树**、4,700 条话语。关键在于它的对话树是一个有向图(**可以有环**),而且**标注细到单条话语级别**——每个节点挂着任务约束陈述和实体传记事实,平均每个实体 7.4 条事实、每条 NPC 话语 1.0 条事实。

这说明"分支结构 + 本体约束"在技术上是成立的、已经有人实现了。区别在于它做的是**生成**(让 NPC 说出符合世界设定的话)而不是**检索**,本体是游戏世界设定而不是用户画像或查询语义。写相关工作时必须处理它。

**MACHIAVELLI**(ICML 2023)则是规模最大的:134 个"选择你自己的冒险"游戏、57 万个场景、286 万条标注,而且**分支是显式发布的**——代码里有 `Node` 类带 `children` 和 `parents`,每个游戏一个完整的图。

---

## 三、分支被用来做什么:CAsT 2022 的路径级评价指标

这是把"分支"从一个数据格式变成一个**可用信号**的关键机制,我把 overview 的数学定义完整读了出来。

CAsT 2022 定义了三个指标,都建立在逐轮指标(实验里用 NDCG@3)之上,先对每条路径算分,再对所有路径求平均。

**第一个是对话累积增益(Conversational Cumulative Gain,CCG)**,就是路径上各轮分数的算术平均:

> CCG(s) = (1/|p|) · Σ_{t∈p} s_t

它的缺点作者自己承认:把各轮当成独立的,轮次之间的依赖只能靠"在路径上取平均"这一点微弱地体现。

**第二个是对话路径分数(Conversational Path Score,CPS)**,用来补上这个缺点。做法是先设一个阈值 θ,把路径切成"连续相关段"和"不相关轮",然后**把每个相关段的长度取 γ 次幂再求和**,用 |p|^γ 归一化。当 γ > 1 时,"连续三轮都答对"的得分会高于"两轮对、断掉、再一轮对"——也就是**奖励对话流不被打断**。当 γ = 1 时它退化成 CCG。

**第三个是轮次偏置累积增益(Turn-Biased CCG,TBCCG)**,这是三个里最有意思的,因为它**把一个用户模型直接编码进了指标**。假设用户在每轮之后以概率 P(c|t) 继续对话,而这个概率取决于系统这一轮答得好不好:答得好(s_t > θ)时是 P_r,答得差时是 P_n,且 P_r ≥ P_n。于是后面的轮次要按"还有多少比例的用户走到了这里"来加权:

> TBCCG(s) = (1/|p|) · ( s_{t₁} + Σ_{i=2}^{n} [ ∏_{j=2}^{i} P(c|t_{j-1}) ] · s_{t_i} )

直观含义是:**如果第二轮答砸了,后面所有轮次的分数都会被大幅折扣,因为用户已经走了。** 当 P_r = P_n = 1 时它退化成 CCG。官方实验用的设置是 NDCG@3 作为逐轮指标、θ = 0.33、γ ∈ {2,3}、P_r = 1、P_n ∈ {0, 0.25}。

**这三个指标是分支结构目前唯一被正经使用的方式。** 而下一节会说明,它们从未被用于可解释性。

### 三点五、这套路径机制其实无人接手(经核实的否定结果)

引文图谱遍历得出了两个很硬的事实,它们把这个方向的"空位"钉死了。

**第一,CAsT 2022 那三个路径指标是零引用提出的。** 我抓了 overview 的 PDF,**它全文只有 6 条参考文献**,而 CCG、CPS、TBCCG 三个指标的提出**没有引用任何先前的路径评价或会话评价工作**——没有引 C/W/L,没有引 Lipani 等人,也没有引任何 session-based 指标。

**第二,更关键:没有任何一篇引用 CAsT 2022 的论文接手了它的分支机制。** 具体做法是把**引用 CAsT 2022 的全部 53 篇论文**的摘要都抓下来,扫描 `tree|branch|path|trajector|stop|continuation|alternative` 这些词。**结果只有一篇命中,而且是误报**(讲澄清问题的 "ungrounded alternatives")。这是摘要级扫描,不能完全排除正文里有,但方向已经很清楚:**这套机制自 2022 年提出后处于休眠状态。**

### 三点六、一个具体的、高价值的空位:P(c|t) 从来没有被真正估计过

这是整个调研里我认为**最值得动手**的一个发现。

回看 TBCCG 的公式:它的全部理论内容都在 P(c|t) 这个"用户是否继续"的概率上。而 **CAsT 官方实验里把 P_r 固定成了 1**——也就是说,"答得好时用户一定继续"。**这等于把这个指标最有意思的那半边关掉了**,只剩 P_n 在起作用。overview 自己也承认这是个 pilot。

而与此同时,**有人独立地把估计这个量所需要的全部经验机制都建好了**:Fu、Pérez-Ortiz、Lipani 的《An Analysis of Stopping Strategies in Conversational Search Systems》(ICTIR 2024)把信息检索里的**停止规则**移植到会话场景,定义了新规则,并在多个会话数据集上测哪些规则真的能预测用户实际停止的位置。他们的结论是文本统计特征占主导——交换的词数、名词数、名词短语数、句子数,其中**用户输出的唯一名词数**是特别强的停止规则。

**这篇论文只有 1 次引用。没有人把它和 CAsT 的 TBCCG 连起来。**

也就是说:**指标框架在那里(TBCCG),经验估计方法在那里(停止规则),两者从未相遇。** 把 P(c|t) 从"拍脑袋常数"换成"从数据拟合的函数",这是一个定义清晰、工作量可控、而且明显没人做过的贡献。

顺带一提,TBCCG 很可能有一个未被承认的思想来源:**C/W/L 框架**(Azzopardi、Thomas、Moffat,SIGIR 2019 的 `cwl_eval`)把所有排序指标统一表述成一个**继续函数**,并由此导出期望效用、期望成本和**期望深度**。而 **Azzopardi 正是 CAsT 2022 的共同作者**,TBCCG 的 P(c|t) 本质上就是把 C/W/L 的继续概率**从"排序位次"抬到"对话轮次"**。这条线索的价值在于:C/W/L 有维护中的实现,而且它提供了一个原则性的方式去定义"**期望对话深度**"——即用户在放弃之前能走多远——作为一个一等公民指标,而不是临时发明一个。

---

## 四、本体 × 会话式查询改写 × 可解释性:一个近乎空白的交叉

### 4.1 空白判断的四条证据

我不想只给一个印象,所以列出四条可核实的独立证据。

**第一**,dblp 的标题精确检索,对以下组合**全部返回零命中**:"ontology conversational search"、"knowledge graph query rewriting conversational search"、"ontology dialogue information retrieval"、"branching conversation retrieval"。

**第二**,dblp 检索 "conversational query rewriting" 返回 25 条标题,时间跨度 2020 到 2026,**其中没有任何一条提到本体或知识图谱**。这条线十年来完全是"神经生成 → 强化学习 → 偏好对齐"的演进路径。

**第三**,Semantic Scholar 检索 "ontology conversational query reformulation",排名前 20 的结果里 19 篇是"会话但无本体"的改写论文,唯一一篇带本体的是 2020 年一篇域特定本体的查询改写,**而它根本不是会话式的**。

**第四**,我把 TREC CAsT 2022 官方 overview 完整抽成文本后检索过:`interpretab`、`explainab`、`explanation`、`transparen` 四个词根**在全文出现零次**。CAsT 引入了树,但从未把树和可解释性联系起来。

再补一条旁证:2022 年那篇权威的《Explainable Information Retrieval: A Survey》全文 35 页,搜索 "conversational / dialogue / session / multi-turn" **一次都没有出现**。可解释信息检索和会话式改写在当时是两个完全不相交的文献。

### 4.2 最接近的一条线:显式结构化的会话需求表示

如果只读两篇,读这两篇。

**CONVINSE**(SIGIR 2022)把当前问题连同会话历史压成一个**四槽的结构化表示**:上下文实体、当前问题实体、谓词、期望答案类型,线性化成一个用竖线分隔的字符串,由微调的 BART 生成。论文原话说这些表示是"self-contained interpretable representations of the user's information need"。

**这里有一个对我们至关重要的设计取舍**:四个槽的值是**表层字符串,刻意不绑定任何 Wikidata 实体编号或本体类型**。这正是它能同时喂给知识库、文本、表格、信息框四种异构源的原因——一旦绑定本体,就只能查知识库了。换句话说,**现有最好的结构化会话需求表示是刻意不本体化的**。教授想加的那一层本体,恰恰是这条线主动放弃的东西。这既是机会也是风险,写论文时必须正面回应"为什么绑本体不会让你失去异构源"。

CONVINSE 还提出了一个叫**会话流图(Conversational Flow Graph)**的东西:问题和答案是节点,边把一个问题连到它所依赖的历史轮次。论文说这个图可以连同结构化表示一起呈现给用户,"either for gaining confidence in final answers, or for scrutinizing error cases"。**这是"会话结构 → 可解释性"最直接的已有先例,而且和 CAsT 2022 同年。** 但它只是论文里的一张示意图,**没有任何评测**。

**EXPLAIGNN**(SIGIR 2023)是同一组的后续,沿用四槽表示,把检索到的异构证据建成图,用迭代图神经网络反复剪枝直到只剩五条证据——这个小证据集就是"解释"。**它是整个调研里唯一一个对解释做了真正人工实验的工作**:亚马逊众包平台,限定通过率 95% 以上的 Master 工人,加蜜罐题剔除乱答,随机采样 1,200 个实例,剔除"用户靠先验知识而非解释做判断"的混淆样本后剩 771 个。任务是让用户看会话历史、当前问题、结构化表示和五条证据,判断系统给的答案对不对。结果:用户判断准确率 **76.1%**,自认"确定"的比例 79.8%,在"确定"条件下准确率升到 **79.2%**。

**如果你们要做可解释的改写并且想让审稿人认可"评过了",EXPLAIGNN 就是必须对标的评测范式。** 但要注意它评的是**答案的解释**,不是**改写的解释**——**改写本身的可解释性,至今没有任何人做过人工评测。**

### 4.3 一个直接的机会:已经有人抱怨过 CAsT-22 的树,但没人利用它

**ZeQR**(ICTIR 2023)把会话式改写显式拆成两个语言学子问题——**指代消解**和**省略补全**——各自转成机器阅读理解题来解。它没有做用户研究,但做了一件比"纯案例展示"更实质的事:**用可解释性当诊断工具**。因为中间表示是显式的,他们能把查询按歧义类型分层统计,再分别测两类查询上的检索表现,最后消融两个模块(去掉省略模块平均掉 20% 的 NDCG@5,去掉指代模块掉 6%)。这是"结构化中间表示带来可归因的错误分析"的一个可复制模板。

而对教授那个提议**最直接相关的一句**在这篇论文的末尾。原文写道:CAsT-22 引入对话树带来了额外难度——"the introduction of a dialogue tree in the CAsT-22 dataset brings additional complexity to the task. This dialogue tree's branching structure allows each conversational session to follow a unique topic flow path, thereby amplifying the challenge for the language model in resolving the resulting ambiguities."

**也就是说:已经有人观察到 CAsT-22 的分支结构让改写变难了,但没有任何人反过来把这个结构当成资源去利用。** 这正是切入口。

### 4.4 一大批"天然可读却从不宣称"的方法

这是调研里最值得注意的一个免费午餐:有一整批改写方法的输出**本身就是一个可读的理由**,但作者的动机全是数据效率、延迟或准确率,从不把可检视性当成待检验的主张。

**QuReTeC**(SIGIR 2020)对会话历史的每个词做二分类,直接输出"该把哪些历史词加进当前查询"——**这是天生的 rationale**。但全文检索 `interpret`、`explain`、`transparent` 等词,**出现零次**。它的卖点是远程监督。而且它其实报告了这个词集的**内在质量**:在 QuAC 衍生集上精确率 71.5、召回 66.1、F1 68.7,在 TREC CAsT 上 F1 78.5——**结构上这就是一个被评测过的解释,只是没人这么称呼它。**

**CRDR**(EMNLP 2022)给上下文每个词打三类标签:REL(历史里可能被指代或省略的表达)、IN(当前查询里的修改入口点)、O(无关),然后在入口点做替换或插入。**这实际上是一份可读的编辑脚本,而且直接从指代与省略的语言学定义推导出来。** 论文把可解释性列为贡献之一,但**没有任何可解释性评测**。

同族的还有《Incomplete Utterance Rewriting as Semantic Segmentation》(EMNLP 2020),它把改写变成预测一个**词级编辑矩阵**——这是文献里最纯粹的可检视中间表示,而它的动机完全是准确率和四倍推理加速。

**所以有一个低风险、高确定性的贡献摆在那里:去评测这些方法早已免费提供的解释。** 问人类"QuReTeC 选出的这组词是不是正确的理由",或者测"这个 rationale 是否忠实于检索器实际用到的信息"。至今没有任何人做过。

### 4.5 完全空白的部分,以及一个直接竞品

必须明确说清楚:**针对改写器的特征归因方法、rationale 抽取方法、忠实性指标,一个都没有。** 没有人对会话历史算显著性来解释一次改写;没有人对改写器做 comprehensiveness/sufficiency 式的评估;没有"这次改写是否忠实于会话内容"的指标。思维链在会话改写里一律被当成**性能手段**,从未被当成解释来验证。趋势甚至在反向走——2026 年有一篇把推理搬进了**潜空间**。

**但有一个直接竞品必须处理**:**LogiCGR**(WWW 2026,《Facilitating Generative Retrieval with Logical Denoising for Interpretable Conversational Search》)。它的摘要把问题定义成两点,和这个立意高度重合:一是上下文噪声,二是"**Poor interpretability**: the lack of transparency in how results are generated undermines user trust"。方法是课程学习加强化学习训练大模型做"逻辑去噪"再接生成式检索。**它的可解释性证据是"intuitive case studies"——又是案例展示,没有人评,没有忠实性指标。**

这篇的存在意味着两件事:第一,"可解释会话检索"这个 framing 在 2026 年已经有人抢占,相关工作必须处理它;第二,它用的是**大模型生成的思维链**这种最弱的可解释性形式,**完全没有结构化表示、没有本体、没有利用会话树**。差异化空间仍然完整。

### 4.6 一个支持这个方向的实证发现

《Towards Self-Contained Answers: Entity-Based Answer Rewriting in Conversational Search》(CHIIR 2024)做的是答案改写而不是查询改写,但它有一个发现对这个方向很重要。

他们建了一套**分级实体显著性**标注(0 不重要 / 1 重要 / 2 必要),在 QReCC 上众包标注,每条由五个工人标。关键结果是:**实体的显著性会随会话轮次演变,相邻两轮之间平均变化 0.36 ± 0.21**,论文的说法是"an entity might become more or less essential as the focus of the conversation changes"。

**这是已发表的、经人工标注验证的证据,说明实体在会话中的角色随结构演进而变化**——正好是"分支结构 × 实体图"这个想法的实证基础。

---

## 五、可解释性在会话检索里的真实状况

### 5.1 这个领域比想象中小得多

有三个可核实的事实。第一,Mo 等人的《A Survey of Conversational Search》里"对话式稠密检索的可解释性"整节**只引了三篇**:LeCoRE、EXPLAIGNN、ConvInv。第二,前面提到的可解释信息检索综述里"conversational"零命中。第三,OpenAlex 上 ConvInv 被引 8 次、LeCoRE 16 次。

### 5.2 做得最扎实的可解释性分析,作者自己都没当成可解释性论文

**ZeCo**(SIGIR 2022)的分析部分是这个领域里方法学最扎实的。他们量化测量了:当会话历史被前置后,每个词的嵌入移动了多远。**移动最大的恰好是指代词**——"they" 移动 0.501,"it" 移动 0.480,"that" 移动 0.440,而宏平均只有 0.185。然后他们问了更尖锐的问题:指代词是**朝正确的先行词**移动,还是随便移动?上下文化之后,指代词与其正确解析词的相似度是 0.372,而与同一会话中随机词的相似度是 0.204。最后他们把这个位移与检索质量关联,报告皮尔逊相关 **R = 0.31,p = 0.005**。

**这是唯一一篇把"模型内部解了指代"变成可测量量、带随机对照、并与端任务效果挂钩的工作**,而且它几乎没被当成可解释性论文引用过。

### 5.3 一个动摇"用人类直觉解释检索器"这个前提的结果

《Learning to Relate to Previous Turns in Conversational Search》(KDD 2023)里藏着一个实质上的**忠实性结论**。他们用"把某个历史查询拼接到当前查询能否提升检索"来自动生成相关性标签,然后把这套自动标签与 TopiOCQA 的**人工话题切换标注**做对比。

重合度是:话题切换 **40.48%**、话题回归 **64.29%**、无切换 **24.25%**。作者自己的结论是"**human-annotated relevant queries are not necessarily the best ones for expansions**",因为检索导出的标签训出来的效果更好。

**这条对教授那个想法是必须正视的警示。** 它说明:人类对会话结构的描述(哪些轮次相关、哪里发生了话题切换)与真正驱动检索的东西,吻合度只有四分之一到三分之二。**任何基于人类话语直觉——包括本体——去解释会话检索器的方案,起点就和数据不符。** 这不否定这个方向,但意味着**本体那一层必须用检索信号去校准,不能只靠语义上的合理性**。

配套的一个数据点来自 HAConvDR(Findings of ACL 2024):**相关的历史轮次最多只占全部历史轮次的 20%**,而且这个比例随对话深度呈"先降后平"的模式。

### 5.4 最大的方法论空白:没有多轮的忠实性指标

所有可信的忠实性指标——ERASER 的 comprehensiveness 和 sufficiency、思维链的 early-answering 与 mistake-injection——**都假设单轮的静态输入**。ERASER 从一篇文档里删词,Lanham 截断一条推理链,Turpin 污染一个提示。**没有任何一篇定义了"擦除第 t−2 轮"意味着什么,或者如何对会话历史做反事实编辑。**

而讽刺的是:**会话检索圈天天在算 leave-one-turn-out 的检索增益**(上面那两篇都用它做训练信号),**只是从没把它当成忠实性指标发表过**。把它翻过来定义"轮次级的 comprehensiveness 与 sufficiency",再拿去评测现有那些自称可解释的系统——这是一篇不存在的论文,而且用现有设施就能做。

### 5.5 一个残酷的警告:解释未必有用,甚至有害

这一点必须放进任何"做可解释性"的规划里,因为邻近领域最严谨的几个用户实验结论都是负面的。

**Łajewska 等人**(SIGIR 2024)在会话信息获取场景做了 160 人的众包实验(用的正是 TREC CAsT 的查询),交叉了解释质量、呈现方式和回复质量。结果:劣质解释会**显著拉低对回复本身几乎所有维度的评分**;解释质量显著影响感知有用性,而**真实的回复质量只在"感知正确性"这一项上达到显著**。最扎心的是:**用户给"完全不提供解释"的回复打了最高的有用性分。**

**Kim 等人**(CHI 2025,预注册,N=308)发现:**解释会同时增加用户对错误回复的依赖**;只有**给出来源**和**暴露解释内部的不一致**才能选择性地降低误信。也就是说,流畅的解释默认是**说服工具**,只有当它暴露出可核查的东西时才变成**校准工具**。

**Seymour 与 Such**(n=1,314)发现:给语音助手用户解释可信度,**反而加重了他们的信任顾虑**。

**Bansal 等人**(CHI 2021)的结论更直白:"explanations increased the chance that humans will accept the AI's recommendation, **regardless of its correctness**"。

**真正闭环的只有三篇**——即证明了"用户检视并修改系统对自己的建模之后,拿到的结果确实变好":Balog 等人(SIGIR 2019,122 名用户,自然语言偏好陈述作为用户模型)、Mysore 等人(SIGIR 2023,LACE,20 名研究者编辑概念瓶颈,NDCG 和 MRR 提升 14% 到 24%)、Schott 等人(MuC 2024,N=135,把回复关联回用户此前的话语,显著提升感知透明度)。

---

## 六、对教授那个想法的评估

### 6.1 真正新的部分(据核实无直接竞品)

**第一,把 CAsT 2022 的分支树当作可解释性装置。** 数据里有 `parent` 字段,CAsT 自己只用它做路径级评测,ZeQR 只把它当作难度来源抱怨过一句,**没有任何人把"这次改写继承了树上哪个祖先节点的什么内容"当成解释来呈现或评测**。

**第二,中等程度的本体化。** 现有工作要么完全不本体化(CONVINSE 的四槽是表层字符串),要么完全绑死 Wikidata(所有知识库问答的语义解析)。**用受控词表、schema.org、FrameNet 这类"轻本体"来描述会话信息需求,这一档没人做。**

**第三,对改写本身做可解释性评测。** EXPLAIGNN 评的是答案的解释,ConvInv 评的是反演文本的保真度和可读性,**没有任何人评过"这次改写的理由是否正确、是否忠实"**。

### 6.2 已经存在、必须在相关工作里处理的

CONVINSE 的四槽表示加会话流图,已经把"结构化会话需求表示 + 用会话结构解释"这个想法说出来了(虽然只是示意图、没评测)。LogiCGR(WWW 2026)已经占据了"可解释会话搜索"这个 framing。ZeQR 已经证明"显式分解成语言学子问题 → 可归因的错误分析"这条路走得通。CRDR 的三类标注和编辑矩阵已经是成熟的可读编辑表示。KNUDGE 已经做了"对话树 + 本体约束"(在生成侧)。交互式路径推理(KDD 2020)在**推荐**侧已经把"知识图谱路径即解释"做完了——**推荐侧成熟而搜索侧几乎为零,这个不对称本身就是引言的好素材。**

### 6.3 我建议的定位

**不要把卖点放在"用本体做改写效果更好"**——几乎肯定打不过大模型改写,而且第 5.3 节那个 40%/64%/24% 的结果说明人类的结构直觉本来就和检索需求不完全对齐。

**把卖点放在:会话树 + 结构化需求表示 = 可归因、可诊断、可审计的改写。** 并且——这是最关键的——**真的把可解释性评了**。这个领域的现状是"人人宣称、无人评测",谁先严肃评测一次谁就立住了。

评测规格可以照抄两个现成的:EXPLAIGNN 的人评规格(实例数、Master 工人、蜜罐题、剔除混淆样本、报告"确定条件下的正确率"),或者 ConvInv 的规格(保真度 + 多维人评)。ConvInv 自己有一个诚实的方法学观察值得记住:**自动相似度分数与人评"持续一致",但当人评从 CAsT-19 到 CAsT-21 下滑时相似度分数却保持稳健——说明廉价的自动代理不足以替代人评。**

### 6.4 三条可能的路线,按风险排序

**路线一(最低风险,最快出结果):去评测别人早已免费提供的解释。** QuReTeC 输出的历史词集、CRDR 输出的编辑脚本,都是现成的 rationale,而且 QuReTeC 甚至已经有内在的 F1。做两件事:一是问人类这些理由对不对;二是定义**轮次级的 comprehensiveness 与 sufficiency**(把会话检索圈已经在算的 leave-one-turn-out 增益翻过来当忠实性指标),测这些 rationale 是否忠实。这条路不需要新数据、不需要新模型,而且填的是第 5.4 节那个明确的方法论空白。

**路线二(中等风险,和我们现有资产结合最紧):审计 iKAT 的画像归因。** iKAT 有一个 PTKB statement ranking 子任务,系统必须声明"我用了画像里的哪几条",而且这个声明**是被 nDCG@3 等指标打分的**,有 NIST 的 gold 标注。**功能上这就是一个带 ground truth 的归因机制,但从来没人当它是可解释性来做。** 没有人问过的问题是:系统声明用了这几条画像,**这几条是不是真正改变了排序的那几条**?考虑到已有研究证明大模型选 PTKB 极不稳定,这个怀疑有根据;而 EXtrA-ShaRC(UMAP 2024)已经提供了方法论——它扩展 ShARC 并**发布了反事实用户画像**,专门测"改画像,输出是否随之正确改变"。**数据我们有、指标现成、方法论有先例。**

**路线一点五(风险低,而且是引文遍历新挖出来的)**:**把 CAsT 的 TBCCG 和停止规则接起来**。第三点六节已经论证了这个空位——CAsT 定义了 P(c|t) 却把 P_r 固定成 1,而 Fu 等人(ICTIR 2024)独立地建好了从数据估计这个量所需的停止规则机制,那篇只有 1 次引用,两者从未相遇。做法是用他们验证过的特征(交换的词数、名词数、**用户输出的唯一名词数**等)在会话数据上拟合 P(c|t),再重算 CAsT-22 的 TBCCG,看系统排名是否改变。往上还可以接 C/W/L 框架,把"**期望对话深度**"作为一等公民指标导出来。

这条路还有一个额外好处:**它天然连着分支**。用户在某轮之后是继续、停止、还是**换一条分支走**,本质上是同一个决策的三种结果,而 CAsT 只建模了前两种。Fu、Lipani、Kando 的 ICTIR 2025 工作已经在 **iKAT 2023** 上做了用户动作标注和下一步动作预测,并**公开了标注数据**——那正是"决定走哪条分支"所需要的机器。

**路线三(最高风险,但最接近教授的原意):分支树 + 轻本体 → 可归因的改写。** 用 CAsT-22 的 `parent` 链把"当前轮的改写从树上哪个祖先继承了什么"显式化,用轻本体(而不是 Wikidata 绑定)描述继承的内容类型,然后用路径级指标(CCG/CPS/TBCCG)和人评双重评估。风险有三:CAsT-22 只有 18 棵树 205 个轮次,统计功效很成问题(我们在 44 个查询上已经吃过这个亏);澄清回答的轮次没有 qrels;以及第 5.3 节那个"人类结构直觉与检索需求不吻合"的根本性挑战。

---

## 七、引用清单(全部经过核实)

**分支数据与分支研究**
- Owoicho, Dalton, Aliannejadi, Azzopardi, Trippas, Vakulenko. *TREC CAsT 2022: Going Beyond User Ask and System Retrieve with Initiative and Response Generation*. TREC 2022. https://trec.nist.gov/pubs/trec31/papers/Overview_cast.pdf
- Meital, Rokach, Vainshtein, Grinberg. *The Branch Not Taken: Predicting Branching in Online Conversations*. arXiv:2404.13613
- Saeidi 等. *Interpretation of Natural Language Rules in Conversational Machine Reading* (ShARC). EMNLP 2018. arXiv:1809.01494
- Dou, Forbes, Holtzman, Choi. *MultiTalk: A Highly-Branching Dialog Testbed*. AAAI 2021. arXiv:2102.01263
- Tuan 等. *CausalDialogue: Modeling Utterance-level Causality in Conversations*. Findings of ACL 2023. arXiv:2212.10515
- Raghu, Agarwal, Joshi, Mausam. *End-to-End Learning of Flowchart Grounded Task-Oriented Dialogs* (FloDial). EMNLP 2021. arXiv:2109.07263
- Väth, Vanderlyn, Vu. *Conversational Tree Search: A New Hybrid Dialog Task*. EACL 2023. arXiv:2303.10227
- Köpf 等. *OpenAssistant Conversations*. NeurIPS 2023 D&B. arXiv:2304.07327
- Kirk 等. *PRISM*. NeurIPS 2024 D&B. arXiv:2404.16019
- Weir 等. *Ontologically Faithful Generation of Non-Player Character Dialogues* (KNUDGE). EMNLP 2024
- Pan 等. *MACHIAVELLI*. ICML 2023. arXiv:2304.03279
- Zhang, Culbertson, Paritosh. *Characterizing Online Discussion Using Coarse Discourse Sequences*. ICWSM 2017
- Kummerfeld 等. *A Large-Scale Corpus for Conversation Disentanglement*. ACL 2019

**结构化会话需求表示与可解释性**
- Christmann, Saha Roy, Weikum. *Conversational Question Answering on Heterogeneous Sources* (CONVINSE). SIGIR 2022. arXiv:2204.11677
- Christmann, Saha Roy, Weikum. *Explainable Conversational Question Answering over Heterogeneous Sources via Iterative GNNs* (EXPLAIGNN). SIGIR 2023. arXiv:2305.01548
- Christmann, Weikum. *ReQAP*. Findings of ACL 2025. arXiv:2505.11900
- Yang, Zhang, Fang. *Zero-shot Query Reformulation for Conversational Search* (ZeQR). ICTIR 2023. arXiv:2307.09384
- Mao 等. *Learning Denoised and Interpretable Session Representation* (LeCoRE). WWW 2023. DOI 10.1145/3543507.3583265
- Cheng, Mao, Dou. *Interpreting Conversational Dense Retrieval by Rewriting-Enhanced Inversion* (ConvInv). ACL 2024. arXiv:2402.12774
- Liu 等. *Facilitating Generative Retrieval with Logical Denoising for Interpretable Conversational Search* (LogiCGR). WWW 2026. DOI 10.1145/3774904.3792544
- Voskarides 等. *Query Resolution for Conversational Search with Limited Supervision* (QuReTeC). SIGIR 2020. arXiv:2005.11723
- Qian, Dou. *Explicit Query Rewriting for Conversational Dense Retrieval* (CRDR). EMNLP 2022
- Liu 等. *Incomplete Utterance Rewriting as Semantic Segmentation*. EMNLP 2020. arXiv:2009.13166
- Krasakis, Yates, Kanoulas. *Zero-shot Query Contextualization for Conversational Search* (ZeCo). SIGIR 2022. arXiv:2204.10613

**会话结构与检索的关系**
- Mo 等. *Learning to Relate to Previous Turns in Conversational Search*. KDD 2023. arXiv:2306.02553
- Mo 等. *History-Aware Conversational Dense Retrieval* (HAConvDR). Findings of ACL 2024. arXiv:2401.16659
- Sekulić, Balog, Crestani. *Towards Self-Contained Answers: Entity-Based Answer Rewriting*. CHIIR 2024. arXiv:2403.01747
- Joko, Hasibi. *Personal Entity, Concept, and Named Entity Linking in Conversations*. CIKM 2022. arXiv:2206.07836
- Ramos, Lipani. *EXtrA-ShaRC: Explainable and Scrutable Reading Comprehension for Conversational Systems*. UMAP 2024. DOI 10.1145/3627043.3659546

**可解释性的评测与警示**
- Łajewska, Spina, Trippas, Balog. *Explainability for Transparent Conversational Information-Seeking*. SIGIR 2024. arXiv:2405.03303
- Kim 等. *Fostering Appropriate Reliance on Large Language Models*. CHI 2025. arXiv:2502.08554
- Seymour, Such. *Ignorance is Bliss? The Effect of Explanations on Perceptions of Voice Assistants*. CSCW 2023. arXiv:2211.12900
- Bansal 等. *Does the Whole Exceed its Parts?*. CHI 2021. DOI 10.1145/3411764.3445717
- Balog, Radlinski, Arakelyan. *Transparent, Scrutable and Explainable User Models*. SIGIR 2019
- Mysore, Jasim, McCallum, Zamani. *Editable User Profiles for Controllable Text Recommendations* (LACE). SIGIR 2023. arXiv:2304.04250
- Jacovi, Goldberg. *Towards Faithfully Interpretable NLP Systems*. ACL 2020
- DeYoung 等. *ERASER*. ACL 2020
- Nauta 等. *From Anecdotal Evidence to Quantitative Evaluation Methods*. ACM CSUR 2023
- Anand 等. *Explainable Information Retrieval: A Survey*. arXiv:2211.02405
- Polyakov, Scells, Eickhoff. *Understanding Wacky Weights*. SIGIR 2026. arXiv:2605.19628
- Lupart 等. *Investigating LLM Variability in Personalized Conversational Information Retrieval*. SIGIR-AP 2025. arXiv:2510.03795

**路径评价、停止行为与用户模型**
- Lipani, Carterette, Yilmaz. *How Am I Doing?: Evaluating Conversational Search Systems Offline*. ACM TOIS 2021. DOI 10.1145/3451160
- Azzopardi, Thomas, Moffat. *cwl_eval: An Evaluation Tool for Information Retrieval* (C/W/L 框架). SIGIR 2019. DOI 10.1145/3331184.3331398
- Fu, Pérez-Ortiz, Lipani. *An Analysis of Stopping Strategies in Conversational Search Systems*. ICTIR 2024. DOI 10.1145/3664190.3672524
- Fu, Lipani, Kando. *Modelling and Predicting User Actions in Conversational Information Retrieval*. ICTIR 2025. DOI 10.1145/3731120.3744622(数据 github.com/RichardFu123/model_predict_user_actions)
- Fu, Lipani. *Priming and Actions: An Analysis in Conversational Search Systems*. SIGIR 2023. DOI 10.1145/3539618.3592041
- Fu, Yilmaz, Lipani. *Evaluating the Cranfield Paradigm for Conversational Search Systems*. ICTIR 2022. DOI 10.1145/3539813.3545126
- Maxwell, Azzopardi, Järvelin, Keskustalo. *Searching and Stopping*. CIKM 2015. DOI 10.1145/2806416.2806476
- Sakai. *SWAN: A Generic Framework for Auditing Textual Conversational Systems*. arXiv:2305.08290
- Li, Gao, Goenka, Chen. *Ditch the Gold Standard: Re-evaluating Conversational Question Answering*. ACL 2022. arXiv:2112.08812
- Abbasiantaeb, Meng, Azzopardi, Aliannejadi. *Improving the Reusability of Conversational Search Test Collections*. ECIR 2025. arXiv:2503.09899
- Vakulenko 等. *QRFA: A Data-Driven Model of Information-Seeking Dialogues*. ECIR 2019. arXiv:1812.10720
- Vakulenko, Kanoulas, de Rijke. *A Large-scale Analysis of Mixed Initiative in Information-Seeking Dialogues*. ACM TOIS 2021. arXiv:2104.07096
- Kim 等. *Tree of Clarifications*. EMNLP 2023. arXiv:2310.14696
- Kruff 等. *Sim4IA-Bench*. arXiv:2511.09329
- Owoicho 等. *ConvSim*. SIGIR 2023. arXiv:2304.13874

**澄清问题与侧面**
- Aliannejadi, Zamani, Crestani, Croft. *Asking Clarifying Questions in Open-Domain Information-Seeking Conversations* (Qulac). SIGIR 2019. arXiv:1907.06554
- Aliannejadi 等. *Building and Evaluating Open-Domain Dialogue Corpora with Clarifying Questions* (ClariQ). EMNLP 2021. arXiv:2109.05794
- Abbasiantaeb 等. *Conversational Gold: Evaluating Personalized Conversational Search Using Gold Nuggets*. arXiv:2503.09902

---

## 八、未核实项(引用前请自行确认)

- **LogiCGR**(WWW 2026)的正文:ACM 数字图书馆拒绝抓取,方法细节与"如何评测可解释性"均未核实,仅摘要与 GitHub 仓库经确认。
- **OntoLLM**(Expert Systems with Applications 2026):ScienceDirect 返回 403,方法、数据、代码、是否评测可解释性全部未核实。
- **EXtrA-ShaRC** 是否包含人工评测:摘要未提及,ACM 页面无法抓取。
- **KNUDGE** 的数据发布地址:ACL Anthology 页面有软件附件但无显式数据链接。
- **GraphWOZ** 的发表场次与年份存在歧义。
- 引文图谱遍历那一路 agent 在本报告完成时仍在运行,其结果未纳入;如有新发现会另行补充。
