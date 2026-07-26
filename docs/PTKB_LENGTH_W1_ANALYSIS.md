# W1:PTKB 长度 × QR 质量 × 检索器 —— 现有结果分析(2026-07-18)

> 假设:"profile 越长,LLM 改写(QR)越差,dense 直连更稳"。
> 数据:现有 per_query_metrics,只算被评 turn(23/24/25 = 176/103/45,配对零丢弃);
> sanity:复算均值与官方 metrics 文件一致(BM25-23 nDCG@3 = 0.1691 ✓)。
> QR = gpt-4o_MQ4CS_persq_rw(BM25/SPLADE;qwen 腿用其 ad-hoc 封装),ANCE 用
> gpt-4o_rar_personalized_cot1_rw(仅有,flavor 不同);direct = qwen_conversation_ptkb
> (base qwen3 与 conv-qwen3-fp32 两个编码器)。指标 nDCG@3。

## 结论(现有数据不支持假设,两个方向都不支持)

1. **QR 不随 profile 变长而衰减**:年内 Spearman(得分 vs PTKB 条数)全部不显著
   (|ρ|<0.09);23 年长 profile 组的 QR 得分反而普遍更高(BM25 +0.030),24 年混合且
   幅度小(BM25 −0.018 / SPLADE +0.041 / ANCE −0.034,均 ns)。gpt-4o 完全扛得住 22 条画像。
2. **dense-direct 全面落后 QR**:conv-qwen3-fp32 直连 0.13(23)/0.25(24),
   低于 QR+任意检索器(0.15–0.43);base qwen3 直连更低。配对 Δ(direct−QR)
   均值 −0.06 ~ −0.25,全年全腿为负。
3. **24 年方向反转警报**:direct 腿随 profile 变长掉得比 QR 腿更狠(qwen-direct
   长短差 −0.064 vs SPLADE-QR +0.041;Δ(direct−SPLADE) 随长度更负,ρ=−0.176, p=0.075)
   ——与假设方向相反。
4. **预注册 W2 触发判定:未触发**(假设方向上无任何 [组间差≥0.02 且 p<0.1] 的格子;
   两年方向也不一致)。唯一接近显著的信号(24 年 SPLADE)方向相反。

## 关键表(完整输出见 w1_out.txt)

年内中位二分(nDCG@3,short→long):
| 腿 | 23 短 | 23 长 | 24 短 | 24 长 |
|---|---|---|---|---|
| BM25(QR) | .155 | .185 | .221 | .203 |
| SPLADE(QR) | .224 | .228 | .385 | .426 |
| ANCE(QR*) | .186 | .203 | .311 | .276 |
| qwen(QR) | .210 | .222 | .331 | .326 |
| convq-direct | .129 | .132 | .267 | .242 |
| qwen-direct | .101 | .116 | .173 | .109 |

## Caveats(为什么这不是假设的死刑)

- **改写器是 gpt-4o(最强档)**:假设真正的适用域可能是**弱改写器**(本地 qwen32b、
  gpt-5-mini minimal)——"弱改写器在长 profile 下崩、强改写器不崩"本身就是可写的结论。
  W2(统一本地 qwen 改写)恰好能回答这个问题,且顺带补齐 25 年空白。
- ANCE 的 QR flavor 不同(rar_personalized);25 年 sparse/ance 无结果;
- 离线没有 PCDR(graded 微调)的 direct 结果,direct 腿以 conv-qwen3-fp32 为上限估计;
- 跨年四档趋势表(输出 D)受年份混杂,只作描述。
- 与 attention 分析的口径差异:attention 说的是"dense 直连会挑着读",本分析说的是
  "直连的绝对检索质量仍不敌 QR+检索器"——两者不矛盾,合起来是:**dense 直连的
  个性化读取机制存在,但离线指标上还打不过强改写器路线**。

## 建议

W2 按预注册标准不触发;但作为"弱改写器压力测试 + 25 年补全"仍有独立价值,
预算:本地 vllm 免费改写 ~738 turns(<1h)+ 12 个检索 run(BM25 快;SPLADE/ANCE
需 GPU 数小时)。是否上,等用户拍板。
