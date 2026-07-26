# Handoff: per-epoch full-corpus 选模功能(perso_dense_val 上批量评一个 run 的所有 epoch + 选最优)

给 TREC_iKAT_2024 session。把现在手动拼的"批量评所有 epoch ckpt、选全库最优 epoch"正式化成一个命令。

## 1. 为什么要这个(背景)
- continual_ir 训 graded personalized dense retriever(query encoder, query-only, 冻结 base-qwen3 doc),**存所有 epoch ckpt**。
- 选最优 epoch **必须用 full-corpus retrieval**(perso_dense_val, 44 held-out turn, 116M ClueWeb)。**绝不能用 pool-rerank proxy**(只重排该 turn ~177 judged docs):已实测 proxy 和全库**反相关**,会选到 query 旋转崩塌的 epoch(全库 NDCG@3 从 ep2 的 11 掉到 ep4 的 1)。
- continual_ir 的 in-training eval(`eval_conv_search`)对 TopiOCQA/QReCC 是 full-corpus,但靠把 corpus(5M docs)常驻训练进程内存。**ClueWeb 116M/450GB 塞不进训练进程** → ikat 必须 offline 批量评。逻辑同 in-training 选 epoch,只是改成训完批量做。

## 2. 现状(我已 ad-hoc 实现的零件,都在 TREC_iKAT_2024)
- **`qwen3_e*` alias 机制**(已加): `dense_search.py:299` / `search.py:604` 接受 `retrieval_model.startswith("qwen3_e")`,都 load `QwenEmbedding(dense_query_encoder_path)`; `evaluation.py` 的 `--retrieval_model` choices 加了 `[f"qwen3_e{i}" for i in range(600)]`。file_name_stem 含 retrieval_model 名 → 每个 alias 出 distinct 文件。**效果:一个 `run_experiments_grouped` invocation(一趟 corpus stream)能评任意多 ckpt**(每个 ckpt 一个 alias,共享 corpus 流)。
- **perso_dense_val 钩子**(已加): `--topics perso_dense_val` + `--retrieval_query_type perso_dense_val_{ptkb,rel_ptkb}` 进了 argparse choices; `evaluation_util.get_query_list` + `evaluation.py topic_name_map` 已有 perso_dense_val 分支; config = `src/apcir/evaluate/fuse_then_eval_config_perso_dense_val_octal40.yaml`(octal40 索引路径已 remap)。
- **但 config 是手动生成的**: 我每次用 python 脚本 glob ckpt → 写 iterate `retrieval_model:[qwen3_e0..eN]` + `param_mapping` 每个 alias→ckpt path → 存 `alias_map.json` 解码 → 跑 → 手动解析 metrics 选最优。见 `continual_ir/sweep_ikat/perso_eval_ALLEPOCHS.yaml` + `alias_map_allepochs.json` 作样例。

## 3. 要正式化的功能
一个命令(建议 `python -m apcir.evaluate.eval_run_epochs`),自动跑通 §2 的手动流程:
- **输入**: 一个或多个 continual_ir run 目录(`/part/01/Tmp/yuchen/continual_ir/<run>/` 含 `checkpoint-step-*` 所有 epoch); 可选 `--refs base conv`(锚点); perso_dense_val config。
- **自动**:
  1. glob 每个 run 的所有 `checkpoint-step-*`(+ base/conv refs)。
  2. 生成 alias config(`qwen3_e{i}`→ckpt path)+ alias_map(`qwen3_e{i}`→`<run>@<step>`)。
  3. **一趟** `run_experiments_grouped --merge topk`(corpus load 一次,串行过所有 ckpt 的 query-encode + 全库检索)。
  4. 解析每 (ckpt, form) 的 full-corpus `ndcg_cut_3`(指标文件 = `results/ClueWeb_ikat/perso_dense_val/metrics/S1[perso_dense_val_{form}]-...-[qwen3_eN]-...json`; 该文件含 "Print this line" header + **两个** JSON 块,取第一个的 ndcg_cut_3)。
- **输出**:
  - 每 run 的 per-epoch full-corpus NDCG@3(rel_ptkb + ptkb)表/曲线 + best `(epoch, NDCG@3)`。
  - 锚点校验行: base/conv 必须复现 rel=13.4/26.0(否则 eval 有问题,别信)。
  - 可选: 把每 run 的 best-epoch ckpt symlink 到 `<run>/best_fullcorpus/`。

## 4. 坑 / 关键设计点
- **一趟评多 ckpt 靠 alias 共享 corpus 流**(已实现)。`partition_by_corpus` 必须是 1 个 group(所有 alias 同 `dense_index_dir_path`)。
- **file-name collision**: 跨**不同次** invocation 复用同一个 retrieval_model 名(如两次都用 `conv-qwen3`)会**覆盖**上一次的输出文件。正式化后用唯一 `qwen3_e{i}`,且**单次 invocation 评完所有**,避免跨次复用名字。
- **epoch 推断**: alias_map 的 `<run>@<step>`; epoch = step 在该 run 所有 step 排序里的位次(各 run steps/epoch 可能不同,别写死 /94 或 /17)。
- **锚点必加**: base(Qwen3-Embedding-0.6B 原始)+ conv(instruct3fp32_qwen_nosched/checkpoint-step-94)作 e0/e1,复现 13.4/26.0 才证明 pipeline 没坏。
- **corpus 每次重载(进阶)**: 现在每个 invocation 从 /part 重读 450GB(~140s)。如果要给**多个 run / 多轮迭代**评,值得搭 **corpus 常驻 server**: 450GB load 进 RAM(octal40 有 491GB 空),holder 进程常驻 + eval 走 client 连它(按需 block→GPU faiss),省每次重读。符合"heavy 资源只 load 一次"。这是单独一块,可后做。

## 5. 我这边(continual_ir)对应的训练侧
训练改在 continual_ir(不在你这):`train_qwen_cl.py` 存所有 epoch(去掉 `--ikat_save_best_only`/`--activate_eval_ikat_graded_while_training`),新加的 anti-collapse(anchor 正则 `--ikat_anchor_weight/--ikat_anchor_emb_file`、全语料负例 `--ikat_global_neg_file/--ikat_global_neg_k`)也在 continual_ir。**你只管 eval 侧**(批量评 + 选 epoch + 可选 corpus server)。

样例数据: `continual_ir/sweep_ikat/{perso_eval_ALLEPOCHS.yaml, alias_map_allepochs.json, perso_eval_134.yaml, alias_map_134.json}`。
