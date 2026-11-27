# 方案二：替换噪声与天然残基交叉熵

## 在固定 E18 面板上比较两种时间轴

`python -m ligandchorus.recovery_benchmark` 沿用既有的训练清单、评测面板、官方冻结编码器及 BLOSUM62 文件。默认运行原五级方案。通过以下选项运行 GraDe-IF 时间轴或同预算对照。

| 参数 | 含义 |
|---|---|
| `--grade-table PATH` | 读取作者发布的 [blosum_substitute.pt](https://raw.githubusercontent.com/ykiiiiii/GraDe_IF/562674761c94c44ad75492324a4a44e8f5a945b5/dataset_src/blosum_substitute.pt)，校验 SHA-256、字母序和 BLOSUM 分数。训练时均匀抽取整数 0…500，推理按 100 跨步。 |
| `--terminal-readout sample` 或 `argmax` | 在同一末轮预测上抽样或取最高分。原五级默认抽样，GraDe-IF 时间轴默认取最高分。首四轮均为随机抽样，区域温度 0.29/0.24。 |
| `--updates N` | 正式默认 25,000 步，配对试验可显式指定相同预算；`--smoke` 固定 8 步。 |

GraDe-IF 发布的代码对温度缩放后的分数做 softmax。前向加噪以天然类别选列，反向后验使用作者的累计表与区间比值近似。项目将其字母序映射到 LigandMPNN 的类别顺序。我们保留 LigandMPNN 的配体条件、三层并行解码器、归一化时间输入和 E18 区域采样温度。每次网络调用同时预测所有有效设计位置的天然身份分布。

在 `--generation` 下保存解码头、优化器和 RNG。对同一检查点切换末轮读出时，将其训练日志及检查点复制到独立输出目录，再用相同训练参数和 `--resume` 运行。逐结构 JSONL 内的 `preterminal_sha256` 覆盖随机初态与前四轮结果，可检验两份对照只在末轮分叉。

正式协作仓库为 LigandChorus。方案一由同学负责，采用分轮揭开掩码。方案二在 `cloud` 分支开发，研究已有氨基酸能否在后续轮次得到修正。

## 训练目标

一次训练输入是氨基酸替换后得到的序列 `noisy_sequence`、结构条件和噪声阶段。监督目标始终为替换前的天然序列 `clean_sequence`。在结构有效、允许设计、天然身份属于 20 种标准氨基酸的位置上，取模型给天然类别的负对数概率，再对这些位置求平均。替换后碰巧保留原身份的位置也参与监督。

例如天然是 L，加噪后变成 I。网络输入 I，交叉熵仍查 L 的概率。如果另一个位置天然是 V，加噪后还是 V，也查 V 的概率。BLOSUM 决定输入中哪些替换更常发生。把 BLOSUM 行归一化后用作交叉熵软标签，则会改变监督目标，应作为独立对照。

## 仓库接口

调用 `model.compute_substitution_loss(features, clean_sequence, noisy_sequence, t)`。两份序列使用相同的 `[B, L]` 整数编码，时间条件使用浮点 `[B]`。沿用 E18 的 21 维头，只对前 20 个 logits 做交叉熵归一化；token 20 是未知或掩码标记。批次没有有效监督时明确报错。

方案二的时间约定拟采用前向噪声阶段编号 1 至 5，归一化为 `t = stage / 5`，1 对应最大噪声。原 E18 掩码训练的 t 表示已知概率，方向相反；复用训练过的 E18 时间嵌入时需要显式对齐语义。仓库的氨基酸顺序为 `ACDEFGHIKLMNPQRSTVWY`，加载替换矩阵时按字母重排。五阶段是本项目设想，原论文与本项目的阶段预算不同。

固定天然序列恢复基准的独立入口为 `python -m ligandchorus.recovery_benchmark`。它核对 E18 的 10,000 条训练清单和 20/22 个开发 PDB 的协议哈希，冻结官方编码器并缓存其 FP32 输出。批大小为 4，共更新 25,000 步；前四步使用由 BLOSUM62 半比特分数构建的替换概率，第五步为均匀终点。每次更新用一个噪声阶段扰动每个样本，预测天然身份。部署时先用 `--smoke` 验证 8 次更新及编码器参数完全一致，再启动正式配置。

使用 `--generation` 在固定 E18 面板执行本分支的五轮生成对照。起始状态在 20 种氨基酸上均匀采样。每一轮从当前状态预测天然身份分布，用该分布混合前向过程的单步反向后验；抽出的上一步状态整体作为下一次输入，有效位置可以重复改写。首版预先锁定 E18 的口袋内外温度 0.29/0.24，施加于天然身份 logits。参考序列用于打分，不参与反向状态更新。

正式训练每 500 步和末步原子写入 `decoder_optimizer.pt`，含解码头、优化器、随机状态、步数及编码器校验值。指定原输出目录和 `--resume` 可恢复。小批验证在第 4 步另存断点，完成第 8 步后重载检查逐轮轨迹。正式评估沿用两种子、每个种子八条候选，按结构等权汇总全局和界面恢复率及候选间归一化汉明距离，分别保存 20/22 两组。在 `--reference-summary` 指向已核验基线 JSON 时，`table2_comparison.json` 输出 22 例的四项指标及配对均值差。独立去噪诊断另存为 `recovery_by_stage.json`。本对照不比较设备耗时，也不将新的采样器温度在检查集上调优。

## 依据

- [GraDe-IF，NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/20888d00c5df685de2c09790040e0327-Abstract-Conference.html) 和[作者实现](https://github.com/ykiiiiii/GraDe_IF/blob/main/diffusion/gradeif.py)。`BlosumTransition` 定义转移，`pred_x0` 模式预测干净序列，默认 `loss_type='CE'`。
- [D3PM，Austin 等](https://arxiv.org/abs/2107.03006)。给出离散转移矩阵的扩散框架，以及变分项与辅助交叉熵的联合目标。本项目首个损失接口对应干净类别预测。
