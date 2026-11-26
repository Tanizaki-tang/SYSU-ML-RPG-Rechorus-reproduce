# RPG 主实验基线与协作说明

本页对应 ReChorus 适配版 RPG 的 `rpg_main_v1` 实验。MovieLens-1M 和 Grocery 的三种子效果比较已完成。图搜索已记录排序指标和访问量，延迟、吞吐与峰值显存待单独测量。

## 数据和评估

| 数据集 | 训练交互 | 验证样本 | 测试样本 | 候选目录 |
|---|---:|---:|---:|---:|
| MovieLens-1M | 568,731 | 2,589 | 2,877 | 3,125 |
| Amazon Reviews 2014 Grocery | 121,892 | 14,681 | 14,681 | 8,713 |

- 三个模型读取同一套冻结切分。MovieLens-1M 将评分 ≥4 视为正反馈，经迭代 5-core 后按全局日历时间切分。Grocery 沿用冻结的 ReChorus 切分。
- 逐交互预测下一物品。模型接收最近 50 条历史，候选过滤使用该时点之前的完整历史，重复交互的当前目标保留。主表统一使用全量候选打分。
- 验证集全量打分的 NDCG@10 用于调参和选 checkpoint。图搜索使用种子 2026 的 checkpoint 在验证集上调参，再将同一搜索配置应用到三个种子。测试指标取种子 2026、2027、2028 的均值和样本标准差。

## 最终配置

RPG 使用 32 位语义 ID，每位 256 个编码，由 `text-embedding-3-large`、PCA 512 维和 OPQ 生成。模型隐藏维度 128，前馈维度 512，2 层、4 头。训练使用多 token 交叉熵、AdamW、梯度裁剪 1.0，以及带预热的余弦学习率调度。正式训练批量大小为 256，最多 150 轮，早停耐心为 20 轮。

| 数据集 | RPG 学习率 | 温度 | 预热步数 | 图搜索邻居数 | 搜索轮数 | beam |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | 0.001 | 0.05 | 10,000 | 100 | 3 | 10 |
| Grocery | 0.003 | 0.07 | 1,110 | 500 | 3 | 10 |

两套数据上，SASRec 采用学习率 0.001、dropout 0.0、嵌入维度 64、1 层、4 头。GRU4Rec 采用学习率 0.003、嵌入和隐状态维度 64。两个基线均使用 ReChorus 的训练损失、Adam 和每个正样本 1 个训练负样本。

调参细节见 [短训入围配置](results/rpg_main_v1/screen.json)与[图搜索配置](results/rpg_main_v1/graph.json)。已使用的环境为 Python 3.11.18、PyTorch 2.15.0+cu132、Transformers 4.56.1，GPU 为 RTX 4060 Laptop。

## 测试结果

| 数据集 | 模型 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| MovieLens-1M | SASRec | 0.09211 ± 0.00125 | 0.06125 ± 0.00055 | 0.14193 ± 0.00289 | 0.07726 ± 0.00116 |
| MovieLens-1M | GRU4Rec | 0.07566 ± 0.00532 | 0.04998 ± 0.00320 | 0.12571 ± 0.00647 | 0.06617 ± 0.00379 |
| MovieLens-1M | RPG | **0.11204** ± 0.00106 | **0.07415** ± 0.00114 | **0.16800** ± 0.00087 | **0.09218** ± 0.00082 |
| Grocery | SASRec | 0.03646 ± 0.00130 | 0.02351 ± 0.00080 | 0.05831 ± 0.00084 | 0.03052 ± 0.00064 |
| Grocery | GRU4Rec | 0.02525 ± 0.00097 | 0.01580 ± 0.00021 | 0.04412 ± 0.00164 | 0.02186 ± 0.00019 |
| Grocery | RPG | **0.04502** ± 0.00192 | **0.03120** ± 0.00105 | **0.06848** ± 0.00201 | **0.03873** ± 0.00094 |

| 数据集 | 图搜索 HR@10 | 图搜索 NDCG@10 | NDCG@10 较全量变化 | 平均访问物品数／目录占比 |
|---|---:|---:|---:|---:|
| MovieLens-1M | 0.16707 ± 0.00040 | 0.09177 ± 0.00083 | −0.00041 | 1,594／51.02% |
| Grocery | 0.06825 ± 0.00185 | 0.03859 ± 0.00087 | −0.00014 | 5,412／62.11% |

[机器可读测试汇总](results/rpg_main_v1/test.json)

## 复用语义 ID 和 checkpoint

1. 从共同的基线 commit 开分支。在仓库根目录运行 `python scripts/freeze_datasets.py`，按原始来源生成两套冻结切分。
2. 从组内共享盘取得 `rpg-team-handoff.zip`，解压到仓库根目录。包内有两套数据的 16/32 位语义 ID、对应 JSON 和各 3 个正式 RPG checkpoint。已有语义 ID 可以直接用于训练和评估。
3. 将本地冻结 manifest 的 `files` 字段与包内 `handoff/manifests/` 的对应 manifest 对照。将语义 ID 和 checkpoint 的 SHA-256 与 `handoff/baseline_assets.json` 对照。语义 ID 第 `i` 行对应冻结物品 ID `i`，第 0 行用于 padding。

公共 GitHub 仓库用于代码、复现说明和汇总结果。交接包通过组内共享盘传递。MovieLens 原始数据由队友从官方来源下载。参考 [MovieLens-1M 数据使用条款](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)。

在激活实验环境后，可以用以下 PowerShell 7 命令重跑种子 2026 的完整验证集。将 `$mode` 改为 `graph` 可复现基线图搜索。将 `$dataset` 改为 `Grocery_and_Gourmet_Food` 可切换数据集。

```powershell
$dataset = 'MovieLens_1M'
$mode = 'full'
$seed = '2026'
$assets = Get-Content -Raw -LiteralPath handoff/baseline_assets.json | ConvertFrom-Json -AsHashtable
$asset = $assets[$dataset]
$graph = $asset.graph_config

python -B src/run_unified.py --model_name RPG --gpu 0 --path data/frozen_v1 `
  --dataset $dataset --sid_file semantic_ids_m32.npy --history_max 50 `
  --emb_size 128 --ffn_size 512 --n_layers 2 --n_heads 4 `
  --temperature $asset.training_config.temperature --random_seed $seed `
  --train 0 --load 1 --model_path $asset.checkpoints[$seed].file `
  --eval_phase dev --eval_limit 0 --decode $mode --eval_batch_size 32 `
  --item_chunk 2048 --topk '5,10' --metric 'NDCG,HR' --main_metric NDCG@10 `
  --graph_beams $graph.beams --graph_edges $graph.edges --graph_steps $graph.steps `
  --decode_seed $graph.decode_seed --log_file "results/handoff-check/${dataset}_${mode}_s$seed.log"
```

种子 2026 的验证 NDCG@10 参考值为 MovieLens-1M 全量 0.07963、图搜索 0.07989，Grocery 全量 0.05362、图搜索 0.05339。

## RPG + 有方向转移重排

RPG 用既有 checkpoint 对全量物品打分，应用相同的历史过滤并保留前 100 个候选。训练集内，将每位用户的交互按时间戳和行号排序，统计同一用户相邻物品对的有向计数 $c(h,i)$。设 $p(i)$ 为训练集加一平滑后的物品频率，对历史物品 $h$ 和候选 $i$ 计算

$$
b(h,i)=\frac{c(h,i)}{c(h,i)+5}\log\left(1+\frac{c(h,i)}{50p(i)}\right).
$$

取请求时最近 5 个历史物品，按 $0.8^j$ 加权平均各自的转移分，$j=0$ 对应最近一次交互。将行为分和 RPG 分数分别在当前候选内做 z-score 标准化，再以 $z_{\mathrm{RPG}}+\lambda z_{\mathrm{behavior}}$ 重排。热门度对照使用 $\log p(i)$ 作为辅助分。三种方法共用候选集及物品过滤规则。

按用户划分验证校准组和留出组。种子 2026 在校准组以 NDCG@10 选择融合权重，随后用种子 2027 和 2028 的 checkpoint 在同一验证留出组复核。正式测试使用冻结的权重和种子 2026、2027、2028。

| 数据集 | 候选数 | 行为历史长度 | 有方向转移权重 | 热门度权重 |
|---|---:|---:|---:|---:|
| MovieLens-1M | 100 | 5 | 1.0 | 0.05 |
| Grocery | 100 | 5 | 0.75 | 0.75 |

[冻结配置](results/rpg_behavior_rerank_config.json) · [种子 2026 校准与消融](results/rpg_behavior_rerank_pilot.json) · [固定权重的三种子验证](results/rpg_behavior_rerank_locked_dev.json)

### 正式测试结果

表中为三种子的均值 ± 样本标准差。

| 数据集 | 方法 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| MovieLens-1M | 原 RPG | 0.11204 ± 0.00106 | 0.07415 ± 0.00114 | 0.16800 ± 0.00087 | 0.09218 ± 0.00082 |
| MovieLens-1M | RPG + 热门度 | 0.11308 ± 0.00100 | 0.07497 ± 0.00109 | 0.16916 ± 0.00178 | 0.09301 ± 0.00043 |
| MovieLens-1M | RPG + 有方向转移 | **0.11470 ± 0.00159** | **0.07630 ± 0.00085** | **0.17426 ± 0.00323** | **0.09543 ± 0.00059** |
| Grocery | 原 RPG | 0.04502 ± 0.00192 | 0.03120 ± 0.00105 | 0.06848 ± 0.00201 | 0.03873 ± 0.00094 |
| Grocery | RPG + 热门度 | 0.05551 ± 0.00106 | 0.03747 ± 0.00094 | 0.08703 ± 0.00063 | 0.04758 ± 0.00046 |
| Grocery | RPG + 有方向转移 | **0.06555 ± 0.00102** | **0.04556 ± 0.00053** | **0.09261 ± 0.00144** | **0.05431 ± 0.00067** |

有方向转移重排的 NDCG@10 相对原 RPG 在 MovieLens-1M 上提高 3.52%，在 Grocery 上提高 40.25%。与热门度重排配对比较时，Grocery 的三个种子各自按用户做 bootstrap 得到的 95% 区间均高于 0，MovieLens-1M 的三个区间均跨过 0。

在 Grocery 上，真实目标存在训练转移证据的样本从 0.15531 提高到 0.25349，缺少训练转移证据的样本从 0.00961 降到 0.00456。该分组用于结果诊断，重排只使用历史、候选与训练转移表。 [逐种子测试明细](results/rpg_behavior_rerank_test.json)

### 复现重排实验

准备两套冻结数据、32 位语义 ID 和交接包中的六个 RPG checkpoint。解压到仓库根目录后，运行下列命令生成一份测试报告。评估脚本读取冻结配置，校验数据、语义 ID、checkpoint 和测试样本，并将原 RPG 指标与正式基线逐项核对。

```powershell
python -B scripts/eval_rpg_behavior_rerank.py --phase test --output results/rpg_behavior_rerank_reproduced.json
```

[评估脚本](scripts/eval_rpg_behavior_rerank.py)
