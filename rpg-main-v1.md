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

## 创新实验怎么对照

- **改解码器**时，两边使用同一 checkpoint、语义 ID、验证样本和历史过滤。先报告验证 NDCG@10、HR@10 与平均访问物品数。如需比较速度，另记录硬件、批量大小和计时边界。图构建耗时单列。
- **改模型或训练损失**时，对照组和实验组使用同一数据、语义 ID、种子与评估流程，各自从头训练。配置通过验证集确定后统一评估测试集。
- 每次实验记录分支或 commit、改动项、对照配置、种子、结果文件和结论，方便把结果回填到课程报告。
