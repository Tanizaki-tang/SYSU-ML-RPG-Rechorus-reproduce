# ReChorus `src/models` 速查清单

> 覆盖 `src/models/` 下全部 45 个模型 `.py` 文件（不含 7 个 `__init__.py`）。
> 每个条目的「签名」均来自源码 AST 解析，方括号内为默认值；`→` 表示上一条目类的成员方法。

---

## 0. 先看命名约定

| 命名模式 | 含义 |
|---|---|
| `XxxBase` | 模型主体：负责 `parse_model_args_xxx` / `_define_init` / `_define_params_xxx` / `forward`，**不单独实例化**。 |
| `XxxCTR` | 把 `XxxBase` 与 `ContextCTRModel` 多继承，用于 **CTR 预测**（输出概率 + BCE/MSE）。 |
| `XxxTopK` | 把 `XxxBase` 与 `ContextModel`（或 `ContextSeqModel`）多继承，用于 **Top-k 排序**（BPR 等 pairwise 损失）。 |
| `XxxGeneral` / `XxxSequential` | 重排模型的两种基座：前者用 general ranker，后者用 sequential ranker。 |
| `XxxImpression` | 复用同一 `Base` 的曝光列表（list-wise）版本。 |
| `reader` / `runner` | 模型类属性，决定数据读取器与训练/评测流程；如 `ContextReader` + `CTRRunner`。 |

**三类任务的基座继承链**

```
BaseModel
├── GeneralModel              # top-k 推荐（BaseReader / BaseRunner）
│   ├── SequentialModel       # 序列推荐（SeqReader）
│   ├── CTRModel              # CTR 预测（CTRRunner）
│   └── ImpressionModel       # 曝光列表排序（ImpressionReader / ImpressionRunner）
│       └── RerankModel       # 重排（加载预训练 ranker）
└── ContextModel 等上下文子类见 §1.2
```

**按任务快速定位**

| 任务 | 模型 |
|---|---|
| 通用 Top-k | BPRMF、BUIR、CFKG、DirectAU、LightGCN、NeuMF、POP |
| 序列推荐 | Caser、Chorus、ComiRec、ContraRec、FPMC、GRU4Rec、KDA、NARM、SASRec、SLRCPlus、TiMiRec、TiSASRec |
| 上下文 CTR / Top-k | AFM、AutoInt、DCN、DCNv2、DeepFM、FM、FinalMLP、SAM、WideDeep、xDeepFM |
| 上下文 + 行为序列 | CAN、DIEN、DIN、ETA、SDIM |
| 重排 | MIR、PRM、SetRank |
| 实验性 | CLRec、FourierTA、S3Rec、SRGNN（`developing/`，接口不一定稳定） |

---

## 1. 基类文件

### 1.1 `BaseModel.py` — 训练骨架与任务基座

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `BaseModel` | `class BaseModel(nn.Module)` | 所有模型的统一骨架：定义参数、前向、损失、存取与参数统计钩子。 |
| → `parse_model_args` | `(parser)` | 注册 `--model_path`、`--buffer`。 |
| → `init_weights` | `(m)` | 对 Linear/Embedding 做均值 0、标准差 0.01 的正态初始化。 |
| → `__init__` | `(self, args, corpus)` | 保存 device、模型路径、buffer 开关、optimizer 与 `check_list`。 |
| → `_define_params` | `(self)` | 子类注册网络参数的钩子（空实现）。 |
| → `forward` | `(self, feed_dict: dict) -> dict` | 前向钩子，要求返回含 `prediction [B, n_candidates]` 的字典。 |
| → `loss` | `(self, out_dict: dict) -> Tensor` | 计算训练损失的钩子。 |
| → `customize_parameters` | `(self) -> list` | 将参数分成 weight/bias 两组，使 bias 不做 weight decay。 |
| → `save_model` / `load_model` | `(self, model_path=None)` | 保存 / 加载 `state_dict`。 |
| → `count_variables` | `(self) -> int` | 统计可训练参数总量。 |
| → `actions_after_train` | `(self)` | 训练结束后的自定义动作钩子。 |
| `BaseModel.Dataset` | `class Dataset(BaseDataset)` | 把 reader 的 DataFrame 包装成逐条产出 `feed_dict` 的数据集。 |
| → `_get_feed_dict` | `(self, index) -> dict` | 构造单条样本输入的钩子。 |
| → `collate_batch` | `(self, feed_dicts: List[dict]) -> dict` | 变长字段用 `pad_sequence` 对齐，其余转 tensor，并附 `batch_size`/`phase`。 |
| → `prepare` | `(self)` | 非训练阶段预先把所有 `feed_dict` 缓存进 `buffer_dict`。 |
| → `actions_before_epoch` | `(self)` | 每个训练 epoch 前的数据预处理钩子（如重采样负例）。 |
| `GeneralModel` | `class GeneralModel(BaseModel)` | Top-k 推荐通用基座：负采样 + 多负例 BPR 损失，参数 `num_neg`/`test_all`/`dropout`。 |
| → `loss` | `(self, out_dict: dict) -> Tensor` | 对正负例分数差做 softmax 加权的 BPR 排序损失。 |
| → `Dataset.actions_before_epoch` | `(self)` | 为每个用户采样 `num_neg` 个未交互负例。 |
| `SequentialModel` | `class SequentialModel(GeneralModel)` | 序列推荐基座：样本额外携带 `history_items`/`history_times`/`lengths`，过滤空历史。 |
| `CTRModel` | `class CTRModel(GeneralModel)` | CTR 基座：单物品输入 + `label`，支持 BCE/MSE 损失（`--loss_n`）。 |

### 1.2 `BaseContextModel.py` — 上下文特征与上下文序列基座

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `get_context_feature` | `(feed_dict, index, corpus, data)` | 把用户特征、情境特征、物品特征（单值或列表）拼进 `feed_dict`。 |
| `ContextModel` | `class ContextModel(GeneralModel)` | 带上下文特征的 Top-k 基座（`ContextReader`），损失可选 BPR 或 BCE。 |
| `ContextCTRModel` | `class ContextCTRModel(CTRModel)` | 带上下文特征的 CTR 基座（`ContextReader` + `CTRRunner`）。 |
| `ContextSeqModel` | `class ContextSeqModel(ContextModel)` | 上下文 + 行为序列的 Top-k 基座（`ContextSeqReader`），额外生成 `history_<特征>`。 |
| `ContextSeqCTRModel` | `class ContextSeqCTRModel(ContextCTRModel)` | 上下文 + 序列的 CTR 基座，过滤无历史样本。 |

### 1.3 `BaseImpressionModel.py` — 曝光列表（list-wise）基座

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `ImpressionModel` | `class ImpressionModel(GeneralModel)` | 一次对整条曝光列表打分，可分别限制训练/测试的正负例数量。 |
| → `loss` | `(self, out_dict, target=None)` | 支持 BPR（含 hard/reweight 变体）、ListNet、softmaxCE、AttentionRank 等列表级损失。 |
| `ImpressionSeqModel` | `class ImpressionSeqModel(ImpressionModel)` | 在曝光列表输入上再拼接用户行为序列的基座。 |

### 1.4 `BaseRerankerModel.py` — 重排基座

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `RerankModel` | `class RerankModel(ImpressionModel)` | 重排基座：先加载预训练 ranker 生成候选表征，再对候选列表重新打分。 |
| → `load_ranker` | `(self, args, corpus)` | 按 yaml 配置与 `.pt` 权重构造基础 ranker，并读取其 embedding 维度。 |
| `RerankSeqModel` | `class RerankSeqModel(RerankModel)` | 带用户行为序列输入的重排基座。 |

---

## 2. `context/` — 上下文特征交互模型（CTR / Top-k 双模式）

### `context/FM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `FMBase` | `class FMBase(object)` | 因子分解机：一阶偏置 + 二阶隐向量内积，是所有上下文模型的公共底座。 |
| → `_get_embeddings_FM` | `(self, feed_dict)` | 取出当前物品与上下文特征的 embedding。 |
| `FMCTR` | `class FMCTR(ContextCTRModel, FMBase)` | FM 的 CTR 版本，输出点击概率并接 BCE 损失。 |
| `FMTopK` | `class FMTopK(ContextModel, FMBase)` | FM 的 Top-k 版本，对候选物品列表打分。 |

### `context/AFM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `AFMBase` | `class AFMBase(FMBase)` | Attentional FM：给每对特征交叉学一个注意力权重再加权求和。 |
| → `build_cross` | `(self, feat_emb)` | 构造所有两两特征的逐元素交叉向量。 |
| → `afm_layer` | `(self, infeature)` | 用注意力分数池化交叉向量，得到 AFM 表征。 |
| `AFMCTR` | `class AFMCTR(ContextCTRModel, AFMBase)` | AFM 的 CTR 版本。 |
| `AFMTopK` | `class AFMTopK(ContextModel, AFMBase)` | AFM 的 Top-k 版本。 |

### `context/AutoInt.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `AutoIntBase` | `class AutoIntBase(FMBase)` | 用多头自注意力在特征 embedding 序列上自动学习高阶特征交互。 |
| `AutoIntCTR` | `class AutoIntCTR(ContextCTRModel, AutoIntBase)` | AutoInt 的 CTR 版本。 |
| `AutoIntTopK` | `class AutoIntTopK(ContextModel, AutoIntBase)` | AutoInt 的 Top-k 版本。 |

### `context/DCN.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `DCNBase` | `class DCNBase(object)` | Deep & Cross：显式 cross 网络建模有界阶特征交叉，并联 deep 分支。 |
| → `cross_net` | `(self, x_0)` | 逐层计算 `x_{l+1} = x_0 · x_lᵀ · w + b` 的交叉特征。 |
| → `l2_reg` | `(self, parameters)` | 对指定参数计算 L2 正则损失。 |
| `DCNCTR` | `class DCNCTR(ContextCTRModel, DCNBase)` | DCN 的 CTR 版本。 |
| `DCNTopK` | `class DCNTopK(ContextModel, DCNBase)` | DCN 的 Top-k 版本。 |

### `context/DCNv2.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `DCNv2Base` | `class DCNv2Base(DCNBase)` | DCN v2：用矩阵分解参数化 cross 层，并引入低秩专家混合（MoE）。 |
| → `cross_net_2` | `(self, x_0)` | 低秩分解版 cross 层。 |
| → `cross_net_mix` | `(self, x_0)` | 多专家加权混合的 cross 层。 |
| `DCNv2CTR` | `class DCNv2CTR(ContextCTRModel, DCNv2Base)` | DCNv2 的 CTR 版本。 |
| `DCNv2TopK` | `class DCNv2TopK(ContextModel, DCNv2Base)` | DCNv2 的 Top-k 版本。 |

### `context/DeepFM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `DeepFMBase` | `class DeepFMBase(WideDeepBase)` | FM 二阶交叉与 deep MLP 共享 embedding，替代 Wide&Deep 的 wide 部分。 |
| `DeepFMCTR` | `class DeepFMCTR(WideDeepCTR, DeepFMBase)` | DeepFM 的 CTR 版本。 |
| `DeepFMTopK` | `class DeepFMTopK(WideDeepTopK, DeepFMBase)` | DeepFM 的 Top-k 版本。 |

### `context/FinalMLP.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `FinalMLPBase` | `class FinalMLPBase(object)` | 双流 MLP：两路独立 MLP 各取不同特征，最后融合成预测。 |
| `FeatureSelection` | `class FeatureSelection(nn.Module)` | 为两路 MLP 分别学习特征门控权重（可用不同上下文特征）。 |
| → `__init__` | `(self, feature_map, feature_dim, embedding_dim, fs_hidden_units=[], fs1_context=[], fs2_context=[], feature_maxn=dict())` | 保存两路特征清单并建门控网络。 |
| → `forward` | `(self, feed_dict, flat_emb)` | 输出两路各自加权的特征向量。 |
| `InteractionAggregation` | `class InteractionAggregation(nn.Module)` | 融合两路 MLP 输出（concat/sum/gate 等，可多头）。 |
| → `__init__` | `(self, x_dim, y_dim, output_dim=1, num_heads=1)` | 定义融合方式与输出维度。 |
| → `forward` | `(self, x, y)` | 返回融合后的标量打分。 |
| `FinalMLPCTR` | `class FinalMLPCTR(ContextCTRModel, FinalMLPBase)` | FinalMLP 的 CTR 版本。 |
| `FinalMLPTopK` | `class FinalMLPTopK(ContextModel, FinalMLPBase)` | FinalMLP 的 Top-k 版本。 |

### `context/SAM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `SAMBase` | `class SAMBase(object)` | 把上下文特征堆叠成序列，用自注意力块（SAM）显式建模特征交互。 |
| `SAMBlock` | `class SAMBlock(nn.Module)` | 堆叠 `num_layers` 层 SAM 交互层，可选残差。 |
| → `__init__` | `(self, num_layers, num_fields, embedding_dim, use_residual=False, interaction_type='SAM2E', aggregation='concat', dropout=0)` | 按交互类型选择 2 元/3 元、加法/元素级注意力。 |
| → `forward` | `(self, F)` | 输入 `[B, item_num, field_num, dim]`，输出交互后的特征。 |
| `SAM2A` / `SAM2E` | `class SAM2A(nn.Module)` / `class SAM2E(nn.Module)` | 二阶（两两特征）注意力交互，A=加法式权重、E=元素级权重。 |
| → `forward` | `(self, F)` | 对特征对做加权组合。 |
| `SAM3A` / `SAM3E` | `class SAM3A(nn.Module)` / `class SAM3E(nn.Module)` | 三阶（三元特征）注意力交互，支持残差。 |
| → `forward` | `(self, F)` | 对特征三元组做加权组合。 |
| `SAMCTR` | `class SAMCTR(ContextCTRModel, SAMBase)` | SAM 的 CTR 版本。 |
| `SAMTopK` | `class SAMTopK(ContextModel, SAMBase)` | SAM 的 Top-k 版本。 |

### `context/WideDeep.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `WideDeepBase` | `class WideDeepBase(FMBase)` | Wide&Deep：wide 线性部分记忆特征共现，deep MLP 部分做泛化。 |
| `WideDeepCTR` | `class WideDeepCTR(ContextCTRModel, WideDeepBase)` | Wide&Deep 的 CTR 版本。 |
| `WideDeepTopK` | `class WideDeepTopK(ContextModel, WideDeepBase)` | Wide&Deep 的 Top-k 版本。 |

### `context/xDeepFM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `xDeepFMBase` | `class xDeepFMBase(FMBase)` | FM + CIN（压缩交互网络，显式高阶交叉）+ DNN 的混合模型。 |
| → `compreseed_interaction_network` | `(self, input_features, activation='nn.ReLU')` | CIN 前向：逐层用卷积做外积式特征交叉。 |
| → `l2_reg` / `reg_loss` | `(self, parameters)` / `(self)` | 计算参数 L2 正则项并加权进总损失。 |
| `xDeepFMCTR` | `class xDeepFMCTR(ContextCTRModel, xDeepFMBase)` | xDeepFM 的 CTR 版本。 |
| `xDeepFMTopK` | `class xDeepFMTopK(ContextModel, xDeepFMBase)` | xDeepFM 的 Top-k 版本。 |

---

## 3. `context_seq/` — 上下文 + 用户行为序列模型

### `context_seq/DIN.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `DINBase` | `class DINBase(object)` | Deep Interest Network：按目标物品对历史行为做注意力加权后池化。 |
| → `attention` | `(self, queries, keys, keys_length, mask_mat, softmax_stag=False, return_seq_weight=False)` | 计算目标-历史注意力分数并屏蔽 padding。 |
| → `get_all_embedding` | `(self, feed_dict, merge_all=True)` | 取出目标、历史、用户与上下文 embedding。 |
| → `att_dnn` | `(self, current_emb, history_emb, all_context, history_lengths)` | 注意力池化后的历史表征接入 DNN 预测。 |
| `DINCTR` | `class DINCTR(ContextSeqCTRModel, DINBase)` | DIN 的 CTR 版本。 |
| `DINTopK` | `class DINTopK(ContextSeqModel, DINBase)` | DIN 的 Top-k 版本。 |

### `context_seq/DIEN.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `DIENBase` | `class DIENBase(DINBase)` | Deep Interest Evolution Network：GRU 抽兴趣 + AUGRU 按目标物品演化兴趣。 |
| → `interest_extractor` | `(self, history_emb, hislens)` | 用 GRU 从行为序列抽取每步兴趣状态。 |
| → `target_attention` | `(self, target_emb, interest_emb, lengths)` | 计算目标物品对每个兴趣状态的注意力。 |
| → `interest_evolving` | `(self, interests, interest_emb, target_emb, lengths)` | 以注意力为门控做 AUGRU 兴趣演化。 |
| → `aux_loss` | `(self, out_dict)` | 用下一时刻真实物品监督每步兴趣表示（辅助损失）。 |
| → `get_all_embeddings` / `get_neg_embeddings` | `(self, feed_dict)` | 获取正样本 / 负样本 embedding。 |
| → `_get_inp` | `(self, feed_dict)` | 组织 DIEN 主干的输入并返回中间结果。 |
| `DynamicGRU` | `class DynamicGRU(nn.Module)` | 支持 GRU / AIGRU / AGRU / AUGRU 四种注意力门控变体的 GRU。 |
| → `__init__` | `(self, input_size, hidden_size, bias=True, gru_type='AUGRU')` | 选择门控类型并建对应 cell。 |
| → `forward` | `(self, packed_seq_emb, attn_score=None, h=None)` | 按注意力分数逐时刻推进隐藏状态。 |
| `AUGRUCell` / `AGRUCell` | `class AUGRUCell(nn.Module)` / `class AGRUCell(nn.Module)` | 带注意力更新门（AUGRU）/ 注意力重置门（AGRU）的单步 GRU cell。 |
| → `forward` | `(self, x, hx, attn)` | 单步更新：`h' = f(x, h, attn)`。 |
| `DIENTopK` | `class DIENTopK(ContextSeqModel, DIENBase)` | DIEN 的 Top-k 版本。 |
| `DIENCTR` | `class DIENCTR(ContextSeqCTRModel, DIENBase)` | DIEN 的 CTR 版本。 |

### `context_seq/CAN.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `CANBase` | `class CANBase(DIENBase)` | 在 DIEN 上增加特征共现（Co-Action）模块：以目标物品诱导向量为条件生成用户/情境/历史交互特征。 |
| → `gen_coaction` | `(self, induction, feed)` | 用目标诱导向量 + 微 MLP 生成单特征的条件交互向量。 |
| → `gen_his_coation` | `(self, induction, feed, mask)` | 与历史序列逐位置做共现交互并屏蔽 padding。 |
| `CANTopK` | `class CANTopK(ContextSeqModel, CANBase)` | CAN 的 Top-k 版本。 |
| `CANCTR` | `class CANCTR(ContextSeqCTRModel, CANBase)` | CAN 的 CTR 版本。 |

### `context_seq/ETA.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `ETABase` | `class ETABase(object)` | End-to-End Target Attention：短序列精确注意力，长序列先用 LSH 检索 top-k 相似行为再注意力。 |
| → `short_interest_attention` | `(self, feature_emb_dict, mask)` | 对近期（短）行为做标准 target attention。 |
| → `long_interest_attention` | `(self, feature_emb_dict, mask, feature_emb)` | 对检索出的长程行为做注意力。 |
| → `topk_retrieval` | `(self, random_rotations, target_item, history_sequence, mask, topk=5)` | 用哈希桶检索与目标最相似的 top-k 历史行为。 |
| → `lsh_hash` | `(self, vecs, random_rotations)` | 随机旋转 + 分桶，得到 SimHash/LSH 桶编号。 |
| → `concat_embedding` | `(self, field, feature_emb_dict)` | 按字段拼接多个特征的 embedding。 |
| `ETACTR` | `class ETACTR(ContextSeqCTRModel, ETABase)` | ETA 的 CTR 版本。 |
| `ETATopK` | `class ETATopK(ContextSeqModel, ETABase)` | ETA 的 Top-k 版本。 |

### `context_seq/SDIM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `SDIMBase` | `class SDIMBase(ETABase)` | Sampling Is All You Need：用 LSH 把长序列分桶，只在同桶内做注意力以降低长序列开销。 |
| → `lsh_attention` | `(self, random_rotations, target_item, history_sequence)` | 目标与历史同桶者才参与注意力计算。 |
| → `long_interest_attention` | `(self, feature_emb_dict, mask, feature_emb)` | 用 LSH 注意力替换 ETA 的检索式长兴趣建模。 |
| `SDIMCTR` | `class SDIMCTR(ContextSeqCTRModel, SDIMBase)` | SDIM 的 CTR 版本。 |
| `SDIMTopK` | `class SDIMTopK(ContextSeqModel, SDIMBase)` | SDIM 的 Top-k 版本。 |

---

## 4. `general/` — 通用协同过滤（Top-k）

### `general/BPRMF.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `BPRMFBase` | `class BPRMFBase(object)` | 矩阵分解 + BPR pairwise 排序损失，用户/物品各一个 embedding。 |
| `BPRMF` | `class BPRMF(GeneralModel, BPRMFBase)` | BPRMF 的通用 Top-k 版本。 |
| `BPRMFImpression` | `class BPRMFImpression(ImpressionModel, BPRMFBase)` | BPRMF 的曝光列表版本。 |

### `general/BUIR.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `BUIR` | `class BUIR(GeneralModel)` | 双分支自举协同过滤：在线/目标两套用户物品表征 + stop-gradient，双向对齐，无需负采样。 |
| → `init_weights` | `(m)` | 对 embedding 做均匀/正态初始化。 |
| → `_update_target` | `(self)` | 以动量方式更新目标网络参数。 |
| → `forward` / `loss` | `(self, feed_dict)` / `(self, output)` | 输出在线/目标表征，并计算用户-物品双向对齐损失。 |

### `general/CFKG.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `CFKG` | `class CFKG(GeneralModel)` | 在知识图谱上用 TransE 式翻译嵌入学习用户/物品/关系向量并用于推荐打分。 |

### `general/DirectAU.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `DirectAU` | `class DirectAU(GeneralModel)` | 直接用 alignment（正对靠近）+ uniformity（整体分散）两个表示学习目标优化 CF 表征。 |
| → `alignment` | `(x, y)` | 计算正样本对的表示对齐损失。 |
| → `uniformity` | `(x)` | 计算表示在球面上的均匀性损失。 |

### `general/LightGCN.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `LightGCNBase` | `class LightGCNBase(object)` | 在 user-item 二部图上做线性图卷积，去掉特征变换与非线性。 |
| → `build_adjmat` | `(user_count, item_count, train_mat, selfloop_flag=False)` | 由训练交互矩阵构造归一化邻接矩阵。 |
| `LGCNEncoder` | `class LGCNEncoder(nn.Module)` | 多层图卷积编码器。 |
| → `__init__` | `(self, user_count, item_count, emb_size, norm_adj, n_layers=3)` | 初始化用户/物品 embedding 与归一化邻接。 |
| → `forward` | `(self, users, items)` | 逐层聚合邻居后返回用户/物品最终表征。 |
| `LightGCN` | `class LightGCN(GeneralModel, LightGCNBase)` | LightGCN 的通用 Top-k 版本。 |
| `LightGCNImpression` | `class LightGCNImpression(ImpressionModel, LightGCNBase)` | LightGCN 的曝光列表版本。 |

### `general/NeuMF.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `NeuMF` | `class NeuMF(GeneralModel)` | 神经协同过滤：GMF 内积路与 MLP 路并行，最后拼接融合。 |

### `general/POP.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `POP` | `class POP(GeneralModel)` | 按物品流行度直接打分的最简 baseline（通常配 `--train 0`）。 |

---

## 5. `reranker/` — 候选列表重排

### `reranker/PRM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `LearnedPositionEncoding` | `class LearnedPositionEncoding(nn.Embedding)` | 可学习的位置编码，给候选列表注入顺序信息。 |
| → `forward` | `(self, x)` | 返回位置向量。 |
| `PRMBase` | `class PRMBase(object)` | Personalized Re-ranking：把基础 ranker 的 PV 向量、候选 embedding 与位置编码送入 Transformer 编码器得到个性化分数。 |
| → `forward` | `(self, feed_dict)` | 拼接 `u_v`/`i_v` 与位置编码，逐层编码候选列表后打分。 |
| `PRMGeneral` | `class PRMGeneral(RerankModel, PRMBase)` | 基于 general ranker 的 PRM。 |
| `PRMSequential` | `class PRMSequential(RerankSeqModel, PRMBase)` | 基于 sequential ranker 的 PRM。 |

### `reranker/SetRank.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `MAB` | `class MAB(nn.Module)` | 多头注意力块（Multihead Attention Block），含残差与前馈。 |
| → `__init__` | `(self, d_model, nhead, d_feedforward, dropout)` | 定义注意力与前馈网络。 |
| → `forward` | `(self, Q, K, V, src_mask=None, src_key_padding_mask=None)` | 计算注意力块输出。 |
| `MSAB` | `class MSAB(nn.Module)` | 自注意力块：MAB 的 Q=K=V 特例，堆叠形成集合编码器。 |
| `IMSAB` | `class IMSAB(nn.Module)` | 诱导式多头注意力块：用 `m_clusters` 个诱导点把复杂度从 O(n²) 降到 O(nm)。 |
| → `__init__` | `(self, d_model, nhead, d_feedforward, dropout, m_clusters=20)` | 设置诱导点数量与注意力子层。 |
| `SetRankBase` | `class SetRankBase(object)` | 排列不变的集合式重排：同一候选集合的分数与输入顺序无关。 |
| → `forward` | `(self, feed_dict)` | 用自注意力编码集合后逐候选打分。 |
| `SetRankGeneral` | `class SetRankGeneral(RerankModel, SetRankBase)` | 基于 general ranker 的 SetRank。 |
| `SetRankSequential` | `class SetRankSequential(RerankSeqModel, SetRankBase)` | 基于 sequential ranker 的 SetRank。 |

### `reranker/MIR.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `SLAttention` | `class SLAttention(nn.Module)` | 支持特征级/物品级与时间衰减的多级注意力层。 |
| → `__init__` | `(self, v_dim, q_dim, v_seq_len, batch_size, hidden_size, fi=True, ii=True, decay=True)` | 开关特征交互、物品交互与时间衰减。 |
| → `forward` | `(self, V, Q, time, usr_prof, his_max)` | 用用户历史对候选列表做多级交互打分。 |
| `MIRBase` | `class MIRBase(object)` | Multi-Level Interaction Reranking：先 ranker 表征，再在用户历史与候选列表间做多级交叉。 |
| → `forward` | `(self, feed_dict)` | 生成候选列表的重排分数。 |
| `MIRGeneral` | `class MIRGeneral(RerankSeqModel, MIRBase)` | 使用 general ranker 的 MIR 变体。 |
| `MIRSequential` | `class MIRSequential(RerankSeqModel, MIRBase)` | 使用 sequential ranker 的 MIR 变体。 |

---

## 6. `sequential/` — 序列推荐

### `sequential/GRU4Rec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `GRU4RecBase` | `class GRU4RecBase(object)` | 用 GRU 编码会话行为序列，取最后隐状态预测下一物品。 |
| `GRU4Rec` | `class GRU4Rec(SequentialModel, GRU4RecBase)` | GRU4Rec 的序列推荐版本。 |
| `GRU4RecImpression` | `class GRU4RecImpression(ImpressionSeqModel, GRU4RecBase)` | GRU4Rec 的曝光列表版本。 |

### `sequential/SASRec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `SASRecBase` | `class SASRecBase(object)` | 因果（单向掩码）自注意力序列推荐，用下一物品预测训练。 |
| `SASRec` | `class SASRec(SequentialModel, SASRecBase)` | SASRec 的序列推荐版本。 |
| `SASRecImpression` | `class SASRecImpression(ImpressionSeqModel, SASRecBase)` | SASRec 的曝光列表版本。 |

### `sequential/Caser.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `Caser` | `class Caser(SequentialModel)` | 用水平卷积捕捉序列局部模式、垂直卷积捕捉跳跃/全局模式（卷积序列嵌入）。 |

### `sequential/FPMC.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `FPMC` | `class FPMC(SequentialModel)` | 因子分解个性化马尔可夫链：融合用户长期偏好与上一物品的转移偏好。 |

### `sequential/NARM.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `NARM` | `class NARM(SequentialModel)` | GRU 编码 + 注意力池化，同时建模序列整体兴趣与最后一次行为（局部兴趣）。 |

### `sequential/ComiRec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `ComiRec` | `class ComiRec(SequentialModel)` | 多兴趣框架：用 K 个注意力头抽取多兴趣向量，训练时选目标兴趣、推理时可控聚合。 |

### `sequential/ContraRec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `ContraRec` | `class ContraRec(SequentialModel)` | 序列推荐 + 多种对比信号（序列-物品 / 序列-序列）增强下一物品预测。 |
| `ContraLoss` | `class ContraLoss(nn.Module)` | 多对比 InfoNCE 损失。 |
| → `__init__` | `(self, device, temperature=0.2)` | 设置温度系数与设备。 |
| → `forward` | `(self, features, labels=None)` | 计算序列/物品特征间的对比损失。 |
| `GRU4RecEncoder` | `class GRU4RecEncoder(nn.Module)` | 可替换的 GRU 序列编码器。 |
| `CaserEncoder` | `class CaserEncoder(nn.Module)` | 可替换的 Caser 卷积序列编码器。 |
| `BERT4RecEncoder` | `class BERT4RecEncoder(nn.Module)` | 可替换的双向 Transformer 序列编码器。 |

### `sequential/Chorus.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `Chorus` | `class Chorus(SequentialModel)` | 知识 + 时间感知：用关系时间间隔核函数动态修正物品向量，再与 MF/GMF 分数结合；两阶段训练。 |
| → `kernel_functions` | `(self, r_interval, betas, sigmas, mus)` | 按关系类型（指数/正态/补货/替代）计算时间衰减核。 |
| → `rec_forward` | `(self, feed_dict)` | 生成时间感知物品向量并计算用户打分。 |
| → `kg_forward` | `(self, feed_dict)` | TransE 式知识图谱三元组打分（stage 1 预训练）。 |
| → `loss` | `(self, out_dict)` | stage 1 用 margin ranking，stage 2 用 BPR。 |
| → `customize_parameters` | `(self)` | 按阶段把 KG 参数与推荐参数分组。 |

### `sequential/KDA.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `KDA` | `class KDA(SequentialModel)` | 知识感知动态聚合：按关系与时间间隔聚合历史实体，再做自注意力与池化。 |
| → `rec_forward` | `(self, feed_dict)` | 第一阶段聚合 + 第二阶段自注意力 + 池化后打分。 |
| → `kg_forward` | `(self, feed_dict)` | DistMult 式知识图谱打分。 |
| → `loss` | `(self, out_dict)` | 推荐 BPR 损失 + KG 损失联合优化。 |
| `RelationalDynamicAggregation` | `class RelationalDynamicAggregation(nn.Module)` | 按关系和时间间隔对历史实体做动态加权聚合。 |
| → `__init__` | `(self, n_relation, n_freq, relation_embeddings, include_val, device)` | 设置关系数、频域维度与是否使用属性值。 |
| → `idft_decay` | `(self, delta_t)` | 由 IDFT 得到时间间隔衰减权重。 |
| → `forward` | `(self, seq, delta_t_n, target, target_value, valid_mask)` | 返回聚合后的历史上下文表示。 |

### `sequential/SLRCPlus.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `SLRCPlus` | `class SLRCPlus(SequentialModel)` | 重复消费建模：为每个物品学时间激励核（指数/正态混合），把激励强度加到 MF 基准强度上。 |
| → `forward` | `(self, feed_dict)` | 基准 MF 分数 + 时间激励项得到最终打分。 |
| → `Dataset._get_feed_dict` | `(self, index)` | 额外收集目标物品相关的重复消费时间间隔。 |

### `sequential/TiSASRec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `TiSASRec` | `class TiSASRec(SequentialModel)` | 时间间隔感知自注意力：把物品间时间间隔编码进注意力的位置/关系项。 |
| `TimeIntervalMultiHeadAttention` | `class TimeIntervalMultiHeadAttention(nn.Module)` | 带时间间隔矩阵的多头注意力。 |
| → `__init__` | `(self, d_model, n_heads, kq_same=False, bias=True)` | 定义 Q/K/V 投影与时间间隔投影。 |
| → `forward` | `(self, q, k, v, pos_k, pos_v, inter_k, inter_v, mask)` | 注意力 logits = 内容项 + 位置项 + 时间间隔项。 |
| → `scaled_dot_product_attention` | `(q, k, v, inter_k, inter_v, d_k, mask)` | 融合时间间隔的缩放点积注意力。 |
| `TimeIntervalTransformerLayer` | `class TimeIntervalTransformerLayer(nn.Module)` | 对应的时间间隔 Transformer 层（注意力 + FFN + 残差/归一化）。 |
| → `forward` | `(self, seq, pos_k, pos_v, inter_k, inter_v, mask)` | 单层前向。 |

### `sequential/TiMiRec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `TiMiRec` | `class TiMiRec(SequentialModel)` | 多兴趣 + 目标兴趣蒸馏：先预训练多兴趣抽取器，再微调时用历史预测目标兴趣分布并做 KL 蒸馏。 |
| → `similarity` | `(a, b)` | 余弦相似度。 |
| → `load_model` | `(self, model_path=None)` | 只加载预训练中存在的参数（部分加载）。 |
| → `forward` | `(self, feed_dict)` | 预训练阶段选目标兴趣；微调阶段按预测兴趣分布加权聚合。 |
| → `loss` | `(self, out_dict: dict)` | 预训练用 BPR；微调用 BPR + 温度缩放 KL 蒸馏损失。 |
| `MultiInterestExtractor` | `class MultiInterestExtractor(nn.Module)` | 用 K 个注意力头从历史序列抽取 K 个兴趣向量。 |
| → `__init__` | `(self, k, item_num, emb_size, attn_size, max_his, add_pos, add_trm)` | 配置兴趣数、位置/Transformer 开关。 |
| → `forward` | `(self, history, lengths)` | 返回 `[B, K, emb]` 多兴趣表示。 |
| `InterestPredictor` | `class InterestPredictor(nn.Module)` | 从历史序列预测目标兴趣分布（用于蒸馏监督）。 |
| → `forward` | `(self, history, lengths)` | 返回 `[B, emb]` 目标兴趣预测。 |

---

## 7. `developing/` — 实验性模型（接口可能变动）

### `developing/S3Rec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `S3Rec` | `class S3Rec(SequentialModel)` | 自监督预训练序列推荐：掩码物品预测 + 片段预测，用 BERT4Rec 编码器。 |
| → `_masked_item_prediction` | `(self, seq_output, target_item_emb)` | 掩码位置物品分类损失。 |
| → `_segment_prediction` | `(self, context, segment_emb)` | 判断片段是否属于同一序列的自监督损失。 |
| → `loss` | `(self, out_dict)` | 预训练多任务损失 / 下游 BPR 损失。 |
| `BERT4RecEncoder` | `class BERT4RecEncoder(nn.Module)` | 双向 Transformer 序列编码器（S3Rec 预训练骨干）。 |

### `developing/CLRec.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `CLRec` | `class CLRec(SequentialModel)` | 纯对比学习序列推荐：用 BERT4Rec 编码历史，再把序列表征与目标物品表征拉近。 |
| → `loss` | `(self, out_dict)` | 对 `[序列表征, 目标物品表征]` 计算对比损失（无 BPR 主损失）。 |
| `ContraLoss` | `class ContraLoss(nn.Module)` | InfoNCE 对比损失。 |
| → `forward` | `(self, features, mask=None)` | 计算批内对比损失。 |
| `BERT4RecEncoder` | `class BERT4RecEncoder(nn.Module)` | 双向 Transformer 序列编码器。 |

### `developing/SRGNN.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `SRGNN` | `class SRGNN(SequentialModel)` | 把会话序列建成图，用 GNN 聚合邻居得到会话/物品表征。 |
| → `_get_slice` | `(self, item_seq)` | 由序列构造邻接矩阵或图切片。 |
| `GNN` | `class GNN(nn.Module)` | 会话图上的图神经网络传播模块。 |
| → `__init__` | `(self, embedding_size, step=1)` | 设置传播步数。 |
| → `gnn_cell` | `(A, hidden)` | 单步邻居聚合。 |
| → `forward` | `(A, hidden)` | 多步传播后返回节点表征。 |

### `developing/FourierTA.py`

| 类 / 函数 | 签名 | 一句话功能 |
|---|---|---|
| `FourierTA` | `class FourierTA(SequentialModel)` | 用傅里叶频域建模时间间隔的时序注意力序列推荐。 |
| `FourierTemporalAttention` | `class FourierTemporalAttention(nn.Module)` | 频域时间注意力层。 |
| → `__init__` | `(self, emb_size, freq_dim, device)` | 设置频域维度。 |
| → `idft_decay` | `(self, delta_t)` | 由 IDFT 计算时间间隔衰减核。 |
| → `forward` | `(self, seq, delta_t_n, target, valid_mask)` | 时间感知注意力聚合历史序列。 |

---

## 8. 常见扩展点

| 想做的事 | 改哪里 |
|---|---|
| 新增一个上下文模型 | 新建 `context/Xxx.py`，仿 `FM.py` 写 `XxxBase` + `XxxCTR`/`XxxTopK`；包内 `__init__.py` 自动发现。 |
| 新增序列模型 | 新建 `sequential/Xxx.py`，继承 `SequentialModel`，实现 `parse_model_args`/`__init__`/`_define_params`/`forward`。 |
| 改样本构造 | 重写对应基座 `Dataset._get_feed_dict` 或 `actions_before_epoch`。 |
| 换损失 | 重写模型 `loss(out_dict)`；Top-k/CTR 基座可优先用 `--loss_n`。 |
| 加辅助任务 | 在 `forward` 里把辅助预测放进 `out_dict`，在 `loss` 中合并（参考 `Chorus`/`KDA`/`TiMiRec`）。 |
| 新重排模型 | 继承 `RerankModel`（或 `RerankSeqModel`），实现 `forward`，并配置 `ranker_name`/`ranker_config_file`/`ranker_model_file`。 |

---

*清单依据：`src/models` 源码 AST 扫描（2026-11-15）。*
