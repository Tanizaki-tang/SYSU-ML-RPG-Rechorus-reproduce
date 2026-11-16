# 源代码

`main.py` 是框架的入口，共有三个主要包。

### 目录结构

- `helpers\`
  - `BaseReader.py`：把数据集 csv 读入 DataFrame，并补充必要信息（如交互历史）
  - `ContextReader.py`：继承自 BaseReader，读取用户与物品的元数据，并统计所有上下文特征的统计量
  - `ContextSeqReader.py`：继承自 ContextReader，在交互历史中附加场景上下文特征
  - `ImpressionReader.py`：继承自 BaseReader，把同一曝光 id（impression id，一次推荐展示的编号）的交互归并为一个实例
  - `BaseRunner.py`：控制模型的训练与评估流程
  - `CTRRunner.py`：继承自 BaseRunner，用二值标签训练和评估模型（点击率预测任务）
  - `ImpressionRunner.py`：继承自 BaseRunner，用基于曝光日志（列表中正负样本长度可变）训练和评估模型
  - `...`：按需自定义有特定功能的辅助模块
- `models\`
  - `BaseModel.py`：基础模型类与数据集类，包含模型的一些通用函数
  - `BaseContextModel.py`：继承自 BaseModel，为基座模型添加上下文特征
  - `BaseImpressionModel.py`：继承自 BaseModel，以曝光（impression）为单位构造数据批次
  - `...`：继承 *BaseModel* 中各类的自定义模型
- `utils\`
  - `layers.py`：模型定义常用模块（如 attention 注意力模块、MLP 多层感知机模块）
  - `utils.py`：若干工具函数
- `main.py`：主入口，串联所有模块
- `exp.py`：重复执行 *run.sh* 中的实验，并把平均结果保存为 csv

### 定义新模型

通常我们定义一个新类继承 *GeneralModel*（*BaseModel* 的子类），并在其中定义内部类 *Dataset*。至少需要实现以下函数：

```python
class NewModel(GeneralModel):
    reader = 'BaseReader'  # 指定 reader 类，默认为 BaseReader
    runner = 'BaseRunner'  # 指定 runner 类，默认为 BaseRunner

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        # 定义模型参数

    def forward(self, feed_dict):
        # 生成预测（根据 feed_dict 中的张量计算排序分数）
        item_id = feed_dict['item_id']  # [batch_size, -1]
        user_id = feed_dict['user_id']  # [batch_size]
        prediction = (...)
        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1)}
        return out_dict

    class Dataset(GeneralModel.Dataset):
        # 为单个样本构造 feed_dict（由 __getitem__ 调用）
        # 之后会整理（collate）成每个批次的完整 feed dict
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            (...)
            return feed_dict
```

如果模型定义更复杂，可以重写 *BaseModel* 中的其他函数（如 `loss` 损失函数、`customize_parameters` 自定义参数）以及 *Dataset* 中的函数（如 `_prepare` 数据准备、`actions_before_epoch` 每个 epoch 前的动作），这需要更深入地理解 [BaseModel.py](https://github.com/THUwangcy/ReChorus/tree/master/src/models/BaseModel.py) 和 [BaseRunner.py](https://github.com/THUwangcy/ReChorus/tree/master/src/helpers/BaseRunner.py)。你也可以实现新的 runner 类以适配不同的实验设置。
