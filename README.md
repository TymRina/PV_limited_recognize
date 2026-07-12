# 光伏限功率识别项目

基于 MiniRocket 算法的光伏限功率事件识别系统，实现从原始数据伪造、限功率模拟到模型训练的完整流程。

## 项目结构

```
├── raw_data/                    # 原始数据目录
│   ├── pv_history.csv           # PV历史功率数据
│   ├── weather_future.csv       # 气象数据（GHI）
│   └── cap_power_on.csv         # 开机容量数据
├── curtailed_data/              # 限功率数据目录
│   └── curtailed_data.csv       # 带限电标签的模拟数据
├── models/                      # 模型目录
│   └── minirocket_pretrained.pkl # 预训练模型
├── generate_fake_data.py        # 原始数据生成器
├── curtailment_simulator.py     # 限功率模拟器
├── data_processor.py            # 数据处理器（特征工程+切片）
├── train_minirocket.py          # MiniRocket训练脚本
├── plot_csv.py                  # 可视化脚本
└── README.md                    # 项目说明
```

## 技术栈

- Python 3.10+
- pandas / numpy：数据处理
- scikit-learn：分类器
- sktime：时序特征提取（MiniRocket）
- plotly：可视化

## 完整流程

### 阶段一：原始数据伪造

**文件**: [generate_fake_data.py](file:///e:/Trae%20Project/Rocket_PV_limited/generate_fake_data.py)

生成2025全年（365天）15分钟粒度的光伏相关数据：

**生成的数据文件**:
- `pv_history.csv`：PV功率数据
- `weather_future.csv`：总辐照度（GHI）数据
- `cap_power_on.csv`：开机容量数据

**核心算法**:

1. **日出日落计算**：基于地理位置（北京纬度39.9°）计算每日日出日落时间
2. **天气模式**：随机生成晴天/多云/雨天三种天气，影响GHI峰值
3. **PV功率曲线**：基于GHI计算PV输出，加入高斯噪声模拟真实波动
4. **容量约束**：PV最大功率不超过容量的95%

**运行方式**:
```bash
python generate_fake_data.py
```

### 阶段二：限功率数据生成

**文件**: [curtailment_simulator.py](file:///e:/Trae%20Project/Rocket_PV_limited/curtailment_simulator.py)

基于马尔可夫链模拟真实的限功率事件，生成带标签的训练数据。

**核心算法**:

1. **马尔可夫链状态转移**：模拟限功率事件的发生概率
   - `p_01`：从正常状态转入限电状态的概率（0.02）
   - `p_11`：持续限电状态的概率（0.85）
   - `weekend_multiplier`：周末限电概率倍增（2.2倍）

2. **日间优先采样**：仅对GHI高于日均中位数的晴天采样为限电日

3. **时间窗口约束**：限功率仅在 09:00-16:00 时段生效

4. **夜间保护**：GHI ≤ 0 或 PV ≤ 0 时不施加限电

5. **离散限电等级**：30%、50%、70% 三档

6. **温漂模拟**：限电过程中加入 -0.5% ~ -1.5% 的向下漂移

**输出数据结构**:

| 字段 | 说明 |
|------|------|
| timestamp | 时间戳（15分钟粒度） |
| pv_data | 原始PV功率 |
| GHI | 总辐照度 |
| cap_power_on | 开机容量 |
| curtailed_pv | 限电后的PV功率 |
| curtailment_label | 限电标签（1=限电中） |
| curtailment_level | 限电百分比（含温漂） |

**运行方式**:
```bash
python curtailment_simulator.py
```

### 阶段三：数据预处理与特征工程

**文件**: [data_processor.py](file:///e:/Trae%20Project/Rocket_PV_limited/data_processor.py)

构建多变量时序特征张量，准备输入MiniRocket模型。

**核心流程**:

1. **特征通道构建**（3通道）：
   - 通道0：`pv_simulated`（限电后的实际功率）
   - 通道1：`GHI`（总辐照度）
   - 通道2：`cap_power_on`（开机容量）

2. **滑动窗口切片**：
   - 窗口长度：32点（8小时）
   - 步长：8点（2小时）
   - 仅对 09:00-16:00 日间时段进行切片

3. **标签聚合**：
   - 窗口内限电点数 ≥ 4（1小时）→ 标签为1（限电窗口）
   - 否则标签为0（正常窗口）

4. **严格时序隔离**：
   - 先按日期划分训练集（70%）、验证集（15%）、测试集（15%）
   - 确保各数据集日期范围无交集
   - 在各自日期块内独立进行窗口切片

**输出维度**:
- 特征张量 X：`(N_samples, 3, Window_Length)`
- 标签数组 Y：`(N_samples,)`

### 阶段四：MiniRocket模型训练

**文件**: [train_minirocket.py](file:///e:/Trae%20Project/Rocket_PV_limited/train_minirocket.py)

使用 MiniRocket 多变量时序分类算法进行限功率识别。

**核心算法**:

1. **MiniRocketMultivariate**：多变量时序特征提取
   - 核数量：10000
   - 输出特征维度：9996

2. **RidgeClassifierCV**：线性分类器
   - 自动选择正则化参数 alpha

3. **模型评估指标**：
   - 混淆矩阵
   - Precision（精准率）
   - Recall（召回率）
   - F1-Score

4. **模型持久化**：保存至 `models/minirocket_pretrained.pkl`

**运行方式**:
```bash
python train_minirocket.py
```

## 训练结果

在严格时序隔离和无泄漏特征的条件下，模型在测试集上取得了优异成绩：

```
测试集评估结果:
  Precision (精准率):  1.0000
  Recall (召回率):     1.0000
  F1-Score:            1.0000

数据分布:
  训练集: 1084 个窗口 (正样本: 45)
  验证集: 84 个窗口 (正样本: 8)
  测试集: 292 个窗口 (正样本: 12)
```

## 关键设计要点

1. **特征泄漏防护**：移除了 `pv_efficiency` 等直接泄漏答案的特征，强迫模型学习曲线形态特征

2. **时序隔离保证**：先按日期分块再切片，确保没有窗口跨越数据集边界

3. **限电日期均衡**：自动调整划分边界，确保每个数据集都包含限电日期

4. **工业物理约束**：模拟真实的限功率物理规律（日间限电、温漂效应、夜间保护）

## 数据可视化

使用 plot_csv.py 生成交互式时间序列图：

```bash
python plot_csv.py
```

生成的 HTML 文件可在浏览器中查看，支持范围选择和鼠标悬停显示详细数据。

## 项目运行顺序

```
1. python generate_fake_data.py    # 生成原始数据
2. python curtailment_simulator.py  # 生成限功率数据
3. python train_minirocket.py       # 训练模型
4. python plot_csv.py               # 可视化结果
```

## 模型部署

预训练模型已保存至 `models/minirocket_pretrained.pkl`，可直接加载用于推理：

```python
import pickle

with open("models/minirocket_pretrained.pkl", "rb") as f:
    model = pickle.load(f)

rocket = model["rocket"]
classifier = model["classifier"]

# 推理
X_transformed = rocket.transform(X_new)
predictions = classifier.predict(X_transformed)
```