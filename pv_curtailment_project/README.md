# 跨电站通用 MiniRocket 限功率识别模型

基于 MiniRocket 时序分类算法的光伏电站限功率识别系统，支持多电站、多容量数据的混合训练与通用模型部署。

## 项目架构

```
pv_curtailment_project/
├── data/
│   ├── 01_raw/              # 原始数据（各电站独立目录）
│   │   └── station_{id}/
│   │       └── capacity_{MW}MW/
│   │           └── merged_data.csv
│   └── 02_simulated/        # 模拟限电数据（自动生成）
├── models/                  # 模型保存目录
│   ├── minirocket_pretrained.pkl
│   └── minirocket_universal.pkl
├── src/
│   ├── __init__.py          # 模块导出
│   ├── simulator.py         # 通用限电模拟引擎
│   ├── data_processor.py    # 数据预处理与切片工具
│   ├── trainer.py           # CurtailmentPipeline 训练类
│   └── legacy/              # 历史版本代码（存档）
└── run_universal_training.py # 跨电站训练主入口
```

## 核心设计

### 1. 严格时序隔离
- **数据划分策略**：先按日期分块，再在各自块内进行窗口切片
- **比例分配**：训练集 70% / 验证集 15% / 测试集 15%
- **无时间交集**：确保训练、验证、测试集在时间戳上完全隔离

### 2. 特征工程
- **输入通道**：`[pv_simulated, GHI, cap_power_on]` 三通道
- **特征张量**：`(N, 3, 32)`，其中 32 点 = 8 小时日间窗口
- **标签规则**：窗口内 `is_curtailed == 1` 点数 ≥ 4 → Y=1，否则 Y=0

### 3. 双模式输入管道

| 管道 | 输入 | 处理流程 | 适用场景 |
|------|------|----------|----------|
| `process_and_fake_raw_sequence` | 原始序列（`pv_data`, `GHI`, `cap_power_on`） | 自动模拟限电 → 切片 | 离线训练、数据增强 |
| `process_labeled_sequence` | 已标记序列（含 `is_curtailed`） | 直接切片 | 生产环境、真实数据 |

### 4. 跨电站混合训练
- 自动发现 `data/01_raw/` 下的所有电站配置
- 支持 60MW ~ 200MW 多容量混合
- 使用 sklearn `shuffle` 彻底打散融合
- 生成通用模型，适配不同容量电站

## 安装依赖

```bash
pip install numpy pandas scikit-learn sktime numba
```

## 快速开始

### 运行跨电站训练

```bash
python run_universal_training.py
```

### 使用 Pipeline

```python
from src.trainer import CurtailmentPipeline

# 初始化管道
pipeline = CurtailmentPipeline(
    window_size=32,
    stride=8,
    label_threshold=4,
    num_kernels=10000
)

# 管道1：原始序列 → 自动模拟限电 → 切片
datasets = pipeline.process_and_fake_raw_sequence(
    raw_df=raw_df,
    station_id="station_001",
    rated_capacity=100.0
)

# 管道2：已标记序列 → 直接切片
datasets = pipeline.process_labeled_sequence(labeled_df)

# 训练模型
pipeline.fit(datasets["X_train"], datasets["Y_train"])

# 评估模型
metrics = pipeline.evaluate(datasets["X_test"], datasets["Y_test"])

# 保存模型
pipeline.save("models/minirocket_universal.pkl")
```

## 数据格式

### 原始数据格式（merged_data.csv）

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | datetime | 时间戳（15分钟间隔） |
| pv_data | float | 光伏实际出力（MW） |
| GHI | float | 总辐照度（W/m²） |
| cap_power_on | float | 开机容量（MW） |

### 已标记数据格式

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | datetime | 时间戳 |
| pv_simulated | float | 限电后功率（MW） |
| GHI | float | 总辐照度（W/m²） |
| cap_power_on | float | 开机容量（MW） |
| is_curtailed | int | 限电标签（0/1） |

## 模型评估

训练完成后输出以下指标：

- **Confusion Matrix**：混淆矩阵
- **Precision**：精准率
- **Recall**：召回率
- **F1-Score**：综合评分

## 技术栈

- **时序分类**：MiniRocket（sktime）
- **分类器**：RidgeClassifierCV（scikit-learn）
- **数据处理**：pandas, numpy
- **并行加速**：numba

## 工程规范

1. **路径设计**：基于 `__file__` 的绝对路径，确保部署灵活性
2. **模块解耦**：simulator / data_processor / trainer 独立模块
3. **可配置化**：核心参数通过类配置，支持灵活调整
4. **数据隔离**：严格时序划分，杜绝数据泄漏

## 许可证

MIT License
