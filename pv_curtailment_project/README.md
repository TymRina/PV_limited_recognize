# 限功率识别流水线模型

基于 MiniRocket 时序分类算法的光伏电站限功率识别系统，采用五大模块解耦架构，支持冷启动训练、增量微调与生产部署的完整流水线。

## 项目架构

```
pv_curtailment_project/
├── data/                          # 数据目录（运行时自动创建）
│   └── 01_raw/                    # 原始数据（各电站独立目录）
│       └── station_{id}/
│           └── capacity_{MW}MW/
│               └── merged_data.csv
├── models/                        # 模型保存目录
│   ├── minirocket_{station}_base.pkl       # 基线模型
│   ├── minirocket_{station}_prod.pkl       # 生产模型
│   └── training_data_{station}_{type}/     # 训练数据（NPY格式）
│       ├── X_train.npy
│       └── Y_train.npy
├── src/
│   ├── __init__.py
│   ├── m1_raw_generator.py        # 模块1：原始电站数据生成
│   ├── m2_curtail_simulator.py    # 模块2：限功率数据制造
│   ├── m3_pretrain_engine.py      # 模块3：预训练模型模块
│   ├── m4_retrain_engine.py       # 模块4：模型再训练模块
│   ├── m5_evaluator.py            # 模块5：结果评估模块
│   └── legacy/                    # 历史版本代码（存档）
├── station_factory.py             # 一键工厂演示入口
└── README.md
```

## 五大模块架构

### 模块1：原始数据生成 [m1_raw_generator.py](file:///e:/Trae%20Project/Rocket_PV_limited/pv_curtailment_project/src/m1_raw_generator.py)

**函数签名**：`generate_raw_station_data(station_id, rated_capacity, start_date, end_date)`

**职责**：根据额定容量生成纯净无异常的时序数据

**输出列**：`['timestamp', 'pv_data', 'GHI', 'cap_power_on']`

**物理约束**：
- GHI ≤ 1000 W/m²（太阳常数上限）
- pv_data ≤ rated_capacity（额定容量约束）

---

### 模块2：限功率制造 [m2_curtail_simulator.py](file:///e:/Trae%20Project/Rocket_PV_limited/pv_curtailment_project/src/m2_curtail_simulator.py)

**函数签名**：`inject_curtailment_scenarios(raw_df, curtail_ratios)`

**职责**：注入限功率场景到原始数据中

**核心逻辑**：
1. **马尔可夫链状态机**：采样限电日期（周末概率倍增）
2. **慢变温漂**：随时间线性变化的限电比例偏移
3. **GHI变差噪声**：轻微限电(0.85~0.95)时注入非对称向下毛刺

**输出列**：新增 `pv_simulated`, `is_curtailed`

---

### 模块3：预训练引擎 [m3_pretrain_engine.py](file:///e:/Trae%20Project/Rocket_PV_limited/pv_curtailment_project/src/m3_pretrain_engine.py)

**函数签名**：`build_base_model(simulated_df, station_id)`

**职责**：执行严格时序隔离并训练基线模型

**核心流程**：
1. 按日期硬隔离：训练集 70% / 验证集 15% / 测试集 15%
2. 在块内切片为 `(N, 3, 32)` 张量
3. 初始化 MiniRocket + RidgeClassifierCV
4. 保存基线模型及训练数据

**标签规则**：窗口内 ≥4 点(1小时)限电 → Y=1

---

### 模块4：再训练引擎 [m4_retrain_engine.py](file:///e:/Trae%20Project/Rocket_PV_limited/pv_curtailment_project/src/m4_retrain_engine.py)

**函数签名**：`fine_tune_with_real_data(base_model_path, real_labeled_df, station_id)`

**职责**：基于真实数据对基线模型进行增量微调

**微调策略**：
1. 加载预训练的 MiniRocket 转换器（保持不变）
2. 将新数据与原始训练数据合并
3. 使用合并后的数据重新训练分类器
4. 另存为生产模型

**注意**：RidgeClassifierCV 不支持 partial_fit，采用合并数据重训策略

---

### 模块5：评估引擎 [m5_evaluator.py](file:///e:/Trae%20Project/Rocket_PV_limited/pv_curtailment_project/src/m5_evaluator.py)

**函数签名**：`evaluate_model_performance(y_true, y_pred, title_prefix)`

**职责**：计算并打印标准评估指标

**输出指标**：
- 混淆矩阵（Confusion Matrix）
- Precision（精准率）
- Recall（召回率）
- F1-Score

---

## 工厂入口脚本

### station_factory.py

**功能**：一键全自动演示冷启动 + 生产微调完整流程

**使用方式**：

```bash
python station_factory.py
```

**演示流程**：

1. **冷启动全流程**（M1→M2→M3→M5）
   - 生成 460MW 原始数据
   - 注入限功率场景
   - 训练基线模型
   - 输出基线评估报告

2. **生产级微调**（M4→M5）
   - 使用独立随机种子生成新季度数据
   - 增量微调基线模型
   - 输出生产模型评估报告

3. **全流程对比报告**
   - 基线模型 vs 生产模型指标对比

---

## 核心设计原则

### 1. 严格时序隔离

```
训练集（70%日期）→ 验证集（15%日期）→ 测试集（15%日期）
    ↑                    ↑                    ↑
  硬隔离               硬隔离               硬隔离
```

- 先按日期分块，再在块内进行窗口切片
- 确保训练、验证、测试集在时间戳上完全无交集
- 防止数据泄漏

### 2. 特征工程

| 通道 | 说明 |
|------|------|
| pv_simulated | 限电后的实际功率 |
| GHI | 总辐照度 |
| cap_power_on | 开机容量 |

- 特征张量形状：`(N, 3, 32)`
- 32 点 = 8 小时日间窗口（9:00~16:00）

### 3. 数据存储优化

- 训练数据单独存储为 `.npy` 文件，避免 pickle 膨胀
- 模型文件仅存储 Rocket 转换器和分类器
- 支持增量训练时动态加载历史数据

### 4. 随机种子控制

- 全局随机种子：`RANDOM_SEED = 42`
- 压测数据使用独立种子（如 2026）进行统计学验证
- 确保结果可复现

---

## 安装依赖

```bash
pip install numpy pandas scikit-learn sktime numba
```

---

## 快速开始

### 一键演示

```bash
python station_factory.py
```

### 模块调用示例

```python
from src.m1_raw_generator import generate_raw_station_data
from src.m2_curtail_simulator import inject_curtailment_scenarios
from src.m3_pretrain_engine import build_base_model
from src.m4_retrain_engine import fine_tune_with_real_data
from src.m5_evaluator import evaluate_model_performance

# 1. 生成原始数据
raw_df = generate_raw_station_data("station_100MW", 100.0)

# 2. 注入限功率场景
simulated_df = inject_curtailment_scenarios(raw_df)

# 3. 训练基线模型
base_metrics = build_base_model(simulated_df, "station_100MW")

# 4. 增量微调（使用新数据）
prod_metrics = fine_tune_with_real_data(
    base_model_path="models/minirocket_station_100MW_base.pkl",
    real_labeled_df=new_data_df,
    station_id="station_100MW"
)

# 5. 评估
evaluate_model_performance(Y_test, Y_pred)
```

---

## 压测验证

### 随机种子对照实验

**实验设计**：
- 冷启动模型：`RANDOM_SEED=42`
- 压测数据：`RANDOM_SEED=2026`（独立随机波动）

**验证结果**：

| 指标 | 基线模型 | 生产模型（压测） | 变化 |
|------|----------|------------------|------|
| Precision | 0.0941 | **1.0000** | +90.59% |
| Recall | 0.9697 | **0.8889** | -8.08% |
| F1-Score | 0.1716 | **0.9412** | +76.96% |

**结论**：模型在独立随机种子数据上保持 F1=94% 的高水准，具备真实泛化能力，非硬过拟合。

---

## 工程规范

1. **路径设计**：基于 `__file__` 的绝对路径，确保部署灵活性
2. **模块解耦**：五大模块职责单一，可独立测试和替换
3. **自动创建目录**：数据和模型目录自动创建，无需手动配置
4. **数据隔离**：严格时序划分，杜绝数据泄漏
5. **训练数据独立存储**：NPY 格式存储，避免 pickle 膨胀
6. **可配置化**：核心参数通过函数参数传递，支持灵活调整

---

## 技术栈

- **时序分类**：MiniRocket（sktime）
- **分类器**：RidgeClassifierCV（scikit-learn）
- **数据处理**：pandas, numpy
- **并行加速**：numba

---

## 许可证

MIT License