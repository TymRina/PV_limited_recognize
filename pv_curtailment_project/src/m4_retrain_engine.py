import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Dict, Optional

from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils import shuffle

from .m3_pretrain_engine import _split_by_date_then_slice
from .m5_evaluator import evaluate_model_performance

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANDOM_SEED = 42


def _load_base_model(base_model_path: str) -> Optional[dict]:
    """
    加载基线模型
    
    Args:
        base_model_path: 基线模型路径
        
    Returns:
        dict or None: 模型字典（包含rocket, classifier, training_data等）
    """
    if not os.path.exists(base_model_path):
        print(f"基线模型不存在: {base_model_path}")
        return None
    
    with open(base_model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"基线模型加载成功 (版本: {model.get('version', 'unknown')})")
    
    if "training_data_path" in model:
        X_train_path = model["training_data_path"]["X_train"]
        Y_train_path = model["training_data_path"]["Y_train"]
        if os.path.exists(X_train_path) and os.path.exists(Y_train_path):
            model["training_data"] = {
                "X_train": np.load(X_train_path),
                "Y_train": np.load(Y_train_path)
            }
            print(f"训练数据加载成功: {len(model['training_data']['X_train'])} 窗口")
        else:
            print(f"训练数据文件不存在，将使用新数据单独训练")
    
    if "training_data" not in model:
        print("警告: 基线模型不包含训练数据，将使用新数据单独训练")
    
    return model


def fine_tune_with_real_data(
    base_model_path: str,
    real_labeled_df: pd.DataFrame,
    station_id: str,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_pretrained_rocket: bool = True,
    random_seed: int = RANDOM_SEED
) -> Dict[str, float]:
    """
    使用真实带标签数据对基线模型进行微调
    
    核心策略：
    1. 加载预训练的MiniRocket转换器（保持不变）
    2. 将新数据与原始训练数据合并
    3. 使用合并后的数据重新训练分类器
    4. 另存为生产模型
    
    注意：RidgeClassifierCV不支持partial_fit，因此采用合并数据后重新训练的策略
    
    Args:
        base_model_path: 基线模型路径
        real_labeled_df: 真实/高仿带标签数据，需包含 ['timestamp', 'pv_simulated', 'GHI', 'cap_power_on', 'is_curtailed']
        station_id: 电站ID
        window_size: 窗口长度
        stride: 滑动步长
        label_threshold: 标签聚合阈值
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
        use_pretrained_rocket: 是否使用预训练的Rocket转换器
        random_seed: 随机种子
        
    Returns:
        Dict[str, float]: 测试集评估指标
    """
    np.random.seed(random_seed)
    
    print(f"\n{'='*60}")
    print(f"再训练引擎启动 - 电站: {station_id}")
    print(f"{'='*60}")
    
    print("\n1. 加载基线模型...")
    base_model = _load_base_model(base_model_path)
    
    if base_model is None:
        print("错误: 无法加载基线模型，终止再训练")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    df = real_labeled_df.copy()
    df["date"] = df["timestamp"].dt.date
    
    print("\n2. 对新数据进行隔离切片...")
    X_new_train, X_new_val, X_new_test, Y_new_train, Y_new_val, Y_new_test = _split_by_date_then_slice(
        df=df,
        window_size=window_size,
        stride=stride,
        label_threshold=label_threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    print(f"   新数据训练集: {len(X_new_train)} 窗口 (正样本: {Y_new_train.sum()})")
    print(f"   新数据验证集: {len(X_new_val)} 窗口 (正样本: {Y_new_val.sum()})")
    print(f"   新数据测试集: {len(X_new_test)} 窗口 (正样本: {Y_new_test.sum()})")
    
    rocket = base_model["rocket"]
    old_classifier = base_model["classifier"]
    
    if use_pretrained_rocket:
        print("\n3. 使用预训练的 MiniRocket 转换器...")
    else:
        print("\n3. 重新训练 MiniRocket 转换器...")
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        rocket = MiniRocketMultivariate(num_kernels=base_model["params"].get("num_kernels", 10000))
    
    X_train_combined = X_new_train
    Y_train_combined = Y_new_train
    
    if "training_data" in base_model and len(base_model["training_data"]["X_train"]) > 0:
        print("\n4. 合并原始训练数据与新数据...")
        X_train_combined = np.concatenate([base_model["training_data"]["X_train"], X_new_train], axis=0)
        Y_train_combined = np.concatenate([base_model["training_data"]["Y_train"], Y_new_train], axis=0)
        
        print(f"   合并后训练集: {len(X_train_combined)} 窗口 (正样本: {Y_train_combined.sum()})")
        
        X_train_combined, Y_train_combined = shuffle(X_train_combined, Y_train_combined, random_state=random_seed)
        print("   已完成打散")
    
    if not use_pretrained_rocket:
        rocket.fit(X_train_combined)
    
    print("\n5. 提取训练特征...")
    X_train_transformed = rocket.transform(X_train_combined)
    print(f"   特征维度: {X_train_transformed.shape}")
    
    print("6. 重新训练分类器...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    classifier.fit(X_train_transformed, Y_train_combined)
    print(f"   最佳正则化参数: alpha = {classifier.alpha_}")
    
    print("\n7. 验证集评估...")
    X_val_transformed = rocket.transform(X_new_val)
    Y_val_pred = classifier.predict(X_val_transformed)
    evaluate_model_performance(Y_new_val, Y_val_pred, title_prefix="新数据验证集")
    
    print("\n8. 测试集评估...")
    X_test_transformed = rocket.transform(X_new_test)
    Y_test_pred = classifier.predict(X_test_transformed)
    metrics = evaluate_model_performance(Y_new_test, Y_test_pred, title_prefix="新数据测试集")
    
    print("\n9. 保存生产模型...")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"minirocket_{station_id}_prod.pkl")
    train_data_dir = os.path.join(model_dir, f"training_data_{station_id}_prod")
    os.makedirs(train_data_dir, exist_ok=True)
    
    X_train_path = os.path.join(train_data_dir, "X_train.npy")
    Y_train_path = os.path.join(train_data_dir, "Y_train.npy")
    np.save(X_train_path, X_train_combined)
    np.save(Y_train_path, Y_train_combined)
    
    model = {
        "rocket": rocket,
        "classifier": classifier,
        "version": "2.0.0",
        "description": f"生产模型 - 电站 {station_id} (基于基线微调)",
        "features": {
            "channels": ["pv_simulated", "GHI", "cap_power_on"],
            "window_size": window_size,
            "stride": stride,
            "label_threshold": label_threshold
        },
        "params": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "num_kernels": base_model["params"].get("num_kernels", 10000),
            "random_seed": random_seed,
            "use_pretrained_rocket": use_pretrained_rocket
        },
        "training_stats": {
            "original_train_samples": base_model["training_stats"]["train_samples"],
            "new_train_samples": len(X_new_train),
            "combined_train_samples": len(X_train_combined),
            "val_samples": len(X_new_val),
            "test_samples": len(X_new_test),
            "combined_positive": int(Y_train_combined.sum()),
            "new_val_positive": int(Y_new_val.sum()),
            "new_test_positive": int(Y_new_test.sum())
        },
        "training_data_path": {
            "X_train": X_train_path,
            "Y_train": Y_train_path
        },
        "base_model_path": base_model_path
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"   模型已保存至: {model_path}")
    
    return metrics


if __name__ == "__main__":
    try:
        from .m1_raw_generator import generate_raw_station_data
        from .m2_curtail_simulator import inject_curtailment_scenarios
        from .m3_pretrain_engine import build_base_model
    except ImportError:
        from m1_raw_generator import generate_raw_station_data
        from m2_curtail_simulator import inject_curtailment_scenarios
        from m3_pretrain_engine import build_base_model
    
    station_id = "demo_100mw_retrain"
    rated_capacity = 100.0
    
    base_model_path = os.path.join(PROJECT_ROOT, "models", "minirocket_demo_100mw_base.pkl")
    
    if not os.path.exists(base_model_path):
        print("先运行基线训练...")
        
        raw_df = generate_raw_station_data("demo_100mw", 100.0)
        simulated_df = inject_curtailment_scenarios(raw_df)
        build_base_model(simulated_df, "demo_100mw")
    
    print(f"\n生成新的真实数据进行微调...")
    raw_df_new = generate_raw_station_data(station_id, rated_capacity)
    real_labeled_df = inject_curtailment_scenarios(raw_df_new)
    
    print(f"\n执行再训练...")
    metrics = fine_tune_with_real_data(base_model_path, real_labeled_df, station_id)
    
    print(f"\n微调后评估结果:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")