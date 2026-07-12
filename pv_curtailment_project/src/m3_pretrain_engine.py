import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Dict, Optional

from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from .m5_evaluator import evaluate_model_performance

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RANDOM_SEED = 42


def _daytime_sliding_slice_on_df(
    df: pd.DataFrame,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    start_hour: int = 9,
    end_hour: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个DataFrame上执行日间滑动窗口切片
    
    Args:
        df: 预处理后的DataFrame
        window_size: 窗口长度（点数）
        stride: 步长（点数）
        label_threshold: 标签聚合阈值
        start_hour: 日间开始小时
        end_hour: 日间结束小时
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 特征张量 (N, 3, L) 和标签数组 (N,)
    """
    channels = ["pv_simulated", "GHI", "cap_power_on"]
    
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    
    X_list = []
    Y_list = []
    
    unique_dates = df["date"].unique()
    
    for date in unique_dates:
        day_data = df[df["date"] == date].copy().reset_index(drop=True)
        
        daytime_mask = (day_data["hour"] >= start_hour) & (day_data["hour"] < end_hour)
        
        if not daytime_mask.any():
            continue
        
        first_daytime_pos = day_data[daytime_mask].index.min()
        last_daytime_pos = day_data[daytime_mask].index.max()
        
        day_start_pos = day_data.index.min()
        day_end_pos = day_data.index.max()
        
        start_search_pos = max(day_start_pos, first_daytime_pos - window_size + 1)
        end_search_pos = min(day_end_pos, last_daytime_pos + 1)
        
        for start_pos in range(start_search_pos, end_search_pos - window_size + 1, stride):
            end_pos = start_pos + window_size
            
            if end_pos > len(day_data):
                break
            
            window_data = day_data.iloc[start_pos:end_pos]
            
            x = window_data[channels].values.T
            
            curtailed_count = window_data["is_curtailed"].sum()
            y = 1 if curtailed_count >= label_threshold else 0
            
            X_list.append(x)
            Y_list.append(y)
    
    if len(X_list) == 0:
        return np.array([]), np.array([])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    return X, Y


def _split_by_date_then_slice(
    df: pd.DataFrame,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    start_hour: int = 9,
    end_hour: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    严格时序隔离：先按日期分块，再在块内切片
    
    Args:
        df: 预处理后的DataFrame（含date列）
        window_size: 窗口长度
        stride: 步长
        label_threshold: 标签聚合阈值
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
        start_hour: 日间开始小时
        end_hour: 日间结束小时
        
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    
    curtail_dates = set(df[df["is_curtailed"] == 1]["date"].unique())
    
    train_date_end = int(n_dates * train_ratio)
    val_date_end = train_date_end + int(n_dates * val_ratio)
    
    train_dates = set(unique_dates[:train_date_end])
    val_dates = set(unique_dates[train_date_end:val_date_end])
    test_dates = set(unique_dates[val_date_end:])
    
    train_has_curtail = len(train_dates & curtail_dates) > 0
    val_has_curtail = len(val_dates & curtail_dates) > 0
    test_has_curtail = len(test_dates & curtail_dates) > 0
    
    if not val_has_curtail or not test_has_curtail:
        all_curtail_dates_sorted = sorted(curtail_dates)
        n_curtail = len(all_curtail_dates_sorted)
        
        if n_curtail >= 3:
            train_curtail_end = max(1, int(n_curtail * train_ratio))
            val_curtail_end = train_curtail_end + max(1, int(n_curtail * val_ratio))
            
            last_train_curtail = all_curtail_dates_sorted[train_curtail_end - 1]
            last_val_curtail = all_curtail_dates_sorted[val_curtail_end - 1]
            
            train_date_end = None
            val_date_end = None
            
            for i, date in enumerate(unique_dates):
                if date > last_train_curtail and train_date_end is None:
                    train_date_end = i
                if date > last_val_curtail and val_date_end is None:
                    val_date_end = i
            
            if train_date_end is None:
                train_date_end = int(n_dates * train_ratio)
            if val_date_end is None:
                val_date_end = train_date_end + int(n_dates * val_ratio)
            
            train_dates = set(unique_dates[:train_date_end])
            val_dates = set(unique_dates[train_date_end:val_date_end])
            test_dates = set(unique_dates[val_date_end:])
    
    train_max_date = max(train_dates)
    val_min_date = min(val_dates)
    val_max_date = max(val_dates)
    test_min_date = min(test_dates)
    
    assert train_max_date < val_min_date, \
        f"训练集最大日期({train_max_date}) >= 验证集最小日期({val_min_date})"
    assert val_max_date < test_min_date, \
        f"验证集最大日期({val_max_date}) >= 测试集最小日期({test_min_date})"
    
    train_df = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)
    val_df = df[df["date"].isin(val_dates)].copy().reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].copy().reset_index(drop=True)
    
    X_train, Y_train = _daytime_sliding_slice_on_df(
        train_df, window_size, stride, label_threshold, start_hour, end_hour
    )
    X_val, Y_val = _daytime_sliding_slice_on_df(
        val_df, window_size, stride, label_threshold, start_hour, end_hour
    )
    X_test, Y_test = _daytime_sliding_slice_on_df(
        test_df, window_size, stride, label_threshold, start_hour, end_hour
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def build_base_model(
    simulated_df: pd.DataFrame,
    station_id: str,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_kernels: int = 10000,
    random_seed: int = RANDOM_SEED
) -> Dict[str, float]:
    """
    构建并训练基线模型
    
    核心流程：
    1. 严格按日期(70%/15%/15%)硬隔离数据
    2. 在块内切片为 (N, 3, 32) 张量
    3. 初始化 MiniRocket 并拟合分类器
    4. 保存模型至 models/minirocket_{station_id}_base.pkl
    
    Args:
        simulated_df: 已注入限功率场景的数据，需包含 ['timestamp', 'pv_simulated', 'GHI', 'cap_power_on', 'is_curtailed']
        station_id: 电站ID
        window_size: 窗口长度（默认32点=8小时）
        stride: 滑动步长（默认8点）
        label_threshold: 标签聚合阈值（默认4点=1小时）
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
        num_kernels: MiniRocket核数量
        random_seed: 随机种子
        
    Returns:
        Dict[str, float]: 测试集评估指标
    """
    np.random.seed(random_seed)
    
    df = simulated_df.copy()
    df["date"] = df["timestamp"].dt.date
    
    print(f"\n{'='*60}")
    print(f"预训练引擎启动 - 电站: {station_id}")
    print(f"{'='*60}")
    
    print("\n1. 按日期分块并切片（严格时序隔离）...")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = _split_by_date_then_slice(
        df=df,
        window_size=window_size,
        stride=stride,
        label_threshold=label_threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    print(f"   训练集: {len(X_train)} 窗口 (正样本: {Y_train.sum()})")
    print(f"   验证集: {len(X_val)} 窗口 (正样本: {Y_val.sum()})")
    print(f"   测试集: {len(X_test)} 窗口 (正样本: {Y_test.sum()})")
    print(f"   特征维度: (N, 3, {window_size})")
    
    print("\n2. 初始化 MiniRocketMultivariate...")
    rocket = MiniRocketMultivariate(num_kernels=num_kernels)
    
    print("3. 训练 Rocket 转换器...")
    rocket.fit(X_train)
    
    print("4. 提取训练特征...")
    X_train_transformed = rocket.transform(X_train)
    print(f"   特征维度: {X_train_transformed.shape}")
    
    print("5. 训练 RidgeClassifierCV 分类器...")
    cv_folds = min(5, max(2, len(Y_train) // 2))
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=cv_folds)
    classifier.fit(X_train_transformed, Y_train)
    print(f"   交叉验证折数: {cv_folds}")
    print(f"   最佳正则化参数: alpha = {classifier.alpha_}")
    
    print("\n6. 验证集评估...")
    X_val_transformed = rocket.transform(X_val)
    Y_val_pred = classifier.predict(X_val_transformed)
    evaluate_model_performance(Y_val, Y_val_pred, title_prefix="验证集")
    
    print("\n7. 测试集评估...")
    X_test_transformed = rocket.transform(X_test)
    Y_test_pred = classifier.predict(X_test_transformed)
    metrics = evaluate_model_performance(Y_test, Y_test_pred, title_prefix="测试集")
    
    print("\n8. 保存模型...")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"minirocket_{station_id}_base.pkl")
    train_data_dir = os.path.join(model_dir, f"training_data_{station_id}_base")
    os.makedirs(train_data_dir, exist_ok=True)
    
    X_train_path = os.path.join(train_data_dir, "X_train.npy")
    Y_train_path = os.path.join(train_data_dir, "Y_train.npy")
    np.save(X_train_path, X_train)
    np.save(Y_train_path, Y_train)
    
    model = {
        "rocket": rocket,
        "classifier": classifier,
        "version": "1.0.0",
        "description": f"基线模型 - 电站 {station_id}",
        "features": {
            "channels": ["pv_simulated", "GHI", "cap_power_on"],
            "window_size": window_size,
            "stride": stride,
            "label_threshold": label_threshold
        },
        "params": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "num_kernels": num_kernels,
            "random_seed": random_seed
        },
        "training_stats": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_positive": int(Y_train.sum()),
            "val_positive": int(Y_val.sum()),
            "test_positive": int(Y_test.sum())
        },
        "training_data_path": {
            "X_train": X_train_path,
            "Y_train": Y_train_path
        }
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"   模型已保存至: {model_path}")
    
    return metrics


if __name__ == "__main__":
    try:
        from .m1_raw_generator import generate_raw_station_data
        from .m2_curtail_simulator import inject_curtailment_scenarios
    except ImportError:
        from m1_raw_generator import generate_raw_station_data
        from m2_curtail_simulator import inject_curtailment_scenarios
    
    station_id = "demo_100mw"
    rated_capacity = 100.0
    
    print(f"步骤1: 生成原始数据...")
    raw_df = generate_raw_station_data(station_id, rated_capacity)
    
    print(f"\n步骤2: 注入限功率场景...")
    simulated_df = inject_curtailment_scenarios(raw_df)
    
    print(f"\n步骤3: 构建基线模型...")
    metrics = build_base_model(simulated_df, station_id)
    
    print(f"\n最终评估结果:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")