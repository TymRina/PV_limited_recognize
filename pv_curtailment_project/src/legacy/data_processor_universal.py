import numpy as np
import pandas as pd
import random
import os
from typing import Tuple, Optional, Dict, List
from scipy.interpolate import interp1d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    加载CSV数据并预处理DatetimeIndex
    """
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    派生基础特征通道，构建3通道多变量时序数据
    
    严格限制为最原始的通道，杜绝泄漏特征：
    - 通道0: pv_simulated（限电后的实际功率）
    - 通道1: GHI（总辐照度）
    - 通道2: cap_power_on（开机容量）
    """
    df = df.copy()
    
    df["pv_simulated"] = df["curtailed_pv"].astype(float)
    df["GHI"] = df["GHI"].astype(float)
    df["cap_power_on"] = df["cap_power_on"].astype(float)
    
    df["is_curtailed"] = df["curtailment_label"].astype(int)
    df["date"] = df["timestamp"].dt.date
    
    return df


def _apply_time_warping(
    window_data: np.ndarray,
    warping_factor: float,
    target_length: int = 32
) -> np.ndarray:
    """
    对时序窗口进行时间轴拉伸/压缩变形
    
    物理逻辑：
    1. 现实中限电时长各异，需要模型对任意限电时长通用
    2. 对限电活跃时段进行随机的时间轴拉伸(>1)或压缩(<1)
    3. 通过线性插值保持序列长度始终为32点
    4. 缩放因子范围：0.6 ~ 1.4（覆盖广泛的时长变化）
    
    Args:
        window_data: 原始窗口数据 (3, window_length)
        warping_factor: 时间轴缩放因子
        target_length: 目标序列长度（默认32点）
    
    Returns:
        变形后的窗口数据 (3, target_length)
    """
    n_channels, original_length = window_data.shape
    
    if warping_factor == 1.0 or original_length == target_length:
        if original_length != target_length:
            t_original = np.linspace(0, 1, original_length)
            t_target = np.linspace(0, 1, target_length)
            warped_data = np.zeros((n_channels, target_length))
            for i in range(n_channels):
                f = interp1d(t_original, window_data[i], kind='linear')
                warped_data[i] = f(t_target)
            return warped_data
        return window_data.copy()
    
    t_original = np.linspace(0, 1, original_length)
    
    if warping_factor > 1.0:
        t_warped = np.linspace(0, warping_factor, original_length)
        t_target = np.linspace(0, warping_factor, target_length)
    else:
        t_warped = np.linspace(0, warping_factor, original_length)
        t_target = np.linspace(0, warping_factor, target_length)
    
    warped_data = np.zeros((n_channels, target_length))
    
    for i in range(n_channels):
        f = interp1d(t_warped, window_data[i], kind='linear', bounds_error=False, fill_value='extrapolate')
        warped_data[i] = f(t_target)
    
    return warped_data


def _daytime_sliding_slice_on_df(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    label_threshold: int,
    apply_time_warping_prob: float = 0.5,
    warping_factor_range: Tuple[float, float] = (0.6, 1.4)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个DataFrame上执行日间滑动窗口切片（含时序变形）
    
    核心改进：
    1. 对部分限电窗口内的限电活跃时段进行随机时间轴变形
    2. 通过线性插值保持序列长度依然为32点
    3. 确保模型对任意限电时长通用
    
    Args:
        df: 预处理后的DataFrame（已按日期过滤）
        window_size: 窗口长度（点数）
        stride: 步长（点数）
        label_threshold: 标签聚合阈值
        apply_time_warping_prob: 应用时序变形的概率（默认50%）
        warping_factor_range: 缩放因子范围（默认0.6~1.4）
    
    Returns:
        X: 特征张量 (N_samples, 3, Window_Length)
        Y: 标签数组 (N,)
    """
    channels = ["pv_simulated", "GHI", "cap_power_on"]
    
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    
    X_list = []
    Y_list = []
    
    unique_dates = df["date"].unique()
    
    for date in unique_dates:
        day_data = df[df["date"] == date].copy()
        
        daytime_mask = (day_data["hour"] >= 9) & (day_data["hour"] < 16)
        
        if not daytime_mask.any():
            continue
        
        first_daytime_idx = day_data[daytime_mask].index.min()
        last_daytime_idx = day_data[daytime_mask].index.max()
        
        day_start_idx = day_data.index.min()
        day_end_idx = day_data.index.max()
        
        start_search_idx = max(day_start_idx, first_daytime_idx - window_size + 1)
        end_search_idx = min(day_end_idx, last_daytime_idx + 1)
        
        for start_idx in range(start_search_idx, end_search_idx - window_size + 1, stride):
            end_idx = start_idx + window_size
            
            if end_idx > len(df):
                break
            
            window_data = df.iloc[start_idx:end_idx]
            
            x = window_data[channels].values.T
            
            curtailed_count = window_data["is_curtailed"].sum()
            y = 1 if curtailed_count >= label_threshold else 0
            
            if y == 1 and random.random() < apply_time_warping_prob:
                warping_factor = random.uniform(warping_factor_range[0], warping_factor_range[1])
                x = _apply_time_warping(x, warping_factor, window_size)
            
            X_list.append(x)
            Y_list.append(y)
    
    if len(X_list) == 0:
        return np.array([]), np.array([])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    return X, Y


def split_by_date_then_slice(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    label_threshold: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    apply_time_warping_prob: float = 0.5,
    warping_factor_range: Tuple[float, float] = (0.6, 1.4)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    严格时序隔离：先按日期分块，再在块内切片（含时序变形）
    
    策略：
    1. 获取所有唯一日期并排序
    2. 计算初始划分边界
    3. 检查每个数据集是否包含限电日期
    4. 如果某个数据集没有限电日期，调整边界确保包含至少一个限电日期
    5. 在各自日期块内进行滑动窗口切片（对限电窗口应用时序变形）
    6. 确保窗口不会跨越划分边界
    7. 确保测试集中包含0.85~0.95的轻微限电样本
    
    Args:
        df: 预处理后的DataFrame（含date列）
        window_size: 窗口长度
        stride: 步长
        label_threshold: 标签聚合阈值
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
        apply_time_warping_prob: 应用时序变形的概率
        warping_factor_range: 缩放因子范围
    
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    
    curtail_dates = set(df[df["is_curtailed"] == 1]["date"].unique())
    
    mild_curtail_dates = set(df[
        (df["is_curtailed"] == 1) & 
        (df["curtail_ratio"] >= 0.85) & 
        (df["curtail_ratio"] <= 0.95)
    ]["date"].unique())
    
    train_date_end = int(n_dates * train_ratio)
    val_date_end = train_date_end + int(n_dates * val_ratio)
    
    train_dates = set(unique_dates[:train_date_end])
    val_dates = set(unique_dates[train_date_end:val_date_end])
    test_dates = set(unique_dates[val_date_end:])
    
    train_has_curtail = len(train_dates & curtail_dates) > 0
    val_has_curtail = len(val_dates & curtail_dates) > 0
    test_has_curtail = len(test_dates & curtail_dates) > 0
    
    train_has_mild = len(train_dates & mild_curtail_dates) > 0
    val_has_mild = len(val_dates & mild_curtail_dates) > 0
    test_has_mild = len(test_dates & mild_curtail_dates) > 0
    
    print(f"\n初始划分检查:")
    print(f"  训练集限电日期数: {len(train_dates & curtail_dates)} (轻微限电: {len(train_dates & mild_curtail_dates)})")
    print(f"  验证集限电日期数: {len(val_dates & curtail_dates)} (轻微限电: {len(val_dates & mild_curtail_dates)})")
    print(f"  测试集限电日期数: {len(test_dates & curtail_dates)} (轻微限电: {len(test_dates & mild_curtail_dates)})")
    
    if not val_has_curtail or not test_has_curtail or not test_has_mild:
        print("调整划分边界以确保每个数据集包含限电日期，特别是测试集包含轻微限电样本...")
        
        all_curtail_dates_sorted = sorted(curtail_dates)
        n_curtail = len(all_curtail_dates_sorted)
        
        all_mild_dates_sorted = sorted(mild_curtail_dates)
        n_mild = len(all_mild_dates_sorted)
        
        train_curtail_end = max(1, int(n_curtail * train_ratio))
        val_curtail_end = train_curtail_end + max(1, int(n_curtail * val_ratio))
        
        last_train_curtail = all_curtail_dates_sorted[train_curtail_end - 1]
        last_val_curtail = all_curtail_dates_sorted[val_curtail_end - 1]
        
        test_mild_min_date = None
        if n_mild >= 1:
            test_mild_idx = max(0, int(n_mild * (train_ratio + val_ratio)))
            test_mild_min_date = all_mild_dates_sorted[test_mild_idx]
        
        train_date_end = None
        val_date_end = None
        
        for i, date in enumerate(unique_dates):
            if date > last_train_curtail and train_date_end is None:
                train_date_end = i
            if date > last_val_curtail and val_date_end is None:
                val_date_end = i
        
        if test_mild_min_date is not None and val_date_end is not None:
            for i, date in enumerate(unique_dates):
                if date >= test_mild_min_date and i > val_date_end:
                    val_date_end = i
                    break
        
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
    
    assert train_max_date < val_min_date, f"训练集最大日期({train_max_date}) >= 验证集最小日期({val_min_date})"
    assert val_max_date < test_min_date, f"验证集最大日期({val_max_date}) >= 测试集最小日期({test_min_date})"
    
    print(f"\n时间轴隔离验证:")
    print(f"  训练集日期范围: {min(train_dates)} ~ {train_max_date}")
    print(f"  验证集日期范围: {val_min_date} ~ {val_max_date}")
    print(f"  测试集日期范围: {test_min_date} ~ {max(test_dates)}")
    print(f"  断言验证: 训练集最大日期 < 验证集最小日期: {train_max_date < val_min_date}")
    print(f"  断言验证: 验证集最大日期 < 测试集最小日期: {val_max_date < test_min_date}")
    
    print(f"\n限电日期分配:")
    print(f"  训练集限电日期数: {len(train_dates & curtail_dates)} (轻微限电: {len(train_dates & mild_curtail_dates)})")
    print(f"  验证集限电日期数: {len(val_dates & curtail_dates)} (轻微限电: {len(val_dates & mild_curtail_dates)})")
    print(f"  测试集限电日期数: {len(test_dates & curtail_dates)} (轻微限电: {len(test_dates & mild_curtail_dates)})")
    
    train_df = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)
    val_df = df[df["date"].isin(val_dates)].copy().reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].copy().reset_index(drop=True)
    
    X_train, Y_train = _daytime_sliding_slice_on_df(
        train_df, window_size, stride, label_threshold,
        apply_time_warping_prob, warping_factor_range
    )
    X_val, Y_val = _daytime_sliding_slice_on_df(
        val_df, window_size, stride, label_threshold,
        apply_time_warping_prob, warping_factor_range
    )
    X_test, Y_test = _daytime_sliding_slice_on_df(
        test_df, window_size, stride, label_threshold,
        apply_time_warping_prob, warping_factor_range
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def prepare_minirocket_datasets_universal(
    file_path: str = os.path.join(PROJECT_ROOT, "data", "02_simulated", "curtailed_data_v2_universal.csv"),
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    apply_time_warping_prob: float = 0.5,
    warping_factor_range: Tuple[float, float] = (0.6, 1.4)
) -> Dict[str, np.ndarray]:
    """
    通用版数据准备流程：加载数据 -> 派生特征 -> 按日期分块 -> 在块内切片（含时序变形）
    
    核心改进：
    1. 时序窗口变形：对部分限电窗口进行时间轴拉伸/压缩
    2. 线性插值保持序列长度为32点
    3. 确保测试集包含0.85~0.95的轻微限电样本
    
    Args:
        file_path: 通用限功率数据CSV路径
        window_size: 窗口长度（默认32点，代表8小时）
        stride: 滑动步长（默认8点）
        label_threshold: 标签聚合阈值（默认4点=1小时）
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
        apply_time_warping_prob: 应用时序变形的概率
        warping_factor_range: 缩放因子范围
    
    Returns:
        包含训练/验证/测试集的字典
    """
    print("1. 加载并预处理数据...")
    df = load_and_preprocess_data(file_path)
    
    print("2. 派生基础特征通道（3通道）...")
    df = derive_features(df)
    
    print("3. 按日期分块并在块内切片（严格时序隔离 + 时序变形）...")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_by_date_then_slice(
        df=df,
        window_size=window_size,
        stride=stride,
        label_threshold=label_threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        apply_time_warping_prob=apply_time_warping_prob,
        warping_factor_range=warping_factor_range
    )
    
    print(f"\n数据准备完成:")
    print(f"  训练集: {len(X_train)} 个窗口 (正样本: {Y_train.sum()})")
    print(f"  验证集: {len(X_val)} 个窗口 (正样本: {Y_val.sum()})")
    print(f"  测试集: {len(X_test)} 个窗口 (正样本: {Y_test.sum()})")
    print(f"  特征维度: (N, 3, {window_size})")
    print(f"  时序变形概率: {apply_time_warping_prob * 100:.0f}%")
    print(f"  变形缩放因子范围: {warping_factor_range[0]} ~ {warping_factor_range[1]}")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test
    }


if __name__ == "__main__":
    datasets = prepare_minirocket_datasets_universal(
        file_path=os.path.join(PROJECT_ROOT, "data", "02_simulated", "curtailed_data_v2_universal.csv"),
        window_size=32,
        stride=8,
        label_threshold=4,
        apply_time_warping_prob=0.5,
        warping_factor_range=(0.6, 1.4)
    )