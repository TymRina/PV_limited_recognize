import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List


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


def _daytime_sliding_slice_on_df(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    label_threshold: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个DataFrame上执行日间滑动窗口切片
    
    Args:
        df: 预处理后的DataFrame（已按日期过滤）
        window_size: 窗口长度（点数）
        stride: 步长（点数）
        label_threshold: 标签聚合阈值
    
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
            X_list.append(x)
            
            curtailed_count = window_data["is_curtailed"].sum()
            y = 1 if curtailed_count >= label_threshold else 0
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
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    严格时序隔离：先按日期分块，再在块内切片
    
    策略：
    1. 获取所有唯一日期并排序
    2. 计算初始划分边界
    3. 检查每个数据集是否包含限电日期
    4. 如果某个数据集没有限电日期，调整边界确保包含至少一个限电日期
    5. 在各自日期块内进行滑动窗口切片
    6. 确保窗口不会跨越划分边界
    
    Args:
        df: 预处理后的DataFrame（含date列）
        window_size: 窗口长度
        stride: 步长
        label_threshold: 标签聚合阈值
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
    
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
    
    print(f"\n初始划分检查:")
    print(f"  训练集限电日期数: {len(train_dates & curtail_dates)}")
    print(f"  验证集限电日期数: {len(val_dates & curtail_dates)}")
    print(f"  测试集限电日期数: {len(test_dates & curtail_dates)}")
    
    if not val_has_curtail or not test_has_curtail:
        print("调整划分边界以确保每个数据集包含限电日期...")
        
        all_curtail_dates_sorted = sorted(curtail_dates)
        n_curtail = len(all_curtail_dates_sorted)
        
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
    
    assert train_max_date < val_min_date, f"训练集最大日期({train_max_date}) >= 验证集最小日期({val_min_date})"
    assert val_max_date < test_min_date, f"验证集最大日期({val_max_date}) >= 测试集最小日期({test_min_date})"
    
    print(f"\n时间轴隔离验证:")
    print(f"  训练集日期范围: {min(train_dates)} ~ {train_max_date}")
    print(f"  验证集日期范围: {val_min_date} ~ {val_max_date}")
    print(f"  测试集日期范围: {test_min_date} ~ {max(test_dates)}")
    print(f"  断言验证: 训练集最大日期 < 验证集最小日期: {train_max_date < val_min_date}")
    print(f"  断言验证: 验证集最大日期 < 测试集最小日期: {val_max_date < test_min_date}")
    
    print(f"\n限电日期分配:")
    print(f"  训练集限电日期数: {len(train_dates & curtail_dates)}")
    print(f"  验证集限电日期数: {len(val_dates & curtail_dates)}")
    print(f"  测试集限电日期数: {len(test_dates & curtail_dates)}")
    
    train_df = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)
    val_df = df[df["date"].isin(val_dates)].copy().reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].copy().reset_index(drop=True)
    
    X_train, Y_train = _daytime_sliding_slice_on_df(
        train_df, window_size, stride, label_threshold
    )
    X_val, Y_val = _daytime_sliding_slice_on_df(
        val_df, window_size, stride, label_threshold
    )
    X_test, Y_test = _daytime_sliding_slice_on_df(
        test_df, window_size, stride, label_threshold
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def prepare_minirocket_datasets(
    file_path: str = "./curtailed_data/curtailed_data.csv",
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, np.ndarray]:
    """
    完整的数据准备流程：加载数据 -> 派生特征 -> 按日期分块 -> 在块内切片
    
    Args:
        file_path: 限功率数据CSV路径
        window_size: 窗口长度（默认32点，代表8小时）
        stride: 滑动步长（默认8点）
        label_threshold: 标签聚合阈值（默认4点=1小时）
        train_ratio: 训练集日期比例
        val_ratio: 验证集日期比例
    
    Returns:
        包含训练/验证/测试集的字典
    """
    print("1. 加载并预处理数据...")
    df = load_and_preprocess_data(file_path)
    
    print("2. 派生基础特征通道（3通道）...")
    df = derive_features(df)
    
    print("3. 按日期分块并在块内切片（严格时序隔离）...")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_by_date_then_slice(
        df=df,
        window_size=window_size,
        stride=stride,
        label_threshold=label_threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    print(f"\n数据准备完成:")
    print(f"  训练集: {len(X_train)} 个窗口 (正样本: {Y_train.sum()})")
    print(f"  验证集: {len(X_val)} 个窗口 (正样本: {Y_val.sum()})")
    print(f"  测试集: {len(X_test)} 个窗口 (正样本: {Y_test.sum()})")
    print(f"  特征维度: (N, 3, {window_size})")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test
    }


if __name__ == "__main__":
    datasets = prepare_minirocket_datasets(
        file_path="./curtailed_data/curtailed_data.csv",
        window_size=32,
        stride=8,
        label_threshold=4
    )