import numpy as np
import pandas as pd
import os
from typing import Tuple, Optional, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def derive_features(df: pd.DataFrame, is_labeled: bool = False) -> pd.DataFrame:
    df = df.copy()
    
    if is_labeled:
        df["pv_simulated"] = df["pv_simulated"].astype(float)
        df["is_curtailed"] = df["is_curtailed"].astype(int)
    else:
        df["pv_simulated"] = df["pv_data"].astype(float)
        df["is_curtailed"] = 0
    
    df["GHI"] = df["GHI"].astype(float)
    df["cap_power_on"] = df["cap_power_on"].astype(float)
    
    df["date"] = df["timestamp"].dt.date
    
    return df


def _daytime_sliding_slice_on_df(
    df: pd.DataFrame,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    start_hour: int = 9,
    end_hour: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    channels = ["pv_simulated", "GHI", "cap_power_on"]
    
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    
    X_list = []
    Y_list = []
    
    unique_dates = df["date"].unique()
    
    for date in unique_dates:
        day_data = df[df["date"] == date].copy()
        
        daytime_mask = (day_data["hour"] >= start_hour) & (day_data["hour"] < end_hour)
        
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
            
            X_list.append(x)
            Y_list.append(y)
    
    if len(X_list) == 0:
        return np.array([]), np.array([])
    
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    return X, Y


def split_by_date_then_slice(
    df: pd.DataFrame,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    start_hour: int = 9,
    end_hour: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def prepare_minirocket_datasets(
    df: pd.DataFrame,
    is_labeled: bool = False,
    window_size: int = 32,
    stride: int = 8,
    label_threshold: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, np.ndarray]:
    print("  派生基础特征通道（3通道）...")
    df = derive_features(df, is_labeled=is_labeled)
    
    print("  按日期分块并在块内切片（严格时序隔离）...")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_by_date_then_slice(
        df=df,
        window_size=window_size,
        stride=stride,
        label_threshold=label_threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    print(f"  训练集: {len(X_train)} 个窗口 (正样本: {Y_train.sum()})")
    print(f"  验证集: {len(X_val)} 个窗口 (正样本: {Y_val.sum()})")
    print(f"  测试集: {len(X_test)} 个窗口 (正样本: {Y_test.sum()})")
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_val": Y_val,
        "Y_test": Y_test
    }
