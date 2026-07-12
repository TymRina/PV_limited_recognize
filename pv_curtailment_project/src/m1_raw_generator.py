import numpy as np
import pandas as pd
import os
import random
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_raw_station_data(
    station_id: str,
    rated_capacity: float,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    interval_minutes: int = 15,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    根据输入的容量生成纯净无异常的时序数据
    
    物理模型：
    - GHI: 基于正弦曲线的日变化模式，叠加季节效应和天气扰动
    - pv_data: GHI × 效率因子 × 容量缩放比例
    - cap_power_on: 白天≈额定容量，夜间=0
    
    Args:
        station_id: 电站ID
        rated_capacity: 额定容量(MW)
        start_date: 开始日期
        end_date: 结束日期
        interval_minutes: 采样间隔(分钟)
        random_seed: 随机种子
        
    Returns:
        DataFrame: 包含 ['timestamp', 'pv_data', 'GHI', 'cap_power_on'] 的时序数据
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq=f"{interval_minutes}min")
    n_samples = len(timestamps)
    
    pv_data = np.zeros(n_samples)
    ghi_data = np.zeros(n_samples)
    cap_power_on = np.zeros(n_samples)
    
    base_ghi_max = 1000.0
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60.0
        day_of_year = ts.dayofyear
        is_weekend = ts.weekday() >= 5
        
        season_factor = 0.6 + 0.4 * np.cos(2 * np.pi * (day_of_year - 81) / 365)
        
        if 5 <= ts.hour < 19:
            sunrise = 6.0 + 0.5 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
            sunset = 18.0 - 0.5 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
            
            if sunrise <= hour <= sunset:
                day_length = sunset - sunrise
                midday = (sunrise + sunset) / 2
                hour_angle = np.pi * (hour - midday) / (day_length / 2)
                
                base_ghi = base_ghi_max * np.cos(hour_angle) * season_factor
                
                weather_noise = np.random.normal(0, 0.08)
                ghi_data[i] = min(base_ghi_max, max(0, base_ghi * (1 + weather_noise)))
                
                pv_data[i] = min(rated_capacity, ghi_data[i] / base_ghi_max * rated_capacity)
                
                cap_power_on[i] = rated_capacity * (0.98 + np.random.normal(0, 0.01))
            else:
                ghi_data[i] = 0.0
                pv_data[i] = 0.0
                cap_power_on[i] = 0.0
        else:
            ghi_data[i] = 0.0
            pv_data[i] = 0.0
            cap_power_on[i] = 0.0
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "pv_data": np.round(pv_data, 4),
        "GHI": np.round(ghi_data, 2),
        "cap_power_on": np.round(cap_power_on, 4)
    })
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def save_raw_station_data(
    df: pd.DataFrame,
    station_id: str,
    rated_capacity: float
) -> str:
    """
    保存原始电站数据到文件系统
    
    Args:
        df: 原始数据DataFrame
        station_id: 电站ID
        rated_capacity: 额定容量(MW)
        
    Returns:
        str: 保存路径
    """
    output_dir = os.path.join(
        PROJECT_ROOT, "data", "01_raw",
        f"station_{station_id}",
        f"capacity_{rated_capacity}MW"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "merged_data.csv")
    df.to_csv(output_path, index=False)
    
    return output_path


if __name__ == "__main__":
    station_id = "demo_100mw"
    rated_capacity = 100.0
    
    print(f"生成电站 {station_id} 原始数据 (容量: {rated_capacity}MW)...")
    raw_df = generate_raw_station_data(station_id, rated_capacity)
    
    print(f"数据长度: {len(raw_df)}")
    print(f"时间范围: {raw_df['timestamp'].min()} ~ {raw_df['timestamp'].max()}")
    print(f"GHI范围: {raw_df['GHI'].min():.2f} ~ {raw_df['GHI'].max():.2f}")
    print(f"PV范围: {raw_df['pv_data'].min():.4f} ~ {raw_df['pv_data'].max():.4f}")
    print(f"Cap范围: {raw_df['cap_power_on'].min():.4f} ~ {raw_df['cap_power_on'].max():.4f}")
    
    save_path = save_raw_station_data(raw_df, station_id, rated_capacity)
    print(f"\n数据已保存至: {save_path}")