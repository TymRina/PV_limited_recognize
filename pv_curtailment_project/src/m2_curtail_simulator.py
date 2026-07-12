import numpy as np
import pandas as pd
import random
from typing import List, Optional

RANDOM_SEED = 42


def _generate_markov_states(
    df: pd.DataFrame,
    p_01: float = 0.06,
    p_11: float = 0.75,
    weekend_multiplier: float = 2.2
) -> pd.DataFrame:
    """
    使用马尔可夫链生成限电状态序列
    
    Args:
        df: 原始数据DataFrame
        p_01: 从正常状态转移到限电状态的概率
        p_11: 限电状态持续的概率
        weekend_multiplier: 周末触发概率乘数
        
    Returns:
        DataFrame: 包含日期和限电状态的结果
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    daily_stats = df.groupby("date")["GHI"].agg(["max", "median"]).reset_index()
    
    ghi_median = daily_stats["median"].median()
    dates = daily_stats["date"].tolist()
    n_days = len(dates)
    states = np.zeros(n_days, dtype=int)
    
    current_state = 0
    for i in range(n_days):
        date = dates[i]
        is_weekend = date.weekday() >= 5
        
        if current_state == 0:
            p_01_effective = p_01
            if is_weekend:
                p_01_effective *= weekend_multiplier
            
            if random.random() < p_01_effective:
                proposed_state = 1
            else:
                proposed_state = 0
        else:
            if random.random() < p_11:
                proposed_state = 1
            else:
                proposed_state = 0
        
        day_ghi_max = daily_stats.loc[i, "max"]
        if proposed_state == 1 and day_ghi_max <= ghi_median:
            proposed_state = 0
        
        states[i] = proposed_state
        current_state = proposed_state
    
    return pd.DataFrame({
        "date": dates,
        "curtailment_state": states
    })


def _calculate_thermal_drift(
    day_start_idx: int,
    current_idx: int,
    day_end_idx: int,
    thermal_drift_min: float = -1.5,
    thermal_drift_max: float = -0.5
) -> float:
    """
    计算温漂效应
    
    Args:
        day_start_idx: 当日起始索引
        current_idx: 当前索引
        day_end_idx: 当日结束索引
        thermal_drift_min: 温漂最小值
        thermal_drift_max: 温漂最大值
        
    Returns:
        float: 温漂值
    """
    total_points = day_end_idx - day_start_idx + 1
    if total_points <= 1:
        return 0.0
    progress = (current_idx - day_start_idx) / total_points
    drift_slope = random.uniform(thermal_drift_min, thermal_drift_max)
    return progress * drift_slope


def _inject_ghi_variation_noise(
    original_limit: float,
    ghi_value: float,
    ghi_noise_std_factor: float = 0.02,
    asymmetric_spike_prob: float = 0.15,
    asymmetric_spike_magnitude: float = 0.03
) -> float:
    """
    基于GHI变差注入非对称向下毛刺噪声
    
    Args:
        original_limit: 原始限电功率上限
        ghi_value: 当前时刻GHI值
        ghi_noise_std_factor: GHI噪声标准差因子
        asymmetric_spike_prob: 非对称毛刺概率
        asymmetric_spike_magnitude: 非对称毛刺幅度
        
    Returns:
        float: 注入噪声后的限电功率上限
    """
    noise_std = ghi_value * ghi_noise_std_factor
    
    if random.random() < asymmetric_spike_prob:
        noise = -abs(np.random.normal(0, noise_std)) * asymmetric_spike_magnitude
    else:
        noise = np.random.normal(0, noise_std * 0.5)
    
    return max(0, original_limit + noise)


def _is_protected_period(
    row,
    start_hour: int = 9,
    end_hour: int = 16
) -> bool:
    """
    判断是否为保护时段（非限电时段）
    
    Args:
        row: 数据行
        start_hour: 限电开始小时
        end_hour: 限电结束小时
        
    Returns:
        bool: 是否为保护时段
    """
    if row.GHI <= 0 or row.pv_data <= 0:
        return True
    hour = row.timestamp.hour
    if hour < start_hour or hour >= end_hour:
        return True
    return False


def inject_curtailment_scenarios(
    raw_df: pd.DataFrame,
    curtail_ratios: Optional[List[float]] = None,
    random_seed: int = RANDOM_SEED,
    p_01: float = 0.06,
    p_11: float = 0.75,
    weekend_multiplier: float = 2.2,
    mild_curtail_min_ratio: float = 0.85,
    mild_curtail_max_ratio: float = 0.95,
    thermal_drift_min: float = -1.5,
    thermal_drift_max: float = -0.5,
    ghi_noise_std_factor: float = 0.02,
    asymmetric_spike_prob: float = 0.15,
    asymmetric_spike_magnitude: float = 0.03,
    start_hour: int = 9,
    end_hour: int = 16
) -> pd.DataFrame:
    """
    注入限功率场景到原始数据中
    
    核心逻辑：
    1. 使用马尔可夫链状态机采样限电日期
    2. 对限电日期施加等比例限电
    3. 注入慢变温漂效应
    4. 轻微限电时注入基于GHI变差的向下毛刺噪声
    
    Args:
        raw_df: 原始数据DataFrame，需包含 ['timestamp', 'pv_data', 'GHI', 'cap_power_on']
        curtail_ratios: 限电比例列表，默认 [0.3, 0.5, 0.7, 0.85, 0.95]
        random_seed: 随机种子
        p_01: 马尔可夫链转移概率(0→1)
        p_11: 马尔可夫链转移概率(1→1)
        weekend_multiplier: 周末触发概率乘数
        mild_curtail_min_ratio: 轻微限电最小比例
        mild_curtail_max_ratio: 轻微限电最大比例
        thermal_drift_min: 温漂最小值
        thermal_drift_max: 温漂最大值
        ghi_noise_std_factor: GHI噪声标准差因子
        asymmetric_spike_prob: 非对称毛刺概率
        asymmetric_spike_magnitude: 非对称毛刺幅度
        start_hour: 限电开始小时
        end_hour: 限电结束小时
        
    Returns:
        DataFrame: 包含 ['pv_simulated', 'is_curtailed'] 列的新数据
    """
    if curtail_ratios is None:
        curtail_ratios = [0.3, 0.5, 0.7, 0.85, 0.95]
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    df = raw_df.copy()
    df["date"] = df["timestamp"].dt.date
    
    markov_result = _generate_markov_states(df, p_01, p_11, weekend_multiplier)
    curtailment_dates = set(markov_result[markov_result["curtailment_state"] == 1]["date"].tolist())
    
    df["pv_simulated"] = df["pv_data"].astype(float)
    df["is_curtailed"] = 0
    
    for date in curtailment_dates:
        day_mask = df["date"] == date
        day_data = df[day_mask].copy()
        
        curtail_ratio = random.choice(curtail_ratios)
        curtailment_level = int(curtail_ratio * 100)
        
        is_mild_curtail = (curtail_ratio >= mild_curtail_min_ratio and 
                          curtail_ratio <= mild_curtail_max_ratio)
        
        curtailment_mask = ~day_data.apply(
            lambda row: _is_protected_period(row, start_hour, end_hour),
            axis=1
        )
        
        if not curtailment_mask.any():
            continue
        
        curtailment_indices = day_data[curtailment_mask].index.tolist()
        day_start_idx = curtailment_indices[0]
        day_end_idx = curtailment_indices[-1]
        
        for idx in curtailment_indices:
            drift = _calculate_thermal_drift(
                day_start_idx, idx, day_end_idx,
                thermal_drift_min, thermal_drift_max
            )
            effective_level = curtailment_level + drift
            effective_level = max(0, min(100, effective_level))
            
            cap = day_data.loc[idx, "cap_power_on"]
            limit_value = cap * effective_level / 100
            
            if is_mild_curtail:
                ghi_value = day_data.loc[idx, "GHI"]
                limit_value = _inject_ghi_variation_noise(
                    limit_value, ghi_value,
                    ghi_noise_std_factor,
                    asymmetric_spike_prob,
                    asymmetric_spike_magnitude
                )
            
            original_pv = day_data.loc[idx, "pv_data"]
            
            df.loc[idx, "pv_simulated"] = min(original_pv, limit_value)
            df.loc[idx, "is_curtailed"] = 1 if original_pv > limit_value else 0
    
    df = df.drop(columns=["date"])
    
    return df


if __name__ == "__main__":
    try:
        from .m1_raw_generator import generate_raw_station_data
    except ImportError:
        from m1_raw_generator import generate_raw_station_data
    
    station_id = "demo_100mw"
    rated_capacity = 100.0
    
    print(f"加载原始数据...")
    raw_df = generate_raw_station_data(station_id, rated_capacity)
    
    print(f"注入限功率场景...")
    simulated_df = inject_curtailment_scenarios(raw_df)
    
    print(f"\n模拟结果统计:")
    print(f"  总记录数: {len(simulated_df)}")
    print(f"  限电样本数: {simulated_df['is_curtailed'].sum()}")
    print(f"  限电比例: {simulated_df['is_curtailed'].mean() * 100:.2f}%")
    
    curtailed_dates = simulated_df[simulated_df["is_curtailed"] == 1]["timestamp"].dt.date.nunique()
    print(f"  限电天数: {curtailed_dates}")