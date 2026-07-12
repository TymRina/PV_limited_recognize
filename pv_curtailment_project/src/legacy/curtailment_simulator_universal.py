import numpy as np
import pandas as pd
import os
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class UniversalCurtailmentConfig:
    raw_data_path: str = os.path.join(PROJECT_ROOT, "data", "01_raw", "station_legacy_v1", "capacity_100MW")
    curtail_ratios: List[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
    start_hour: int = 9
    end_hour: int = 16
    thermal_drift_min: float = -1.5
    thermal_drift_max: float = -0.5
    random_seed: Optional[int] = None
    p_01: float = 0.06
    p_11: float = 0.75
    weekend_multiplier: float = 2.2
    mild_curtail_min_ratio: float = 0.85
    mild_curtail_max_ratio: float = 0.95
    ghi_noise_std_factor: float = 0.02
    asymmetric_spike_prob: float = 0.15
    asymmetric_spike_magnitude: float = 0.03


class UniversalCurtailmentSimulator:
    def __init__(self, config: Optional[UniversalCurtailmentConfig] = None):
        self.config = config or UniversalCurtailmentConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        self.data: Optional[pd.DataFrame] = None
        self.curtailment_dates: List[pd.Timestamp] = []

    def _load_single_csv(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.config.raw_data_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        df = pd.read_csv(filepath)
        df = df.rename(columns={"dtime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def load_and_merge_data(self) -> pd.DataFrame:
        try:
            pv_df = self._load_single_csv("pv_history.csv")
            weather_df = self._load_single_csv("weather_future.csv")
            cap_df = self._load_single_csv("cap_power_on.csv")
        except FileNotFoundError:
            merged_path = os.path.join(self.config.raw_data_path, "merged_data.csv")
            if os.path.exists(merged_path):
                self.data = pd.read_csv(merged_path)
                self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
                return self.data
            raise

        self.data = pv_df.merge(weather_df, on="timestamp", how="outer")
        self.data = self.data.merge(cap_df, on="timestamp", how="outer")
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)
        return self.data

    def _get_daily_ghi_stats(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("数据未加载，请先调用 load_and_merge_data()")

        daily = self.data.copy()
        daily["date"] = daily["timestamp"].dt.date
        daily_stats = daily.groupby("date")["GHI"].agg(["max", "median"]).reset_index()
        return daily_stats

    def _generate_markov_states(self) -> pd.DataFrame:
        daily_stats = self._get_daily_ghi_stats()
        ghi_median = daily_stats["median"].median()

        dates = daily_stats["date"].tolist()
        n_days = len(dates)
        states = np.zeros(n_days, dtype=int)

        current_state = 0
        for i in range(n_days):
            date = dates[i]
            is_weekend = date.weekday() >= 5

            if current_state == 0:
                p_01_effective = self.config.p_01
                if is_weekend:
                    p_01_effective *= self.config.weekend_multiplier

                if random.random() < p_01_effective:
                    proposed_state = 1
                else:
                    proposed_state = 0
            else:
                if random.random() < self.config.p_11:
                    proposed_state = 1
                else:
                    proposed_state = 0

            day_ghi_max = daily_stats.loc[i, "max"]
            if proposed_state == 1 and day_ghi_max <= ghi_median:
                proposed_state = 0

            states[i] = proposed_state
            current_state = proposed_state

        result = pd.DataFrame({
            "date": dates,
            "curtailment_state": states
        })
        return result

    def sample_curtailment_days(self) -> List[pd.Timestamp]:
        markov_result = self._generate_markov_states()
        self.curtailment_dates = markov_result[markov_result["curtailment_state"] == 1]["date"].tolist()
        return self.curtailment_dates

    def _is_protected_period(self, row) -> bool:
        if row.GHI <= 0 or row.pv_data <= 0:
            return True
        hour = row.timestamp.hour
        if hour < self.config.start_hour or hour >= self.config.end_hour:
            return True
        return False

    def _calculate_thermal_drift(self, day_start_idx: int, current_idx: int, day_end_idx: int) -> float:
        total_points = day_end_idx - day_start_idx + 1
        if total_points <= 1:
            return 0.0
        progress = (current_idx - day_start_idx) / total_points
        drift_slope = random.uniform(self.config.thermal_drift_min, self.config.thermal_drift_max)
        drift = progress * drift_slope
        return drift

    def _inject_ghi_variation_noise(self, original_limit: float, ghi_value: float) -> float:
        """
        基于GHI变差注入非对称向下毛刺噪声
        
        物理逻辑：
        - 轻微限电时，实际功率接近上限，微小的辐照度波动会被放大
        - 非对称：向下毛刺概率更高，模拟系统响应延迟导致的瞬时功率下降
        - 噪声幅度与GHI成正比，模拟真实物理环境的扰动

        Args:
            original_limit: 原始限电功率上限
            ghi_value: 当前时刻GHI值

        Returns:
            注入噪声后的限电功率上限
        """
        noise_std = ghi_value * self.config.ghi_noise_std_factor
        
        if random.random() < self.config.asymmetric_spike_prob:
            noise = -abs(np.random.normal(0, noise_std)) * self.config.asymmetric_spike_magnitude
        else:
            noise = np.random.normal(0, noise_std * 0.5)
        
        return max(0, original_limit + noise)

    def _apply_curtailment_to_day(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """
        对单日数据施加限电功率
        
        核心改进：
        1. 限电比例拓展至 [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        2. 轻微限电(0.85~0.95)时注入微观温漂和GHI变差噪声
        3. 逼迫模型在极度微弱的平顶形态下提取差分特征

        Args:
            day_data: 单日数据

        Returns:
            施加限电后的单日数据
        """
        day_data = day_data.copy()
        
        curtail_ratio = random.choice(self.config.curtail_ratios)
        curtailment_level = int(curtail_ratio * 100)
        
        is_mild_curtail = (curtail_ratio >= self.config.mild_curtail_min_ratio and 
                          curtail_ratio <= self.config.mild_curtail_max_ratio)

        curtailment_mask = ~day_data.apply(self._is_protected_period, axis=1)
        if not curtailment_mask.any():
            day_data["curtailment_label"] = 0
            day_data["curtailment_level"] = 0
            day_data["curtailed_pv"] = day_data["pv_data"]
            day_data["curtail_ratio"] = 1.0
            return day_data

        curtailment_indices = day_data[curtailment_mask].index.tolist()
        day_start_idx = curtailment_indices[0]
        day_end_idx = curtailment_indices[-1]

        for idx in curtailment_indices:
            drift = self._calculate_thermal_drift(day_start_idx, idx, day_end_idx)
            effective_level = curtailment_level + drift
            effective_level = max(0, min(100, effective_level))

            cap = day_data.loc[idx, "cap_power_on"]
            limit_value = cap * effective_level / 100
            
            if is_mild_curtail:
                ghi_value = day_data.loc[idx, "GHI"]
                limit_value = self._inject_ghi_variation_noise(limit_value, ghi_value)

            original_pv = day_data.loc[idx, "pv_data"]

            day_data.loc[idx, "curtailed_pv"] = min(original_pv, limit_value)
            day_data.loc[idx, "curtailment_label"] = 1 if original_pv > limit_value else 0
            day_data.loc[idx, "curtailment_level"] = effective_level
            day_data.loc[idx, "curtail_ratio"] = effective_level / 100

        protected_mask = ~curtailment_mask
        day_data.loc[protected_mask, "curtailed_pv"] = day_data.loc[protected_mask, "pv_data"]
        day_data.loc[protected_mask, "curtailment_label"] = 0
        day_data.loc[protected_mask, "curtailment_level"] = 0
        day_data.loc[protected_mask, "curtail_ratio"] = 1.0

        return day_data

    def generate_curtailment_data(self) -> pd.DataFrame:
        """
        生成通用限电数据集
        
        生成逻辑：
        1. 加载合并后的数据
        2. 基于Markov链采样限电日期
        3. 对每个限电日期施加不同程度的限电
        4. 轻微限电时注入微观噪声，增强模型泛化能力

        Returns:
            包含限电信息的完整数据集
        """
        if self.data is None:
            raise ValueError("数据未加载，请先调用 load_and_merge_data()")
        if not self.curtailment_dates:
            self.sample_curtailment_days()

        result = self.data.copy()
        result["curtailed_pv"] = result["pv_data"].astype(float)
        result["curtailment_label"] = 0
        result["curtailment_level"] = 0.0
        result["curtail_ratio"] = 1.0

        for date in self.curtailment_dates:
            day_mask = result["timestamp"].dt.date == date
            day_data = result[day_mask].copy()
            day_data = self._apply_curtailment_to_day(day_data)
            result.loc[day_mask, ["curtailed_pv", "curtailment_label", "curtailment_level", "curtail_ratio"]] = (
                day_data[["curtailed_pv", "curtailment_label", "curtailment_level", "curtail_ratio"]].values
            )

        return result

    def save_curtailment_data(self, output_path: str = os.path.join(PROJECT_ROOT, "data", "02_simulated", "curtailed_data_v2_universal.csv")) -> None:
        """
        保存通用限电数据集

        Args:
            output_path: 输出路径，默认保存在 universal_stage/ 目录下
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = self.generate_curtailment_data()
        result.to_csv(output_path, index=False)


def main():
    """
    通用限电数据模拟引擎主入口
    
    生成覆盖全谱系限电场景的训练数据：
    - 重度限电(0.3~0.7)：明显的功率截断特征
    - 中度限电(0.8)：中等程度的功率限制
    - 轻微限电(0.85~0.95)：极度微弱的功率平顶，需注入微观噪声
    
    确保模型能够学习到从明显到微弱的各种限电形态特征。
    """
    config = UniversalCurtailmentConfig(
        raw_data_path=os.path.join(PROJECT_ROOT, "data", "01_raw", "station_legacy_v1", "capacity_100MW"),
        curtail_ratios=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        start_hour=9,
        end_hour=16,
        thermal_drift_min=-1.5,
        thermal_drift_max=-0.5,
        random_seed=42,
        p_01=0.02,
        p_11=0.85,
        weekend_multiplier=2.2,
        mild_curtail_min_ratio=0.85,
        mild_curtail_max_ratio=0.95,
        ghi_noise_std_factor=0.02,
        asymmetric_spike_prob=0.15,
        asymmetric_spike_magnitude=0.03,
    )

    simulator = UniversalCurtailmentSimulator(config)
    simulator.load_and_merge_data()
    simulator.sample_curtailment_days()
    result = simulator.generate_curtailment_data()
    simulator.save_curtailment_data()

    print(f"通用限功率数据生成完成，共 {len(result)} 条记录")
    print(f"限功率天数: {len(simulator.curtailment_dates)}")
    print(f"限功率时段数: {result['curtailment_label'].sum()}")
    
    mild_curtail_mask = (result["curtail_ratio"] >= 0.85) & (result["curtail_ratio"] <= 0.95) & (result["curtailment_label"] == 1)
    print(f"轻微限电(0.85~0.95)样本数: {mild_curtail_mask.sum()}")
    
    print("\n限电比例分布统计:")
    for ratio in config.curtail_ratios:
        mask = (result["curtail_ratio"] >= ratio - 0.02) & (result["curtail_ratio"] <= ratio + 0.02)
        count = mask.sum()
        print(f"  限电比例 {ratio:.2f}: {count} 条记录")


if __name__ == "__main__":
    main()