import numpy as np
import pandas as pd
import os
import random
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class UniversalCurtailmentConfig:
    station_id: str = "default"
    rated_capacity: float = 100.0
    min_capacity: float = 60.0
    max_capacity: float = 200.0
    
    raw_data_base_path: str = os.path.join(PROJECT_ROOT, "data", "01_raw")
    simulated_data_base_path: str = os.path.join(PROJECT_ROOT, "data", "02_simulated")
    
    curtail_ratios: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
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
    
    pv_csv_name: str = "pv_history.csv"
    weather_csv_name: str = "weather_future.csv"
    cap_csv_name: str = "cap_power_on.csv"
    merged_csv_name: str = "merged_data.csv"
    
    def __post_init__(self):
        if self.rated_capacity < self.min_capacity or self.rated_capacity > self.max_capacity:
            raise ValueError(f"额定容量 {self.rated_capacity}MW 超出有效范围 [{self.min_capacity}, {self.max_capacity}]MW")


class UniversalCurtailmentSimulator:
    def __init__(self, config: Optional[UniversalCurtailmentConfig] = None):
        self.config = config or UniversalCurtailmentConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        self.data: Optional[pd.DataFrame] = None
        self.curtailment_dates: List[pd.Timestamp] = []
        
        self.station_raw_path = os.path.join(
            self.config.raw_data_base_path, 
            f"station_{self.config.station_id}",
            f"capacity_{self.config.rated_capacity}MW"
        )
        self.station_simulated_path = os.path.join(
            self.config.simulated_data_base_path, 
            f"station_{self.config.station_id}",
            f"capacity_{self.config.rated_capacity}MW"
        )
        
        os.makedirs(self.station_raw_path, exist_ok=True)
        os.makedirs(self.station_simulated_path, exist_ok=True)

    def _load_single_csv(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.station_raw_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        df = pd.read_csv(filepath)
        df = df.rename(columns={"dtime": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def load_and_merge_data(self) -> pd.DataFrame:
        try:
            pv_df = self._load_single_csv(self.config.pv_csv_name)
            weather_df = self._load_single_csv(self.config.weather_csv_name)
            cap_df = self._load_single_csv(self.config.cap_csv_name)
        except FileNotFoundError:
            merged_path = os.path.join(self.station_raw_path, self.config.merged_csv_name)
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
        noise_std = ghi_value * self.config.ghi_noise_std_factor
        
        if random.random() < self.config.asymmetric_spike_prob:
            noise = -abs(np.random.normal(0, noise_std)) * self.config.asymmetric_spike_magnitude
        else:
            noise = np.random.normal(0, noise_std * 0.5)
        
        return max(0, original_limit + noise)

    def _apply_curtailment_to_day(self, day_data: pd.DataFrame) -> pd.DataFrame:
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

    def save_curtailment_data(self, output_filename: str = "curtailed_data.csv") -> str:
        output_path = os.path.join(self.station_simulated_path, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = self.generate_curtailment_data()
        result.to_csv(output_path, index=False)
        return output_path

    def run(self) -> Dict[str, Union[str, int, float]]:
        print(f"\n{'='*60}")
        print(f"限功率模拟引擎启动")
        print(f"电站ID: {self.config.station_id}")
        print(f"额定容量: {self.config.rated_capacity}MW")
        print(f"{'='*60}")
        
        print("\n1. 加载数据...")
        self.load_and_merge_data()
        print(f"   数据记录数: {len(self.data)}")
        
        print("\n2. 采样限电日期...")
        self.sample_curtailment_days()
        print(f"   限电天数: {len(self.curtailment_dates)}")
        
        print("\n3. 生成限电数据...")
        result = self.generate_curtailment_data()
        
        print("\n4. 保存结果...")
        output_path = self.save_curtailment_data()
        print(f"   保存路径: {output_path}")
        
        mild_curtail_mask = (result["curtail_ratio"] >= 0.85) & (result["curtail_ratio"] <= 0.95) & (result["curtailment_label"] == 1)
        
        return {
            "station_id": self.config.station_id,
            "rated_capacity": self.config.rated_capacity,
            "total_records": len(result),
            "curtailment_days": len(self.curtailment_dates),
            "curtailed_periods": int(result["curtailment_label"].sum()),
            "mild_curtail_samples": int(mild_curtail_mask.sum()),
            "output_path": output_path
        }


def batch_simulate(
    configs: List[UniversalCurtailmentConfig],
    verbose: bool = True
) -> List[Dict[str, Union[str, int, float]]]:
    results = []
    for i, config in enumerate(configs):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"批次处理: {i+1}/{len(configs)}")
            print(f"{'#'*60}")
        
        try:
            simulator = UniversalCurtailmentSimulator(config)
            result = simulator.run()
            results.append(result)
            
            if verbose:
                print(f"\n处理完成:")
                print(f"  电站ID: {result['station_id']}")
                print(f"  额定容量: {result['rated_capacity']}MW")
                print(f"  限电天数: {result['curtailment_days']}")
                print(f"  限电时段: {result['curtailed_periods']}")
        except Exception as e:
            print(f"\n处理失败 [{config.station_id} - {config.rated_capacity}MW]: {str(e)}")
            results.append({
                "station_id": config.station_id,
                "rated_capacity": config.rated_capacity,
                "error": str(e)
            })
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"批次处理完成")
        print(f"{'='*60}")
        success_count = sum(1 for r in results if "error" not in r)
        print(f"成功: {success_count}/{len(configs)}")
    
    return results


def main():
    configs = [
        UniversalCurtailmentConfig(
            station_id="legacy_v1",
            rated_capacity=100,
            random_seed=42,
            p_01=0.02,
            p_11=0.85,
            weekend_multiplier=2.2
        )
    ]
    
    results = batch_simulate(configs)
    
    for result in results:
        if "error" in result:
            print(f"错误: {result['station_id']} - {result['rated_capacity']}MW: {result['error']}")
        else:
            print(f"\n统计结果 [{result['station_id']} - {result['rated_capacity']}MW]:")
            print(f"  总记录数: {result['total_records']}")
            print(f"  限电天数: {result['curtailment_days']}")
            print(f"  限电时段: {result['curtailed_periods']}")
            print(f"  轻微限电样本: {result['mild_curtail_samples']}")
            print(f"  输出路径: {result['output_path']}")


if __name__ == "__main__":
    main()
