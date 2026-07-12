import numpy as np
import pandas as pd
import os
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CurtailmentConfig:
    raw_data_path: str = "./raw_data"
    curtailment_levels: List[int] = (30, 50, 70)
    start_hour: int = 9
    end_hour: int = 16
    thermal_drift_min: float = -1.5
    thermal_drift_max: float = -0.5
    random_seed: Optional[int] = None
    p_01: float = 0.06
    p_11: float = 0.75
    weekend_multiplier: float = 2.2


class CurtailmentSimulator:
    def __init__(self, config: Optional[CurtailmentConfig] = None):
        self.config = config or CurtailmentConfig()
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

    def _apply_curtailment_to_day(self, day_data: pd.DataFrame) -> pd.DataFrame:
        day_data = day_data.copy()
        curtailment_level = random.choice(self.config.curtailment_levels)

        curtailment_mask = ~day_data.apply(self._is_protected_period, axis=1)
        if not curtailment_mask.any():
            day_data["curtailment_label"] = 0
            day_data["curtailment_level"] = 0
            day_data["curtailed_pv"] = day_data["pv_data"]
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
            original_pv = day_data.loc[idx, "pv_data"]

            day_data.loc[idx, "curtailed_pv"] = min(original_pv, limit_value)
            day_data.loc[idx, "curtailment_label"] = 1 if original_pv > limit_value else 0
            day_data.loc[idx, "curtailment_level"] = effective_level

        protected_mask = ~curtailment_mask
        day_data.loc[protected_mask, "curtailed_pv"] = day_data.loc[protected_mask, "pv_data"]
        day_data.loc[protected_mask, "curtailment_label"] = 0
        day_data.loc[protected_mask, "curtailment_level"] = 0

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

        for date in self.curtailment_dates:
            day_mask = result["timestamp"].dt.date == date
            day_data = result[day_mask].copy()
            day_data = self._apply_curtailment_to_day(day_data)
            result.loc[day_mask, ["curtailed_pv", "curtailment_label", "curtailment_level"]] = (
                day_data[["curtailed_pv", "curtailment_label", "curtailment_level"]].values
            )

        return result

    def save_curtailment_data(self, output_path: str = "./curtailed_data.csv") -> None:
        result = self.generate_curtailment_data()
        result.to_csv(output_path, index=False)


def main():
    config = CurtailmentConfig(
        raw_data_path="./raw_data",
        curtailment_levels=[30, 50, 70],
        start_hour=9,
        end_hour=16,
        thermal_drift_min=-1.5,
        thermal_drift_max=-0.5,
        random_seed=42,
        p_01=0.02,
        p_11=0.85,
        weekend_multiplier=2.2,
    )

    simulator = CurtailmentSimulator(config)
    simulator.load_and_merge_data()
    simulator.sample_curtailment_days()
    result = simulator.generate_curtailment_data()
    simulator.save_curtailment_data()

    print(f"限功率数据生成完成，共 {len(result)} 条记录")
    print(f"限功率天数: {len(simulator.curtailment_dates)}")
    print(f"限功率时段数: {result['curtailment_label'].sum()}")


if __name__ == "__main__":
    main()
