import warnings
import numpy as np
import pandas as pd
import os
import random
import time
from typing import Dict, Optional

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.seterr(all='ignore')

warnings.filterwarnings('ignore', message='.*Ill-conditioned matrix.*')
warnings.filterwarnings('ignore', message='.*Singular matrix.*')

from src.m1_raw_generator import generate_raw_station_data, save_raw_station_data
from src.m2_curtail_simulator import inject_curtailment_scenarios
from src.m3_pretrain_engine import build_base_model
from src.m4_retrain_engine import fine_tune_with_real_data
from src.m5_evaluator import evaluate_model_performance, aggregate_metrics

RANDOM_SEED = 42


class StationModelFactory:
    """
    电站限功率识别模型工厂
    
    提供一站式冷启动训练和生产级微调能力，
    为业务方提供完整的平台化功能演示。
    """
    
    def __init__(self, station_id: str, rated_capacity: float):
        """
        初始化模型工厂
        
        Args:
            station_id: 电站唯一标识
            rated_capacity: 额定容量(MW)
        """
        self.station_id = station_id
        self.rated_capacity = rated_capacity
        self.base_model_path = os.path.join(
            os.path.dirname(__file__), "models",
            f"minirocket_{station_id}_base.pkl"
        )
        self.prod_model_path = os.path.join(
            os.path.dirname(__file__), "models",
            f"minirocket_{station_id}_prod.pkl"
        )
        
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        self._print_welcome_banner()
    
    def _print_welcome_banner(self):
        """打印工厂初始化欢迎横幅"""
        print("\n" + "="*70)
        print(f"🚀 限功率识别模型工厂 v2.0")
        print("="*70)
        print(f"  电站ID:       {self.station_id}")
        print(f"  额定容量:     {self.rated_capacity}MW")
        print(f"  随机种子:     {RANDOM_SEED}")
        print(f"  输入通道:     pv_simulated | GHI | cap_power_on")
        print(f"  窗口规格:     32点 (8小时日间)")
        print("="*70 + "\n")
    
    def _print_stage_banner(self, stage_name: str, stage_num: int, total_stages: int):
        """打印阶段横幅"""
        print(f"\n{'─'*70}")
        print(f"📌 阶段 {stage_num}/{total_stages}: {stage_name}")
        print(f"{'─'*70}")
    
    def _print_success_banner(self, message: str):
        """打印成功横幅"""
        print(f"\n✅ {message}")
    
    def _print_divider(self):
        """打印分隔线"""
        print(f"\n{'─'*70}")
    
    def run_cold_start_pipeline(
        self,
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31"
    ) -> Dict[str, float]:
        """
        冷启动全流程：从原始数据到基线模型
        
        执行流程：
        1. M1: 生成原始电站数据
        2. M2: 注入限功率场景
        3. M3: 隔离切片并训练基线模型
        4. M5: 打印基线评估报告
        
        Args:
            start_date: 数据开始日期
            end_date: 数据结束日期
            
        Returns:
            Dict[str, float]: 测试集评估指标
        """
        total_stages = 4
        
        self._print_stage_banner("原始数据生成 (M1)", 1, total_stages)
        print(f"  时间范围: {start_date} ~ {end_date}")
        print(f"  额定容量: {self.rated_capacity}MW")
        
        start_time = time.time()
        raw_df = generate_raw_station_data(
            station_id=self.station_id,
            rated_capacity=self.rated_capacity,
            start_date=start_date,
            end_date=end_date,
            random_seed=RANDOM_SEED
        )
        elapsed = time.time() - start_time
        
        print(f"  生成记录数: {len(raw_df):,}")
        print(f"  耗时: {elapsed:.2f}秒")
        print(f"  GHI范围: {raw_df['GHI'].min():.0f} ~ {raw_df['GHI'].max():.0f} W/m²")
        print(f"  PV范围: {raw_df['pv_data'].min():.2f} ~ {raw_df['pv_data'].max():.2f} MW")
        self._print_success_banner("原始数据生成完成")
        
        self._print_stage_banner("限功率场景注入 (M2)", 2, total_stages)
        print(f"  限电比例: [0.3, 0.5, 0.7, 0.85, 0.95]")
        print(f"  注入策略: 马尔可夫链 + 温漂 + GHI变差噪声")
        
        start_time = time.time()
        simulated_df = inject_curtailment_scenarios(
            raw_df=raw_df,
            curtail_ratios=[0.3, 0.5, 0.7, 0.85, 0.95],
            p_01=0.15,
            p_11=0.85,
            random_seed=RANDOM_SEED
        )
        elapsed = time.time() - start_time
        
        curtailed_count = simulated_df["is_curtailed"].sum()
        curtailed_ratio = curtailed_count / len(simulated_df) * 100
        curtailed_dates = simulated_df[simulated_df["is_curtailed"] == 1]["timestamp"].dt.date.nunique()
        
        print(f"  总记录数: {len(simulated_df):,}")
        print(f"  限电样本数: {curtailed_count:,} ({curtailed_ratio:.2f}%)")
        print(f"  限电天数: {curtailed_dates}")
        print(f"  耗时: {elapsed:.2f}秒")
        self._print_success_banner("限功率场景注入完成")
        
        self._print_stage_banner("基线模型训练 (M3)", 3, total_stages)
        print(f"  数据划分: 训练集70% | 验证集15% | 测试集15%")
        print(f"  窗口规格: 32点/窗口, 步长8点")
        print(f"  标签规则: ≥4点(1小时)限电标记为正样本")
        
        start_time = time.time()
        base_metrics = build_base_model(
            simulated_df=simulated_df,
            station_id=self.station_id,
            random_seed=RANDOM_SEED
        )
        elapsed = time.time() - start_time
        
        print(f"  耗时: {elapsed:.2f}秒")
        self._print_success_banner("基线模型训练完成")
        
        self._print_stage_banner("基线评估报告 (M5)", 4, total_stages)
        print(f"  模型路径: {self.base_model_path}")
        print(f"\n📊 基线模型测试集指标:")
        print(f"  Precision:  {base_metrics['precision']:.4f}")
        print(f"  Recall:     {base_metrics['recall']:.4f}")
        print(f"  F1-Score:   {base_metrics['f1_score']:.4f}")
        
        self._print_divider()
        print("🎉 冷启动全流程完成！")
        self._print_divider()
        
        return base_metrics
    
    def run_production_tuning(
        self,
        real_data_df: pd.DataFrame,
        use_pretrained_rocket: bool = True
    ) -> Dict[str, float]:
        """
        生产级微调流程：基于真实数据增量优化
        
        执行流程：
        1. 加载基线模型
        2. M4: 合并新旧数据并微调分类器
        3. M5: 打印微调后终极压测报告
        
        Args:
            real_data_df: 真实/高仿带标签数据，需包含标准列
            use_pretrained_rocket: 是否复用预训练转换器
            
        Returns:
            Dict[str, float]: 测试集评估指标
        """
        total_stages = 3
        
        if not os.path.exists(self.base_model_path):
            print(f"❌ 基线模型不存在: {self.base_model_path}")
            print(f"   请先调用 run_cold_start_pipeline() 生成基线模型")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        self._print_stage_banner("生产数据准备", 1, total_stages)
        print(f"  真实数据记录数: {len(real_data_df):,}")
        print(f"  限电样本数: {real_data_df['is_curtailed'].sum():,}")
        
        self._print_stage_banner("增量微调训练 (M4)", 2, total_stages)
        print(f"  基线模型: {os.path.basename(self.base_model_path)}")
        print(f"  复用转换器: {'是' if use_pretrained_rocket else '否'}")
        
        start_time = time.time()
        prod_metrics = fine_tune_with_real_data(
            base_model_path=self.base_model_path,
            real_labeled_df=real_data_df,
            station_id=self.station_id,
            use_pretrained_rocket=use_pretrained_rocket,
            random_seed=RANDOM_SEED
        )
        elapsed = time.time() - start_time
        
        print(f"  耗时: {elapsed:.2f}秒")
        self._print_success_banner("生产模型微调完成")
        
        self._print_stage_banner("终极压测报告 (M5)", 3, total_stages)
        print(f"  生产模型路径: {self.prod_model_path}")
        print(f"\n🏆 生产模型测试集指标:")
        print(f"  Precision:  {prod_metrics['precision']:.4f}")
        print(f"  Recall:     {prod_metrics['recall']:.4f}")
        print(f"  F1-Score:   {prod_metrics['f1_score']:.4f}")
        
        self._print_divider()
        print("🎯 生产级微调全流程完成！")
        self._print_divider()
        
        return prod_metrics


def generate_demo_real_data(
    station_id: str,
    rated_capacity: float,
    start_date: str = "2026-01-01",
    end_date: str = "2026-03-31",
    random_seed: int = 2026
) -> pd.DataFrame:
    """
    生成演示用真实数据（模拟真实运营场景）
    
    Args:
        station_id: 电站ID
        rated_capacity: 额定容量
        start_date: 开始日期
        end_date: 结束日期
        random_seed: 随机种子（与训练数据不同）
        
    Returns:
        pd.DataFrame: 带标签的真实数据
    """
    print(f"\n{'─'*70}")
    print(f"📦 生成演示用真实数据")
    print(f"{'─'*70}")
    print(f"  时间范围: {start_date} ~ {end_date}")
    print(f"  随机种子: {random_seed} (与训练数据不同)")
    
    raw_df = generate_raw_station_data(
        station_id=station_id,
        rated_capacity=rated_capacity,
        start_date=start_date,
        end_date=end_date,
        random_seed=random_seed
    )
    
    real_df = inject_curtailment_scenarios(
        raw_df=raw_df,
        curtail_ratios=[0.4, 0.6, 0.8, 0.9],
        random_seed=random_seed
    )
    
    print(f"  生成记录数: {len(real_df):,}")
    print(f"  限电样本数: {real_df['is_curtailed'].sum():,}")
    print(f"  生成完成 ✓")
    
    return real_df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("⚡ 限功率识别系统 - 一键工厂演示")
    print("="*70)
    print("  演示目标: 460MW 大型光伏电站")
    print("  流程1: 冷启动全流程 (M1→M2→M3→M5)")
    print("  流程2: 生产级微调 (M4→M5)")
    print("="*70)
    
    station_id = "station_460MW"
    rated_capacity = 460.0
    
    factory = StationModelFactory(station_id, rated_capacity)
    
    print("\n" + "="*70)
    print("📋 流程1: 冷启动全流程")
    print("="*70)
    base_metrics = factory.run_cold_start_pipeline(
        start_date="2025-01-01",
        end_date="2025-12-31"
    )
    
    print("\n" + "="*70)
    print("📋 流程2: 生产级微调 (随机种子=2026，模拟真实季度波动)")
    print("="*70)
    print(f"  ⚠️  压测模式: 使用独立随机种子 {2026} 生成全新一季度数据")
    print(f"  ⚠️  验证目标: 检验模型在纯粹形态随机波动下的泛化能力")
    real_data_df = generate_demo_real_data(
        station_id=station_id + "_real",
        rated_capacity=rated_capacity,
        start_date="2026-01-01",
        end_date="2026-03-31",
        random_seed=2026
    )
    
    prod_metrics = factory.run_production_tuning(real_data_df)
    
    print("\n" + "="*70)
    print("📊 全流程对比报告")
    print("="*70)
    print(f"{'指标':<15} {'基线模型':<12} {'生产模型':<12} {'变化':<10}")
    print(f"{'─'*47}")
    
    metrics_names = ["Precision", "Recall", "F1-Score"]
    metrics_keys = ["precision", "recall", "f1_score"]
    
    for name, key in zip(metrics_names, metrics_keys):
        base_val = base_metrics[key]
        prod_val = prod_metrics[key]
        change = f"+{(prod_val - base_val) * 100:.2f}%" if prod_val >= base_val else f"{(prod_val - base_val) * 100:.2f}%"
        print(f"{name:<15} {base_val:<12.4f} {prod_val:<12.4f} {change:<10}")
    
    print(f"\n📁 模型文件清单:")
    print(f"  ├── 基线模型: {factory.base_model_path}")
    print(f"  └── 生产模型: {factory.prod_model_path}")
    
    print("\n" + "="*70)
    print("🎊 演示完成！")
    print("="*70)
    print("  系统已成功完成以下功能:")
    print("  ✓ 原始数据自动生成 (M1)")
    print("  ✓ 限功率场景动态制造 (M2)")
    print("  ✓ 基线模型训练与评估 (M3+M5)")
    print("  ✓ 生产级增量微调 (M4)")
    print("  ✓ 全流程指标对比报告")
    print("="*70)