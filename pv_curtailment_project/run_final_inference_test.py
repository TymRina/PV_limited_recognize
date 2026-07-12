import numpy as np
import pandas as pd
import pickle
import os
import random

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.simulator import UniversalCurtailmentConfig, UniversalCurtailmentSimulator
from src.trainer import CurtailmentPipeline


RANDOM_SEED = 42


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_ARENA_DIR = os.path.join(PROJECT_ROOT, "test_arena")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_base_raw_data(capacity_mw: float, start_date: str = "2025-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    data = []
    for dt in dates:
        hour = dt.hour
        day_of_year = dt.dayofyear
        
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365) * 0.3 + 0.85
        
        if hour >= 6 and hour < 18:
            sun_position = (hour - 6) / 12
            ghi_base = 1000 * np.sin(np.pi * sun_position) * seasonal_factor
            
            cloud_cover = 0.3 + np.random.normal(0, 0.15)
            cloud_cover = max(0.1, min(0.8, cloud_cover))
            
            ghi = max(0, ghi_base * (1 - cloud_cover))
            
            pv_efficiency = 0.15 + np.random.normal(0, 0.01)
            pv_data = min(capacity_mw * 0.95, ghi * pv_efficiency * 1.2)
            pv_data = max(0, pv_data)
        else:
            ghi = 0
            pv_data = 0
        
        cap_power_on = capacity_mw * (0.95 + np.random.normal(0, 0.02))
        
        data.append({
            'timestamp': dt,
            'pv_data': round(pv_data, 2),
            'GHI': round(ghi, 2),
            'cap_power_on': round(cap_power_on, 2)
        })
    
    return pd.DataFrame(data)


def generate_test_station_data(
    station_id: str,
    capacity_mw: float,
    curtail_ratios: list,
    seed: int
) -> pd.DataFrame:
    config = UniversalCurtailmentConfig(
        station_id=station_id,
        rated_capacity=capacity_mw,
        random_seed=seed,
        p_01=0.08,
        p_11=0.75,
        curtail_ratios=curtail_ratios
    )
    
    simulator = UniversalCurtailmentSimulator(config)
    
    raw_df = generate_base_raw_data(capacity_mw)
    simulator.data = raw_df.copy()
    
    simulator.sample_curtailment_days()
    
    labeled_df = simulator.generate_curtailment_data()
    
    labeled_df = labeled_df.rename(columns={
        "curtailed_pv": "pv_simulated",
        "curtailment_label": "is_curtailed"
    })
    
    labeled_df["station_id"] = station_id
    labeled_df["rated_capacity"] = capacity_mw
    
    return labeled_df


def evaluate_single_station(
    rocket,
    classifier,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    station_id: str,
    capacity_mw: float
) -> dict:
    X_test_transformed = rocket.transform(X_test)
    Y_pred = classifier.predict(X_test_transformed)
    
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    
    return {
        "station_id": station_id,
        "capacity_mw": capacity_mw,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_samples": len(Y_test),
        "positive_samples": int(Y_test.sum()),
        "negative_samples": int(len(Y_test) - Y_test.sum())
    }


def print_diagnostic_report(reports: list):
    print("\n" + "="*70)
    print("跨电站通用模型 - 新容量电站诊断报告")
    print("="*70)
    
    for report in reports:
        print(f"\n{'#'*60}")
        print(f"电站ID: {report['station_id']}")
        print(f"额定容量: {report['capacity_mw']}MW")
        print(f"{'#'*60}")
        
        cm = report["confusion_matrix"]
        print("\n混淆矩阵:")
        print("-"*30)
        print(f"                预测正常      预测限电")
        print(f"实际正常        {cm[0,0]:>12}        {cm[0,1]:>10}")
        print(f"实际限电        {cm[1,0]:>12}        {cm[1,1]:>10}")
        print("-"*30)
        
        print("\n分类指标:")
        print(f"  Precision (精准率):  {report['precision']:.4f}")
        print(f"  Recall (召回率):     {report['recall']:.4f}")
        print(f"  F1-Score:            {report['f1_score']:.4f}")
        
        print("\n数据分布:")
        print(f"  总样本: {report['total_samples']}")
        print(f"  正常样本: {report['negative_samples']}")
        print(f"  限电样本: {report['positive_samples']}")
        print(f"  限电比例: {report['positive_samples'] / report['total_samples'] * 100:.2f}%")
    
    avg_precision = np.mean([r["precision"] for r in reports])
    avg_recall = np.mean([r["recall"] for r in reports])
    avg_f1 = np.mean([r["f1_score"] for r in reports])
    
    print("\n" + "="*70)
    print("综合评估结果")
    print("="*70)
    print(f"\n平均 Precision: {avg_precision:.4f}")
    print(f"平均 Recall:    {avg_recall:.4f}")
    print(f"平均 F1-Score:  {avg_f1:.4f}")


def main():
    print("="*70)
    print("跨电站通用模型 - 端到端压测脚本")
    print("="*70)
    print(f"\n全局随机种子: {RANDOM_SEED}")
    
    set_global_seed(RANDOM_SEED)
    
    os.makedirs(TEST_ARENA_DIR, exist_ok=True)
    print(f"测试目录已创建: {TEST_ARENA_DIR}")
    
    test_stations = [
        {"station_id": "test_001", "capacity_mw": 90.0},
        {"station_id": "test_002", "capacity_mw": 130.0},
        {"station_id": "test_003", "capacity_mw": 175.0},
    ]
    
    curtail_ratios = [0.3, 0.5, 0.7]
    print(f"\n测试配置:")
    print(f"  限电比例: {curtail_ratios}")
    print(f"  新容量电站: {[s['capacity_mw'] for s in test_stations]}MW")
    
    print("\n1. 生成新容量电站限电数据...")
    all_labeled_data = []
    for i, station in enumerate(test_stations):
        seed = RANDOM_SEED + i
        print(f"\n  生成 {station['station_id']} ({station['capacity_mw']}MW)...")
        labeled_df = generate_test_station_data(
            station_id=station["station_id"],
            capacity_mw=station["capacity_mw"],
            curtail_ratios=curtail_ratios,
            seed=seed
        )
        all_labeled_data.append(labeled_df)
        print(f"    记录数: {len(labeled_df)}, 限电样本: {labeled_df['is_curtailed'].sum()}")
    
    merged_labeled_df = pd.concat(all_labeled_data, ignore_index=True)
    merged_csv_path = os.path.join(TEST_ARENA_DIR, "new_unseen_stations_labeled.csv")
    merged_labeled_df.to_csv(merged_csv_path, index=False)
    print(f"\n  合并数据已保存: {merged_csv_path}")
    print(f"  总记录数: {len(merged_labeled_df)}")
    
    print("\n2. 加载通用模型...")
    model_path = os.path.join(PROJECT_ROOT, "models", "minirocket_universal.pkl")
    pipeline = CurtailmentPipeline(
        window_size=32,
        stride=8,
        label_threshold=4,
        random_seed=RANDOM_SEED
    )
    
    success = pipeline.load(model_path)
    if not success:
        print("模型加载失败，退出")
        return
    
    print("\n3. 切片测试数据（管道2: process_labeled_sequence）...")
    datasets = pipeline.process_labeled_sequence(merged_labeled_df)
    
    X_test = datasets["X_test"]
    Y_test = datasets["Y_test"]
    
    print(f"\n  测试张量维度: {X_test.shape}")
    print(f"  测试标签维度: {Y_test.shape}")
    print(f"  特征通道: ['pv_simulated', 'GHI', 'cap_power_on']")
    
    print("\n4. 推理与评估...")
    print("\n" + "="*70)
    print("混合测试集综合评估")
    print("="*70)
    
    metrics = pipeline.evaluate(X_test, Y_test, "跨电站通用模型")
    
    print("\n5. 按电站详细诊断...")
    diagnostic_reports = []
    
    for station in test_stations:
        station_mask = merged_labeled_df["station_id"] == station["station_id"]
        station_df = merged_labeled_df[station_mask].copy().reset_index(drop=True)
        
        station_datasets = pipeline.process_labeled_sequence(station_df)
        station_X_test = station_datasets["X_test"]
        station_Y_test = station_datasets["Y_test"]
        
        report = evaluate_single_station(
            pipeline.rocket,
            pipeline.classifier,
            station_X_test,
            station_Y_test,
            station["station_id"],
            station["capacity_mw"]
        )
        diagnostic_reports.append(report)
    
    print_diagnostic_report(diagnostic_reports)
    
    print("\n" + "="*70)
    print("压测完成！")
    print("="*70)
    print(f"\n测试数据保存位置: {TEST_ARENA_DIR}")
    print(f"模型路径: {model_path}")


if __name__ == "__main__":
    main()
