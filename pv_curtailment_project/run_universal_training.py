import os
import numpy as np
import pandas as pd

from src.data_processor import load_and_preprocess_data
from src.trainer import CurtailmentPipeline, load_pretrained_model, merge_and_shuffle_datasets


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def discover_stations(raw_data_base_path: str) -> list:
    """
    自动发现 data/01_raw/ 下的所有电站配置
    
    Args:
        raw_data_base_path: 原始数据基础路径
        
    Returns:
        电站配置列表 [(station_id, capacity_mw, data_path), ...]
    """
    stations = []
    
    if not os.path.exists(raw_data_base_path):
        print(f"原始数据目录不存在: {raw_data_base_path}")
        return stations
    
    for station_folder in sorted(os.listdir(raw_data_base_path)):
        station_path = os.path.join(raw_data_base_path, station_folder)
        if not os.path.isdir(station_path):
            continue
        
        station_id = station_folder.replace("station_", "")
        
        for capacity_folder in sorted(os.listdir(station_path)):
            capacity_path = os.path.join(station_path, capacity_folder)
            if not os.path.isdir(capacity_path):
                continue
            
            try:
                capacity_mw = float(capacity_folder.replace("capacity_", "").replace("MW", ""))
            except ValueError:
                continue
            
            merged_path = os.path.join(capacity_path, "merged_data.csv")
            if not os.path.exists(merged_path):
                print(f"警告: {capacity_path} 下未找到 merged_data.csv")
                continue
            
            stations.append({
                "station_id": station_id,
                "capacity_mw": capacity_mw,
                "data_path": merged_path,
                "folder_path": capacity_path
            })
    
    return stations


def main():
    """
    跨电站通用模型训练主入口
    
    流程：
    1. 自动发现所有电站配置
    2. 对每个电站调用 Pipeline 生成带标签的数据集
    3. 合并并打散所有电站的数据集
    4. 训练 MiniRocket 模型
    5. 在混合测试集上评估
    6. 保存通用模型
    """
    print("="*60)
    print("跨电站通用 MiniRocket 限功率识别模型训练")
    print("="*60)
    print("\n核心特性:")
    print("  - 多电站自动发现与批量处理")
    print("  - 跨容量(60~200MW)混合训练")
    print("  - 数据集彻底打散融合")
    print("  - 严格时序隔离(70%/15%/15%)")
    print("  - 双模式输入管道支持")
    
    raw_data_base_path = os.path.join(PROJECT_ROOT, "data", "01_raw")
    
    print("\n1. 自动发现电站配置...")
    stations = discover_stations(raw_data_base_path)
    
    if not stations:
        print("未发现任何电站配置，退出")
        return
    
    print(f"  发现 {len(stations)} 个电站配置:")
    for station in stations:
        print(f"    - {station['station_id']} ({station['capacity_mw']}MW): {station['data_path']}")
    
    pipeline = CurtailmentPipeline(
        window_size=32,
        stride=8,
        label_threshold=4,
        train_ratio=0.7,
        val_ratio=0.15,
        num_kernels=10000,
        random_seed=42
    )
    
    datasets_list = []
    
    print("\n2. 逐个处理电站数据...")
    for i, station in enumerate(stations):
        print(f"\n{'#'*50}")
        print(f"处理电站 {i+1}/{len(stations)}")
        print(f"电站ID: {station['station_id']}")
        print(f"额定容量: {station['capacity_mw']}MW")
        print(f"{'#'*50}")
        
        print("加载原始数据...")
        raw_df = load_and_preprocess_data(station["data_path"])
        print(f"  数据记录数: {len(raw_df)}")
        
        datasets = pipeline.process_and_fake_raw_sequence(
            raw_df=raw_df,
            station_id=station["station_id"],
            rated_capacity=station["capacity_mw"],
            p_trigger=0.06
        )
        
        datasets_list.append(datasets)
    
    print("\n3. 合并并打散所有电站数据集...")
    merged_datasets = merge_and_shuffle_datasets(
        datasets_list=datasets_list,
        random_seed=42
    )
    
    X_train = merged_datasets["X_train"]
    Y_train = merged_datasets["Y_train"]
    X_val = merged_datasets["X_val"]
    Y_val = merged_datasets["Y_val"]
    X_test = merged_datasets["X_test"]
    Y_test = merged_datasets["Y_test"]
    
    print("\n4. 加载预训练模型（可选）...")
    pretrained_path = os.path.join(PROJECT_ROOT, "models", "minirocket_pretrained.pkl")
    pretrained_model = load_pretrained_model(pretrained_path)
    
    print("\n5. 训练 MiniRocket 模型...")
    rocket, classifier = pipeline.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        pretrained_model=pretrained_model,
        use_pretrained_rocket=True
    )
    
    print("\n6. 在混合测试集上评估模型...")
    metrics = pipeline.evaluate(X_test, Y_test, "跨电站通用模型")
    
    print("\n7. 保存通用模型...")
    save_path = pipeline.save(os.path.join(PROJECT_ROOT, "models", "minirocket_universal.pkl"))
    
    print("\n" + "="*60)
    print("跨电站通用模型训练完成！")
    print("="*60)
    print("\n最终综合评估指标:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\n模型保存路径: {save_path}")


if __name__ == "__main__":
    main()
