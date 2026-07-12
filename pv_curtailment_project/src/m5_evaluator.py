import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from typing import Dict, Union


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = ""
) -> Dict[str, Union[np.ndarray, float]]:
    """
    评估模型性能，计算并打印标准指标
    
    核心指标：
    - 混淆矩阵 (Confusion Matrix)
    - Precision (精准率)
    - Recall (召回率)
    - F1-Score
    
    Args:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        title_prefix: 标题前缀（用于区分不同数据集的评估）
        
    Returns:
        Dict[str, Union[np.ndarray, float]]: 评估指标字典
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"\n{title_prefix}评估警告: 标签数组为空")
        return {
            "confusion_matrix": np.array([]),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if title_prefix:
        title = f"{title_prefix}评估结果"
    else:
        title = "模型评估结果"
    
    print("\n" + "="*60)
    print(title)
    print("="*60)
    
    print("\n混淆矩阵:")
    print("-"*30)
    print(f"                预测正常      预测限电")
    print(f"实际正常        {cm[0,0]:>12}        {cm[0,1]:>10}")
    print(f"实际限电        {cm[1,0]:>12}        {cm[1,1]:>10}")
    print("-"*30)
    
    print("\n分类指标:")
    print(f"  Precision (精准率):  {precision:.4f}")
    print(f"  Recall (召回率):     {recall:.4f}")
    print(f"  F1-Score:            {f1:.4f}")
    
    print("\n完整分类报告:")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["正常", "限电"], zero_division=0))
    
    print("\n数据分布:")
    print(f"  总样本数: {len(y_true)}")
    print(f"  正常样本: {len(y_true) - y_true.sum()}")
    print(f"  限电样本: {int(y_true.sum())}")
    print(f"  限电样本比例: {y_true.sum() / len(y_true) * 100:.2f}%")
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def aggregate_metrics(
    metrics_list: list
) -> Dict[str, float]:
    """
    聚合多个数据集的评估指标
    
    Args:
        metrics_list: 多个评估指标字典的列表
        
    Returns:
        Dict[str, float]: 聚合后的平均指标
    """
    if not metrics_list:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1_score": 0.0
        }
    
    precisions = [m["precision"] for m in metrics_list]
    recalls = [m["recall"] for m in metrics_list]
    f1_scores = [m["f1_score"] for m in metrics_list]
    
    return {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
        "avg_precision": np.mean(precisions),
        "avg_recall": np.mean(recalls),
        "avg_f1_score": np.mean(f1_scores)
    }


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
    
    print("测试评估器...")
    metrics = evaluate_model_performance(y_true, y_pred, title_prefix="测试集")
    
    print(f"\n返回的指标:")
    print(f"  混淆矩阵:\n{metrics['confusion_matrix']}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")