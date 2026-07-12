import numpy as np
import pickle
import os
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sktime.transformations.panel.rocket import MiniRocketMultivariate

from data_processor_universal import prepare_minirocket_datasets_universal


def load_pretrained_model(model_path: str = "./models/minirocket_pretrained.pkl") -> dict:
    """
    加载预训练模型（MiniRocket转换器 + 分类器）
    
    Args:
        model_path: 预训练模型路径
    
    Returns:
        包含rocket和classifier的字典
    """
    if not os.path.exists(model_path):
        print(f"预训练模型不存在: {model_path}")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"预训练模型加载成功 (版本: {model.get('version', 'unknown')})")
    return model


def train_minirocket_classifier_universal(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray = None,
    Y_val: np.ndarray = None,
    num_kernels: int = 10000,
    pretrained_model: dict = None,
    use_pretrained_rocket: bool = True
) -> tuple:
    """
    训练通用版MiniRocket多变量时序分类器
    
    核心改进：
    1. 支持加载预训练模型进行微调
    2. 可以选择使用预训练的Rocket转换器（保持形态特征提取能力）
    3. 仅重新训练分类器或完整重新训练
    4. 使用上过强度的通用数据集进行训练
    
    Args:
        X_train: 训练特征张量 (N, 3, L)
        Y_train: 训练标签 (N,)
        X_val: 验证特征张量
        Y_val: 验证标签
        num_kernels: Rocket核数量
        pretrained_model: 预训练模型字典
        use_pretrained_rocket: 是否使用预训练的Rocket转换器
    
    Returns:
        rocket: 训练好的MiniRocket转换器
        classifier: 训练好的分类器
    """
    if pretrained_model is not None and use_pretrained_rocket:
        print(f"\n使用预训练的 MiniRocket 转换器...")
        rocket = pretrained_model["rocket"]
        print(f"  预训练核数量: {rocket.num_kernels}")
        
        print("提取训练特征（使用预训练转换器）...")
        X_train_transformed = rocket.transform(X_train)
    else:
        print(f"\n初始化新的 MiniRocketMultivariate (核数量: {num_kernels})...")
        rocket = MiniRocketMultivariate(num_kernels=num_kernels)
        
        print("训练 Rocket 转换器...")
        rocket.fit(X_train)
        
        print("提取训练特征...")
        X_train_transformed = rocket.transform(X_train)
    
    print(f"特征维度: {X_train_transformed.shape}")
    
    print("训练 RidgeClassifierCV 分类器...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
    classifier.fit(X_train_transformed, Y_train)
    
    print(f"最佳正则化参数: alpha = {classifier.alpha_}")
    
    if X_val is not None and Y_val is not None:
        print("\n验证集评估:")
        X_val_transformed = rocket.transform(X_val)
        Y_val_pred = classifier.predict(X_val_transformed)
        
        print(classification_report(Y_val, Y_val_pred))
    
    return rocket, classifier


def evaluate_model_universal(
    rocket: MiniRocketMultivariate,
    classifier,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    model_name: str = "通用版模型"
) -> dict:
    """
    在测试集上评估模型性能（含轻微限电样本）
    
    核心改进：
    1. 在包含0.85~0.95轻微限电样本的测试集上进行全面评估
    2. 进行最硬核的极限压测
    3. 打印详细的混淆矩阵和分类指标
    
    Args:
        rocket: MiniRocket转换器
        classifier: 分类器
        X_test: 测试特征张量
        Y_test: 测试标签
        model_name: 模型名称（用于输出）
    
    Returns:
        评估指标字典
    """
    print("\n" + "="*60)
    print(f"{model_name} 测试集评估结果")
    print("="*60)
    
    X_test_transformed = rocket.transform(X_test)
    Y_pred = classifier.predict(X_test_transformed)
    
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)
    
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
    print(classification_report(Y_test, Y_pred, labels=[0, 1], target_names=["正常", "限电"], zero_division=0))
    
    print("\n数据分布:")
    print(f"  测试集总样本: {len(Y_test)}")
    print(f"  正常样本: {len(Y_test) - Y_test.sum()}")
    print(f"  限电样本: {Y_test.sum()}")
    print(f"  限电样本比例: {Y_test.sum() / len(Y_test) * 100:.2f}%")
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def save_model_universal(
    rocket: MiniRocketMultivariate,
    classifier,
    save_path: str = "./models/minirocket_universal.pkl"
) -> None:
    """
    保存通用版训练好的模型（包括Rocket转换器和分类器）
    
    Args:
        rocket: MiniRocket转换器
        classifier: 分类器
        save_path: 保存路径（默认保存在 ./models/minirocket_universal.pkl）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = {
        "rocket": rocket,
        "classifier": classifier,
        "version": "2.0.0",
        "description": "通用限功率识别模型 - 支持全谱系限电场景（0.3~0.95）",
        "features": {
            "channels": ["pv_simulated", "GHI", "cap_power_on"],
            "window_size": 32,
            "curtail_ratios": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
            "time_warping": True,
            "warping_factor_range": (0.6, 1.4)
        }
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n通用版模型已保存至: {save_path}")


def main():
    """
    通用版MiniRocket限功率识别模型训练主入口
    
    训练流程：
    1. 加载通用限电数据集（含全谱系限电比例和时序变形）
    2. 加载预训练模型（可选）
    3. 使用预训练的Rocket转换器提取特征（保持形态特征提取能力）
    4. 重新训练分类器以适配通用场景
    5. 在包含轻微限电样本的测试集上进行极限评估
    6. 保存通用版模型
    
    关键设计：
    - 保持预训练的Rocket转换器：保留已学习的形态特征提取能力
    - 重新训练分类器：适配新的通用数据分布
    - 测试集包含0.85~0.95轻微限电样本：验证模型泛化能力
    """
    print("="*60)
    print("通用版 MiniRocket 限功率识别模型训练")
    print("="*60)
    print("\n核心特性:")
    print("  - 全谱系限电比例: 0.3 ~ 0.95")
    print("  - 轻微限电(0.85~0.95)微观噪声注入")
    print("  - 时序窗口变形: 时间轴拉伸/压缩 (0.6~1.4)")
    print("  - 严格时序隔离: 训练/验证/测试日期无交集")
    print("  - 基于预训练模型微调: 保留形态特征提取能力")
    
    print("\n1. 数据准备阶段...")
    datasets = prepare_minirocket_datasets_universal(
        file_path="./universal_stage/curtailed_data_universal.csv",
        window_size=32,
        stride=8,
        label_threshold=4,
        apply_time_warping_prob=0.5,
        warping_factor_range=(0.6, 1.4)
    )
    
    X_train = datasets["X_train"]
    Y_train = datasets["Y_train"]
    X_val = datasets["X_val"]
    Y_val = datasets["Y_val"]
    X_test = datasets["X_test"]
    Y_test = datasets["Y_test"]
    
    print("\n2. 加载预训练模型...")
    pretrained_model = load_pretrained_model("./models/minirocket_pretrained.pkl")
    
    print("\n3. 模型训练阶段...")
    rocket, classifier = train_minirocket_classifier_universal(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        num_kernels=10000,
        pretrained_model=pretrained_model,
        use_pretrained_rocket=True
    )
    
    print("\n4. 模型评估阶段...")
    metrics = evaluate_model_universal(rocket, classifier, X_test, Y_test, "通用版模型")
    
    print("\n5. 模型保存阶段...")
    save_model_universal(rocket, classifier, "./models/minirocket_universal.pkl")
    
    print("\n" + "="*60)
    print("通用版模型训练完成！")
    print("="*60)
    print("\n最终评估指标:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()