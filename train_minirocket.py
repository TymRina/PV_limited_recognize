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

from data_processor import prepare_minirocket_datasets


def train_minirocket_classifier(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray = None,
    Y_val: np.ndarray = None,
    num_kernels: int = 10000
) -> tuple:
    """
    训练MiniRocket多变量时序分类器
    
    Args:
        X_train: 训练特征张量 (N, 4, L)
        Y_train: 训练标签 (N,)
        X_val: 验证特征张量
        Y_val: 验证标签
        num_kernels: Rocket核数量
    
    Returns:
        rocket: 训练好的MiniRocket转换器
        classifier: 训练好的分类器
    """
    print(f"\n初始化 MiniRocketMultivariate (核数量: {num_kernels})...")
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


def evaluate_model(
    rocket: MiniRocketMultivariate,
    classifier,
    X_test: np.ndarray,
    Y_test: np.ndarray
) -> dict:
    """
    在测试集上评估模型性能
    
    Args:
        rocket: MiniRocket转换器
        classifier: 分类器
        X_test: 测试特征张量
        Y_test: 测试标签
    
    Returns:
        评估指标字典
    """
    print("\n" + "="*60)
    print("测试集评估结果")
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
    
    return {
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def save_model(
    rocket: MiniRocketMultivariate,
    classifier,
    save_path: str = "./models/minirocket_pretrained.pkl"
) -> None:
    """
    保存训练好的模型（包括Rocket转换器和分类器）
    
    Args:
        rocket: MiniRocket转换器
        classifier: 分类器
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model = {
        "rocket": rocket,
        "classifier": classifier,
        "version": "1.0.0"
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\n模型已保存至: {save_path}")


def main():
    print("="*60)
    print("MiniRocket 限功率识别模型预训练")
    print("="*60)
    
    print("\n1. 数据准备阶段...")
    datasets = prepare_minirocket_datasets(
        file_path="./curtailed_data/curtailed_data.csv",
        window_size=32,
        stride=8,
        label_threshold=4
    )
    
    X_train = datasets["X_train"]
    Y_train = datasets["Y_train"]
    X_val = datasets["X_val"]
    Y_val = datasets["Y_val"]
    X_test = datasets["X_test"]
    Y_test = datasets["Y_test"]
    
    print("\n2. 模型训练阶段...")
    rocket, classifier = train_minirocket_classifier(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        num_kernels=10000
    )
    
    print("\n3. 模型评估阶段...")
    metrics = evaluate_model(rocket, classifier, X_test, Y_test)
    
    print("\n4. 模型保存阶段...")
    save_model(rocket, classifier, "./models/minirocket_pretrained.pkl")
    
    print("\n" + "="*60)
    print("预训练完成！")
    print("="*60)


if __name__ == "__main__":
    main()