import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple, Optional, Dict, List, Union
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.utils import shuffle

from sktime.transformations.panel.rocket import MiniRocketMultivariate

from .simulator import UniversalCurtailmentConfig, UniversalCurtailmentSimulator
from .data_processor import prepare_minirocket_datasets, load_and_preprocess_data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class CurtailmentPipeline:
    def __init__(
        self,
        window_size: int = 32,
        stride: int = 8,
        label_threshold: int = 4,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_kernels: int = 10000,
        random_seed: int = 42
    ):
        self.window_size = window_size
        self.stride = stride
        self.label_threshold = label_threshold
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_kernels = num_kernels
        self.random_seed = random_seed
        
        self.rocket: Optional[MiniRocketMultivariate] = None
        self.classifier = None
        
        np.random.seed(random_seed)
    
    def process_and_fake_raw_sequence(
        self,
        raw_df: pd.DataFrame,
        station_id: str = "default",
        rated_capacity: float = 100.0,
        p_trigger: float = 0.06,
        curtail_ratios: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        管道1：接收纯净原始序列，自动模拟限电数据后切片
        
        Args:
            raw_df: 原始数据DataFrame，需包含 ['timestamp', 'pv_data', 'GHI', 'cap_power_on']
            station_id: 电站ID
            rated_capacity: 额定容量(MW)
            p_trigger: 限电触发概率
            curtail_ratios: 限电比例列表
            
        Returns:
            包含训练/验证/测试集的字典
        """
        print(f"\n{'='*50}")
        print(f"管道1: 原始序列模拟 + 切片")
        print(f"{'='*50}")
        
        config = UniversalCurtailmentConfig(
            station_id=station_id,
            rated_capacity=rated_capacity,
            random_seed=self.random_seed,
            p_01=p_trigger
        )
        
        if curtail_ratios is not None:
            config.curtail_ratios = curtail_ratios
        
        simulator = UniversalCurtailmentSimulator(config)
        simulator.data = raw_df.copy()
        
        print("  采样限电日期...")
        simulator.sample_curtailment_days()
        print(f"  限电天数: {len(simulator.curtailment_dates)}")
        
        print("  生成限电数据...")
        labeled_df = simulator.generate_curtailment_data()
        
        labeled_df = labeled_df.rename(columns={
            "curtailed_pv": "pv_simulated",
            "curtailment_label": "is_curtailed"
        })
        
        print("  调用切片器...")
        datasets = prepare_minirocket_datasets(
            labeled_df,
            is_labeled=True,
            window_size=self.window_size,
            stride=self.stride,
            label_threshold=self.label_threshold,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )
        
        return datasets
    
    def process_labeled_sequence(
        self,
        labeled_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        管道2：接收已标记的限功率时序数据，直接切片
        
        Args:
            labeled_df: 已标记数据DataFrame，需包含 ['pv_simulated', 'GHI', 'cap_power_on', 'is_curtailed']
            
        Returns:
            包含训练/验证/测试集的字典
        """
        print(f"\n{'='*50}")
        print(f"管道2: 已标记序列直接切片")
        print(f"{'='*50}")
        
        print("  调用切片器...")
        datasets = prepare_minirocket_datasets(
            labeled_df,
            is_labeled=True,
            window_size=self.window_size,
            stride=self.stride,
            label_threshold=self.label_threshold,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )
        
        return datasets
    
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray = None,
        Y_val: np.ndarray = None,
        pretrained_model: Optional[dict] = None,
        use_pretrained_rocket: bool = True
    ) -> Tuple[MiniRocketMultivariate, RidgeClassifierCV]:
        """
        训练 MiniRocket 分类器
        
        Args:
            X_train: 训练特征张量 (N, 3, L)
            Y_train: 训练标签 (N,)
            X_val: 验证特征张量
            Y_val: 验证标签
            pretrained_model: 预训练模型字典
            use_pretrained_rocket: 是否使用预训练的 Rocket 转换器
            
        Returns:
            rocket: 训练好的 MiniRocket 转换器
            classifier: 训练好的分类器
        """
        print(f"\n{'='*50}")
        print(f"模型训练阶段")
        print(f"{'='*50}")
        
        if pretrained_model is not None and use_pretrained_rocket:
            print(f"使用预训练的 MiniRocket 转换器...")
            self.rocket = pretrained_model["rocket"]
            
            print("提取训练特征（使用预训练转换器）...")
            X_train_transformed = self.rocket.transform(X_train)
        else:
            print(f"初始化新的 MiniRocketMultivariate (核数量: {self.num_kernels})...")
            self.rocket = MiniRocketMultivariate(num_kernels=self.num_kernels)
            
            print("训练 Rocket 转换器...")
            self.rocket.fit(X_train)
            
            print("提取训练特征...")
            X_train_transformed = self.rocket.transform(X_train)
        
        print(f"特征维度: {X_train_transformed.shape}")
        
        print("训练 RidgeClassifierCV 分类器...")
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=5)
        self.classifier.fit(X_train_transformed, Y_train)
        
        print(f"最佳正则化参数: alpha = {self.classifier.alpha_}")
        
        if X_val is not None and Y_val is not None:
            print("\n验证集评估:")
            X_val_transformed = self.rocket.transform(X_val)
            Y_val_pred = self.classifier.predict(X_val_transformed)
            print(classification_report(Y_val, Y_val_pred))
        
        return self.rocket, self.classifier
    
    def evaluate(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        model_name: str = "模型"
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        在测试集上评估模型性能
        
        Args:
            X_test: 测试特征张量
            Y_test: 测试标签
            model_name: 模型名称（用于输出）
            
        Returns:
            评估指标字典
        """
        print("\n" + "="*60)
        print(f"{model_name} 测试集评估结果")
        print("="*60)
        
        X_test_transformed = self.rocket.transform(X_test)
        Y_pred = self.classifier.predict(X_test_transformed)
        
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
    
    def save(
        self,
        save_path: str = None
    ) -> str:
        """
        保存训练好的模型
        
        Args:
            save_path: 保存路径
            
        Returns:
            实际保存路径
        """
        if save_path is None:
            save_path = os.path.join(PROJECT_ROOT, "models", "minirocket_universal.pkl")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model = {
            "rocket": self.rocket,
            "classifier": self.classifier,
            "version": "3.0.0",
            "description": "跨电站通用限功率识别模型 - 支持多容量混合训练",
            "features": {
                "channels": ["pv_simulated", "GHI", "cap_power_on"],
                "window_size": self.window_size,
                "stride": self.stride,
                "label_threshold": self.label_threshold
            },
            "pipeline_params": {
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "num_kernels": self.num_kernels,
                "random_seed": self.random_seed
            }
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"\n模型已保存至: {save_path}")
        return save_path
    
    def load(self, model_path: str) -> bool:
        """
        加载已保存的模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        self.rocket = model["rocket"]
        self.classifier = model["classifier"]
        
        print(f"模型加载成功 (版本: {model.get('version', 'unknown')})")
        return True


def load_pretrained_model(model_path: str) -> Optional[dict]:
    """
    加载预训练模型
    
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


def merge_and_shuffle_datasets(
    datasets_list: List[Dict[str, np.ndarray]],
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    合并多个电站的数据集并打散
    
    Args:
        datasets_list: 多个电站的数据集列表
        random_seed: 随机种子
        
    Returns:
        合并并打散后的数据集字典
    """
    print(f"\n{'='*50}")
    print(f"合并并打散 {len(datasets_list)} 个电站的数据集")
    print(f"{'='*50}")
    
    X_train_list = []
    Y_train_list = []
    X_val_list = []
    Y_val_list = []
    X_test_list = []
    Y_test_list = []
    
    for i, ds in enumerate(datasets_list):
        print(f"  电站{i+1}: 训练集 {len(ds['X_train'])} 窗口, "
              f"验证集 {len(ds['X_val'])} 窗口, 测试集 {len(ds['X_test'])} 窗口")
        
        X_train_list.append(ds['X_train'])
        Y_train_list.append(ds['Y_train'])
        X_val_list.append(ds['X_val'])
        Y_val_list.append(ds['Y_val'])
        X_test_list.append(ds['X_test'])
        Y_test_list.append(ds['Y_test'])
    
    X_train = np.concatenate(X_train_list, axis=0)
    Y_train = np.concatenate(Y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    Y_val = np.concatenate(Y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)
    
    print(f"\n  合并后:")
    print(f"    训练集: {len(X_train)} 窗口 (正样本: {Y_train.sum()})")
    print(f"    验证集: {len(X_val)} 窗口 (正样本: {Y_val.sum()})")
    print(f"    测试集: {len(X_test)} 窗口 (正样本: {Y_test.sum()})")
    
    X_train, Y_train = shuffle(X_train, Y_train, random_state=random_seed)
    X_val, Y_val = shuffle(X_val, Y_val, random_state=random_seed)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=random_seed)
    
    print(f"  已完成打散")
    
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test
    }
