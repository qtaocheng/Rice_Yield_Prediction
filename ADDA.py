#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/11/13 11:15
# @Author  : ywg
# @Site    : 
# @File    : 这里为模拟数据，请修改成自己的数据
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

###############字体设置#####################
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['figure.autolayout'] = True  # 使显示图标自适应

# 忽略警告
import warnings
import os
warnings.filterwarnings("ignore")

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置随机种子确保结果可重现
def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ========================== 数据加载和预处理 ==========================

def load_data():
    """
    加载数据并进行预处理
    由于原始Excel文件路径不可访问，这里使用模拟数据
    """
    print("正在加载数据...")
    
    # 生成模拟数据 - 替代原始Excel数据
    # 假设有4个特征：GPP相关特征
    n_samples = 1000  # 样本数量
    
    # 生成模拟的GPP相关特征数据
    # 特征1: 累积GPP (kg C/m²)
    feature1 = np.random.normal(2.5, 0.8, n_samples)
    # 特征2: 最大GPP值 (g C/m²/day)
    feature2 = np.random.normal(15.0, 3.0, n_samples)
    # 特征3: GPP变异系数
    feature3 = np.random.normal(0.3, 0.1, n_samples)
    # 特征4: 生长季长度 (days)
    feature4 = np.random.normal(120, 15, n_samples)
    
    # 组合特征
    X = np.column_stack([feature1, feature2, feature3, feature4])
    
    # 生成对应的产量数据（基于特征的线性组合加噪声）
    y = (2.0 * feature1 + 0.3 * feature2 + 
         1.5 * (1/feature3) + 0.02 * feature4 + 
         np.random.normal(0, 1.0, n_samples)).reshape(-1, 1)
    
    # 确保产量为正值
    y = np.abs(y)
    
    print(f"数据加载完成，样本数量: {n_samples}")
    print(f"特征维度: {X.shape}")
    print(f"标签维度: {y.shape}")
    
    return X, y

def add_noise_augmentation(X, y, noise_factor=0.1):
    """
    数据增强：向输入数据添加噪声
    
    Args:
        X: 输入特征
        y: 标签
        noise_factor: 噪声系数
    
    Returns:
        增强后的数据
    """
    print(f"正在进行数据增强，噪声系数: {noise_factor}")
    
    # 为特征添加高斯噪声
    noise = np.random.normal(0, noise_factor, X.shape)
    X_augmented = X + noise * np.std(X, axis=0)  # 按特征标准差缩放噪声
    
    # 为标签添加较小的噪声
    label_noise = np.random.normal(0, noise_factor * 0.5, y.shape)
    y_augmented = y + label_noise * np.std(y)
    
    # 合并原始数据和增强数据
    X_combined = np.vstack([X, X_augmented])
    y_combined = np.vstack([y, y_augmented])
    
    print(f"数据增强完成，新样本数量: {X_combined.shape[0]}")
    
    return X_combined, y_combined

def preprocess_data(X, y, use_augmentation=True, noise_factor=0.1):
    """
    数据预处理流程
    
    Args:
        X: 输入特征
        y: 标签
        use_augmentation: 是否使用数据增强
        noise_factor: 噪声系数
    
    Returns:
        预处理后的训练和验证数据
    """
    # 数据增强
    if use_augmentation:
        X, y = add_noise_augmentation(X, y, noise_factor)
    
    # 特征标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    print("数据标准化完成")
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    # 划分训练集和验证集 (7:3)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.3, random_state=43, shuffle=True
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val, scaler_X, scaler_y

# ========================== 模型定义 ==========================

class RegressionNet(nn.Module):
    """
    深度神经网络回归模型
    用于GPP到产量的映射预测
    """
    def __init__(self, input_dim=4, hidden_dims=[512, 256, 32], dropout_rates=[0.001, 0.0003, 0.0003]):
        """
        初始化神经网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rates: 各层dropout率
        """
        super(RegressionNet, self).__init__()
        
        # 第一个隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])  # 批标准化
        self.dropout1 = nn.Dropout(dropout_rates[0])
        
        # 第二个隐藏层
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        
        # 第三个隐藏层
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout_rates[2])
        
        # 输出层 - 回归问题输出维度为1
        self.fc4 = nn.Linear(hidden_dims[2], 1)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 第一层：线性变换 -> 批标准化 -> ReLU激活 -> Dropout
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # 第二层
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # 第三层
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # 输出层（无激活函数，因为是回归问题）
        x = self.fc4(x)
        
        return x

# ========================== 训练和评估函数 ==========================

def evaluate_model(model, loader, scaler_y=None):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        loader: 数据加载器
        scaler_y: 标签标准化器（用于反标准化）
    
    Returns:
        R2分数, RMSE, RRMSE, MAE
    """
    model.eval()
    predictions, labels = [], []
    
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = inputs.to(device)
            batch_predictions = model(inputs)
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 如果提供了标准化器，进行反标准化
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(predictions)
        labels = scaler_y.inverse_transform(labels)
    
    # 计算评估指标
    r2 = r2_score(labels, predictions)
    rmse = sqrt(np.mean((labels - predictions) ** 2))
    mae = mean_absolute_error(labels, predictions)
    rrmse = rmse / np.mean(np.abs(labels)) * 100
    
    return r2, rmse, rrmse, mae

def train_model(model, train_loader, val_loader, scaler_y, epochs=100, patience=15):
    """
    训练模型主函数
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        scaler_y: 标签标准化器
        epochs: 最大训练轮数
        patience: 早停耐心值
    """
    print("开始训练模型...")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adam优化器
    
    # 学习率调度器 - 每15个epoch将学习率乘以0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 早停相关变量
    best_val_r2 = -np.inf
    trigger_times = 0
    train_losses = []
    val_r2_scores = []
    
    for epoch in range(epochs):
        # ============ 训练阶段 ============
        model.train()
        total_loss = 0
        batch_count = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        average_loss = total_loss / batch_count
        train_losses.append(average_loss)
        
        # ============ 验证阶段 ============
        val_r2, val_rmse, val_rrmse, val_mae = evaluate_model(model, val_loader)
        val_r2_scores.append(val_r2)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{epochs}: '
                  f'Loss={average_loss:.6f}, '
                  f'Val_R2={val_r2:.4f}, '
                  f'Val_RMSE={val_rmse:.4f}, '
                  f'LR={current_lr:.6f}')
        
        # ============ 早停检查 ============
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            trigger_times = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best validation R2: {best_val_r2:.4f}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    print("训练完成，已加载最佳模型")
    
    return train_losses, val_r2_scores

def plot_results(train_losses, val_r2_scores):
    """绘制训练过程图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制验证R2曲线
    ax2.plot(val_r2_scores, 'r-', label='Validation R²')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Validation R² Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_scatter(model, val_loader, scaler_y):
    """绘制预测值vs真实值散点图"""
    model.eval()
    predictions, labels = [], []
    
    with torch.no_grad():
        for inputs, batch_labels in val_loader:
            inputs = inputs.to(device)
            batch_predictions = model(inputs)
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # 反标准化
    predictions = scaler_y.inverse_transform(predictions).flatten()
    labels = scaler_y.inverse_transform(labels).flatten()
    
    # 绘制散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, predictions, alpha=0.6, s=30)
    
    # 绘制理想线 (y=x)
    min_val = min(min(labels), min(predictions))
    max_val = max(max(labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ========================== 主程序 ==========================

def main():
    """主程序"""
    print("=" * 60)
    print("GPP到产量预测神经网络模型")
    print("=" * 60)
    
    # 1. 数据加载和预处理
    X, y = load_data()
    X_train, X_val, y_train, y_val, scaler_X, scaler_y = preprocess_data(
        X, y, use_augmentation=True, noise_factor=0.05
    )
    
    # 2. 创建数据加载器
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"批次大小: {batch_size}")
    
    # 3. 创建模型
    model = RegressionNet(input_dim=4).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 4. 训练模型
    train_losses, val_r2_scores = train_model(
        model, train_loader, val_loader, scaler_y, epochs=200, patience=20
    )
    
    # 5. 最终评估
    print("\n" + "=" * 60)
    print("最终模型评估结果")
    print("=" * 60)
    
    train_r2, train_rmse, train_rrmse, train_mae = evaluate_model(model, train_loader, scaler_y)
    val_r2, val_rmse, val_rrmse, val_mae = evaluate_model(model, val_loader, scaler_y)
    
    print(f'训练集 - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, RRMSE: {train_rrmse:.2f}%, MAE: {train_mae:.4f}')
    print(f'验证集 - R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}, RRMSE: {val_rrmse:.2f}%, MAE: {val_mae:.4f}')
    
    # 6. 绘制结果
    plot_results(train_losses, val_r2_scores)
    plot_prediction_scatter(model, val_loader, scaler_y)
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()