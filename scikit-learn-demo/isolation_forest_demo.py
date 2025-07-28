import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 生成模拟数据
def generate_anomaly_data(n_samples=1000, n_features=2, contamination=0.05, random_state=42):
    """生成包含异常值的模拟数据集"""
    np.random.seed(random_state)
    
    # 生成正常数据（聚类）
    X_normal, _ = make_blobs(
        n_samples=int(n_samples * (1 - contamination)),
        n_features=n_features,
        centers=3,
        cluster_std=0.6,
        random_state=random_state
    )
    
    # 生成异常数据（离群点）
    X_anomaly = np.random.uniform(
        low=-10, 
        high=10, 
        size=(int(n_samples * contamination), n_features)
    )
    
    # 合并数据
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.ones(len(X_normal)), -np.ones(len(X_anomaly))])  # 1=正常, -1=异常
    
    # 打乱数据顺序
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

# 二维数据可视化
def visualize_anomalies(X, y_true, y_pred, contamination, dataset_name="合成数据"):
    """可视化异常检测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 真实标签
    normal_mask = y_true == 1
    anomaly_mask = y_true == -1
    
    axes[0].scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', s=30, label='正常数据')
    axes[0].scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], c='red', s=50, marker='x', label='异常数据')
    axes[0].set_title(f'{dataset_name} - 真实标签 (污染率: {contamination*100:.1f}%)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 预测结果
    normal_mask = y_pred == 1
    anomaly_mask = y_pred == -1
    
    axes[1].scatter(X[normal_mask, 0], X[normal_mask, 1], c='blue', s=30, label='预测正常')
    axes[1].scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], c='red', s=50, marker='x', label='预测异常')
    axes[1].set_title(f'{dataset_name} - 孤立森林预测结果')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 评估模型性能
    print("\n模型性能评估:")
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print("混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # 绘制混淆矩阵热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["预测异常", "预测正常"], 
                yticklabels=["真实异常", "真实正常"])
    plt.title("混淆矩阵")
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=['异常', '正常']))

# 使用孤立森林进行异常检测
def anomaly_detection_with_isolation_forest(X, y_true, contamination=0.05, random_state=42):
    """使用孤立森林算法检测异常数据"""
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 初始化并训练孤立森林模型
    model = IsolationForest(
        n_estimators=100,        # 树的数量
        contamination=contamination,  # 异常样本比例
        max_samples='auto',      # 每棵树使用的样本数
        random_state=random_state,
        n_jobs=-1                # 使用所有CPU核心
    )
    
    # 拟合模型并预测
    model.fit(X_scaled)
    y_pred = model.predict(X_scaled)  # 预测结果: 1=正常, -1=异常
    
    # 计算异常分数
    anomaly_scores = model.decision_function(X_scaled)  # 分数越低越可能是异常
    
    return y_pred, anomaly_scores

# 主函数
def main():
    # 1. 使用合成数据演示
    print("正在生成合成数据集...")
    contamination = 0.05  # 异常样本比例
    X, y_true = generate_anomaly_data(
        n_samples=1000, 
        n_features=2, 
        contamination=contamination,
        random_state=42
    )
    
    print("\n使用孤立森林检测异常...")
    y_pred, anomaly_scores = anomaly_detection_with_isolation_forest(
        X, y_true, contamination=contamination
    )
    
    print("\n可视化异常检测结果...")
    visualize_anomalies(X, y_true, y_pred, contamination, "合成数据集")
    
    # 2. 在电力数据上应用
    print("\n\n在电力数据上应用孤立森林...")
    # 生成模拟电力数据
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # 正常用电模式（考虑时间因素）
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    
    # 工作日白天用电高，夜间和周末用电低
    base_power = 50 + 30 * np.sin(np.pi * hour_of_day / 12)
    base_power[day_of_week >= 5] *= 0.7  # 周末用电减少30%
    
    # 添加随机噪声
    normal_power = base_power + np.random.normal(0, 10, n_samples)
    
    # 添加异常用电（随机时间点的高用电量）
    n_anomalies = int(n_samples * contamination)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    normal_power[anomaly_indices] += np.random.uniform(50, 100, n_anomalies)
    
    # 创建电力数据集
    power_data = pd.DataFrame({
        'date': dates,
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'power_consumption': normal_power
    })
    
    # 特征工程
    X_power = power_data[['hour', 'day_of_week', 'power_consumption']].values
    
    # 使用孤立森林检测异常
    print("\n检测电力数据中的异常...")
    y_pred_power, anomaly_scores_power = anomaly_detection_with_isolation_forest(
        X_power, np.ones(n_samples), contamination=contamination
    )
    
    # 将预测结果添加到数据中
    power_data['is_anomaly'] = y_pred_power
    power_data['anomaly_score'] = anomaly_scores_power
    
    # 可视化电力数据异常检测结果
    plt.figure(figsize=(15, 6))
    normal_data = power_data[power_data['is_anomaly'] == 1]
    anomaly_data = power_data[power_data['is_anomaly'] == -1]
    
    plt.plot(normal_data['date'], normal_data['power_consumption'], 
             'b-', alpha=0.7, label='正常用电')
    plt.scatter(anomaly_data['date'], anomaly_data['power_consumption'], 
                c='red', s=50, marker='x', label='异常用电')
    
    plt.title('电力数据异常检测结果')
    plt.xlabel('日期')
    plt.ylabel('用电量')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('power_data_anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"检测到的电力异常数量: {len(anomaly_data)} / {n_samples}")
    
    # 保存结果
    power_data.to_csv('power_data_with_anomalies.csv', index=False)
    print("电力数据异常检测结果已保存至 'power_data_with_anomalies.csv'")

if __name__ == "__main__":
    main()    