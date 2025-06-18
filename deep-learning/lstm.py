import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 定义LSTM模型
class LSTMPowerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMPowerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 数据预处理函数
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# 异常检测函数
def detect_anomalies(model, data_loader, threshold, device):
    model.eval()
    anomalies = []
    reconstruction_errors = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # 计算重构误差
            error = torch.mean(torch.abs(outputs - inputs[:, -1, :]), dim=1)
            reconstruction_errors.extend(error.cpu().numpy())

            # 标记异常
            anomalies.extend((error > threshold).cpu().numpy())

    return np.array(anomalies), np.array(reconstruction_errors)


# 主函数
def main():
    # 1. 数据加载与预处理
    # 这里假设我们有一个电力监控数据集
    # 实际应用中请替换为真实数据
    data = pd.read_csv('power_monitoring_data.csv')
    # 提取电力负荷列
    power_data = data['power_load'].values.reshape(-1, 1)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    power_data_scaled = scaler.fit_transform(power_data)

    # 创建序列
    seq_length = 24  # 使用前24小时数据预测下一小时
    X, y = create_sequences(power_data_scaled, seq_length)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 2. 模型训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    input_size = 1  # 特征维度
    hidden_size = 64  # 隐藏层大小
    num_layers = 2  # LSTM层数
    output_size = 1  # 输出维度

    model = LSTMPowerModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # 3. 模型评估
    model.eval()
    test_loss = 0
    predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # 收集预测结果
            predictions.extend(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    # 反归一化预测结果
    predictions = scaler.inverse_transform(np.array(predictions))
    actual = scaler.inverse_transform(y_test.numpy())

    # 4. 异常检测
    # 使用重构误差作为异常指标
    # 首先需要将模型转换为自编码器模式
    # 这里简化处理，直接使用预测误差
    threshold = np.mean(np.abs(predictions - actual)) + 3 * np.std(np.abs(predictions - actual))

    anomalies = np.abs(predictions - actual) > threshold
    anomaly_indices = np.where(anomalies)[0]

    print(f"检测到的异常数量: {len(anomaly_indices)}")

    # 5. 可视化结果
    plt.figure(figsize=(15, 10))

    # 绘制训练损失
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 绘制预测结果
    plt.subplot(2, 2, 2)
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Power Load Prediction')
    plt.xlabel('Time')
    plt.ylabel('Power Load')
    plt.legend()

    # 绘制异常点
    plt.subplot(2, 2, 3)
    plt.plot(actual, label='Actual')
    plt.scatter(anomaly_indices, actual[anomaly_indices], color='red', label='Anomalies')
    plt.title('Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Power Load')
    plt.legend()

    # 绘制误差分布
    plt.subplot(2, 2, 4)
    errors = np.abs(predictions - actual)
    sns.histplot(errors, kde=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.savefig('power_monitoring_results.png')
    plt.show()


if __name__ == "__main__":
    main()
