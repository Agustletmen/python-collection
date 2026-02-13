# tsfresh 对数据格式有严格要求，必须包含 3 类列：
# id：区分不同时间序列（比如设备 ID、用户 ID）；
# time：时间戳（用于排序序列）；
# value：时序数值（比如传感器读数、指标值）。

import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 数据准备（不变）
np.random.seed(42)
data = pd.DataFrame({
    "id": np.repeat(range(10), 50),
    "time": np.tile(range(50), 10),
    "value": np.random.randn(500)
})
labels = pd.Series([0,1,0,1,0,1,0,1,0,1], index=range(10))

# 2. 特征提取（修复n_jobs参数）
features = extract_features(
    data,                       
    column_id="id",              
    column_sort="time",          
    default_fc_parameters=MinimalFCParameters(),
    n_jobs=1,                    # 关键修复：改为1（单进程），避免多进程报错
    impute_function=None         
)

# 3. 后续步骤（不变）
features_imputed = impute(features)
filtered_features = select_features(features_imputed, labels, ml_task="classification")

# 4. 一站式操作（同样修复n_jobs）
relevant_features = extract_relevant_features(
    data,
    labels,
    column_id="id",
    column_sort="time",
    default_fc_parameters=MinimalFCParameters(),
    ml_task="classification",
    n_jobs=1  # 同样改为1
)

# 5. 建模评估
X_train, X_test, y_train, y_test = train_test_split(
    filtered_features, labels, test_size=0.3, random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 打印结果
print(f"提取的特征数量：{features.shape[1]}")
print(f"筛选后的特征数量：{filtered_features.shape[1]}")
print(f"模型分类准确率：{accuracy_score(y_test, y_pred):.2f}")