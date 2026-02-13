# 文件名：tsfresh_demo.py
# from tsfresh import extract_features
import pandas as pd


from tsfresh.convenience.relevant_extraction import (  # noqa: E402 一站式完成 “特征提取 + 特征筛选”，直接输出有效特征
    extract_relevant_features,
)
from tsfresh.feature_extraction import extract_features  # noqa: E402 特征提取
from tsfresh.feature_selection import select_features  # noqa: E402 特征筛选

# 核心修复：将执行代码放入 if __name__ == '__main__' 代码块
if __name__ == '__main__':
    # 创建示例数据
    df = pd.DataFrame({
        'id': [i for i in range(10) for _ in range(100)],
        'time': [t for _ in range(10) for t in range(100)],
        'value': [float(t) for _ in range(10) for t in range(100)]
    })
    
    # 启用多进程（n_jobs=-1 表示使用所有CPU核心）
    features = extract_features(
        df, 
        column_id='id', 
        column_sort='time',
        n_jobs=1  # 多进程加速，需保证代码在 if __name__ 内
    )
    
    print("多进程提取完成！特征数量：", features.shape[1])
    print(features.shape[0])
    # print("\n提取的特征示例：")
    # print(features.head())