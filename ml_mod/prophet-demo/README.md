pip install prophet pandas numpy matplotlib scikit-learn
pip install plotly


NeuralProphet


经典统计方法：
Prophet（Facebook 开源，适配强季节性 + 异常值场景）
ARIMA（适配线性趋势 + 周期特征，需数据平稳）

机器学习：
LSTM/Transformer（适配非线性、长序列数据）
XGBoost/LightGBM（需手动构造时间特征，适配混合影响因素场景）