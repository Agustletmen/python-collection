# pip install prophet pandas matplotlib scikit-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

# ----------------------
# 1. 准备数据（ Prophet 要求输入两列：ds(时间)、y(目标值) ）
# ----------------------
# 模拟数据：假设2018-2023年的日销售额（含季节性和趋势）
data = pd.DataFrame()
# 生成日期序列（2018-01-01 到 2023-12-31）
data['ds'] = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
# 模拟销售额（含年度增长趋势 + 周季节性 + 随机波动）
data['y'] = (
    500 + 0.2 * data['ds'].astype('int64') // 10**9  # 线性增长趋势
    + 100 * data['ds'].dt.dayofweek.map({0:1, 1:0.5, 2:0.6, 3:0.7, 4:0.8, 5:2, 6:1.5})  # 周季节性（周末高）
    + 200 * ((data['ds'].dt.month == 12) & (data['ds'].dt.day >= 20)).astype(int)  # 圣诞季高峰
    + pd.Series(range(len(data))).apply(lambda x: 50 * (x % 100 < 10))  # 随机促销波动
)

# 查看数据前5行
print(data.head())


# ----------------------
# 2. 初始化并训练模型
# ----------------------
# 添加自定义节假日（如双十一，影响3天）
holidays = pd.DataFrame({
    'holiday': 'double11',
    'ds': pd.to_datetime(['2018-11-11', '2019-11-11', '2020-11-11', '2021-11-11', '2022-11-11', '2023-11-11']),
    'lower_window': 0,  # 节假日当天
    'upper_window': 2   # 后延2天（共3天影响）
})

# 初始化模型（可自定义参数，如季节性周期、趋势类型等）
model = Prophet(
    yearly_seasonality=True,  # 年度季节性（默认开启）
    weekly_seasonality=True,  # 周度季节性（默认开启）
    daily_seasonality=False,  # 本例无需日度季节性（可根据数据调整）
    seasonality_mode='additive',  # 季节性叠加模式（默认加法，可选乘法 multiplicative）
    holidays=holidays  # 在初始化时添加自定义节假日
)

# 添加中国法定节假日（如春节）
model.add_country_holidays(country_name='CN')

# 训练模型
model.fit(data)


# ----------------------
# 3. 预测未来数据
# ----------------------
# 生成未来365天的日期（预测周期）
future = model.make_future_dataframe(periods=365)
# 预测（返回包含预测值 yhat、上下界 yhat_lower/yhat_upper 的数据框）
forecast = model.predict(future)

# 查看预测结果的关键列（ds:时间, yhat:预测值, trend:趋势项, weekly:周季节性）
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'weekly']].tail())


# ----------------------
# 4. 可视化结果
# ----------------------
# 基础预测图（历史数据 + 预测趋势 + 置信区间）
fig1 = model.plot(forecast)
plt.title('销售额预测（历史数据 + 未来趋势）')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.show()

# 组件分解图（趋势 + 季节性 + 节假日影响）
fig2 = model.plot_components(forecast)
plt.show()

# （可选）交互式可视化（需安装 plotly）
# plot_plotly(model, forecast)  # 交互式预测图
# plot_components_plotly(model, forecast)  # 交互式组件分解图


# =============================================================================
# 扩展功能示例：更多Prophet高级特性
# =============================================================================
print("\n" + "="*50)
print("扩展功能演示开始")
print("="*50)

# ----------------------
# 示例1：乘法季节性模型（适用于季节性幅度随趋势增长的情况）
# ----------------------
print("\n1. 乘法季节性模型示例")
data_multiplicative = data.copy()
# 使季节性幅度随趋势增长（更适合乘法模型）
data_multiplicative['y'] = data_multiplicative['y'] * (1 + 0.0001 * np.arange(len(data_multiplicative)))**2

# 初始化模型
model_multi = Prophet(
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays  # 添加相同的节假日信息
)
model_multi.add_country_holidays(country_name='CN')
model_multi.fit(data_multiplicative)
future_multi = model_multi.make_future_dataframe(periods=365)
forecast_multi = model_multi.predict(future_multi)

fig_multi = model_multi.plot(forecast_multi)
plt.title('乘法季节性模型预测')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.show()

fig_multi_components = model_multi.plot_components(forecast_multi)
plt.show()


# ----------------------
# 示例2：交叉验证和模型评估
# ----------------------
print("\n2. 交叉验证和模型评估")
# 初始训练期为4年，每次预测365天，间隔180天
cv_results = cross_validation(
    model=model,
    initial='1460 days',  # 初始训练期
    period='180 days',    # 评估间隔
    horizon='365 days'    # 预测范围
)

# 计算模型性能指标
metrics = performance_metrics(cv_results)
print("交叉验证性能指标:")
print(metrics[['horizon', 'mse', 'rmse', 'mae', 'mape']].head())

# 可视化交叉验证结果
fig_cv = plot_cross_validation_metric(cv_results, metric='mape')
plt.title('交叉验证MAPE指标')
plt.show()


# ----------------------
# 示例3：异常值检测和处理
# ----------------------
print("\n3. 异常值检测和处理")
# 创建带异常值的数据集
data_outliers = data.copy()
# 随机添加一些异常值
np.random.seed(42)
outlier_indices = np.random.choice(data_outliers.index, size=20, replace=False)
data_outliers.loc[outlier_indices, 'y'] = data_outliers.loc[outlier_indices, 'y'] * 3

# 方法1：使用内置的异常值处理（设置changepoint_prior_scale来降低异常值影响）
model_robust = Prophet(
    changepoint_prior_scale=0.01,  # 降低趋势灵活性，减少异常值影响
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays  # 添加相同的节假日信息
)
model_robust.add_country_holidays(country_name='CN')
model_robust.fit(data_outliers)

future_robust = model_robust.make_future_dataframe(periods=365)
forecast_robust = model_robust.predict(future_robust)

# 可视化原始数据和预测结果
plt.figure(figsize=(12, 6))
plt.scatter(data_outliers['ds'], data_outliers['y'], label='原始数据（含异常值）', alpha=0.6, s=10)
plt.plot(forecast_robust['ds'], forecast_robust['yhat'], 'r-', label='Prophet预测')
plt.fill_between(forecast_robust['ds'], forecast_robust['yhat_lower'], forecast_robust['yhat_upper'], color='r', alpha=0.2)
plt.title('异常值鲁棒性测试')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.legend()
plt.show()


# ----------------------
# 示例4：自定义季节性模式
# ----------------------
print("\n4. 自定义季节性模式")
# 创建带有月度季节性的数据
data_custom = data.copy()

# 初始化模型
model_custom = Prophet(
    yearly_seasonality=False, 
    weekly_seasonality=True,
    holidays=holidays  # 添加相同的节假日信息
)
model_custom.add_country_holidays(country_name='CN')

# 添加月度季节性（周期为30天）
model_custom.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 添加季度季节性（周期为90天）
model_custom.add_seasonality(name='quarterly', period=90, fourier_order=8)

model_custom.fit(data_custom)
future_custom = model_custom.make_future_dataframe(periods=365)
forecast_custom = model_custom.predict(future_custom)

# 可视化组件分解
fig_custom_components = model_custom.plot_components(forecast_custom)
plt.show()


# ----------------------
# 示例5：添加外部协变量
# ----------------------
print("\n5. 添加外部协变量")
# 创建带有外部协变量的数据
data_external = data.copy()
# 模拟营销支出作为外部协变量
data_external['marketing_spend'] = 1000 + 50 * np.sin(np.arange(len(data_external)) / 30) + np.random.normal(0, 100, len(data_external))
# 模拟竞争对手活动
competitor_activity = np.zeros(len(data_external))
competitor_indices = np.random.choice(data_external.index, size=50, replace=False)
competitor_activity[competitor_indices] = 1
data_external['competitor_promotion'] = competitor_activity

# 初始化模型
model_external = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True,
    holidays=holidays  # 添加相同的节假日信息
)
model_external.add_country_holidays(country_name='CN')
# 添加外部协变量
model_external.add_regressor('marketing_spend')
model_external.add_regressor('competitor_promotion')

# 训练模型
model_external.fit(data_external)

# 为预测创建未来的外部协变量
def create_future_covariates(future_df, past_data):
    # 基于历史数据的模式生成未来的营销支出
    last_idx = len(past_data)
    future_df['marketing_spend'] = 1000 + 50 * np.sin((np.arange(len(future_df)) + last_idx) / 30) + np.random.normal(0, 100, len(future_df))
    # 随机生成竞争对手促销活动
    future_df['competitor_promotion'] = np.random.choice([0, 1], size=len(future_df), p=[0.9, 0.1])
    return future_df

# 创建未来数据框并添加协变量
future_external = model_external.make_future_dataframe(periods=365)
future_external = create_future_covariates(future_external, data_external)

# 预测
forecast_external = model_external.predict(future_external)

# 可视化预测结果
fig_external = model_external.plot(forecast_external)
plt.title('带外部协变量的预测')
plt.show()


# ----------------------
# 示例6：参数调优
# ----------------------
print("\n6. 参数调优示例")
# 定义要尝试的参数组合
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
}

# 简单的参数调优过程（在实际应用中可使用网格搜索或贝叶斯优化）
best_params = None
best_mape = float('inf')

print("尝试不同参数组合:")
for cp in param_grid['changepoint_prior_scale']:
    for sp in param_grid['seasonality_prior_scale']:
        try:
            # 训练模型
            model_param = Prophet(
                changepoint_prior_scale=cp,
                seasonality_prior_scale=sp,
                yearly_seasonality=True,
                weekly_seasonality=True,
                holidays=holidays  # 添加相同的节假日信息
            )
            model_param.add_country_holidays(country_name='CN')
            model_param.fit(data)
            
            # 交叉验证
            cv_param = cross_validation(
                model=model_param,
                initial='730 days',
                period='180 days',
                horizon='90 days'
            )
            
            # 计算性能
            metrics_param = performance_metrics(cv_param)
            current_mape = metrics_param['mape'].mean()
            
            print(f"  changepoint_prior_scale={cp}, seasonality_prior_scale={sp} => MAPE={current_mape:.4f}")
            
            # 更新最佳参数
            if current_mape < best_mape:
                best_mape = current_mape
                best_params = {'changepoint_prior_scale': cp, 'seasonality_prior_scale': sp}
        except Exception as e:
            print(f"  参数组合 {cp}, {sp} 失败: {e}")

print(f"\n最佳参数: {best_params}")
print(f"最佳MAPE: {best_mape:.4f}")


print("\n" + "="*50)
print("Prophet扩展功能演示完成")
print("="*50)