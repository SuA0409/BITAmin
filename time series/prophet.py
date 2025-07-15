# Installation
pip install prophet

import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error

# colab 환경에서 실행 시
# from google.colab import drive
# drive.mount('/content/drive')


# 1. data load
data =  pd.read_csv("C:/Users/user/Desktop/BITAmin 15기/1_2025-冬/세션활동/시계열 프로젝트/[02.22] 프로젝트 발표/final/주요국 통화의 대원화환율_06183901.csv")   # data path
# 2. data preprocessing
data['변환'] = pd.to_datetime(data['변환']).dt.normalize()
data['원자료'] = data['원자료'].str.replace(',', '').astype(float)
data.columns = ['ds', 'y']

# 3. train prophet model
m = Prophet()
m.fit(data)

# 4. predict by prophet
future = m.make_future_dataframe(periods=365)
# future.tail()
forecast = m.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

# 5. visualization
plot_plotly(m, forecast)

# Filtering
# Define the date range
# start_date = '2025-02-06'
# end_date = '2026-02-06'
# Filter the DataFrame to include only rows where 'ds' is within the specified range
# filtered_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
# filtered_forecast= filtered_forecast[['ds','yhat','yhat_lower', 'yhat_upper']]
# filtered_forecast.tail()


# prophet 모델의 성능 지표
# 실제 값과 예측 값 추출
y_true = data['y']  # 학습 데이터의 실제 값
y_pred = forecast['yhat'].iloc[:-365]  # 예측값에서 마지막 365일 제외
# MAE 계산
mae = mean_absolute_error(y_true, y_pred)
# RMSE 계산
rmse = mean_squared_error(y_true, y_pred)** 0.5  # RMSE는 MSE의 제곱근
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
# 예측 결과와 원본 데이터프레임을 'ds' 열을 기준으로 병합
forecast_value = forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']]
prophet_prediction = pd.merge(data, forecast_value, on='ds', how='outer')


# 데이터 추출
# period 열 추가
prophet_prediction['period'] = range(1, len(prophet_prediction)+1)
# MAE와 RMSE 값을 모든 행에 넣기
prophet_prediction['MAE'] = mae
prophet_prediction['RMSE'] = rmse
prophet_prediction
# 결과를 엑셀 파일로 저장
output_file = '/content/drive/MyDrive/BITAmin/Project/시계열1조/prophet_prediction.xlsx'
prophet_prediction.to_excel(output_file, index=False)
print(f"엑셀 파일 '{output_file}'이 저장되었습니다.")