# Installation
!git clone https://github.com/DC-research/TEMPO.git
%cd TEMPO
!pip install -r requirements.txt
!pip install omegaconf
!pip install openai tqdm
!pip install timeagi

import os
import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import google.generativeai as genai
import openai
import json
from tqdm import tqdm
from tempo.models.TEMPO import TEMPO
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 1. Load and preprocess exchange rate data
exchange_rate_df = pd.read_csv('/content/주요국 통화의 대원화환율_06183901.csv')   # dataset path
exchange_rate_df['변환'] = pd.to_datetime(exchange_rate_df['변환'])
exchange_rate_df['원자료'] = exchange_rate_df['원자료'].str.replace(',', '').astype(float)
exchange_rate_df = exchange_rate_df.rename(columns={'변환': 'date', '원자료': 'exchange_rate'})
# Ensure the data is sorted
exchange_rate_df = exchange_rate_df.sort_values('date').reset_index(drop=True)


# 2. OpenAI API 키 설정
API_KEY = "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"
# OpenAI API 클라이언트 초기화
client = openai.OpenAI(api_key=API_KEY)


# 3. 뉴스 데이터를 수집할 연도 및 국가 설정
monthly_news = {}
year_start = 2022
year_end = 2024
country = "Korea"
currency1 = "KRW"
currency2 = "USD"

# 4. OpenAI GPT-4 모델을 활용하여 뉴스 데이터 수집
for year in tqdm(range(year_start, year_end + 1), desc="Year Progress"):
    for month in range(1, 13):
        month_name = f"{month:02d}"
        prompt = f"""
        Suppose you are living in {year}. Please summarize the most important news events from {year}-{month_name}
        related to {country} and their impact on the exchange rate between {currency1} and {currency2}.
        Provide a concise summary in 2-3 sentences.
        """

        # API 호출 및 응답 받기
        for attempt in range(3):  # 최대 3번 재시도
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in financial markets."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )

                # 응답 데이터 추출
                news_summary = response.choices[0].message.content.strip()
                break  # 성공하면 루프 종료
            except Exception as e:
                print(f"ERROR: {e}. 재시도 ({attempt + 1}/3)...")
                time.sleep(5)  # 5초 대기 후 재시도
        else:
            news_summary = "Failed to retrieve data after multiple attempts."

        # 데이터 저장
        monthly_news[f"{year}-{month_name}"] = news_summary
        print(f"{year}-{month_name}: {news_summary}")

        time.sleep(5)  # API 요청 간격 유지


# 5. 뉴스 데이터를 JSON 파일로 저장
news_json_path = "news_data_openai.json"
with open(news_json_path, "w", encoding="utf-8") as f:
    json.dump(monthly_news, f, indent=4, ensure_ascii=False)

print(f"\n뉴스 데이터가 '{news_json_path}' 파일로 저장되었습니다.")


# 6. 수집한 데이터를 저장할 디렉토리 설정
# 작업 디렉터리 확인 및 변경
if 'TEMPO' in os.getcwd():
    os.chdir('/content')
    print("Changed working directory to:", os.getcwd())
# 'data' 디렉터리 생성
data_dir = '/content/data'
os.makedirs(data_dir, exist_ok=True)
# 뉴스 데이터 저장 (news.txt)
news_txt_path = f'{data_dir}/news.txt'
with open(news_txt_path, 'w', encoding='utf-8') as f:
    for date, news in monthly_news.items():
        f.write(f"{date}: {news}\n")
print(f"\n뉴스 데이터가 '{news_txt_path}' 파일로 저장되었습니다.")


# 7. 환율 데이터 정규화 및 저장
# Min-Max 정규화 수행
scaler = MinMaxScaler()
exchange_rate_df['scaled_rate'] = scaler.fit_transform(exchange_rate_df[['exchange_rate']])
# 환율 데이터 저장 (exchange_rate.csv)
exchange_rate_save_path = f'{data_dir}/exchange_rate.csv'
exchange_rate_df[['date', 'scaled_rate']].to_csv(exchange_rate_save_path, index=False)
print(f"환율 데이터가 '{exchange_rate_save_path}' 파일로 저장되었습니다.")


# 8. tempo_predict.py 실행
!python tempo_predict.py --time_series_data ../data/exchange_rate.csv --text_data ../data/news.txt --output ../data/predictions.csv
# Min-Max Scaling 확인 (정규화 범위 체크)
print("Scaling Check:")
print(f"Original Min: {exchange_rate_df['exchange_rate'].min()}, Max: {exchange_rate_df['exchange_rate'].max()}")
print(f"Scaled Min: {exchange_rate_df['scaled_rate'].min()}, Max: {exchange_rate_df['scaled_rate'].max()}")


# 9. TEMPO 모델 설치 및 로드
model = TEMPO.load_pretrained_model(
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    repo_id="Melady/TEMPO",
    filename="TEMPO-80M_v1.pth",
    cache_dir="./checkpoints/TEMPO_checkpoints"
)


# 10. TEMPO 모델을 사용한 예측
# 최근 336일 데이터 사용하여 예측 준비
input_data = exchange_rate_df['scaled_rate'].values[-336:]
# 예측 수행
with torch.no_grad():
    predicted_values = model.predict(input_data, pred_length=365)
# 모델이 예측한 값 확인
print("Predicted Values (Before Clipping) - Min/Max Check:")
print(f"Min: {predicted_values.min()}, Max: {predicted_values.max()}")
# 예측값 범위를 0~1 사이로 강제 제한하여 비정상적인 값 방지
predicted_values = np.clip(predicted_values, 0, 1)
# MinMaxScaler를 기존보다 넓은 범위로 적용하여 비정상적인 변환 방지
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(exchange_rate_df[['exchange_rate']])
predicted_exchange_rates = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
# 최근 변동성을 반영한 랜덤 노이즈 추가 (강도를 낮춤)
recent_mean = exchange_rate_df['exchange_rate'].iloc[-30:].mean()
recent_std = exchange_rate_df['exchange_rate'].iloc[-30:].std()
predicted_exchange_rates += np.random.normal(0, recent_std * 0.1, len(predicted_exchange_rates))
# 변동성 범위 조정 (±40%로 축소)
predicted_exchange_rates = np.clip(predicted_exchange_rates,
                                   recent_mean * (1 - 0.4),
                                   recent_mean * (1 + 0.4))
# 최종 예측값과 최소/최대값 확인
print("Final Adjusted Predicted Values (First 5):")
print(predicted_exchange_rates[:5])
print("Predicted Values (Scaled) - Min/Max Check:")
print(f"Min: {predicted_values.min()}, Max: {predicted_values.max()}")
# 예측된 데이터 저장
predictions_df = pd.DataFrame({
    'date': pd.date_range(start=exchange_rate_df['date'].max() + timedelta(days=1), periods=365, freq='D'),
    'predicted_exchange_rate': predicted_exchange_rates
})
# 시각화
plt.figure(figsize=(14, 7))
plt.plot(exchange_rate_df['date'], exchange_rate_df['exchange_rate'], label='Historical Exchange Rate', color='blue')
plt.plot(predictions_df['date'], predictions_df['predicted_exchange_rate'], label='Predicted Exchange Rate', linestyle='--', color='red')
plt.title('Exchange Rate Forecast for 2025')
plt.xlabel('Date')
plt.ylabel('Exchange Rate (KRW/USD)')
plt.legend()
plt.grid(True)
plt.show()