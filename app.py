import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from streamlit_option_menu import option_menu
import chardet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 페이지 설정
st.set_page_config(layout="wide")

# css 파일 읽어오기
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# css 적용하기
local_css("style.css")


# 파일 인코딩 감지 함수
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # 파일 포인터를 다시 처음으로
    return result['encoding']

# LSTM 모델용 전처리
def create_lstm_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

# 파일 업로드
st.markdown("<h1 class='title'>AI Health data Monitoring and Prediction System</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    try:
        # CSV 파일 인코딩 감지 및 데이터프레임으로 읽기
        encoding = detect_encoding(uploaded_file)
        df = pd.read_csv(uploaded_file, encoding=encoding)

        # 데이터 확인
        st.write("업로드된 데이터:")
        st.write(df.head())

        # 삭제 기준 설정
        delete_thredholds = {
            "수축기혈압": {"min": 50, "max":200},
            "이완기혈압": {"min": 30, "max": 120},
            "맥박": {"min": 20, "max": 150},
            "체온": {"min": 35, "max": 40},
            "혈당": {"min": 20, "max": 500},
            "호흡": {"min": 5, "max":60},
            "체중": {"min": 10, "max": 200},
        }

        # 환자 선택 기능
        option = st.selectbox("이름 또는 번호로 선택하세요", ["이름", "번호"])
        if option == "이름":
            patients = df["이름"].unique()
        else:
            patients = df["번호"].unique()

        selected_patient = st.selectbox(f"{option}을 선택하세요", patients)

        # 선택한 환자의 데이터 필터링
        if option == "이름":
            patient_data = df[df["이름"] == selected_patient]
        else:
            patient_data = df[df["번호"] == selected_patient]

        # 시계열 예측을 위한 데이터 준비
        patient_data.loc[:, '측정날짜'] = pd.to_datetime(patient_data['측정날짜']) # 원본 데이터 쓰게끔 변경
        patient_data = patient_data.sort_values('측정날짜')

        metrics = ["수축기혈압", "이완기혈압", "맥박", "체온", "혈당", "호흡", "체중"]
        
        # 이상치 기준선 설정
        thresholds = {
            "수축기혈압": {"upper": 140, "lower": 100},
            "이완기혈압": {"upper": 90, "lower": 60},
            "맥박": {"upper": 90, "lower": 50},
            "체온": {"upper": 37, "lower": 36},
        }

        # 예측 모델 선택
        model_option = st.selectbox("예측 모델을 선택하세요", ["Prophet", "LSTM"])

        # 다양한 look_back 값 설정 (설정 기준은 주간, 월간, 계절, 상반하반, 연별 패턴)
        look_back_candidates = [7, 30, 90, 180, 365]

        # Plotly layout 설정 (한글 폰트)
        font_family = "Noto Sans KR"  # 한글 폰트를 사용
        layout = go.Layout(
            font=dict(family=font_family, size=14),
            title_font=dict(size=17),
            xaxis=dict(title_font=dict(size=13), tickfont=dict(size=11)),
            yaxis=dict(title_font=dict(size=13), tickfont=dict(size=11)),
        )

        for metric in metrics:
            st.subheader(f"{metric} 예측")

            # 데이터 전처리
            data = patient_data[["측정날짜", metric]].rename(columns={"측정날짜": "ds", metric: "y"})

            if metric in delete_thredholds:
                limits = delete_thredholds[metric]
                data.loc[(data['y'] < limits["min"]) | (data['y'] > limits["max"]), 'y'] = None

            valid_data = data.dropna(subset=['y'])

            if len(valid_data) > 0:
                if model_option == "Prophet":
                    model = Prophet()
                    model.fit(valid_data)
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=valid_data['ds'], y=valid_data['y'], mode='markers', name='실제', marker=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='예측', line=dict(color='orange')))
                
                elif model_option == "LSTM":
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(valid_data[['y']].values)

                    # look_back 후보 값들 보다 작은 경우 고려
                    look_back_values = [lb for lb in look_back_candidates if lb < len(valid_data)]

                    if not look_back_values:
                        st.write(f"{metric}에 대한 유효한 look_back 값이 없습니다.")
                        continue

                    # 최적값 찾기 위한 셋팅
                    best_look_back = 0
                    best_mse = float("inf")
                    best_predictions = None
                    best_future_dates = None

                    for look_back in look_back_values:
                        X, Y = create_lstm_dataset(scaled_data, look_back)
                        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

                        model = Sequential()
                        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
                        model.add(Dropout(0.2))  # 드롭아웃 추가
                        model.add(LSTM(50, return_sequences=False))
                        model.add(Dropout(0.2))  # 드롭아웃 추가
                        model.add(Dense(25))
                        model.add(Dense(1))

                        model.compile(optimizer='adam', loss='mean_squared_error')

                        # 조기 종료 콜백 추가
                        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                        model.fit(X, Y, batch_size=1, epochs=100, callbacks=[early_stopping], verbose=0)

                        # 미래 예측
                        test_data = scaled_data[-look_back:]
                        future_predictions = []
                        for _ in range(30):  # 30일 예측
                            test_data = np.reshape(test_data, (1, look_back, 1))
                            prediction = model.predict(test_data)
                            future_predictions.append(prediction[0][0])
                            test_data = np.append(test_data[0][1:], prediction)[np.newaxis, :, np.newaxis]

                        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                        future_dates = pd.date_range(start=valid_data['ds'].iloc[-1], periods=30) # closed = 'right' 제거
                        future_df = pd.DataFrame(data={'ds': future_dates, 'yhat': future_predictions.flatten()})

                        mse = mean_squared_error(valid_data['y'][-30:], future_predictions[:30])

                        # 최적값 적용
                        if mse < best_mse:
                            best_mse = mse
                            best_look_back = look_back
                            best_predictions = future_predictions
                            best_future_dates = future_dates

                    st.write(f"Best look_back for {metric}: {best_look_back} with MSE: {best_mse}")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=valid_data['ds'], y=valid_data['y'], mode='markers', name='실제', marker=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=best_future_dates, y=best_predictions.flatten(), mode='lines', name='예측', line=dict(color='orange')))

                if metric in thresholds:
                    fig.add_hline(y=thresholds[metric]["upper"], line=dict(color='green', dash='dash'), name='상한선')
                    fig.add_hline(y=thresholds[metric]["lower"], line=dict(color='green', dash='dash'), name='하한선')
                    outliers = valid_data[(valid_data['y'] > thresholds[metric]["upper"]) | (valid_data['y'] < thresholds[metric]["lower"])]
                    fig.add_trace(go.Scatter(x=outliers['ds'], y=outliers['y'], mode='markers', name='이상치', marker=dict(color='red')))

                fig.update_layout(layout)
                fig.update_layout(title=f"{metric} 예측 그래프", xaxis_title='날짜', yaxis_title=metric)
                st.plotly_chart(fig)

                st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

            else:
                st.write(f"{metric}에 대한 유효한 데이터가 없습니다.")
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
else:
    st.info("CSV 파일을 업로드하세요.")