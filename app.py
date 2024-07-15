import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from streamlit_option_menu import option_menu
import chardet
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 페이지 설정
st.set_page_config(layout="wide")

# css 파일 읽어오기
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# css 적용하기
local_css("style.css")

# 한글폰트 적용하기
font_path = 'font/NotoSansKR-VariableFont_wght.ttf'
font_properties = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_properties.get_name()
# fontprop = fm.FontProperties(fname=font_path, size=12)
# plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.titlesize'] = 17  # 그래프 제목 크기
plt.rcParams['axes.labelsize'] = 13  # 축 제목 크기
plt.rcParams['xtick.labelsize'] = 11  # x축 눈금 크기
plt.rcParams['ytick.labelsize'] = 11  # y축 눈금 크기

# 파일 인코딩 감지 함수
def detect_encoding(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # 파일 포인터를 다시 처음으로
    return result['encoding']

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
        patient_data['측정날짜'] = pd.to_datetime(patient_data['측정날짜'])
        patient_data = patient_data.sort_values('측정날짜')

        metrics = ["수축기혈압", "이완기혈압", "맥박", "체온", "호흡"]
        
        # 이상치 기준선 설정
        thresholds = {
            "수축기혈압": {"upper": 140, "lower": 100},
            "이완기혈압": {"upper": 90, "lower": 60},
            "맥박": {"upper": 90, "lower": 50},
            "체온": {"upper": 37, "lower": 36},
        }


        for metric in metrics:
            st.subheader(f"{metric} 예측")
            
            # Prophet 모델을 사용하기 위한 데이터 준비
            data = patient_data[["측정날짜", metric]].rename(columns={"측정날짜": "ds", metric: "y"})

            # 각 컬럼별로 말도 안되는 값 None으로 지정하는 작업
            if metric in delete_thredholds:
                limits = delete_thredholds[metric]
                data.loc[(data['y'] < limits["min"]) | (data['y'] > limits["max"]), 'y'] = None

            # None으로 지정된 값을 포함한 행 제거(어차피 독립적으로 수행되기 때문에 행단위로 삭제해도 됨)
            valid_data = data.dropna(subset=['y'])

            # 유효한 데이터가 있는 경우에만 예측 수행
            if len(valid_data) > 0: #데이터 길이로 판단
                # Prophet 모델 생성 및 학습
                model = Prophet()
                model.fit(data)


                # 미래 데이터프레임 생성 및 예측
                future = model.make_future_dataframe(periods=30)  # 30일 예측
                forecast = model.predict(future)

                # 예측 결과 시각화
                fig, ax = plt.subplots(figsize=(16, 8)) # 가로폭 조정
                ax.plot(data['ds'], data['y'], label='실제', color='blue')
                ax.plot(forecast['ds'], forecast['yhat'], label='예측', color='orange')

                # 이상치 기준선이 존재하는 경우에만 기준선 및 이상치 표시
                if metric in thresholds:
                    # 기준선 표시
                    ax.axhline(y=thresholds[metric]["upper"], color='green', linestyle='--', label='상한선')
                    ax.axhline(y=thresholds[metric]["lower"], color='green', linestyle='--', label='하한선')

                    # 이상치 표시
                    outliers = data[(data['y'] > thresholds[metric]["upper"]) | (data['y'] < thresholds[metric]["lower"])]
                    ax.scatter(outliers['ds'], outliers['y'], color='red', label='이상치')
            
                ax.legend(prop=font_properties, loc='upper right') # 범례박스 고정
                ax.set_title(f"{metric} 예측 그래프", fontproperties=font_properties)
                plt.xticks(fontsize=14)  # x축 눈금 크기 설정
                plt.yticks(fontsize=14)  # y축 눈금 크기 설정
                plt.subplots_adjust(hspace=0.8)  # 그래프 사이에 간격 추가
                st.pyplot(fig)
                # 그래프 사이에 간격 추가
                st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

                # # 모델 성능 평가
                # y_true = valid_data['y'].values
                # y_pred = forecast.loc[forecast['ds'].isin(valid_data['ds']), 'yhat'].values

                # mae = mean_absolute_error(y_true, y_pred)
                # mse = mean_squared_error(y_true, y_pred)
                # rmse = np.sqrt(mse)

                # st.write(f"### {metric} 성능 평가")
                # st.write(f"MAE (평균 절대 오차): {mae}")
                # st.write(f"MSE (평균 제곱 오차): {mse}")
                # st.write(f"RMSE (제곱근 평균 제곱 오차): {rmse}")

            else:
                st.write(f"{metric}에 대한 유효한 데이터가 없습니다.")
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
else:
    st.info("CSV 파일을 업로드하세요.")