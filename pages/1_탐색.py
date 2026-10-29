import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide",
)

# 데이터 URL 설정
DATA_URL = "https://raw.githubusercontent.com/hanseongchan9-ops/data/refs/heads/main/stroke.csv"


# 데이터 로드 (캐싱 적용)
@st.cache_data
def get_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df = get_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 페이지 제목
st.title("📊 뇌졸중 데이터 탐색")
st.markdown("수집된 뇌졸중 예측 데이터를 다양한 시각화와 통계로 탐색하는 공간입니다.")

st.divider()

# 1. 나이와 평균 혈당의 분포를 히스토그램 두 개로 나란히
st.subheader("1. 나이 및 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30, title="나이(age) 분포", labels={"age": "나이"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# 2. 뇌졸중 유무에 따른 나이와 평균 혈당 상자그림 및 평균값 표
st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
df_stroke_label = df.copy()
df_stroke_label["stroke_str"] = df_stroke_label["stroke"].map(
    {0: "정상 (0)", 1: "뇌졸중 (1)"}
)

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df_stroke_label,
        x="stroke_str",
        y="age",
        color="stroke_str",
        title="뇌졸중 유무별 나이 상자그림",
        labels={"stroke_str": "뇌졸중 여부", "age": "나이"},
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df_stroke_label,
        x="stroke_str",
        y="avg_glucose_level",
        color="stroke_str",
        title="뇌졸중 유무별 평균 혈당 상자그림",
        labels={"stroke_str": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 그룹별 평균값 표
mean_summary = (
    df.groupby("stroke")[["age", "avg_glucose_level"]]
    .mean()
    .reset_index()
)
mean_summary["stroke"] = mean_summary["stroke"].map(
    {0: "정상 (0)", 1: "뇌졸중 (1)"}
)
mean_summary.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.markdown("##### 📌 그룹별 평균값 요약")
st.dataframe(
    mean_summary.set_index("뇌졸중 여부"), use_container_width=True
)

st.divider()

# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("3. 기질적 요인(고혈압, 심장병)에 따른 뇌졸중 비율")
col5, col6 = st.columns(2)

with col5:
    hypertension_grouped = (
        df.groupby("hypertension")["stroke"].mean().reset_index()
    )
    hypertension_grouped["hypertension"] = hypertension_grouped[
        "hypertension"
    ].map({0: "고혈압 없음 (0)", 1: "고혈압 있음 (1)"})
    hypertension_grouped["stroke_pct"] = (
        hypertension_grouped["stroke"] * 100
    )

    fig_hyp = px.bar(
        hypertension_grouped,
        x="hypertension",
        y="stroke_pct",
        text_auto=".2f%",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={"hypertension": "고혈압 여부", "stroke_pct": "뇌졸중 비율 (%)"},
    )
    st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
    heart_grouped = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_grouped["heart_disease"] = heart_grouped["heart_disease"].map(
        {0: "심장병 없음 (0)", 1: "심장병 있음 (1)"}
    )
    heart_grouped["stroke_pct"] = heart_grouped["stroke"] * 100

    fig_heart = px.bar(
        heart_grouped,
        x="heart_disease",
        y="stroke_pct",
        text_auto=".2f%",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={"heart_disease": "심장병 여부", "stroke_pct": "뇌졸중 비율 (%)"},
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# 4. 체질량지수(bmi) 결측치 그룹과 전체 그룹의 뇌졸중 비율 비교
st.subheader("4. BMI 결측치 여부에 따른 뇌졸중 비율 비교")
bmi_null_df = df[df["bmi"].isnull()]
bmi_null_count = len(bmi_null_df)
total_count = len(df)

bmi_null_stroke_ratio = (
    (bmi_null_df["stroke"].mean() * 100) if bmi_null_count > 0 else 0.0
)
total_stroke_ratio = (df["stroke"].mean() * 100) if total_count > 0 else 0.0

comparison_data = {
    "구분": ["BMI 결측치 그룹", "전체 데이터 그룹"],
    "인원 수": [bmi_null_count, total_count],
    "뇌졸중 비율 (%)": [
        round(bmi_null_stroke_ratio, 2),
        round(total_stroke_ratio, 2),
    ],
}
comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, hide_index=True, use_container_width=True)

st.divider()

# 5. 흡연 상태별 사람 수 표
st.subheader("5. 흡연 상태(smoking_status)별 인원 수")
smoking_counts = (
    df["smoking_status"]
    .value_counts()
    .reset_index(name="사람 수")
)
smoking_counts.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking_counts, hide_index=True, use_container_width=True)
