import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 페이지 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🤖", layout="wide"
)

# 데이터 URL 설정
DATA_URL = "https://raw.githubusercontent.com/hanseongchan9-ops/data/refs/heads/main/stroke.csv"

# 속성 한글 매핑 딕셔너리
feature_mapping = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}


# 데이터 로드
@st.cache_data
def get_raw_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


try:
    df_raw = get_raw_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 페이지 제목
st.title("🤖 뇌졸중 예측 분류 모델 실습")
st.markdown(
    "입력 속성을 선택하고 로지스틱 회귀와 의사결정트리 모델을 학습 및 평가하는 공간입니다."
)

st.divider()

# 1. 속성 선택 UI (기본값: bmi 제외한 넷)
st.subheader("1. 입력 속성 선택")
default_features_kor = ["나이", "평균 혈당", "고혈압", "심장병"]
all_features_kor = list(feature_mapping.values())

selected_features_kor = st.multiselect(
    "학습에 사용할 속성을 골라주세요 (2개 이상 선택):",
    options=all_features_kor,
    default=default_features_kor,
)

# 두 개보다 적게 고르면 안내문 표시 후 중단
if len(selected_features_kor) < 2:
    st.warning("⚠️ 속성을 **2개 이상** 선택해 주세요.")
    st.stop()

selected_features = [
    reverse_feature_mapping[feat] for feat in selected_features_kor
]

st.divider()

# 2. 데이터 분할 규칙 적용 (번호 순 정렬 -> 열 명씩 묶어 앞 3명 테스트, 나머지 7명 훈련)
# id 등을 기준으로 정렬되어 있다고 가정하고 인덱스 순서대로 처리
df_sorted = df_raw.sort_values(by="id").reset_index(drop=True)

train_indices = []
test_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i : i + 10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3])
        train_indices.extend(chunk.index[3:])
    else:
        # 10개가 안 되는 마지막 묶음 처리
        test_indices.extend(chunk.index[: min(3, len(chunk))])
        if len(chunk) > 3:
            train_indices.extend(chunk.index[3:])

train_df = df_sorted.loc[train_indices].copy()
test_df = df_sorted.loc[test_indices].copy()

# 3. 전처리 (bmi 중앙값 처리 등)
# bmi가 선택되었거나, 혹은 결정 경계 계산/중앙값 대체를 위해 훈련용의 bmi 중앙값 계산 필요
bmi_median = train_df["bmi"].median()

for dataset in [train_df, test_df]:
    if "bmi" in dataset.columns:
        dataset["bmi"] = dataset["bmi"].fillna(bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]

X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 4. 모델 학습
# 로지스틱 회귀
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_train_acc = lr_model.score(X_train, y_train)
lr_test_acc = lr_model.score(X_test, y_test)

# 의사결정트리 (질문 3번 제한, 마디당 최소 샘플 5명 미만 분할 금지, 난수 고정)
dt_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_model.fit(X_train, y_train)
dt_train_acc = dt_model.score(X_train, y_train)
dt_test_acc = dt_model.score(X_test, y_test)

# 기준 모델 (다수 클래스로만 답하는 모델)
majority_class = y_train.mode()[0]
baseline_train_acc = (y_train == majority_class).mean()
baseline_test_acc = (y_test == majority_class).mean()

# 5. 정확도 카드 표시 (큰 숫자 카드, 아래에 훈련/테스트 나란히)
st.subheader("2. 모델 성능 비교 (정확도)")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{lr_test_acc * 100:.2f}%",
        help="테스트 데이터 기준 정확도",
    )
    st.caption(
        f"훈련 정확도: {lr_train_acc * 100:.2f}% | 테스트 정확도: {lr_test_acc * 100:.2f}%"
    )

with c2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{dt_test_acc * 100:.2f}%",
        help="테스트 데이터 기준 정확도",
    )
    st.caption(
        f"훈련 정확도: {dt_train_acc * 100:.2f}% | 테스트 정확도: {dt_test_acc * 100:.2f}%"
    )

with c3:
    st.metric(
        label="다수 클래스 기준 모델",
        value=f"{baseline_test_acc * 100:.2f}%",
        help="훈련용에서 많은 쪽으로만 답하는 모델",
    )
    st.caption(
        f"훈련 정확도: {baseline_train_acc * 100:.2f}% | 테스트 정확도: {baseline_test_acc * 100:.2f}%"
    )

st.
